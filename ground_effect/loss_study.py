"""Compare losses and inputs for the downforce: U-Nets trained with the plain L1 loss and with the
additions of train_unet.py (--cl-weight, --wall-weight, --distance), all tested again on the same
validation and test samples with every measure of train_unet.evaluate; then a table, a csv and a
figure.

--plain is a learning_curve.py folder whose runs f1_pretrained_s0 and f1_pretrained_s1 (all the
training samples, from the free-air model) used the plain loss. --loss names the other runs: a
learning_curve.py folder, or a folder with one such folder per loss, made the same way (see the
README for Kaggle), for example
    python learning_curve.py ... --fractions 1 --starts pretrained --wall-weight 10 --out <loss>/wall10
--starts chooses their runs with all the samples: f1_pretrained_s<seed> (from the free-air model,
the default), f1_scratch_s<seed> (from random weights), or both. The samples are those in the plain
runs' errors_val.csv and errors_test.csv, loaded from --data and --test, so that samples added to
these folders later do not change the comparison. The measures (see train_unet.py): the field
error of the original test script ('all'), the same with the output set to 0 inside the wing and
the ground, the pressure error in the air cells next to the wing, the Cl error, and the number of
samples whose Cl is closer to the simulation than with the plain loss (same seed).

From the ground_effect folder (Windows; a few seconds per model on the CPU):
    ..\\.venv\\Scripts\\python.exe loss_study.py --data data\\dataset\\train --test data\\dataset\\test --plain data\\unet\\learning_curve --loss data\\unet\\loss --plot ..\\results\\ground_loss_study.png
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

import unet_data
from train_unet import evaluate, predict
from unet import TurbNetG

RUN = "f1_{}_s{}"  # the run with all the samples, from a start (pretrained or scratch) and a seed
COLOR, INK, GRID = "#eb6834", "#333333", "#d9d9d9"  # the colour of 'from the free-air model' in learning_curve.py
PANELS = (("cl", "downforce (Cl) error"), ("p_near_wing", "pressure error next to the wing"),
          ("all_zeroed", "field error, output 0 in the solid"))


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="folder with the training samples (the validation samples are among them)")
    ap.add_argument("--test", required=True, help="folder with the test samples")
    ap.add_argument("--plain", required=True,
                    help="learning_curve.py folder with the runs f1_pretrained_s<seed> of the plain L1 loss")
    ap.add_argument("--loss", required=True, nargs="+",
                    help="learning_curve.py folders of the other runs, or folders with one such folder per loss")
    ap.add_argument("--starts", nargs="+", choices=("pretrained", "scratch"), default=["pretrained"],
                    help="the runs of --loss from the free-air model (pretrained, the default), from random "
                         "weights (scratch), or both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="seeds of the runs (default 0 1)")
    ap.add_argument("--csv", help="one line per model and sample set (default <first --loss>/loss_study.csv)")
    ap.add_argument("--plot", help="draw the figure into this image")
    ap.add_argument("--title", default="Losses for the downforce", help="the start of the figure's title")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for torch (default 2)")
    ap.add_argument("--airfoil-dirs", nargs="+", default=list(unet_data.AIRFOIL_DIRS),
                    help="folders with the airfoil files, for models trained with --distance")
    return ap.parse_args(argv)


def label(settings, start="pretrained"):
    """A short name for the loss, the inputs and the start of a run, from its settings."""
    parts = [f"Cl term {settings['cl_weight']:g}"] if settings.get("cl_weight") else []
    if settings.get("wall_weight"):
        cells = settings["wall_cells"]
        parts.append(f"wall x{1 + settings['wall_weight']:g}, {cells} cell{'s' if cells != 1 else ''}")
    name = " + ".join(parts) or "plain L1"
    name += " + distance input" if settings.get("distance") else ""
    return name + (", from random weights" if start == "scratch" else "")


def losses(args):
    """(name, folder, start, settings) of the plain loss and of each run in --loss: the plain loss
    first, then the Cl terms by weight, then the wall weightings; within each, without the distance
    input first, and from the free-air model before from random weights."""
    folders = [("plain", args.plain, ["pretrained"])]
    for path in args.loss:
        if any(os.path.exists(os.path.join(path, RUN.format(start, args.seeds[0]))) for start in args.starts):
            folders.append((os.path.basename(os.path.normpath(path)), path, args.starts))  # a learning_curve.py folder
        else:
            folders += [(n, os.path.join(path, n), args.starts) for n in sorted(os.listdir(path))]
    found = []
    for name, folder, starts in folders:
        for start in starts:
            path = os.path.join(folder, RUN.format(start, args.seeds[0]), "summary.json")
            if os.path.exists(path):
                with open(path) as f:
                    found.append((name, folder, start, json.load(f)["settings"]))
    order = lambda x: (x[0] != "plain", bool(x[3].get("wall_weight")), x[3].get("cl_weight", 0),
                       x[3].get("wall_weight", 0), x[3].get("wall_cells", 0), bool(x[3].get("distance")),
                       x[2] == "scratch")
    return sorted(found, key=order)


def samples(args, p_offset, distance=False):
    """The validation and test samples of the plain runs (their errors_<set>.csv), normalised;
    with distance=True, input 1 is the signed distance (unet_data.load)."""
    sets = {}
    for name, folder in (("val", args.data), ("test", args.test)):
        with open(os.path.join(args.plain, RUN.format("pretrained", args.seeds[0]), f"errors_{name}.csv")) as f:
            files = [r["file"] for r in csv.DictReader(f)]
        sets[name] = unet_data.load(folder, p_offset, files=files, distance=distance, airfoil_dirs=args.airfoil_dirs)
    return sets


def test_all(args):
    """One row per loss, start, seed and sample set, with every measure."""
    found = losses(args)
    p_offset = found[0][3].get("p_offset", "fluid")
    loaded = {}  # the samples, without and with the distance input, loaded when first needed
    rows, plain_err = [], {}
    for name, folder, start, settings in found:
        distance = bool(settings.get("distance"))
        if distance not in loaded:
            loaded[distance] = samples(args, p_offset, distance)
        sets = loaded[distance]
        for seed in args.seeds:
            run = os.path.join(folder, RUN.format(start, seed))
            if not os.path.exists(os.path.join(run, "model.pt")):
                print(f"{run}: no model.pt, left out")
                continue
            with open(os.path.join(run, "summary.json")) as f:
                n_train = json.load(f)["n_train"]
            net = TurbNetG(channelExponent=settings.get("expo", 5))
            net.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location="cpu", weights_only=True))
            for set_name, d in sets.items():
                s, _, cl_pred, cl_cfd = evaluate(predict(net, torch.from_numpy(d["inputs"])), d)
                err = np.abs(cl_pred - cl_cfd)
                plain = plain_err.setdefault((set_name, seed), err)  # the plain loss comes first
                rows.append({"loss": name, "start": start, "label": label(settings, start), "seed": seed,
                             "set": set_name, "n": s["n"],
                             "n_train": n_train, "p": s["p"], "v": s["v"], "all": s["all"],
                             "p_zeroed": s["zeroed"]["p"], "v_zeroed": s["zeroed"]["v"],
                             "all_zeroed": s["zeroed"]["all"], "p_near_wing": s["p_near_wing"],
                             "cl": s["cl"]["rel"], "cl_mae": s["cl"]["mae"], "cl_median": s["cl"]["median_rel"],
                             "cl_r2": s["cl"]["r2"], "cl_bias": float(np.mean(cl_pred - cl_cfd)),
                             "cl_closer": int((err < plain).sum())})
        print(f"tested {name}: {label(settings, start)}", flush=True)
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in r.items()})
    print("saved", path)


def by_loss(rows):
    """{(loss, start): {'label': ..., 'val': {measure: [values of the seeds]}, 'test': {...}}}, in the
    order of rows."""
    out = {}
    for r in rows:
        m = out.setdefault((r["loss"], r["start"]), {"label": r["label"]}).setdefault(r["set"], {})
        for k, v in r.items():
            if k not in ("loss", "start", "label", "set", "seed"):
                m.setdefault(k, []).append(v)
    return out


def table(rows, which):
    print(f"\n{which} samples (%, mean over the seeds; '0 in solid': 'all' with the output set to 0 inside the wing "
          f"and the ground; 'p at wing': pressure in the air cells next to the wing; 'closer': samples whose Cl is "
          f"closer to the simulation than with the plain loss, each seed):")
    width = max(24, *(len(m["label"]) for m in by_loss(rows).values()))
    print(f"  {'loss':{width}s} {'p':>6s} {'v':>5s} {'all':>5s} {'0 in solid':>10s} {'p at wing':>9s} {'Cl':>6s} {'R2':>5s}  closer")
    for (name, _), m in by_loss(rows).items():
        s = m[which]
        avg = lambda k: np.mean(s[k])
        closer = "-" if name == "plain" else "/".join(str(c) for c in s["cl_closer"]) + f" of {s['n'][0]}"
        print(f"  {m['label']:{width}s} {100 * avg('p'):6.2f} {100 * avg('v'):5.2f} {100 * avg('all'):5.2f} "
              f"{100 * avg('all_zeroed'):10.2f} {100 * avg('p_near_wing'):9.2f} {100 * avg('cl'):6.2f} "
              f"{avg('cl_r2'):5.3f}  {closer}")


def wrap(text, width=24):
    """A label on several lines: broken before ' + ' and before ', from', where a line would be longer
    than width characters."""
    chunks = text.replace(" + ", "\n+ ").replace(", from ", ",\nfrom ").split("\n")
    lines = chunks[:1]
    for c in chunks[1:]:
        if len(lines[-1]) + 1 + len(c) <= width:
            lines[-1] += " " + c
        else:
            lines.append(c)
    return "\n".join(lines)


def plot(rows, path, heading="Losses for the downforce"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    m = by_loss(rows)
    names = list(m)
    best = min(names, key=lambda n: np.mean(m[n]["val"]["cl"]))  # the choice, on the validation samples
    group = lambda n: (m[n]["label"].split()[0], "distance input" in m[n]["label"])
    labels = [wrap(m[n]["label"].replace(" x", " ×")) for n in names]
    row = 0.36 + 0.12 * (max(t.count("\n") for t in labels))  # inches per row: more for labels on several lines
    plt.rcParams.update({"axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "text.color": INK, "font.size": 9})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(10, row * len(names) + 1.6), sharey=True)
    for ax, (key, title) in zip(axes, PANELS):
        top = 0
        for which, shift, filled in (("val", -0.15, False), ("test", 0.15, True)):
            for i, name in enumerate(names):
                v = 100 * np.array(m[name][which][key])
                top = max(top, v.max())
                ax.plot([v.min(), v.max()], [i + shift] * 2, color=COLOR, lw=1.2, solid_capstyle="round")
                ax.plot(v.mean(), i + shift, "o", ms=5.5, color=COLOR, mfc=COLOR if filled else "white", mew=1.3)
        ax.axvline(100 * np.mean(m[names[0]]["test"][key]), color=INK, lw=0.8, ls="--")
        for i in range(1, len(names)):  # lines between the plain loss, the Cl terms, the wall weightings and the inputs
            if group(names[i]) != group(names[i - 1]):
                ax.axhline(i - 0.5, color=GRID, lw=0.8)
        ax.set_xlim(0, 1.08 * top)
        ax.set(title=title, xlabel="%")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(names)), labels)
    axes[0].set_ylim(len(names) - 0.5, -0.5)  # the plain loss at the top
    for tick, name in zip(axes[0].get_yticklabels(), names):
        tick.set_fontweight("bold" if name == best else "normal")
    handles = [Line2D([], [], marker="o", ls="none", color=COLOR, mfc="white", mew=1.3, label="validation samples"),
               Line2D([], [], marker="o", ls="none", color=COLOR, mew=1.3, label="test samples (airfoils not trained on)"),
               Line2D([], [], color=INK, lw=0.8, ls="--", label="plain L1 loss, test samples")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    start = (" from the free-air model" if all(r["start"] == "pretrained" for r in rows) else
             ", from the free-air model unless marked")
    fig.suptitle(f"{heading}: U-Nets trained on {rows[0]['n_train']} samples{start} "
                 f"(mean and range of {len(m[names[0]]['test']['cl'])} seeds)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=110)
    print("saved", path)


def main(argv=None):
    args = parse(argv)
    torch.set_num_threads(args.threads)
    rows = test_all(args)
    if not rows:
        raise SystemExit("no runs found")
    write_csv(rows, args.csv or os.path.join(args.loss[0], "loss_study.csv"))
    for which in ("test", "val"):
        table(rows, which)
    if args.plot:
        plot(rows, args.plot, args.title)


if __name__ == "__main__":
    main()
