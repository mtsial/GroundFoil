"""Report on a ride-height sweep (sweep_naca4412.sh): the downforce coefficient and the share of
the wing surface with reversed flow against the ride height, with the free-air run as the
reference. Reads the samples.csv of each run folder (<sweep>/<run>/samples.csv), prints the table,
and --plot draws the figure, --csv writes the table.

--model adds the U-Net: each model (a model.pt of train_unet.py) predicts the flow of the runs in
ground effect whose Cl became steady, and Cl comes from the predicted pressure on the 128 x 128
grid (unet_data.lift_weights). It is compared with Cl from the simulated pressure on the same grid,
which is a few percent off OpenFOAM's Cl; with several models, the figure shows their mean and
their range. The free-air run is left out: every training sample has a ground.

On Windows, from the ground_effect folder:
    ..\\.venv\\Scripts\\python.exe sweep_report.py --plot ..\\results\\sweep_naca4412.png --csv ..\\results\\sweep_naca4412.csv
with the two U-Nets trained on all the samples from the free-air model (learning_curve.py):
    ... --model data\\unet\\learning_curve_wall\\f1_pretrained_s0\\model.pt data\\unet\\learning_curve_wall\\f1_pretrained_s1\\model.pt
"""
import argparse
import csv
import glob
import json
import math
import os

BLUE, ORANGE, RED, INK, GRID = "#2a78d6", "#eb6834", "#d03b3b", "#333333", "#d9d9d9"  # categorical slots 1, 2; red as 'failed' in param_scan_report.py
COLUMNS = ["run", "h_over_c", "downforce", "vs_free_air", "Cd", "reversed_flow_pct", "iterations", "converged",
           "cl_range_pct", "downforce_grid", "downforce_unet", "unet_min", "unet_max", "unet_vs_grid_pct",
           "unet_field_error_pct"]


def mean(values):
    return sum(values) / len(values)


def load(folder):
    """One row per run, sorted by ride height, free air last."""
    rows = []
    for path in glob.glob(os.path.join(folder, "*", "samples.csv")):
        run = os.path.basename(os.path.dirname(path))
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                h = float(r["h_over_c"])
                converged = r["converged"] == "1"
                rows.append({"run": run, "h": None if math.isinf(h) else h, "airfoil": r["airfoil"],
                             "alpha": float(r["alpha_deg"]), "U": float(r["U"]), "downforce": -float(r["Cl"]),
                             "Cd": float(r["Cd"]), "converged": converged, "cl_range": 100 * float(r["cl_range"]),
                             "reversed": 100 * float(r["reversed_flow"]), "iterations": int(r["iterations"]),
                             "file": f"{run}/{r['file']}" if converged else f"{run}/not_converged/{r['file']}"})
    rows.sort(key=lambda r: (r["h"] is None, r["h"] or 0.0))
    free = next((r for r in rows if r["h"] is None), None)
    for r in rows:
        r["vs_free_air"] = r["downforce"] / free["downforce"] if free else float("nan")
    return rows


def add_unet(rows, folder, models):
    """Adds to each run in ground effect with a steady Cl: 'downforce_grid', -Cl from the simulated
    pressure on the 128 x 128 grid, and for each model (model.pt files of train_unet.py) in 'unet'
    -Cl from the predicted pressure and in 'unet_field' the error of the predicted fields with the
    output set to 0 in the wing and the ground (%, unet_data.relative_errors)."""
    import torch
    import unet_data
    from train_unet import predict
    from unet import TurbNetG

    settings = []
    for path in models:  # the settings of the training, in summary.json next to the model
        info = os.path.join(os.path.dirname(path), "summary.json")
        if os.path.exists(info):
            with open(info) as f:
                settings.append(json.load(f)["settings"])
        else:
            settings.append({})
    offsets = {s.get("p_offset", "fluid") for s in settings}
    if len(offsets) > 1:
        raise SystemExit("the models were trained with different pressure offsets (--p-offset)")
    use = [r for r in rows if r["h"] is not None and r["converged"]]
    data = unet_data.load(folder, offsets.pop(), files=[r["file"] for r in use])
    inputs = torch.from_numpy(data["inputs"])
    for r, h in zip(use, data["h"]):
        assert abs(r["h"] - h) < 1e-3, (r["file"], h)
        r["unet"], r["unet_field"] = [], []
    for path, s in zip(models, settings):
        net = TurbNetG(channelExponent=s.get("expo", 5))
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        pred = predict(net, inputs)
        cl_pred, cl_cfd = unet_data.lift_errors(pred, data["targets"], data["inputs"])
        field = unet_data.relative_errors(unet_data.zero_solid(pred, data["inputs"]), data["targets"])["all"]
        for i, r in enumerate(use):
            r["downforce_grid"] = -float(cl_cfd[i])
            r["unet"].append(-float(cl_pred[i]))
            r["unet_field"].append(100 * float(field[i]))
        cl_error = 100 * mean([abs(r["unet"][-1] / r["downforce_grid"] - 1) for r in use])
        print(f"U-Net {path}: Cl {cl_error:.2f}% off that of the simulated pressure on average, "
              f"field error {mean([r['unet_field'][-1] for r in use]):.2f}%")


def table(rows):
    unet = any(r.get("unet") for r in rows)
    print(f"{'h/c':>8} {'-Cl':>7} {'x free air':>10} {'Cd':>7} {'reversed':>9} {'its':>5}  "
          + ("-Cl grid  U-Net (range)        vs grid  field err  " if unet else "") + "Cl steady")
    for r in rows:
        h = "free air" if r["h"] is None else f"{r['h']:.3f}"
        steady = "yes" if r["converged"] else f"no (varies by {r['cl_range']:.1f}%)"
        line = (f"{h:>8} {r['downforce']:7.3f} {r['vs_free_air']:10.2f} {r['Cd']:7.4f} {r['reversed']:8.0f}%"
                f" {r['iterations']:5d}  ")
        if unet:
            u = r.get("unet")
            line += (f"{r['downforce_grid']:8.3f}  {mean(u):5.3f} ({min(u):.3f}-{max(u):.3f})"
                     f" {100 * (mean(u) / r['downforce_grid'] - 1):+7.1f}% {mean(r['unet_field']):9.2f}%  " if u else
                     f"{'-':>8s}  {'-':21s} {'-':>8s} {'-':>10s}  ")
        print(line + steady)


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            u = r.get("unet")
            w.writerow([r["run"], "inf" if r["h"] is None else f"{r['h']:.3f}", f"{r['downforce']:.4f}",
                        f"{r['vs_free_air']:.3f}", f"{r['Cd']:.4f}", f"{r['reversed']:.1f}", r["iterations"],
                        int(r["converged"]), f"{r['cl_range']:.3f}"] +
                       ([f"{r['downforce_grid']:.4f}", f"{mean(u):.4f}", f"{min(u):.4f}", f"{max(u):.4f}",
                         f"{100 * (mean(u) / r['downforce_grid'] - 1):.2f}", f"{mean(r['unet_field']):.3f}"]
                        if u else [""] * 6))
    print("saved", path)


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "text.color": INK, "font.size": 9})
    free = next((r for r in rows if r["h"] is None), None)
    ok = [r for r in rows if r["h"] is not None and r["converged"]]
    bad = [r for r in rows if r["h"] is not None and not r["converged"]]
    unet = [r for r in ok if r.get("unet")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    right = 1.05 * max(r["h"] for r in rows if r["h"] is not None)
    handles = {}
    for ax, key, ylabel in zip(axes, ("downforce", "reversed"),
                               ("downforce coefficient, -Cl", "wing surface with reversed flow (%)")):
        handles["cfd"], = ax.plot([r["h"] for r in ok], [r[key] for r in ok], "-o", lw=2, ms=6, color=BLUE,
                                  mec="white", mew=1.2, zorder=3)
        if bad:
            handles["bad"], = ax.plot([r["h"] for r in bad], [r[key] for r in bad], "x", ms=8, mew=2, color=RED)
        ax.set(xlabel="ride height h/c", ylabel=ylabel, xlim=(0, right))
        ax.grid(color=GRID, lw=0.6)
    ax = axes[0]
    labels = {"cfd": "CFD, OpenFOAM's force", "bad": "CFD, Cl not steady (left out)"}
    top = max(r["downforce"] for r in ok)
    if unet:
        h = [r["h"] for r in unet]
        handles["grid"], = ax.plot(h, [r["downforce_grid"] for r in unet], ":", lw=1.6, color=BLUE)
        labels["grid"] = "CFD, from its pressure on the U-Net's grid"
        line, = ax.plot(h, [mean(r["unet"]) for r in unet], "-o", lw=2, ms=6, color=ORANGE, mec="white", mew=1.2,
                        zorder=4)
        n = len(unet[0]["unet"])
        handles["unet"], labels["unet"] = line, "U-Net, from its predicted pressure"
        if n > 1:
            band = ax.fill_between(h, [min(r["unet"]) for r in unet], [max(r["unet"]) for r in unet], color=ORANGE,
                                   alpha=0.25, lw=0)
            handles["unet"] = (band, line)
            labels["unet"] += f" ({n} models: mean and range)"
        top = max(top, max(max(r["unet"]) for r in unet))
    if free:
        handles["free"] = ax.axhline(free["downforce"], color=INK, lw=1, ls="--")
        labels["free"] = "free air (no ground)"
    peak = max(ok, key=lambda r: r["downforce"])
    note = f"most downforce (CFD):\n{peak['downforce']:.2f} at h/c {peak['h']:.3f}"
    if free:
        note += f",\n{peak['vs_free_air']:.2f} times free air"
    ax.annotate(note, (peak["h"], peak["downforce"]), xytext=(0.01, top + 0.06), fontsize=8,
                arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.6, "shrinkB": 4})
    lo = min(r["downforce"] for r in rows)
    ax.set_ylim(lo - 0.1, top + 0.3)
    axes[1].set_ylim(bottom=0)
    order = [k for k in ("cfd", "grid", "bad", "unet", "free") if k in handles]  # the CFD in the first column
    fig.legend([handles[k] for k in order], [labels[k] for k in order], loc="lower center", ncol=2, frameon=False,
               fontsize=8)
    a = rows[0]
    name = a["airfoil"].lower()
    name = (name.upper().replace("NACA", "NACA ") if name.startswith("naca") else
            "Tyrrell 026" if name.startswith("tyrrell026") else a["airfoil"])
    fig.suptitle(f"{name} upside down at {a['alpha']:g} deg and {a['U']:g} m/s "
                 f"(Re {a['U'] * 1e5 / 1e6:g} million), not among the U-Net's airfoils", fontsize=10)
    fig.tight_layout(rect=(0, 0.035 * ((len(order) + 1) // 2) + 0.02, 1, 1))  # room for the legend's rows
    fig.savefig(path, dpi=110)
    print("saved", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sweep", default=os.path.join("data", "sweep_naca4412"), help="folder of the sweep's runs")
    ap.add_argument("--model", nargs="+", help="U-Nets (model.pt of train_unet.py) to compare with the CFD")
    ap.add_argument("--plot", help="draw the figure into this image")
    ap.add_argument("--csv", help="write the table into this file")
    args = ap.parse_args(argv)
    rows = load(args.sweep)
    if not rows:
        raise SystemExit(f"no samples.csv in {args.sweep}/*/")
    if args.model:
        add_unet(rows, args.sweep, args.model)
    table(rows)
    if args.csv:
        write_csv(rows, args.csv)
    if args.plot:
        plot(rows, args.plot)


if __name__ == "__main__":
    main()
