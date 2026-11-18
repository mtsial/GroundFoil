"""A comparison model for the downforce: a small fully connected network (MLP) that predicts Cl
directly from the airfoil's shape, the ride height, the angle of attack and the speed, without the
flow field. Does predicting the whole flow with the U-Net cost accuracy in the downforce?

The samples are the U-Net's: those in --data (or only those listed in --files), split into
training and validation samples as train_unet.py splits them (unet_data.is_validation), and those
in --test. By default the network learns Cl from the simulated pressure on the U-Net's 128 x 128
grid (unet_data.lift), the Cl that the U-Net's is compared with, so the errors compare directly;
--target openfoam makes it learn OpenFOAM's own Cl instead.

The network sees, for each sample:
- the shape, from the airfoil file the simulation used (unet_data.AIRFOIL_DIRS): the camber line
  and the thickness at --stations points along the chord, closer together at the leading and the
  trailing edge;
- the ride height h/c (the gap between the lowest point of the wing and the ground, measured on
  the outline) and its logarithm, the angle of attack and the speed.
Inputs and Cl are scaled to mean 0 and standard deviation 1 over the training samples. Training:
AdamW on the mean absolute error of Cl in mini-batches of --batch samples, the learning rate
falling along a cosine to 1/100 of --lr; the weights with the lowest Cl error on the validation
samples (checked every 10 epochs) are kept.

--search tries every combination of --widths, --depths, --weight-decays and --lrs, each with every
seed, and trains the final models with the one whose mean validation error is the lowest; the test
samples play no part in it. --sweep adds the ride-height sweeps (folders of sweep_naca4412.sh and
sweep_tyrrell026.sh), and --compare U-Net runs (folders with errors_test.csv and errors_val.csv of
train_unet.py) to compare with, sample by sample.

Writes into --out: search_runs.csv and search.csv (with --search); s<seed>/model.pt,
errors_val.csv and errors_test.csv (file, airfoil, h_over_c, alpha_deg, U, clipped, cl_cfd = the Cl
learnt, cl_grid, cl_openfoam, cl_pred) and summary.json (settings, input scales, errors);
sweep_<name>.csv; compare.json (with --compare).

On a GPU (--device auto takes one if there is one; with --jobs, the models are spread over all the
GPUs) a model takes well under a minute; on one CPU thread 2-15 minutes, by its size. --load takes
the models already in --out (trained elsewhere, e.g. on Kaggle) instead of training them, checks
that they give the same Cl here, and adds --sweep and --compare, which need files that stay local.

On Kaggle (see the README), then on Windows, from the ground_effect folder:
    !python baseline_cl.py --data {data}/train --test {data}/test --airfoil-dirs {smooth} {test_smooth} --search --jobs 4 --out /kaggle/working/baseline_cl
    ..\\.venv\\Scripts\\python.exe baseline_cl.py --data data\\dataset\\train --files data\\unet\\baseline_cl\\train_files.txt --test data\\dataset\\test --load --out data\\unet\\baseline_cl --sweep data\\sweep_naca4412 data\\sweep_tyrrell026 --compare data\\unet\\distance\\f1_scratch_s0 data\\unet\\distance\\f1_scratch_s1
"""
import argparse
import csv
import itertools
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

import unet_data

H_BINS = (0.05, 0.08, 0.135, 0.22, 0.365, 0.601)  # bins of h/c for the comparison, as in the docs
DATA = {}  # the scaled samples, in each process of --jobs


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="folder with the training samples (the validation samples are among them)")
    ap.add_argument("--files", help="text file with the samples of --data to use, one path per line as in 'file' "
                                    "(default: all the samples in --data)")
    ap.add_argument("--test", required=True, help="folder with the test samples")
    ap.add_argument("--out", required=True, help="folder for the results")
    ap.add_argument("--val-percent", type=float, default=10, help="percent of the samples kept for validation (default 10, as train_unet.py)")
    ap.add_argument("--target", choices=("grid", "openfoam"), default="grid",
                    help="the Cl to learn: from the simulated pressure on the U-Net's grid (default) or OpenFOAM's")
    ap.add_argument("--stations", type=int, default=32, help="points along the chord for the camber and the thickness (default 32)")
    ap.add_argument("--width", type=int, default=256, help="neurons per hidden layer (default 256)")
    ap.add_argument("--depth", type=int, default=3, help="hidden layers (default 3)")
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default 1e-4)")
    ap.add_argument("--lr", type=float, default=1e-3, help="starting learning rate (default 1e-3)")
    ap.add_argument("--epochs", type=int, default=1000, help="passes over the training samples (default 1000)")
    ap.add_argument("--batch", type=int, default=128, help="samples per step (default 128)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="seeds of the final models (default 0 1)")
    ap.add_argument("--search", action="store_true", help="choose width, depth, weight decay and learning rate on the validation samples")
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--weight-decays", type=float, nargs="+", default=[0, 1e-4, 1e-3, 1e-2])
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--jobs", type=int, default=1, help="models trained at the same time during --search (default 1)")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads for torch in each process (default 1)")
    ap.add_argument("--device", default="auto", help="cpu, cuda, or auto (default: a GPU if there is one)")
    ap.add_argument("--load", action="store_true", help="use the models already in --out instead of training them")
    ap.add_argument("--sweep", nargs="*", default=[], help="folders of ride-height sweeps to predict")
    ap.add_argument("--compare", nargs="*", default=[], help="U-Net run folders to compare with, sample by sample")
    ap.add_argument("--airfoil-dirs", nargs="+", default=list(unet_data.AIRFOIL_DIRS), help="folders with the airfoil files")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- inputs and Cl of the samples

def stations(n):
    """n points along the chord (0 to 1, ends left out), closer together at both ends."""
    return 0.5 * (1 - np.cos(np.pi * np.arange(1, n + 1) / (n + 1)))


def airfoil_points(name, airfoil_dirs):
    """The points of the airfoil as the simulation read them (see unet_data.wing_outline)."""
    import groundGen  # numpy and the standard library only
    path = next((p for p in (os.path.join(k, name + ".dat") for k in airfoil_dirs) if os.path.isfile(p)), name)
    try:
        return groundGen.load_airfoil(path)
    except OSError:
        raise SystemExit(f"airfoil file {name}.dat not found in {list(airfoil_dirs)} (see --airfoil-dirs)")


def shape_features(points, n):
    """The camber line and the thickness (in chords) at n stations along the chord of an airfoil
    (its points as in the file): 2n values. The outline is split at the leading edge (its
    smallest x); the side with the larger mean y is the upper one, whatever the order in the file."""
    le = int(np.argmin(points[:, 0]))
    sides = sorted([points[:le + 1], points[le:]], key=lambda s: s[:, 1].mean(), reverse=True)
    x0, x1 = points[:, 0].min(), points[:, 0].max()
    xs = x0 + (x1 - x0) * stations(n)
    upper, lower = (np.interp(xs, *s[np.argsort(s[:, 0], kind="stable")].T) for s in sides)
    return np.concatenate([(upper + lower) / 2, upper - lower])


def record(folder, rel, airfoil_dirs, n, shapes):
    """The network's inputs and the Cl of one sample (rel: its path in folder, as in 'file')."""
    with np.load(os.path.join(folder, *rel.split("/"))) as d:
        name = str(d["airfoil"])
        if name not in shapes:
            shapes[name] = shape_features(airfoil_points(name, airfoil_dirs), n)
        outline, ground_y = unet_data.wing_outline(d, airfoil_dirs)
        inputs, targets, _ = unet_data.normalise(d["a"])
        p = targets[0].astype(np.float64) * unet_data.MAX_TARGETS[0]  # p / U^2, as unet_data.lift_errors
        return {"file": rel, "airfoil": name, "h": float(outline[:, 1].min() - ground_y), "h_file": float(d["h"]),
                "alpha": float(d["alpha"]), "U": float(d["U"]), "clipped": bool(d["clipped"]),
                "cl_grid": float(unet_data.lift(p, inputs[2] > 0.5)), "cl_openfoam": float(d["Cl"]),
                "shape": shapes[name]}


def features(recs):
    """The inputs of the network (N x (2 stations + 4)), unscaled."""
    h = np.array([r["h"] for r in recs])
    cond = np.column_stack([h, np.log(h), [r["alpha"] for r in recs], [r["U"] for r in recs]])
    return np.column_stack([np.array([r["shape"] for r in recs]), cond])


def load_sets(args):
    """The records of the training, validation and test samples, and of each sweep's runs in
    ground effect whose Cl became steady."""
    if args.files:
        with open(args.files) as f:
            files = [line.strip() for line in f if line.strip()]
    else:
        files = [os.path.relpath(f, args.data).replace(os.sep, "/") for f in unet_data.sample_files(args.data)]
    test = [os.path.relpath(f, args.test).replace(os.sep, "/") for f in unet_data.sample_files(args.test)]
    shapes = {}
    rec = lambda folder, rel: record(folder, rel, args.airfoil_dirs, args.stations, shapes)
    all_train = [rec(args.data, f) for f in files]
    sets = {"train": [r for r in all_train if not unet_data.is_validation(r["file"], args.val_percent)],
            "val": [r for r in all_train if unet_data.is_validation(r["file"], args.val_percent)],
            "test": [rec(args.test, f) for f in test]}
    import sweep_report
    for folder in args.sweep:
        runs = [r for r in sweep_report.load(folder) if r["h"] is not None and r["converged"]]
        sets["sweep:" + folder] = [dict(rec(folder, r["file"]), run=r["run"], h_run=r["h"]) for r in runs]
    return sets


def scale(sets, target):
    """Tensors of the scaled inputs and Cl of each set, and the scales (from the training samples)."""
    x_train = features(sets["train"])
    mean, std = x_train.mean(axis=0), np.maximum(x_train.std(axis=0), 1e-12)
    cl_train = np.array([r["cl_" + target] for r in sets["train"]])
    out = {"scale": {"x_mean": mean, "x_std": std, "y_mean": float(cl_train.mean()), "y_std": float(cl_train.std())}}
    for name, recs in sets.items():
        cl = np.array([r["cl_" + target] for r in recs])
        out[name] = {"x": torch.tensor((features(recs) - mean) / std, dtype=torch.float32), "cl": cl,
                     "y": torch.tensor((cl - out["scale"]["y_mean"]) / out["scale"]["y_std"], dtype=torch.float32)}
    return out


# ---------------------------------------------------------------- the network

def make_net(n_in, width, depth, n_out=1):
    layers, k = [], n_in
    for _ in range(depth):
        layers += [torch.nn.Linear(k, width), torch.nn.SiLU()]
        k = width
    return torch.nn.Sequential(*layers, torch.nn.Linear(k, n_out))


def predict(net, x, s):
    """Cl predicted for the scaled inputs x (on the network's device)."""
    net.eval()
    with torch.no_grad():
        return net(x)[:, 0].double().cpu().numpy() * s["y_std"] + s["y_mean"]


def rel_error(cl_pred, cl):
    """mean |error| / mean |Cl|, as unet_data.lift_summary's 'rel'."""
    return float(np.abs(cl_pred - cl).mean() / np.abs(cl).mean())


def fit(data, width, depth, weight_decay, lr, seed, epochs, batch, device="cpu"):
    """One network, trained on device; returns it (on the CPU), its validation Cl error and the
    epoch of its weights."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, y, x_val = (t.to(device) for t in (data["train"]["x"], data["train"]["y"], data["val"]["x"]))
    net = make_net(x.shape[1], width, depth).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * math.ceil(len(x) / batch), eta_min=lr / 100)
    best = (math.inf, None, 0)
    for epoch in range(1, epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(len(x))).to(device)
        for i in range(0, len(x), batch):
            idx = order[i:i + batch]
            loss = (net(x[idx])[:, 0] - y[idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if epoch % 10 == 0 or epoch == epochs:
            err = rel_error(predict(net, x_val, data["scale"]), data["val"]["cl"])
            if err < best[0]:
                best = (err, {k: v.detach().clone() for k, v in net.state_dict().items()}, epoch)
    net.load_state_dict(best[1])
    return net.cpu(), best[0], best[2]


def devices(name):
    """The devices to train on: all the GPUs for cuda (or auto, if there are any), else the CPU."""
    if name in ("cuda", "auto") and torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if name == "cuda":
        raise SystemExit("--device cuda: no GPU found")
    return [name if name != "auto" else "cpu"]


def _init(data, threads, device_list, counter=None):
    """Each process of --jobs: the samples, and a device (the GPUs in turn)."""
    torch.set_num_threads(threads)
    DATA.update(data)
    k = 0
    if counter is not None:
        with counter.get_lock():
            k = counter.value
            counter.value += 1
    DATA["device"] = device_list[k % len(device_list)]


def _search_one(job):
    (width, depth, weight_decay, lr), seed, epochs, batch = job
    start = time.time()
    _, err, epoch = fit(DATA, width, depth, weight_decay, lr, seed, epochs, batch, DATA["device"])
    return (width, depth, weight_decay, lr), seed, err, epoch, time.time() - start


SEARCH_COLUMNS = ["width", "depth", "weight_decay", "lr", "seed", "epochs", "batch", "val_cl_error", "epoch", "seconds"]


def search(args, data):
    """The combination with the lowest mean validation error over the seeds; writes search.csv.
    Each model's result goes into search_runs.csv as soon as it is trained, and models already
    there (same settings, seed, epochs and batch) are not trained again: a stopped search goes on
    where it stopped."""
    combos = list(itertools.product(args.widths, args.depths, args.weight_decays, args.lrs))
    key = lambda combo, seed: (*combo, seed, args.epochs, args.batch)
    path = os.path.join(args.out, "search_runs.csv")
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                k = (int(r["width"]), int(r["depth"]), float(r["weight_decay"]), float(r["lr"]), int(r["seed"]),
                     int(r["epochs"]), int(r["batch"]))
                done[k] = (float(r["val_cl_error"]), int(r["epoch"]), float(r["seconds"]))
    jobs = [(c, s, args.epochs, args.batch) for c in combos for s in args.seeds if key(c, s) not in done]
    print(f"search: {len(combos)} combinations x {len(args.seeds)} seeds; {len(jobs)} models to train "
          f"({len(combos) * len(args.seeds) - len(jobs)} trained before), {args.jobs} at a time", flush=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(SEARCH_COLUMNS)

        def save(k, result):
            combo, seed, err, epoch, seconds = result
            done[key(combo, seed)] = (err, epoch, seconds)
            w.writerow([*combo, seed, args.epochs, args.batch, err, epoch, round(seconds, 1)])
            f.flush()
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} models", flush=True)

        device_list = devices(args.device)
        if args.jobs == 1:  # no extra process: least memory
            _init(data, args.threads, device_list)
            for k, job in enumerate(jobs, 1):
                save(k, _search_one(job))
        else:  # new processes started afresh ('spawn'), which a GPU needs
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(args.jobs, mp_context=ctx, initializer=_init,
                                     initargs=(data, args.threads, device_list, ctx.Value("i", 0))) as pool:
                for k, fut in enumerate(as_completed([pool.submit(_search_one, j) for j in jobs]), 1):
                    save(k, fut.result())
    results = {c: [(s, *done[key(c, s)]) for s in args.seeds] for c in combos}
    rows = []
    for combo, res in results.items():
        rows.append({"width": combo[0], "depth": combo[1], "weight_decay": combo[2], "lr": combo[3],
                     "val_cl_error": float(np.mean([r[1] for r in res])),
                     **{f"val_cl_error_s{r[0]}": r[1] for r in res}, **{f"epoch_s{r[0]}": r[2] for r in res},
                     "seconds": float(np.mean([r[3] for r in res]))})
    rows.sort(key=lambda r: r["val_cl_error"])
    with open(os.path.join(args.out, "search.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("best on the validation samples (mean of the seeds):")
    for r in rows[:5]:
        print(f"  width {r['width']}, depth {r['depth']}, weight decay {r['weight_decay']:g}, lr {r['lr']:g}: "
              f"{100 * r['val_cl_error']:.2f}%")
    return rows[0]


# ---------------------------------------------------------------- results

def write_errors(path, recs, cl_pred, target):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "airfoil", "h_over_c", "alpha_deg", "U", "clipped", "cl_cfd", "cl_grid", "cl_openfoam", "cl_pred"])
        for r, c in zip(recs, cl_pred):
            w.writerow([r["file"], r["airfoil"], f"{r['h_file']:.4f}", f"{r['alpha']:.4f}", f"{r['U']:.3f}", int(r["clipped"]),
                        f"{r['cl_' + target]:.5f}", f"{r['cl_grid']:.5f}", f"{r['cl_openfoam']:.5f}", f"{c:.5f}"])


def sweep_table(name, recs, preds, target, out):
    """-Cl of each run: simulated (OpenFOAM's and on the grid) and from the networks (mean, range)."""
    p = np.array(preds)  # seeds x runs
    rows = []
    print(f"\n{name}: -Cl against the ride height (networks: mean and range of {len(preds)})")
    print(f"  {'h/c':>6s} {'OpenFOAM':>9s} {'grid':>7s} {'network':>8s} {'(range)':>15s} {'vs ' + target:>8s}")
    for i, r in enumerate(recs):
        ref = r["cl_" + target]
        rows.append({"run": r["run"], "h_over_c": r["h_run"], "downforce": -r["cl_openfoam"], "downforce_grid": -r["cl_grid"],
                     "downforce_net": -p[:, i].mean(), "net_min": -p[:, i].max(), "net_max": -p[:, i].min(),
                     f"net_vs_{target}_pct": 100 * (p[:, i].mean() / ref - 1)})
        print(f"  {r['h_run']:6.3f} {-r['cl_openfoam']:9.3f} {-r['cl_grid']:7.3f} {-p[:, i].mean():8.3f} "
              f"({-p[:, i].max():.3f}-{-p[:, i].min():.3f}) {100 * (p[:, i].mean() / ref - 1):+7.1f}%")
    dev = [r[f"net_vs_{target}_pct"] for r in rows]
    print(f"  mean {np.mean(dev):+.1f}%, mean |.| {np.mean(np.abs(dev)):.1f}%")
    base = os.path.basename(os.path.normpath(name))  # e.g. sweep_naca4412
    with open(os.path.join(out, (base if base.startswith("sweep") else "sweep_" + base) + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def compare(args, sets, preds):
    """Sample by sample against the U-Net runs: |dCl| as the mean over the networks' seeds and over
    the U-Net runs; Cl errors, samples closer, Wilcoxon signed-rank test, and by bins of h/c.

    With --target grid both are compared with Cl from the simulated pressure on the grid. With
    --target openfoam both are compared with OpenFOAM's Cl: the U-Net's Cl (from its pressure on the
    grid) is first mapped to OpenFOAM's by the straight line that fits OpenFOAM's Cl against the
    grid's best over the training samples, which removes the grid's average offset (about 3%) but
    not its scatter."""
    from scipy import stats
    train = sets["train"]
    line = np.polyfit([r["cl_grid"] for r in train], [r["cl_openfoam"] for r in train], 1)
    to_target = (lambda c: c) if args.target == "grid" else (lambda c: np.polyval(line, c))
    out = {"target": args.target, "grid_to_openfoam_line": [float(v) for v in line]}
    if args.target == "openfoam":
        print(f"\nOpenFOAM's Cl = {line[0]:.4f} x the grid's {line[1]:+.4f} (fit over the training samples)")
    for name in ("test", "val"):
        unet, unet_cl = {}, {}
        recs = sets[name]
        for run in args.compare:
            with open(os.path.join(run, f"errors_{name}.csv")) as f:
                for r in csv.DictReader(f):
                    unet.setdefault(r["file"], []).append(float(r["cl_pred"]))
                    unet_cl[r["file"]] = float(r["cl_cfd"])
        missing = [r["file"] for r in recs if r["file"] not in unet]
        if missing:
            raise SystemExit(f"{len(missing)} {name} samples missing in the U-Net runs, e.g. {missing[0]}")
        off = max(abs(unet_cl[r["file"]] - r["cl_grid"]) for r in recs)
        if off > 1e-4:  # the csv files keep 5 decimals
            raise SystemExit(f"{name}: the U-Net runs' Cl from the simulated pressure differs from ours by up to {off:.1e}")
        cl = np.array([r["cl_" + args.target] for r in recs])
        e_net = np.mean([np.abs(p - cl) for p in preds[name]], axis=0)
        e_unet = np.array([np.mean(np.abs(to_target(np.array(unet[r["file"]])) - c)) for r, c in zip(recs, cl)])
        h = np.array([r["h"] for r in recs])
        w = stats.wilcoxon(e_net, e_unet)
        res = {"n": len(recs), "cl_error_net": float(e_net.mean() / np.abs(cl).mean()),
               "cl_error_unet": float(e_unet.mean() / np.abs(cl).mean()), "net_closer": int((e_net < e_unet).sum()),
               "wilcoxon_p": float(w.pvalue), "by_h": []}
        print(f"\n{name}: Cl error, network {100 * res['cl_error_net']:.2f}%, U-Net {100 * res['cl_error_unet']:.2f}% "
              f"(each the mean |dCl| of its runs / mean |Cl|); the network is closer on {res['net_closer']} of "
              f"{res['n']} samples, Wilcoxon p = {res['wilcoxon_p']:.1e}")
        for lo, hi in zip(H_BINS[:-1], H_BINS[1:]):
            sel = (h >= lo) & (h < hi)
            if sel.any():
                b = {"lo": lo, "hi": hi, "n": int(sel.sum()), "net": float(e_net[sel].mean() / np.abs(cl[sel]).mean()),
                     "unet": float(e_unet[sel].mean() / np.abs(cl[sel]).mean())}
                res["by_h"].append(b)
                print(f"  h/c {lo:5.3g}-{hi:<5.3g} n={b['n']:3d}: network {100 * b['net']:5.2f}%, U-Net {100 * b['unet']:5.2f}%")
        out[name] = res
    with open(os.path.join(args.out, "compare.json"), "w") as f:
        json.dump({"unet_runs": args.compare, **out}, f, indent=1)


def load_models(args, data, sets):
    """The networks saved in --out (s<seed>), after checking that the inputs are scaled as when
    they were trained and that they predict the Cl in their errors_<set>.csv."""
    nets = {}
    for seed in args.seeds:
        folder = os.path.join(args.out, f"s{seed}")
        with open(os.path.join(folder, "summary.json")) as f:
            summary = json.load(f)
        s = summary["settings"]
        saved = {k: np.asarray(v) for k, v in summary["scale"].items()}
        off = max(float(np.max(np.abs(saved[k] - np.asarray(data["scale"][k])) / (np.abs(saved[k]) + 1e-9)))
                  for k in saved)
        net = make_net(data["train"]["x"].shape[1], s["width"], s["depth"])
        net.load_state_dict(torch.load(os.path.join(folder, "model.pt"), map_location="cpu", weights_only=True))
        worst = 0.0
        for name in ("val", "test"):
            with open(os.path.join(folder, f"errors_{name}.csv")) as f:
                stored = {r["file"]: float(r["cl_pred"]) for r in csv.DictReader(f)}
            cl_pred = predict(net, data[name]["x"], data["scale"])
            worst = max(worst, max(abs(stored[r["file"]] - c) for r, c in zip(sets[name], cl_pred)))
        print(f"seed {seed}: loaded (width {s['width']}, depth {s['depth']}); input scales as in training within "
              f"{off:.1e}; Cl as in its errors files within {worst:.1e}", flush=True)
        if worst > 1e-3:
            raise SystemExit(f"seed {seed}: the saved model does not reproduce its own Cl here (inputs differ?)")
        nets[seed] = (net, summary)
    return nets


def main(argv=None):
    args = parse(argv)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    if args.load:  # the saved models decide the inputs and the Cl learnt
        with open(os.path.join(args.out, f"s{args.seeds[0]}", "summary.json")) as f:
            saved = json.load(f)["settings"]
        args.target, args.stations, args.val_percent = saved["target"], saved["stations"], saved["val_percent"]
    start = time.time()
    sets = load_sets(args)
    far = max(abs(r["h"] - r["h_file"]) for name in ("train", "val", "test") for r in sets[name])
    print(f"samples: {len(sets['train'])} training, {len(sets['val'])} validation, {len(sets['test'])} test "
          f"({len({r['airfoil'] for r in sets['train']})} training airfoils), loaded in {time.time() - start:.0f} s; "
          f"ride height from the outline within {far:.1e} of the files'", flush=True)
    data = scale(sets, args.target)
    preds = {name: [] for name in sets if name != "train"}
    if args.load:
        for seed, (net, summary) in load_models(args, data, sets).items():
            for name in preds:
                preds[name].append(predict(net, data[name]["x"], data["scale"]))
    else:
        device = devices(args.device)[0]
        hyper = {"width": args.width, "depth": args.depth, "weight_decay": args.weight_decay, "lr": args.lr}
        if args.search:
            best = search(args, {k: v for k, v in data.items() if k in ("train", "val", "scale")})
            hyper = {k: best[k] for k in hyper}
        print(f"final models: {hyper}, on {device}", flush=True)
        for seed in args.seeds:
            t = time.time()
            net, val_err, epoch = fit(data, hyper["width"], hyper["depth"], hyper["weight_decay"], hyper["lr"],
                                      seed, args.epochs, args.batch, device)
            folder = os.path.join(args.out, f"s{seed}")
            os.makedirs(folder, exist_ok=True)
            summary = {"settings": {**vars(args), **hyper}, "n_train": len(sets["train"]), "best_epoch": epoch,
                       "train_seconds": time.time() - t, "device": device,
                       "scale": {k: np.asarray(v).tolist() for k, v in data["scale"].items()}}
            for name in preds:
                cl_pred = predict(net, data[name]["x"], data["scale"])
                preds[name].append(cl_pred)
                if name in ("val", "test"):
                    write_errors(os.path.join(folder, f"errors_{name}.csv"), sets[name], cl_pred, args.target)
                    summary[name] = unet_data.lift_summary(cl_pred, data[name]["cl"])
            torch.save(net.state_dict(), os.path.join(folder, "model.pt"))
            with open(os.path.join(folder, "summary.json"), "w") as f:
                json.dump(summary, f, indent=1)
            print(f"seed {seed}: Cl error validation {100 * summary['val']['rel']:.2f}%, test {100 * summary['test']['rel']:.2f}% "
                  f"(median {100 * summary['test']['median_rel']:.2f}%, R2 {summary['test']['r2']:.3f}); weights of epoch "
                  f"{epoch}, {summary['train_seconds']:.0f} s", flush=True)
    for name in ("val", "test"):
        errs = [rel_error(p, data[name]["cl"]) for p in preds[name]]
        print(f"{name}: Cl error {100 * np.mean(errs):.2f}% (mean of {len(errs)} seeds: "
              + " ".join(f"{100 * e:.2f}" for e in errs) + ")")
    for name in preds:
        if name.startswith("sweep:"):
            sweep_table(name[len("sweep:"):], sets[name], preds[name], args.target, args.out)
    if args.compare:
        compare(args, sets, preds)


if __name__ == "__main__":
    main()
