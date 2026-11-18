"""The surface model: a small network that predicts the pressure and the friction along the wing's
surface from the same inputs as the comparison model of step 14 (baseline_cl.py: the shape, the ride
height, the angle and the speed). Its outputs are those of surface_data.targets: Cp and Cf at
--points points on each side (the side facing the ground and the side facing up) and the base's mean
Cp. Cl, Cd and Cm follow by adding up over the mesh's faces as OpenFOAM does (surface_data.forces),
and where Cf turns negative the flow has separated.

Samples, split and inputs as baseline_cl.py (with --files, the U-Net's training samples). Each output
is scaled to mean 0 and spread 1 over the training samples, and the loss is the mean absolute error
of the scaled outputs. Training as in baseline_cl.py: AdamW, the learning rate falling along a cosine,
--steps steps of --batch samples, keeping the weights with the lowest validation loss (checked every
250 steps). --search tries every combination of --widths, --depths and --weight-decays with every seed
and keeps the one with the lowest mean validation loss; each model's result goes into search_runs.csv
as soon as it is trained, and models already there are not trained again.

Measures on the validation and test samples (mean of the seeds): the error of Cp (mean |dCp| over the
points of both sides) and of Cf; Cl and Cd against OpenFOAM's, Cm against that of the sample's own
surface data; the share of the surface with reversed flow (Cf < 0 between 2% and 98% of each side)
against that of the simulation's Cf. --compare-net (a baseline_cl.py folder trained with --target
openfoam) and --compare-unet (U-Net runs; their Cl from the grid mapped to OpenFOAM's as in
baseline_cl.compare) add the other models' Cl, sample by sample. --sweep adds the ride-height sweeps.
--load takes the models already in --out instead of training them.

Writes into --out: search_runs.csv and search.csv; s<seed>/model.pt, summary.json (settings, scales,
measures), errors_val.csv and errors_test.csv (per sample); sweep_<name>.csv; compare.json.

From the ground_effect folder (Windows, one process; a few minutes per model on the CPU):
    ..\\.venv\\Scripts\\python.exe surface_model.py --data data\\dataset\\train --files data\\unet\\baseline_cl\\train_files.txt --test data\\dataset\\test --search --out data\\unet\\surface_model --sweep data\\sweep_naca4412 data\\sweep_tyrrell026 --compare-net data\\unet\\baseline_cl_openfoam --compare-unet data\\unet\\distance\\f1_scratch_s0 data\\unet\\distance\\f1_scratch_s1
"""
import argparse
import csv
import itertools
import json
import math
import os
import time

import numpy as np
import torch

import baseline_cl
import surface_data
import unet_data

H_BINS = baseline_cl.H_BINS
SEARCH_COLUMNS = ["width", "depth", "weight_decay", "lr", "seed", "steps", "val_loss", "step", "seconds"]


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="folder with the training samples (the validation samples are among them)")
    ap.add_argument("--files", help="text file with the samples of --data to use (see baseline_cl.py)")
    ap.add_argument("--test", required=True, help="folder with the test samples")
    ap.add_argument("--out", required=True, help="folder for the results")
    ap.add_argument("--points", type=int, default=128, help="points per side (default 128)")
    ap.add_argument("--width", type=int, default=256, help="neurons per hidden layer (default 256)")
    ap.add_argument("--depth", type=int, default=4, help="hidden layers (default 4)")
    ap.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay (default 1e-5)")
    ap.add_argument("--lr", type=float, default=1e-3, help="starting learning rate (default 1e-3)")
    ap.add_argument("--steps", type=int, default=25000, help="training steps (default 25,000)")
    ap.add_argument("--batch", type=int, default=128, help="samples per step (default 128)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--search", action="store_true", help="choose width, depth and weight decay on the validation samples")
    ap.add_argument("--widths", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--depths", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--weight-decays", type=float, nargs="+", default=[1e-5, 1e-4])
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for torch (default 2)")
    ap.add_argument("--device", default="auto", help="cpu, cuda, or auto (default: a GPU if there is one)")
    ap.add_argument("--load", action="store_true", help="use the models already in --out instead of training them")
    ap.add_argument("--sweep", nargs="*", default=[], help="folders of ride-height sweeps to predict")
    ap.add_argument("--compare-net", help="baseline_cl.py folder (trained with --target openfoam) to compare with")
    ap.add_argument("--compare-unet", nargs="*", default=[], help="U-Net run folders to compare with")
    ap.add_argument("--airfoil-dirs", nargs="+", default=list(unet_data.AIRFOIL_DIRS), help="folders with the airfoil files")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- samples

def load_sets(args):
    """The records of baseline_cl.load_sets with the surface targets 'y' (4 x points + 1 numbers) and,
    for the samples not trained on, what the measures need (the outline, the faces' size, Cm)."""
    loader = argparse.Namespace(data=args.data, files=args.files, test=args.test, airfoil_dirs=args.airfoil_dirs,
                                stations=32, val_percent=10, sweep=args.sweep)
    sets = baseline_cl.load_sets(loader)
    for name, recs in sets.items():
        folder = args.data if name in ("train", "val") else args.test if name == "test" else name[len("sweep:"):]
        for r in recs:
            with np.load(os.path.join(folder, *r["file"].split("/"))) as d:
                t = surface_data.targets(d, args.points, args.airfoil_dirs)
                base = t["cp_base"] if np.isfinite(t["cp_base"]) else 0.5 * (t["cp"][0, -1] + t["cp"][1, -1])
                r["y"] = np.concatenate([t["cp"].ravel(), t["cf"].ravel(), [base]])
                r["cd_openfoam"] = float(d["Cd"])
                if name != "train":
                    r["outline"], r["wall"] = t["outline"], surface_data.wall_size(d)
                    r["cm_surface"] = surface_data.full_forces(d, t["outline"])[2]
                    t["outline"].dense = t["outline"].dense_s = None  # not needed any more (memory)
    return sets


def unpack(y, n):
    """Cp (2 x n), Cf (2 x n) and the base's Cp of one output vector."""
    return y[:2 * n].reshape(2, n), y[2 * n:4 * n].reshape(2, n), float(y[4 * n])


def reversed_share(cf):
    """The share of the surface between 2% and 98% of each side where Cf < 0 (by the spacing of u)."""
    n = cf.shape[-1]
    u = surface_data.stations(n)
    w = np.gradient(u)
    sel = (u > 0.02) & (u < 0.98)
    return float((w[sel] * (cf[:, sel] < 0)).sum() / (cf.shape[0] * w[sel].sum()))


def scaling(train):
    x = baseline_cl.features(train)
    y = np.array([r["y"] for r in train])
    return {"x_mean": x.mean(axis=0), "x_std": np.maximum(x.std(axis=0), 1e-12),
            "y_mean": y.mean(axis=0), "y_std": np.maximum(y.std(axis=0), 1e-6)}


def tensors(recs, sc):
    x = (baseline_cl.features(recs) - sc["x_mean"]) / sc["x_std"]
    y = (np.array([r["y"] for r in recs]) - sc["y_mean"]) / sc["y_std"]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ---------------------------------------------------------------- the network

def fit(data, width, depth, weight_decay, lr, seed, steps, batch, device, check=250):
    """One network; returns it (on the CPU), its lowest validation loss and the step of its weights."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, y = (t.to(device) for t in data["train"])
    xv, yv = (t.to(device) for t in data["val"])
    net = baseline_cl.make_net(x.shape[1], width, depth, y.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 100)
    best, step, order = (math.inf, None, 0), 0, np.array([], int)
    while step < steps:
        if len(order) < batch:
            order = np.concatenate([order, rng.permutation(len(x))])
        idx, order = torch.from_numpy(order[:batch]).to(device), order[batch:]
        net.train()
        loss = (net(x[idx]) - y[idx]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step % check == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                v = float((net(xv) - yv).abs().mean())
            if v < best[0]:
                best = (v, {k: t.detach().clone() for k, t in net.state_dict().items()}, step)
    net.load_state_dict(best[1])
    return net.cpu(), best[0], best[2]


def predict(net, x, sc):
    net.eval()
    with torch.no_grad():
        return net(x).double().numpy() * sc["y_std"] + sc["y_mean"]


def search(args, data, device):
    """The settings with the lowest mean validation loss (search_runs.csv, search.csv)."""
    combos = list(itertools.product(args.widths, args.depths, args.weight_decays))
    path = os.path.join(args.out, "search_runs.csv")
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                done[(int(r["width"]), int(r["depth"]), float(r["weight_decay"]), float(r["lr"]), int(r["seed"]),
                      int(r["steps"]))] = float(r["val_loss"])
    key = lambda c, s: (*c, args.lr, s, args.steps)
    todo = [(c, s) for c in combos for s in args.seeds if key(c, s) not in done]
    print(f"search: {len(combos)} combinations x {len(args.seeds)} seeds; {len(todo)} to train", flush=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(SEARCH_COLUMNS)
        for k, (c, s) in enumerate(todo, 1):
            t = time.time()
            _, v, step = fit(data, c[0], c[1], c[2], args.lr, s, args.steps, args.batch, device)
            done[key(c, s)] = v
            w.writerow([*c, args.lr, s, args.steps, v, step, round(time.time() - t, 1)])
            f.flush()
            print(f"  {k}/{len(todo)}: width {c[0]}, depth {c[1]}, weight decay {c[2]:g}, seed {s}: "
                  f"validation loss {v:.4f} ({time.time() - t:.0f} s)", flush=True)
    rows = sorted(({"width": c[0], "depth": c[1], "weight_decay": c[2], "lr": args.lr,
                    "val_loss": float(np.mean([done[key(c, s)] for s in args.seeds]))} for c in combos),
                  key=lambda r: r["val_loss"])
    with open(os.path.join(args.out, "search.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows[:4]:
        print(f"  width {r['width']}, depth {r['depth']}, weight decay {r['weight_decay']:g}: {r['val_loss']:.4f}")
    return rows[0]


# ---------------------------------------------------------------- measures

def measure(recs, y_pred, n):
    """One row per sample: forces and errors of the predicted surface."""
    rows = []
    for r, yp in zip(recs, y_pred):
        cp, cf, base = unpack(yp, n)
        cpt, cft, _ = unpack(r["y"], n)
        cl, cd, cm = surface_data.forces(cp, cf, base, r["outline"], r["wall"])
        rows.append({"file": r["file"], "airfoil": r["airfoil"], "h_over_c": r["h_file"], "alpha_deg": r["alpha"],
                     "U": r["U"], "cl_openfoam": r["cl_openfoam"], "cl_pred": cl, "cd_openfoam": r["cd_openfoam"],
                     "cd_pred": cd, "cm_surface": r["cm_surface"], "cm_pred": cm,
                     "cp_error": float(np.abs(cp - cpt).mean()), "cf_error": float(np.abs(cf - cft).mean()),
                     "reversed_cfd": reversed_share(cft), "reversed_pred": reversed_share(cf)})
    return rows


def summarise(rows):
    a = lambda k: np.array([r[k] for r in rows], float)
    cl, cd, cm = a("cl_openfoam"), a("cd_openfoam"), a("cm_surface")
    return {"n": len(rows), "cl_error": float(np.abs(a("cl_pred") - cl).mean() / np.abs(cl).mean()),
            "cd_error": float(np.abs(a("cd_pred") - cd).mean() / cd.mean()),
            "cm_error": float(np.abs(a("cm_pred") - cm).mean() / np.abs(cm).mean()),
            "cp_error": float(a("cp_error").mean()), "cf_error": float(a("cf_error").mean()),
            "reversed_error": float(np.abs(a("reversed_pred") - a("reversed_cfd")).mean()),
            "reversed_mean_cfd": float(a("reversed_cfd").mean())}


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def compare(args, sets, rows_by_seed):
    """Cl sample by sample against the comparison model of step 14 and the U-Net runs."""
    from scipy import stats
    train = sets["train"]
    line = np.polyfit([r["cl_grid"] for r in train], [r["cl_openfoam"] for r in train], 1)
    out = {"grid_to_openfoam_line": [float(v) for v in line]}
    for name in ("test", "val"):
        files = [r["file"] for r in sets[name]]
        cl = np.array([r["cl_openfoam"] for r in sets[name]])
        errs = {"surface model": np.mean([[abs(r["cl_pred"] - r["cl_openfoam"]) for r in rows[name]]
                                          for rows in rows_by_seed], axis=0)}
        if args.compare_net:
            pred = []
            for s in (0, 1):
                with open(os.path.join(args.compare_net, f"s{s}", f"errors_{name}.csv")) as f:
                    by = {r["file"]: float(r["cl_pred"]) for r in csv.DictReader(f)}
                pred.append([by[x] for x in files])
            errs["network (step 14)"] = np.mean(np.abs(np.array(pred) - cl), axis=0)
        if args.compare_unet:
            pred = []
            for run in args.compare_unet:
                with open(os.path.join(run, f"errors_{name}.csv")) as f:
                    by = {r["file"]: float(r["cl_pred"]) for r in csv.DictReader(f)}
                pred.append(np.polyval(line, np.array([by[x] for x in files])))
            errs["U-Net (step 13)"] = np.mean(np.abs(np.array(pred) - cl), axis=0)
        res = {}
        print(f"\n{name}: Cl error against OpenFOAM's force (mean |dCl| of each model's runs / mean |Cl|)")
        for k, e in errs.items():
            res[k] = {"cl_error": float(e.mean() / np.abs(cl).mean())}
            line_txt = f"  {k:20s} {100 * res[k]['cl_error']:5.2f}%"
            if k != "surface model":
                w = stats.wilcoxon(errs["surface model"], e)
                res[k]["surface_closer"] = int((errs["surface model"] < e).sum())
                res[k]["wilcoxon_p"] = float(w.pvalue)
                line_txt += f"; the surface model is closer on {res[k]['surface_closer']} of {len(e)} (p = {w.pvalue:.1e})"
            print(line_txt)
        out[name] = res
    with open(os.path.join(args.out, "compare.json"), "w") as f:
        json.dump(out, f, indent=1)


def sweep_table(name, recs, rows_by_seed, out):
    pred = np.array([[[r["cl_pred"], r["cd_pred"]] for r in rows] for rows in rows_by_seed])  # seeds x runs x 2
    rows = []
    print(f"\n{name}: OpenFOAM against the surface model (mean and range of {len(rows_by_seed)})")
    print(f"  {'h/c':>6s} {'-Cl':>7s} {'model':>7s} {'':>15s} {'Cd':>7s} {'model':>7s}")
    for i, r in enumerate(recs):
        cl, cd = pred[:, i, 0], pred[:, i, 1]
        rows.append({"run": r["run"], "h_over_c": r["h_run"], "downforce": -r["cl_openfoam"], "downforce_model": -cl.mean(),
                     "model_min": -cl.max(), "model_max": -cl.min(), "cd": r["cd_openfoam"], "cd_model": cd.mean(),
                     "cd_model_min": cd.min(), "cd_model_max": cd.max(),
                     "model_vs_openfoam_pct": 100 * (cl.mean() / r["cl_openfoam"] - 1),
                     "cd_model_vs_openfoam_pct": 100 * (cd.mean() / r["cd_openfoam"] - 1)})
        print(f"  {r['h_run']:6.3f} {-r['cl_openfoam']:7.3f} {-cl.mean():7.3f} ({-cl.max():.3f}-{-cl.min():.3f}) "
              f"{r['cd_openfoam']:7.4f} {cd.mean():7.4f}")
    for k, txt in (("model_vs_openfoam_pct", "-Cl"), ("cd_model_vs_openfoam_pct", "Cd")):
        dev = np.array([r[k] for r in rows])
        print(f"  {txt}: mean {dev.mean():+.1f}%, mean |.| {np.abs(dev).mean():.1f}%, {dev.min():+.1f} to {dev.max():+.1f}%")
    base = os.path.basename(os.path.normpath(name))
    write_rows(os.path.join(out, (base if base.startswith("sweep") else "sweep_" + base) + ".csv"), rows)


# ---------------------------------------------------------------- main

def main(argv=None):
    args = parse(argv)
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    os.makedirs(args.out, exist_ok=True)
    if args.load:
        with open(os.path.join(args.out, f"s{args.seeds[0]}", "summary.json")) as f:
            args.points = json.load(f)["settings"]["points"]
    t = time.time()
    sets = load_sets(args)
    n = args.points
    print(f"samples: {len(sets['train'])} training, {len(sets['val'])} validation, {len(sets['test'])} test; "
          f"{4 * n + 1} outputs each; loaded in {time.time() - t:.0f} s", flush=True)
    sc = scaling(sets["train"])
    data = {k: tensors(v, sc) for k, v in sets.items()}
    rows_by_seed, others = [], [k for k in sets if k != "train"]
    preds = {k: [] for k in others}
    if not args.load:
        hyper = {"width": args.width, "depth": args.depth, "weight_decay": args.weight_decay}
        if args.search:
            best = search(args, data, device)
            hyper = {k: best[k] for k in hyper}
        print(f"final models: {hyper}, on {device}", flush=True)
    for seed in args.seeds:
        folder = os.path.join(args.out, f"s{seed}")
        os.makedirs(folder, exist_ok=True)
        if args.load:
            with open(os.path.join(folder, "summary.json")) as f:
                summary = json.load(f)
            st = summary["settings"]
            net = baseline_cl.make_net(data["train"][0].shape[1], st["width"], st["depth"], 4 * n + 1)
            net.load_state_dict(torch.load(os.path.join(folder, "model.pt"), map_location="cpu", weights_only=True))
            saved = {k: np.asarray(v) for k, v in summary["scale"].items()}
            off = max(float(np.max(np.abs(saved[k] - sc[k]))) for k in saved)
            print(f"seed {seed}: loaded (width {st['width']}, depth {st['depth']}); scales as in training within {off:.1e}")
        else:
            t = time.time()
            net, val_loss, step = fit(data, hyper["width"], hyper["depth"], hyper["weight_decay"], args.lr, seed,
                                      args.steps, args.batch, device)
            summary = {"settings": {**vars(args), **hyper}, "n_train": len(sets["train"]), "best_step": step,
                       "val_loss": val_loss, "train_seconds": time.time() - t, "device": device,
                       "scale": {k: np.asarray(v).tolist() for k, v in sc.items()}}
            torch.save(net.state_dict(), os.path.join(folder, "model.pt"))
        rows = {}
        for k in others:
            y_pred = predict(net, data[k][0], sc)
            preds[k].append(y_pred)
            rows[k] = measure(sets[k], y_pred, n)
            if k in ("val", "test"):
                write_rows(os.path.join(folder, f"errors_{k}.csv"), rows[k])
                summary[k] = summarise(rows[k])
        rows_by_seed.append(rows)
        with open(os.path.join(folder, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        s = summary["test"]
        print(f"seed {seed}: test Cl {100 * s['cl_error']:.2f}%, Cd {100 * s['cd_error']:.2f}%, Cm {100 * s['cm_error']:.2f}%, "
              f"Cp error {s['cp_error']:.4f}, Cf error {s['cf_error']:.5f}, reversed flow off by "
              f"{100 * s['reversed_error']:.1f} points (CFD {100 * s['reversed_mean_cfd']:.1f}% of the surface)", flush=True)
    for k in ("val", "test"):
        ss = [summarise(rows[k]) for rows in rows_by_seed]
        print(f"{k}: " + ", ".join(f"{m} {100 * np.mean([s[m] for s in ss]):.2f}%" for m in ("cl_error", "cd_error", "cm_error"))
              + f", Cp error {np.mean([s['cp_error'] for s in ss]):.4f} (mean of {len(ss)} seeds)")
    for k in others:
        if k.startswith("sweep:"):
            sweep_table(k[len("sweep:"):], sets[k], [rows[k] for rows in rows_by_seed], args.out)
    if args.compare_net or args.compare_unet:
        compare(args, sets, rows_by_seed)


if __name__ == "__main__":
    main()
