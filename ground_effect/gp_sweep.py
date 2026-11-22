#!/usr/bin/env python3
"""The Gaussian process of cl_learning_curve.py, fitted on all the training samples, on the
ride-height sweeps: its downforce with a standard deviation at every ride height.

The fit (Adam on the likelihood, 250 steps in double precision, as in cl_learning_curve.py) needs a
GPU with all the samples (about half an hour on a Kaggle T4), so --fit runs there and writes the
fitted settings to gp.json: the kernel's length scales, signal, noise and mean, the scaling of Cl, a
checksum of the list of training samples, and the predictions for the test samples. Everything
else runs where the sweeps are: --params gp.json rebuilds the Gaussian process from those settings
on the same training samples (one Cholesky factorisation, seconds on a CPU), checks that it gives
the fit's test predictions, and predicts each --sweep. It writes sweep_<wing>.csv into --out, with
OpenFOAM's downforce, the mean and the standard deviation of the Gaussian process, its deviation
from OpenFOAM, and whether OpenFOAM's value lies within 1.96 standard deviations.

On Kaggle (both datasets, a GPU; data, smooth and test_smooth found as in the howto):
    !python gp_sweep.py --fit --data {data}/train --test {data}/test --airfoil-dirs {smooth} {test_smooth} --out /kaggle/working/gp
On Windows, from the ground_effect folder, with gp.json from Kaggle in data\\unet\\final\\gp:
    ..\\.venv\\Scripts\\python.exe gp_sweep.py --params data\\unet\\final\\gp\\gp.json --data data\\dataset\\train --files data\\unet\\final\\train_files.txt --test data\\dataset\\test --sweep data\\sweep_naca4412 data\\sweep_tyrrell026 --out data\\unet\\final\\gp
"""
import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np
import torch

import baseline_cl
import unet_data
from cl_learning_curve import GaussianProcess


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="folder with the training samples (the validation samples are among them)")
    ap.add_argument("--files", help="text file with the samples of --data to use (see baseline_cl.py)")
    ap.add_argument("--test", required=True, help="folder with the test samples")
    ap.add_argument("--out", required=True, help="folder for gp.json (--fit) or for the sweep tables (--params)")
    ap.add_argument("--fit", action="store_true", help="fit the Gaussian process and write gp.json (a GPU with all the samples)")
    ap.add_argument("--params", help="gp.json of a fit: rebuild the Gaussian process from it")
    ap.add_argument("--iterations", type=int, default=250, help="Adam steps of the fit (default 250, as cl_learning_curve.py)")
    ap.add_argument("--sweep", nargs="*", default=[], help="folders of ride-height sweeps to predict (with --params)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for torch (default 4)")
    ap.add_argument("--device", default="auto", help="cpu, cuda, or auto (default: a GPU if there is one)")
    ap.add_argument("--airfoil-dirs", nargs="+", default=list(unet_data.AIRFOIL_DIRS), help="folders with the airfoil files")
    args = ap.parse_args(argv)
    if args.fit == bool(args.params):
        ap.error("give either --fit or --params")
    return args


def checksum(train):
    """MD5 of the training samples' names, one per line, in the order of the fit."""
    return hashlib.md5("\n".join(r["file"] for r in train).encode()).hexdigest()


def rebuild(x, y, p, device="cpu"):
    """The Gaussian process on the scaled inputs x and targets y with the fitted settings p,
    without fitting again: the settings, then the Cholesky factor as at the end of the fit."""
    gp = GaussianProcess.__new__(GaussianProcess)
    f64 = {"dtype": torch.float64, "device": device}
    gp.x, gp.y, gp.jitter = torch.as_tensor(x, **f64), torch.as_tensor(y, **f64), p["jitter"]
    gp.log_length = torch.tensor(p["log_length"], **f64)
    gp.log_signal = torch.tensor(p["log_signal"], **f64)
    gp.log_noise = torch.tensor(p["log_noise"], **f64)
    gp.mean = torch.tensor(p["mean"], **f64)
    with torch.no_grad():
        gp.chol = torch.linalg.cholesky(gp.k_train())
        gp.alpha = torch.cholesky_solve((gp.y - gp.mean)[:, None], gp.chol)
    return gp


def rel_error(cl_pred, recs):
    cl = np.array([r["cl_openfoam"] for r in recs])
    return float(np.abs(cl_pred - cl).mean() / np.abs(cl).mean())


def sweep_table(name, recs, cl_pred, sd, out):
    """One line per ride height: OpenFOAM's downforce (-Cl), the Gaussian process's mean and
    standard deviation, the deviation in percent, and whether OpenFOAM lies within 1.96 sd."""
    rows = []
    for r, c, s in sorted(zip(recs, cl_pred, sd), key=lambda t: t[0]["h_run"]):
        down, down_gp = -r["cl_openfoam"], -c  # downforce is -Cl
        rows.append({"run": r["run"], "h_over_c": f"{r['h_run']:.3f}", "downforce": f"{down:.4f}",
                     "gp": f"{down_gp:.4f}", "gp_sd": f"{s:.4f}",
                     "gp_vs_openfoam_pct": f"{100 * (down_gp - down) / down:.2f}",  # positive: more downforce
                     "within_1.96sd": int(abs(down_gp - down) <= 1.96 * s)})
    path = os.path.join(out, f"sweep_{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    dev = np.array([abs(float(r["gp_vs_openfoam_pct"])) for r in rows])
    sdr = np.array([100 * float(r["gp_sd"]) / float(r["downforce"]) for r in rows])
    print(f"{name}: {len(rows)} ride heights; Gaussian process {dev.mean():.2f}% off OpenFOAM on average, at most "
          f"{dev.max():.2f}%; standard deviation {sdr.min():.2f}-{sdr.max():.2f}% of the downforce; OpenFOAM within "
          f"1.96 sd at {sum(r['within_1.96sd'] for r in rows)} of {len(rows)} -> {path}")


def main(argv=None):
    args = parse(argv)
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(args.out, exist_ok=True)
    loader = argparse.Namespace(data=args.data, files=args.files, test=args.test, airfoil_dirs=args.airfoil_dirs,
                                stations=32, val_percent=10, sweep=args.sweep if args.params else [])
    t = time.time()
    sets = baseline_cl.load_sets(loader)
    train = sorted(sets["train"], key=lambda r: r["file"])  # the order of the fit
    print(f"samples: {len(train)} training, {len(sets['val'])} validation, {len(sets['test'])} test; "
          f"loaded in {time.time() - t:.0f} s", flush=True)
    x = baseline_cl.features(train)
    x_mean, x_std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-12)  # as cl_learning_curve.scaled
    cl = np.array([r["cl_openfoam"] for r in train])
    scaled = lambda recs: (baseline_cl.features(recs) - x_mean) / x_std
    if args.fit:
        y_mean, y_std = float(cl.mean()), float(cl.std())
        t = time.time()
        gp = GaussianProcess((x - x_mean) / x_std, (cl - y_mean) / y_std, args.iterations, device=device)
        print(f"fitted in {time.time() - t:.0f} s", flush=True)
        p = {"n": len(train), "files_md5": checksum(train), "y_mean": y_mean, "y_std": y_std,
             "log_length": gp.log_length.detach().cpu().tolist(), "log_signal": gp.log_signal.item(),
             "log_noise": gp.log_noise.item(), "mean": gp.mean.item(), "jitter": gp.jitter}
    else:
        with open(args.params) as f:
            p = json.load(f)
        if p["n"] != len(train) or p["files_md5"] != checksum(train):
            raise SystemExit(f"the training samples are not those of the fit ({len(train)} here, {p['n']} there; "
                             f"checksum {checksum(train)} here, {p['files_md5']} there): see --files")
        if abs(p["y_mean"] - cl.mean()) > 1e-9 or abs(p["y_std"] - cl.std()) > 1e-9:
            raise SystemExit("the scaling of Cl differs from the fit's")
        y_mean, y_std = p["y_mean"], p["y_std"]
        t = time.time()
        gp = rebuild((x - x_mean) / x_std, (cl - y_mean) / y_std, p, device)
        print(f"rebuilt from {args.params} in {time.time() - t:.0f} s", flush=True)
    preds = {}
    for name, recs in [("val", sets["val"]), ("test", sets["test"])]:
        m, s = gp.predict(scaled(recs))
        preds[name] = (m * y_std + y_mean, s * y_std)
        print(f"{name}: Cl error {100 * rel_error(preds[name][0], recs):.4f}%", flush=True)
    test_files = [r["file"] for r in sets["test"]]
    if args.fit:
        p.update({"test_files": test_files, "test_pred": preds["test"][0].tolist(), "test_sd": preds["test"][1].tolist()})
        with open(os.path.join(args.out, "gp.json"), "w") as f:
            json.dump(p, f)
        print("GPJSON " + json.dumps(p))
        return
    if p.get("test_files") == test_files:  # the rebuilt process must give the fit's predictions
        d_mean = np.abs(preds["test"][0] - np.array(p["test_pred"])).max()
        d_sd = np.abs(preds["test"][1] - np.array(p["test_sd"])).max()
        print(f"test predictions against the fit's: largest difference {d_mean:.1e} (mean), {d_sd:.1e} (sd)")
        if d_mean > 1e-6 or d_sd > 1e-6:
            raise SystemExit("the rebuilt Gaussian process does not give the fit's test predictions")
    for folder in args.sweep:
        recs = sets["sweep:" + folder]
        m, s = gp.predict(scaled(recs))
        name = os.path.basename(os.path.normpath(folder)).replace("sweep_", "")
        sweep_table(name, recs, m * y_std + y_mean, s * y_std, args.out)


if __name__ == "__main__":
    main()
