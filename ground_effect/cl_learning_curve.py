"""Learning curves for the downforce: how the error of each model falls with the number of training
samples. A Gaussian process is said to need less data than a neural network; this tests it.

The models, all against OpenFOAM's Cl on the same validation and test samples as baseline_cl.py:
- 'network': the small network of baseline_cl.py with the settings its tuning chose (4 layers of
  128 neurons, weight decay 1e-5, learning rate 1e-3), trained for --steps steps whatever the number
  of samples, keeping the weights that do best on the validation samples (checked every 250 steps).
- 'gp': a Gaussian process (kriging) on the same inputs: a Matern 5/2 kernel with one length per
  input, a constant mean and a noise term, whose settings maximise the likelihood of the training
  samples (Adam, --gp-iterations steps, in double precision). Its predictions come with a standard
  deviation: the table says how often the error is within 1.96 of them (95% if they are right).
  --gp-components > 0 first reduces the 64 shape numbers to that many principal components (fitted
  on the training samples), as a Gaussian process does best with few inputs.
- the U-Net runs of learning_curve.py (--unet, a folder of runs f<fraction>_scratch_s<seed>), whose
  Cl from the grid is mapped to OpenFOAM's by the straight line of baseline_cl.compare (fitted on all
  the training samples); --unet-extra adds single U-Net runs (e.g. those with the distance input).

At each fraction the training samples are those train_unet.split takes with the same seed: the same
samples as the U-Net runs. --csv gets one row per model, fraction and seed as soon as each is done;
rows already in it are not done again, so a stopped run goes on where it stopped, and runs with
other --models (or --gp-components) add their rows to the same table. --plot draws the test error
against the number of samples (--from-csv: only the figure, from the table).

The Gaussian process's memory and time grow with the cube of the number of samples (about 6 minutes
for 1,548 samples on two CPU threads, eight times that for twice the samples), so with all the
samples it runs on a GPU (--device auto takes one if there is one). On Kaggle (see the howto), then
on Windows, from the ground_effect folder, to add the U-Net's rows and draw the figure:
    !python cl_learning_curve.py --data {data}/train --test {data}/test --airfoil-dirs {smooth} {test_smooth} --csv /kaggle/working/cl_learning_curve.csv
    ..\\.venv\\Scripts\\python.exe cl_learning_curve.py --data data\\dataset\\train --files data\\unet\\baseline_cl\\train_files.txt --test data\\dataset\\test --models --unet data\\unet\\learning_curve_wall --unet-extra data\\unet\\distance\\f1_scratch_s0 data\\unet\\distance\\f1_scratch_s1 --csv ..\\results\\downforce_learning_curve.csv --plot ..\\results\\downforce_learning_curve.png
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

import baseline_cl
import unet_data
from train_unet import split

FRACTIONS = (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0)  # as learning_curve.py: 97 ... 3,096 samples
UNET_LABEL = "U-Net, from its pressure (loss of step 10)"  # the U-Net runs of step 15's figure
COLUMNS = ["model", "n_train", "fraction", "seed", "seconds"] + [
    f"{s}_{k}" for s in ("val", "test") for k in ("cl_error", "cl_median", "cl_r2", "within_2sd", "mean_sd_rel")]
BLUE, ORANGE, AQUA, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#333333", "#d9d9d9"


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", help="folder with the training samples (the validation samples are among them)")
    ap.add_argument("--files", help="text file with the samples of --data to use (see baseline_cl.py)")
    ap.add_argument("--test", help="folder with the test samples")
    ap.add_argument("--out", help="folder for the table, if --csv is not given")
    ap.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="seeds: which samples, and the network's start")
    ap.add_argument("--models", nargs="*", default=["network", "gp"], choices=("network", "gp"),
                    help="models to train (none: only the U-Net rows, e.g. after training elsewhere)")
    ap.add_argument("--steps", type=int, default=25000, help="training steps of the network (default 25,000, as 1000 epochs of all the samples)")
    ap.add_argument("--gp-iterations", type=int, default=250, help="Adam steps for the Gaussian process's settings (default 250)")
    ap.add_argument("--gp-components", type=int, default=0, help="principal components of the shape for the Gaussian process (0: all 64 numbers)")
    ap.add_argument("--unet", help="learning_curve.py folder of U-Net runs to compare with (f<fraction>_scratch_s<seed>)")
    ap.add_argument("--unet-extra", nargs="*", default=[], help="single U-Net runs to show too (with all the samples)")
    ap.add_argument("--unet-label", default=UNET_LABEL, help=f"the U-Net's name in the figure (default: {UNET_LABEL})")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for torch (default 2)")
    ap.add_argument("--device", default="auto", help="cpu, cuda, or auto (default: a GPU if there is one)")
    ap.add_argument("--plot", help="draw the figure into this image")
    ap.add_argument("--csv", help="write the table into this file")
    ap.add_argument("--from-csv", help="only draw --plot, from a table written before")
    ap.add_argument("--airfoil-dirs", nargs="+", default=list(unet_data.AIRFOIL_DIRS), help="folders with the airfoil files")
    args = ap.parse_args(argv)
    if args.from_csv and not args.plot:
        ap.error("--from-csv needs --plot")
    if not args.from_csv and not (args.data and args.test and (args.out or args.csv)):
        ap.error("--data, --test and --out (or --csv) are needed, unless --from-csv")
    return args


# ---------------------------------------------------------------- the Gaussian process

class GaussianProcess:
    """Exact Gaussian process regression with a Matern 5/2 kernel, one length per input, a constant
    mean and Gaussian noise; inputs and targets should be scaled to about mean 0 and spread 1."""

    def __init__(self, x, y, iterations=250, lr=0.05, jitter=1e-6, device="cpu"):
        f64 = {"dtype": torch.float64, "device": device}
        self.x = torch.as_tensor(x, **f64)
        self.y = torch.as_tensor(y, **f64)
        d = self.x.shape[1]
        self.jitter = jitter
        self.log_length = torch.full((d,), 0.5 * math.log(d), **f64, requires_grad=True)
        self.log_signal = torch.zeros((), **f64, requires_grad=True)
        self.log_noise = torch.tensor(math.log(1e-2), **f64, requires_grad=True)
        self.mean = torch.zeros((), **f64, requires_grad=True)
        params = [self.log_length, self.log_signal, self.log_noise, self.mean]
        opt = torch.optim.Adam(params, lr=lr)
        self.history = []
        for _ in range(iterations):
            opt.zero_grad()
            loss = self.neg_log_likelihood()
            loss.backward()
            opt.step()
            self.history.append(float(loss.detach()))
        with torch.no_grad():
            self.chol = torch.linalg.cholesky(self.k_train())
            self.alpha = torch.cholesky_solve((self.y - self.mean)[:, None], self.chol)

    def kernel(self, a, b):
        length = self.log_length.exp()
        a, b = a / length, b / length
        r2 = ((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T).clamp_min(0)
        r5 = torch.sqrt(5 * r2 + 1e-12)
        return self.log_signal.exp() * (1 + r5 + r5 * r5 / 3) * torch.exp(-r5)

    def k_train(self):
        n = len(self.x)
        eye = torch.eye(n, dtype=torch.float64, device=self.x.device)
        return self.kernel(self.x, self.x) + (self.log_noise.exp() + self.jitter) * eye

    def neg_log_likelihood(self):
        """Per sample, without the constant."""
        chol = torch.linalg.cholesky(self.k_train())
        r = (self.y - self.mean)[:, None]
        alpha = torch.cholesky_solve(r, chol)
        return (0.5 * (r * alpha).sum() + torch.log(torch.diagonal(chol)).sum()) / len(self.x)

    @torch.no_grad()
    def predict(self, x):
        """Mean and standard deviation (with the noise) at the inputs x."""
        x = torch.as_tensor(x, dtype=torch.float64, device=self.x.device)
        ks = self.kernel(x, self.x)
        mean = self.mean + (ks @ self.alpha)[:, 0]
        v = torch.cholesky_solve(ks.T, self.chol)
        var = self.log_signal.exp() - (ks * v.T).sum(1) + self.log_noise.exp()
        return mean.cpu().numpy(), torch.sqrt(var.clamp_min(0)).cpu().numpy()


# ---------------------------------------------------------------- one model on one subset

def scaled(train, others, target, components=0):
    """Inputs of the training samples and of the other sets, scaled on the training samples; with
    components > 0 the shape is reduced to that many principal components first."""
    x_tr = baseline_cl.features(train)
    xs = {k: baseline_cl.features(v) for k, v in others.items()}
    if components:
        n_shape = x_tr.shape[1] - 4  # the last 4 are h, log h, alpha, U
        mu = x_tr[:, :n_shape].mean(axis=0)
        _, _, vt = np.linalg.svd(x_tr[:, :n_shape] - mu, full_matrices=False)
        proj = lambda x: np.column_stack([(x[:, :n_shape] - mu) @ vt[:components].T, x[:, n_shape:]])
        x_tr, xs = proj(x_tr), {k: proj(v) for k, v in xs.items()}
    mean, std = x_tr.mean(axis=0), np.maximum(x_tr.std(axis=0), 1e-12)
    cl = np.array([r["cl_" + target] for r in train])
    return ((x_tr - mean) / std, {k: (v - mean) / std for k, v in xs.items()},
            float(cl.mean()), float(cl.std()), cl)


def fit_network(x, y, x_val, cl_val, y_mean, y_std, seed, steps, width=128, depth=4, weight_decay=1e-5, lr=1e-3,
                batch=128, check=250, device="cpu"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, y, x_val = (torch.tensor(a, dtype=torch.float32, device=device) for a in (x, y, x_val))
    net = baseline_cl.make_net(x.shape[1], width, depth).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 100)
    scale = {"y_mean": y_mean, "y_std": y_std}
    best, step, order = (math.inf, None), 0, np.array([], int)
    while step < steps:
        if len(order) < batch:
            order = np.concatenate([order, rng.permutation(len(x))])
        idx, order = torch.from_numpy(order[:min(batch, len(x))]).to(device), order[min(batch, len(x)):]
        net.train()
        loss = (net(x[idx])[:, 0] - y[idx]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step % check == 0 or step == steps:
            err = baseline_cl.rel_error(baseline_cl.predict(net, x_val, scale), cl_val)
            if err < best[0]:
                best = (err, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    return lambda xs: baseline_cl.predict(net, torch.tensor(xs, dtype=torch.float32, device=device), scale)


def run_one(model, train, sets, args, seed, target="openfoam"):
    """Predictions (and for the Gaussian process, standard deviations) of one model on val and test."""
    comps = args.gp_components if model == "gp" else 0
    x_tr, xs, y_mean, y_std, cl = scaled(train, {k: sets[k] for k in ("val", "test")}, target, comps)
    y = (cl - y_mean) / y_std
    if model == "network":
        cl_val = np.array([r["cl_" + target] for r in sets["val"]])
        f = fit_network(x_tr, y, xs["val"], cl_val, y_mean, y_std, seed, args.steps, device=args.device)
        return {k: (f(v), None) for k, v in xs.items()}
    gp = GaussianProcess(x_tr, y, args.gp_iterations, device=args.device)
    out = {}
    for k, v in xs.items():
        m, s = gp.predict(v)
        out[k] = (m * y_std + y_mean, s * y_std)
    return out


# ---------------------------------------------------------------- the U-Net runs

def unet_errors(run, sets, line):
    """The U-Net run's Cl (from the grid, mapped to OpenFOAM's by line) for val and test."""
    out = {}
    for name in ("val", "test"):
        with open(os.path.join(run, f"errors_{name}.csv")) as f:
            pred = {r["file"]: float(r["cl_pred"]) for r in csv.DictReader(f)}
        out[name] = (np.polyval(line, np.array([pred[r["file"]] for r in sets[name]])), None)
    return out


def summarise(model, n, fraction, seed, preds, sets, seconds, target="openfoam"):
    row = {"model": model, "n_train": n, "fraction": fraction, "seed": seed, "seconds": round(seconds, 1)}
    for name in ("val", "test"):
        cl = np.array([r["cl_" + target] for r in sets[name]])
        pred, std = preds[name]
        s = unet_data.lift_summary(pred, cl)
        row[f"{name}_cl_error"] = s["rel"]
        row[f"{name}_cl_median"] = s["median_rel"]
        row[f"{name}_cl_r2"] = s["r2"]
        if std is not None:
            row[f"{name}_within_2sd"] = float(np.mean(np.abs(pred - cl) <= 1.96 * std))
            row[f"{name}_mean_sd_rel"] = float(np.mean(std) / np.abs(cl).mean())
    return row


def read_rows(path):
    """The rows of a --csv written before (numbers as numbers)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "model":
                r[k] = float(v) if v not in ("", None) else None
        r["n_train"] = int(r["n_train"])
    return rows


def main(argv=None):
    args = parse(argv)
    if args.from_csv:  # only the figure, from a table written before
        plot(read_rows(args.from_csv), args.plot, args.unet_label)
        return
    torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    loader = argparse.Namespace(data=args.data, files=args.files, test=args.test, airfoil_dirs=args.airfoil_dirs,
                                stations=32, val_percent=10, sweep=[])
    t = time.time()
    sets = baseline_cl.load_sets(loader)
    print(f"samples: {len(sets['train'])} training, {len(sets['val'])} validation, {len(sets['test'])} test; "
          f"loaded in {time.time() - t:.0f} s", flush=True)
    all_train = sets["train"]
    line = np.polyfit([r["cl_grid"] for r in all_train], [r["cl_openfoam"] for r in all_train], 1)
    # the same subsets as train_unet.split: it takes the training samples in the order of the files
    files = [r["file"] for r in sorted(all_train + sets["val"], key=lambda r: r["file"])]
    by_file = {r["file"]: r for r in all_train}
    # each row goes into the table as soon as it is done; rows already there are not done again
    path = args.csv or os.path.join(args.out, "cl_learning_curve.csv")
    rows = read_rows(path) if os.path.exists(path) else []
    done = {(r["model"], r["fraction"], int(r["seed"])) for r in rows}
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()

        def add(row):
            rows.append(row)
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
            f.flush()

        for fraction in args.fractions:
            for seed in args.seeds:
                train_part, _ = split({"file": files}, 10, fraction, seed)
                train = [by_file[name] for name in train_part["file"]]
                for model in args.models:
                    label = model if model == "network" or not args.gp_components else f"gp{args.gp_components}"
                    if (label, fraction, seed) in done:
                        continue
                    t = time.time()
                    preds = run_one(model, train, sets, args, seed)
                    add(summarise(label, len(train), fraction, seed, preds, sets, time.time() - t))
                    r = rows[-1]
                    extra = (f", within 1.96 sd: {100 * r['test_within_2sd']:.0f}% (sd {100 * r['test_mean_sd_rel']:.1f}%)"
                             if model == "gp" else "")
                    print(f"{label:8s} n={len(train):5d} seed {seed}: Cl error test {100 * r['test_cl_error']:.2f}%, "
                          f"val {100 * r['val_cl_error']:.2f}%{extra} ({r['seconds']:.0f} s)", flush=True)
                run = os.path.join(args.unet or "", f"f{fraction:g}_scratch_s{seed}")
                if args.unet and ("unet", fraction, seed) not in done and os.path.exists(os.path.join(run, "errors_test.csv")):
                    add(summarise("unet", len(train), fraction, seed, unet_errors(run, sets, line), sets, 0))
                    print(f"unet     n={len(train):5d} seed {seed}: Cl error test {100 * rows[-1]['test_cl_error']:.2f}%", flush=True)
        for k, run in enumerate(args.unet_extra):
            if ("unet_distance", 1.0, k) not in done:
                add(summarise("unet_distance", len(all_train), 1.0, k, unet_errors(run, sets, line), sets, 0))
                print(f"unet + distance n={len(all_train)} ({run}): Cl error test {100 * rows[-1]['test_cl_error']:.2f}%")
    print("saved", path)
    if args.plot:
        plot(rows, args.plot, args.unet_label)


def plot(rows, path, unet_label=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    series = [("unet", ORANGE, "s", "-", unet_label or UNET_LABEL),
              ("network", AQUA, "^", "-", "small network (step 14)"), ("gp", BLUE, "o", "-", "Gaussian process")]
    series += [(m, BLUE, "o", "--", f"Gaussian process, shape as {m[2:]} numbers")
               for m in sorted({r["model"] for r in rows if r["model"].startswith("gp") and r["model"] != "gp"})]
    for model, color, marker, style, name in series:
        rs = [r for r in rows if r["model"] == model]
        if not rs:
            continue
        ns = sorted({r["n_train"] for r in rs})
        err = [[100 * r["test_cl_error"] for r in rs if r["n_train"] == n] for n in ns]
        mean = [np.mean(e) for e in err]
        ax.fill_between(ns, [min(e) for e in err], [max(e) for e in err], color=color, alpha=0.18, lw=0)
        ax.plot(ns, mean, style, color=color, lw=1.6, marker=marker, ms=7, mec="white", mew=1.0,
                mfc=color if style == "-" else "white", label=name)
    extra = [r for r in rows if r["model"] == "unet_distance"]
    if extra:
        ax.plot([extra[0]["n_train"]], [100 * np.mean([r["test_cl_error"] for r in extra])], "D", color=ORANGE,
                ms=8, mec=INK, mew=0.8, label="U-Net with the distance input (step 13)", zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ns = sorted({r["n_train"] for r in rows})
    ax.set_xticks(ns, [f"{n:,}" for n in ns])
    ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.set_xlabel("training samples")
    ax.set_ylabel("downforce error on the test samples (% of mean |Cl|)")
    ax.set_title("Downforce error against the amount of data, against OpenFOAM's force\n"
                 "(mean and range of 2 seeds; the same samples for every model)", fontsize=10)
    ax.grid(color=GRID, lw=0.6, which="both")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("saved", path)


if __name__ == "__main__":
    main()
