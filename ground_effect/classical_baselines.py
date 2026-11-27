"""Classical regressions for the forces, and every model of the final runs scored alike (step 19).

Polynomial regression (with a ridge penalty), a random forest and support vector regression (with a
Gaussian kernel), from scikit-learn, learn OpenFOAM's Cl, and with all the training samples also its
Cd, from the inputs of the downforce network (baseline_cl.features: the camber and the thickness at
32 points along the chord, h/c, log h/c, the angle and the speed), on the samples of the final runs:
the training and validation samples split as there, and the test samples. For the downforce they
also learn from the parts of the training samples of the learning curve (the same subsets as
cl_learning_curve.py: train_unet.split with seeds 0 and 1). Each method's settings are chosen on the
validation samples by the relative error (mean |error| / mean |coefficient|), at every size and for
the drag separately; the test samples play no part in it. Support vector regression starts from the
range of SVR_START and grows it where the best setting lies at an end of a range and the error
still falls there by more than SVR_FLAT, up to SVR_MAX_STEPS steps; C stays at 3000 at most, as
fits take tens of minutes above it. Settings that stay at an end of their range are reported.

Then the test predictions of every model are scored alike, against OpenFOAM's forces: the final
runs' models from their errors_test.csv (the U-Net's Cl, from its pressure, mapped to OpenFOAM's by
the straight line of baseline_cl.compare, and the same for the network trained on the grid's Cl),
the Gaussian process rebuilt from gp.json as in gp_sweep.py (its test predictions checked against
the fit's) and the three new methods: the relative error, the mean absolute error, the root-mean-
square error, R^2, the median relative error and the largest error, for each seed and then their
mean. Each model is compared with the downforce (drag) network sample by sample and, as the samples
of one airfoil are not independent, airfoil by airfoil (the mean |error| of each test airfoil): on
how many the network is closer, and the Wilcoxon signed-rank test (the error of a sample is the mean
over the seeds).

Writes into --out: preds/ (the new models' predictions and chosen settings, and the Gaussian
process's; finished ones are not done again, so a stopped run goes on where it stopped) and
search.csv (every setting tried, with its validation error); --csv: the scores, one row per model
and coefficient; --learning-curve-csv: the downforce's test error against the number of training
samples, the mean over the seeds, with the network's and the Gaussian process's from
<final>/cl_learning_curve.csv.

On the laptop, from the ground_effect folder (needs scikit-learn; about 2.5 hours, nearly all of it
the two searches of the support vector regression with all the samples, 8 fits at a time):
    ..\\.venv\\Scripts\\python.exe classical_baselines.py --data data\\dataset\\train --files data\\unet\\final\\train_files.txt --test data\\dataset\\test --final data\\unet\\final --out data\\unet\\final\\classical --csv ..\\results\\classical_baselines.csv --learning-curve-csv ..\\results\\classical_learning_curve.csv
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
import time

import numpy as np

import baseline_cl
import gp_sweep
import unet_data
from train_unet import split

FRACTIONS = (0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0)  # 97 ... 6,177 samples, as cl_learning_curve.py
SEEDS = (0, 1)
METHODS = ("polynomial", "random_forest", "svr")
LABELS = {"polynomial": "Polynomial regression", "random_forest": "Random forest",
          "svr": "Support vector regression"}
KEYS = {"cl": "cl_openfoam", "cd": "cd_openfoam"}  # the coefficient each target is scored against

# Support vector regression starts from these values (gamma as a multiple of 1 / number of inputs,
# scikit-learn's 'scale' for scaled inputs) and grows its range where needed: while the best setting
# lies at an end of a range and beats the next value inside by more than SVR_FLAT (0.01 percentage
# points of the relative error), the next value beyond (in steps of 1-3-10) is tried too, up to
# SVR_MAX_STEPS times at each end. Where the error is flat, a better value cannot lie beyond.
SVR_START = {"C": [1, 3, 10, 30, 100, 300, 1000, 3000], "g": [0.1, 0.3, 1, 3],
             "epsilon": [0.001, 0.003, 0.01, 0.03, 0.1]}
SVR_FLAT, SVR_MAX_STEPS = 1e-4, 3
# C does not grow above 3000: with 3,088 samples a fit takes minutes at C = 3000 and over 20 minutes
# at C = 10,000; with up to 3,088 samples the best C was 10 to 1000.
SVR_LIMITS = {("C", True): 0}
# a setting at the end of its range may mean that a better one lies beyond it, except where the
# range cannot go further: all the inputs or one sample per leaf of a tree, polynomial degree 1 or 2
# (degree 3 of 68 inputs would be 57,000 terms)
NATURAL_ENDS = {("random_forest", "max_features", 1.0), ("random_forest", "min_samples_leaf", 1),
                ("polynomial", "degree", 1), ("polynomial", "degree", 2)}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="folder of the training and validation samples")
    ap.add_argument("--files", help="only these samples of --data, one per line (train_files.txt of the final runs)")
    ap.add_argument("--test", required=True, help="folder of the test samples")
    ap.add_argument("--airfoil-dirs", nargs="+", default=list(unet_data.AIRFOIL_DIRS),
                    help="folders of the airfoil files the simulations used")
    ap.add_argument("--final", required=True, help="folder of the final runs (final_unet, final_unet_160k, "
                    "baseline_cl_openfoam, baseline_cl, baseline_cd, surface_model, gp/gp.json, cl_learning_curve.csv)")
    ap.add_argument("--out", required=True, help="folder for the predictions and the settings tried")
    ap.add_argument("--csv", help="write the scores of all the models into this file")
    ap.add_argument("--learning-curve-csv", help="write the downforce's test error against the number of samples")
    ap.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS),
                    help="parts of the training samples for the learning curve (1 is needed for the scores)")
    ap.add_argument("--jobs", type=int, default=8,
                    help="fits of the support vector regression at a time (each up to a few hundred MB)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- data

def load(args):
    loader = argparse.Namespace(data=args.data, files=args.files, test=args.test, airfoil_dirs=args.airfoil_dirs,
                                stations=32, val_percent=10, sweep=[])
    t = time.time()
    sets = baseline_cl.load_sets(loader)
    print(f"samples: {len(sets['train'])} training, {len(sets['val'])} validation, {len(sets['test'])} test "
          f"({time.time() - t:.0f} s)", flush=True)
    return sets


def subsets(sets, fractions):
    """{(fraction, seed): training records}, as cl_learning_curve.py takes them."""
    files = [r["file"] for r in sorted(sets["train"] + sets["val"], key=lambda r: r["file"])]
    by_file = {r["file"]: r for r in sets["train"]}
    out = {}
    for fraction in fractions:
        for seed in SEEDS:
            part, _ = split({"file": files}, 10, fraction, seed)
            out[(fraction, seed)] = [by_file[f] for f in part["file"]]
    return out


def scaled(train, others, key):
    """Inputs and target scaled to zero mean and unit spread over the training samples."""
    x = baseline_cl.features(train)
    m, s = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-12)
    y = np.array([r[key] for r in train])
    ym, ys = float(y.mean()), float(y.std())
    return (x - m) / s, (y - ym) / ys, {k: (baseline_cl.features(v) - m) / s for k, v in others.items()}, ym, ys


def rel(pred, true):
    return float(np.abs(pred - true).mean() / np.abs(true).mean())


# ---------------------------------------------------------------- the three methods

def grid(method, n_in):
    """The settings tried (for support vector regression: the starting ones)."""
    if method == "polynomial":
        return [{"degree": d, "alpha": a} for d in (1, 2) for a in np.logspace(-4, 3, 8).tolist()]
    if method == "random_forest":
        return [{"max_features": f, "min_samples_leaf": k} for f in (1.0, 0.5, 0.2) for k in (1, 3, 10)]
    return [{"C": c, "gamma": g / n_in, "epsilon": e}
            for c in SVR_START["C"] for g in SVR_START["g"] for e in SVR_START["epsilon"]]


def step(v, up):
    """The next value of the 1-3-10 sequence above (up) or below v."""
    k = int(np.floor(np.log10(v) + 1e-9))
    m = round(v / 10 ** k)  # 1 or 3
    nxt = (3 * 10 ** k if m == 1 else 10 ** (k + 1)) if up else (10 ** k if m == 3 else 3 * 10 ** (k - 1))
    return float(f"{nxt:.6g}")


def grid_tag(method, n_in=68):
    """A short checksum of the method's settings tried: saved predictions of another range are not reused."""
    text = json.dumps(grid(method, n_in))
    if method == "svr":
        text += f"|growing range, flat {SVR_FLAT:g}, {SVR_MAX_STEPS} steps, limits {sorted(SVR_LIMITS.items())}"
    return hashlib.md5(text.encode()).hexdigest()[:8]


def at_edges(method, setting, n_in=68):
    """The settings chosen that lie at an end of their (fixed) range, where it could go further."""
    values = {}
    for s in grid(method, n_in):
        for k, v in s.items():
            values.setdefault(k, set()).add(v)
    return [f"{k}={v:g} ({'lowest' if v == min(values[k]) else 'highest'} tried)" for k, v in setting.items()
            if len(values[k]) > 1 and v in (min(values[k]), max(values[k])) and (method, k, v) not in NATURAL_ENDS]


def _svr(setting):
    from sklearn.svm import SVR
    return SVR(kernel="rbf", cache_size=1000, **setting)


def _fit_svr(x, y, xv, setting):
    """Validation predictions of one setting (the model itself stays in the worker, to save memory)."""
    t = time.time()
    m = _svr(setting).fit(x, y)
    return m.predict(xv), time.time() - t


def svr_search(x, y, xs, ym, ys, true_val, jobs):
    """Support vector regression over a growing range (see SVR_START); returns as search."""
    from joblib import Parallel, delayed
    n_in = x.shape[1]
    vals = {k: list(v) for k, v in SVR_START.items()}
    steps = {(k, up): 0 for k in vals for up in (False, True)}
    as_setting = lambda c: {"C": c[0], "gamma": c[1] / n_in, "epsilon": c[2]}
    err, secs = {}, {}
    while True:
        combos = [c for c in itertools.product(vals["C"], vals["g"], vals["epsilon"]) if c not in err]
        res = Parallel(n_jobs=jobs)(delayed(_fit_svr)(x, y, xs["val"], as_setting(c)) for c in combos)
        for c, (pval, sec) in zip(combos, res):
            err[c], secs[c] = rel(pval * ys + ym, true_val), sec
        best = min(err, key=err.get)
        grown, edges = False, []
        for i, k in enumerate(("C", "g", "epsilon")):
            lst = sorted(vals[k])
            for up, end, inner in ((False, lst[0], lst[1]), (True, lst[-1], lst[-2])):
                if best[i] != end:
                    continue
                neighbour = list(best)
                neighbour[i] = inner
                if err[tuple(neighbour)] - err[best] <= SVR_FLAT:
                    continue  # flat towards the end: nothing to gain beyond it
                limit = SVR_LIMITS.get((k, up), SVR_MAX_STEPS)
                if steps[(k, up)] < limit:
                    vals[k].append(step(end, up))
                    steps[(k, up)] += 1
                    grown = True
                else:
                    name, v = ("gamma", end / n_in) if k == "g" else (k, end)
                    why = (f"after {SVR_MAX_STEPS} extra steps" if limit == SVR_MAX_STEPS
                           else "not grown further, for the run time")
                    edges.append(f"{name}={v:g} ({'lowest' if not up else 'highest'} tried, {why}; "
                                 f"{100 * (err[tuple(neighbour)] - err[best]):.3f} points better than the next "
                                 f"value inside)")
        if not grown:
            break
    m = _svr(as_setting(best)).fit(x, y)
    pv = {name: m.predict(v) * ys + ym for name, v in xs.items()}
    assert abs(rel(pv["val"], true_val) - err[best]) < 1e-9, "the refit differs from the search"
    return pv, as_setting(best), [(as_setting(c), err[c], secs[c]) for c in err], edges


def search(method, x, y, xs, ym, ys, true_val, seed, jobs):
    """(predictions on val and test of the setting with the lowest validation error, that setting,
    rows of every setting tried, the settings chosen at an end of their range)."""
    if method == "svr":
        return svr_search(x, y, xs, ym, ys, true_val, jobs)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    tried, best = [], None
    if method == "polynomial":
        for degree in (1, 2):
            if degree == 1:
                tr, ev = x, xs
            else:
                poly = PolynomialFeatures(2, include_bias=False).fit(x)
                sc = StandardScaler().fit(poly.transform(x))
                tr = sc.transform(poly.transform(x))
                ev = {k: sc.transform(poly.transform(v)) for k, v in xs.items()}
            for setting in grid(method, x.shape[1]):
                if setting["degree"] != degree:
                    continue
                t = time.time()
                m = Ridge(alpha=setting["alpha"]).fit(tr, y)
                pv = {k: m.predict(v) * ys + ym for k, v in ev.items()}
                tried.append((setting, rel(pv["val"], true_val), time.time() - t))
                if best is None or tried[-1][1] < best[1]:
                    best = (pv, tried[-1][1], setting)
    else:
        for setting in grid(method, x.shape[1]):
            t = time.time()
            m = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1, **setting).fit(x, y)
            pv = {k: m.predict(v) * ys + ym for k, v in xs.items()}
            tried.append((setting, rel(pv["val"], true_val), time.time() - t))
            if best is None or tried[-1][1] < best[1]:
                best = (pv, tried[-1][1], setting)
    return best[0], best[2], tried, at_edges(method, best[2], x.shape[1])


def run_new(sets, parts, args):
    """Train the three methods where their predictions are not saved yet; returns
    {(method, target, fraction, seed): {"val", "test", "setting", "edges", "n"}}."""
    os.makedirs(os.path.join(args.out, "preds"), exist_ok=True)
    search_path = os.path.join(args.out, "search.csv")
    new_search = not os.path.exists(search_path)
    jobs = [(m, "cl", f, s) for f in args.fractions for s in SEEDS for m in METHODS]
    jobs += [(m, "cd", 1.0, s) for s in SEEDS for m in METHODS]
    out, memo = {}, {}
    with open(search_path, "a", newline="") as fs:
        w = csv.writer(fs)
        if new_search:
            w.writerow(["method", "target", "fraction", "seed", "n_train", "setting", "val_rel_error", "seconds"])
        for method, target, fraction, seed in jobs:
            path = os.path.join(args.out, "preds", f"{method}_{grid_tag(method)}_{target}_f{fraction:g}_s{seed}.npz")
            train = parts[(fraction, seed)]
            if os.path.exists(path):
                with np.load(path, allow_pickle=False) as d:
                    setting = json.loads(str(d["setting"]))
                    edges = json.loads(str(d["edges"])) if "edges" in d.files else at_edges(method, setting)
                    out[(method, target, fraction, seed)] = {"val": d["val"], "test": d["test"], "n": int(d["n"]),
                                                             "setting": setting, "edges": edges}
                assert out[(method, target, fraction, seed)]["n"] == len(train), path
                continue
            # polynomial and SVR do not depend on the seed: the same subset gives the same model
            ids = hashlib.md5("\n".join(r["file"] for r in train).encode()).hexdigest()
            memo_key = (method, target, ids) if method != "random_forest" else None
            t = time.time()
            if memo_key in memo:
                pv, setting, edges = memo[memo_key]
            else:
                key = KEYS[target]
                x, y, xs, ym, ys = scaled(train, {"val": sets["val"], "test": sets["test"]}, key)
                true_val = np.array([r[key] for r in sets["val"]])
                pv, setting, tried, edges = search(method, x, y, xs, ym, ys, true_val, seed, args.jobs)
                for s, err, sec in tried:
                    w.writerow([method, target, fraction, seed, len(train), json.dumps(s), f"{err:.6g}", f"{sec:.1f}"])
                fs.flush()
                if memo_key:
                    memo[memo_key] = (pv, setting, edges)
            np.savez(path, val=pv["val"], test=pv["test"], n=len(train), setting=json.dumps(setting),
                     edges=json.dumps(edges))
            out[(method, target, fraction, seed)] = {"val": pv["val"], "test": pv["test"], "n": len(train),
                                                     "setting": setting, "edges": edges}
            true = {k: np.array([r[KEYS[target]] for r in sets[k]]) for k in ("val", "test")}
            print(f"{method:14s} {target} n={len(train):5d} seed {seed}: val {100 * rel(pv['val'], true['val']):.2f}%, "
                  f"test {100 * rel(pv['test'], true['test']):.2f}%  {setting}  ({time.time() - t:.0f} s)"
                  + (f"  AT END OF RANGE: {', '.join(edges)}" if edges else ""), flush=True)
    return out


# ---------------------------------------------------------------- the models of the final runs

def saved(folder, name, column, recs, true_column, true_key):
    """Predictions in column of folder/errors_<name>.csv, in the order of recs (checking the file's
    true values against the records)."""
    with open(os.path.join(folder, f"errors_{name}.csv")) as f:
        rows = {r["file"]: r for r in csv.DictReader(f)}
    if set(rows) != {r["file"] for r in recs}:
        raise SystemExit(f"{folder}: other {name} samples than here")
    off = max(abs(float(rows[r["file"]][true_column]) - r[true_key]) for r in recs)
    if off > 2e-4:  # the files keep 5-6 significant digits
        raise SystemExit(f"{folder}: the {name} samples' {true_column} differs from ours by up to {off:.1e}")
    return np.array([float(rows[r["file"]][column]) for r in recs])


def final_models(sets, final, out):
    """{(id, target): {"label", "val": [per seed], "test": [per seed]}} of the final runs' models."""
    train = sets["train"]
    line = np.polyfit([r["cl_grid"] for r in train], [r["cl_openfoam"] for r in train], 1)
    compare = os.path.join(final, "baseline_cl_openfoam", "compare.json")
    if os.path.exists(compare):  # written by baseline_cl.py --compare (not in the Zenodo record)
        with open(compare) as f:
            if not np.allclose(line, json.load(f)["grid_to_openfoam_line"], rtol=0, atol=1e-10):
                raise SystemExit("the line from the grid's Cl to OpenFOAM's differs from compare.json")
    models = {}
    runs = ("s0", "s1")
    unet_runs = ("f1_scratch_s0", "f1_scratch_s1")
    for mid, label, target, folder, column, true_column, true_key, mapped, seeds in (
            ("unet", "U-Net, 80,000 iterations (Cl from its pressure)", "cl", "final_unet", "cl_pred", "cl_cfd",
             "cl_grid", True, unet_runs),
            ("unet_160k", "U-Net, 160,000 iterations (Cl from its pressure)", "cl", "final_unet_160k", "cl_pred",
             "cl_cfd", "cl_grid", True, unet_runs),
            ("network", "Downforce network", "cl", "baseline_cl_openfoam", "cl_pred", "cl_openfoam", "cl_openfoam",
             False, runs),
            ("network_grid", "Downforce network trained on the grid's Cl", "cl", "baseline_cl", "cl_pred",
             "cl_openfoam", "cl_openfoam", True, runs),
            ("surface", "Surface model", "cl", "surface_model", "cl_pred", "cl_openfoam", "cl_openfoam", False, runs),
            ("network", "Drag network", "cd", "baseline_cd", "cd_pred", "cd_openfoam", "cd_openfoam", False, runs),
            ("surface", "Surface model", "cd", "surface_model", "cd_pred", "cd_openfoam", "cd_openfoam", False, runs)):
        m = {"label": label, "val": [], "test": []}
        for s in seeds:
            for name in ("val", "test"):
                p = saved(os.path.join(final, folder, s), name, column, sets[name], true_column, true_key)
                m[name].append(np.polyval(line, p) if mapped else p)
        models[(mid, target)] = m

    # the Gaussian process, rebuilt from gp.json (about 2 GB of memory for its 6,177 x 6,177
    # matrices) once: its checked predictions are kept in preds/gp_<checksum of gp.json>.npz
    gp_json = os.path.join(final, "gp", "gp.json")
    if not os.path.exists(gp_json):  # the fit of gp_sweep.py --fit (not in the Zenodo record)
        print(f"no {gp_json} (the Gaussian process's fit, gp_sweep.py --fit): the Gaussian process is left out",
              flush=True)
        return models
    with open(gp_json, "rb") as f:
        raw = f.read()
    p = json.loads(raw)
    cache = os.path.join(out, "preds", f"gp_{hashlib.md5(raw).hexdigest()[:8]}.npz")
    m = {"label": "Gaussian process", "val": [], "test": []}
    c = p["check"]
    if os.path.exists(cache):
        with np.load(cache) as d:
            m["val"], m["test"] = [d["val"]], [d["test"]]
        source = "predictions from " + os.path.basename(cache)
    else:
        tr = sorted(train, key=lambda r: r["file"])
        if gp_sweep.checksum(tr) != p["files_md5"] or len(tr) != p["n"]:
            raise SystemExit("gp.json was fitted on other training samples")
        x = baseline_cl.features(tr)
        xm, xsd = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-12)
        cl = np.array([r["cl_openfoam"] for r in tr])
        gp = gp_sweep.rebuild((x - xm) / xsd, (cl - p["y_mean"]) / p["y_std"], p)
        for name in ("val", "test"):
            mean, sd = gp.predict((baseline_cl.features(sets[name]) - xm) / xsd)
            m[name].append(mean * p["y_std"] + p["y_mean"])
        if abs((sd * p["y_std"]).sum() - c["sum_sd"]) > 1e-6:
            raise SystemExit("the rebuilt Gaussian process's standard deviations differ from the fit's")
        del gp  # free its matrices before the training starts
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, val=m["val"][0], test=m["test"][0])
        source = "rebuilt"
    pred = m["test"][0]
    if abs(pred.sum() - c["sum_pred"]) > 1e-6 or abs((pred ** 2).sum() - c["sum_pred2"]) > 1e-6:
        raise SystemExit("the Gaussian process's test predictions differ from the fit's (gp.json)")
    models[("gp", "cl")] = m
    print(f"Gaussian process: {source}, checked against gp.json", flush=True)
    return models


# ---------------------------------------------------------------- scores

def score(pred, true):
    e = pred - true
    a = np.abs(e)
    return {"rel_error": float(a.mean() / np.abs(true).mean()), "mae": float(a.mean()),
            "rmse": float(np.sqrt((e ** 2).mean())), "r2": float(1 - (e ** 2).sum() / ((true - true.mean()) ** 2).sum()),
            "median_rel_error": float(np.median(a / np.abs(true))), "largest_error": float(a.max())}


def mean_scores(preds, true):
    """The scores of each seed's predictions, then their mean."""
    per = [score(p, true) for p in preds]
    return {k: float(np.mean([s[k] for s in per])) for k in per[0]}


SCORE_COLUMNS = ["model", "label", "coefficient", "seeds", "test_rel_error", "test_mae", "test_rmse", "test_r2",
                 "test_median_rel_error", "test_largest_error", "val_rel_error", "network_closer_samples",
                 "n_samples", "wilcoxon_p_samples", "network_closer_airfoils", "n_airfoils", "wilcoxon_p_airfoils",
                 "settings", "settings_at_end_of_range"]


def scores(models, sets):
    """One row per model and coefficient, ranked by the test error."""
    from scipy import stats
    test = sets["test"]
    airfoils = np.array([r["airfoil"] for r in test])
    names = sorted(set(airfoils))
    rows = []
    for (mid, target), m in models.items():
        row = {"model": mid, "label": m["label"], "coefficient": target.capitalize(), "seeds": len(m["test"])}
        true = np.array([r[KEYS[target]] for r in test])
        row.update({"test_" + k: v for k, v in mean_scores(m["test"], true).items()})
        row["val_rel_error"] = mean_scores(m["val"], np.array([r[KEYS[target]] for r in sets["val"]]))["rel_error"]
        if mid != "network":  # against the network for the same coefficient
            e = np.mean([np.abs(p - true) for p in m["test"]], axis=0)
            e_ref = np.mean([np.abs(p - true) for p in models[("network", target)]["test"]], axis=0)
            a = np.array([e[airfoils == n].mean() for n in names])
            a_ref = np.array([e_ref[airfoils == n].mean() for n in names])
            row.update({"network_closer_samples": int((e_ref < e).sum()), "n_samples": len(e),
                        "wilcoxon_p_samples": float(stats.wilcoxon(e, e_ref).pvalue),
                        "network_closer_airfoils": int((a_ref < a).sum()), "n_airfoils": len(names),
                        "wilcoxon_p_airfoils": float(stats.wilcoxon(a, a_ref).pvalue)})
        if "settings" in m:
            row["settings"] = json.dumps(m["settings"])
            row["settings_at_end_of_range"] = "; ".join(m["edges"])
        rows.append(row)
    return sorted(rows, key=lambda r: (r["coefficient"] != "Cl", r["test_rel_error"]))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("saved", path)


def learning_curve(new, sets, final):
    """Rows (model, n_train, seeds, test_rel_error) of the downforce, the mean over the seeds."""
    lc = {}
    with open(os.path.join(final, "cl_learning_curve.csv")) as f:
        for r in csv.DictReader(f):
            if r["model"] in ("network", "gp"):
                lc.setdefault((r["model"], int(r["n_train"])), []).append(float(r["test_cl_error"]))
    true = np.array([r["cl_openfoam"] for r in sets["test"]])
    for (method, target, fraction, seed), v in new.items():
        if target == "cl":
            lc.setdefault((method, v["n"]), []).append(rel(v["test"], true))
    order = {"network": 0, "gp": 1, **{m: 2 + i for i, m in enumerate(METHODS)}}
    names = {"network": "Downforce network (fixed settings, the runs of cl_learning_curve.py)",
             "gp": "Gaussian process (the runs of cl_learning_curve.py)", **LABELS}
    return [{"model": m, "label": names[m], "n_train": n, "seeds": len(v), "test_rel_error": float(np.mean(v))}
            for (m, n), v in sorted(lc.items(), key=lambda kv: (order[kv[0][0]], kv[0][1]))]


def main(argv=None):
    args = parse(argv)
    if 1.0 not in args.fractions:
        raise SystemExit("--fractions needs 1 (all the training samples) for the scores")
    sets = load(args)
    parts = subsets(sets, args.fractions)
    models = final_models(sets, args.final, args.out)
    new = run_new(sets, parts, args)
    for method in METHODS:
        for target in ("cl", "cd"):
            runs = [new[(method, target, 1.0, s)] for s in SEEDS]
            keep = runs if method == "random_forest" else runs[:1]  # the others: the same model twice
            models[(method, target)] = {"label": LABELS[method], "val": [r["val"] for r in keep],
                                        "test": [r["test"] for r in keep], "settings": runs[0]["setting"],
                                        "edges": runs[0]["edges"]}
    rows = scores(models, sets)
    lc = learning_curve(new, sets, args.final)
    if args.csv:
        write_csv(args.csv, SCORE_COLUMNS, rows)
    if args.learning_curve_csv:
        write_csv(args.learning_curve_csv, ["model", "label", "n_train", "seeds", "test_rel_error"], lc)

    print("\nTest samples, against OpenFOAM's forces; 'closer': samples and airfoils on which the downforce "
          "(drag) network is closer, Wilcoxon p over the airfoils")
    print(f"{'':3s}{'model':50s} {'error':>7s} {'RMSE':>8s} {'R^2':>6s} {'largest':>8s} {'closer':>16s}  p (airfoils)")
    for r in rows:
        closer = (f"{r['network_closer_samples']:>3d}/{r['n_samples']}, {r['network_closer_airfoils']:>2d}/{r['n_airfoils']}"
                  if "network_closer_samples" in r else "(reference)")
        p = f"{r['wilcoxon_p_airfoils']:.1e}" if "wilcoxon_p_airfoils" in r else ""
        print(f"{r['coefficient']} {r['label']:50s} {100 * r['test_rel_error']:6.2f}% {r['test_rmse']:8.5f} "
              f"{r['test_r2']:6.4f} {r['test_largest_error']:8.5f} {closer:>16s}  {p}")
    print("\nDownforce, test error against the number of training samples (mean over the seeds):")
    sizes = sorted({r["n_train"] for r in lc})
    print(f"{'':28s}" + "".join(f"{n:>8d}" for n in sizes))
    for m in ("network", "gp", *METHODS):
        got = {r["n_train"]: r["test_rel_error"] for r in lc if r["model"] == m}
        print(f"{m:28s}" + "".join(f"{100 * got[n]:7.2f}%" if n in got else " " * 8 for n in sizes))
    ends = [f"{LABELS[k[0]]} {k[1]}, n={v['n']}, seed {k[3]}: {'; '.join(v['edges'])}"
            for k, v in sorted(new.items()) if v["edges"]]
    print(f"\nsettings chosen at an end of their range: {len(ends)} of {len(new)}")
    for line in ends:
        print("  " + line)


if __name__ == "__main__":
    main()
