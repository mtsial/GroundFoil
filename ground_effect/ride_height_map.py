"""Where does the downforce peak? Ride-height sweeps of every airfoil of the data, predicted by the
force networks of baseline_cl.py, at chosen incidences and speeds; and the same networks against
simulated ride-height sweeps (sweep_peaks.sh, sweep_naca4412.sh, sweep_tyrrell026.sh).

The map (--data). The airfoils are those with converged runs in the folders of --data (the
samples.csv files of the w<N> folders of dataset.sh): the training airfoils and, with the test
folder, the test airfoils. For each airfoil, incidence (--alphas) and speed (--speeds), the
networks of --models (the s<seed> folders of baseline_cl.py; the final ones learnt OpenFOAM's Cl)
predict Cl at --heights ride heights spaced evenly on a log scale over the range of the data, h/c
0.05 to 0.6. Along each sweep the downforce -Cl (the mean of the networks) peaks at its largest
value; between two heights the peak is placed by a parabola in log h through the largest value
and its two neighbours. A peak at the lowest height means that the downforce still rises at
h/c = 0.05, one at the highest that it falls all the way down.

The networks learnt only from runs whose Cl became steady. Close to the ground at high incidence
many runs did not (Fig. 2a of the paper): there the predictions rest on fewer samples, and the
real flow may be unsteady. Each sweep therefore notes the share of the training runs that did not
converge in the bin of log h and alpha around its peak (the bins of that figure).

The comparison (--sweeps). Each folder of --sweeps is a sweep (runs in <run>/samples.csv, as
sweep_naca4412.sh writes them) or holds sweeps in its subfolders (as sweep_peaks.sh writes them).
The networks predict each run in ground effect whose Cl became steady, from its airfoil file, its
ride height (the gap between the lowest point of the wing and the ground, from the outline, as in
training: with --h-alpha, as for the Tyrrell 026, it differs from the height set), incidence and
speed. The peak of each sweep is found in the same way for the simulations and for the networks:
the largest downforce among the converged runs, and the parabola in log h through it and the runs
on either side. The networks' peak along a fine sweep (200 heights over the same range) is given
as well. A peak with a run that did not converge next to it (or below it, when the peak is at the
lowest converged run) is marked: the largest downforce may lie where RANS has no steady solution.

Writes into --out:
- with --data: peaks.csv, one row per airfoil, incidence and speed: the peak ride height and
  downforce, each network's peak ride height, the downforce at the two ends (h/c 0.6 and 0.05),
  the gain of the peak over h/c = 0.6, the loss from the peak down to h/c = 0.05, the share of the
  training runs near the peak that did not converge, and the airfoil's shape: largest camber and
  thickness (in percent of the chord) and where along the chord they lie, and the angle of the
  camber line at the trailing edge (between 90% and 99% of the chord; positive when it turns
  towards the side that faces the ground, as a flap would); and curves.npz, the downforce along
  every sweep (mean of the networks, and the range between them), for the figures;
- with --sweeps: sweeps.csv, one row per simulated sweep (its runs, the networks' error along it,
  the simulated and the predicted peak, the gain of the peak over the highest run), and
  sweep_points.csv, one row per run (also those that did not converge: the networks' downforce at
  the height set, without a simulated one to compare).
Each prints a summary: for the map, the peak ride heights by incidence and speed, what they
follow, how far the curves collapse onto one when scaled by their peak, and the test airfoils; for
the comparison, each sweep, then the errors and the peaks over all the sweeps, and how the peaks
move with the incidence and the speed, simulated and predicted.

Before either, each network predicts its own test samples again from the inputs built here (the
shape from the airfoil file, h/c, alpha and U as stored in errors_test.csv, rounded): they must
agree with its stored predictions.

On Windows, from the ground_effect folder (about a minute on the laptop's CPU):
    ..\\.venv\\Scripts\\python.exe ride_height_map.py --data data\\dataset\\train data\\dataset\\test --models data\\unet\\final\\baseline_cl_openfoam --out data\\ride_height_map
    ..\\.venv\\Scripts\\python.exe ride_height_map.py --sweeps data\\sweep_peaks data\\sweep_naca4412 data\\sweep_tyrrell026 --models data\\unet\\final\\baseline_cl_openfoam --out data\\ride_height_map
"""
import argparse
import csv
import glob
import json
import math
import os
import time

import numpy as np
import torch

import baseline_cl
import unet_data

H_EDGES = np.geomspace(0.05, 0.6, 9)  # bins of log h and alpha of Fig. 2a of the paper
A_EDGES = np.linspace(0.0, 10.0, 11)
SHAPE_STATIONS = 200  # stations for the shape descriptors (the networks see 32)
FINE = 200  # heights of the networks' fine sweep along a simulated sweep


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", nargs="*", default=[],
                    help="for the map: dataset folders with w<N>/samples.csv (training first; a folder named test gives the test airfoils)")
    ap.add_argument("--sweeps", nargs="*", default=[],
                    help="for the comparison: folders of simulated sweeps (<run>/samples.csv), or folders of such folders")
    ap.add_argument("--models", required=True, help="folder with the networks of baseline_cl.py (s<seed>/model.pt, summary.json)")
    ap.add_argument("--out", required=True, help="folder for the tables")
    ap.add_argument("--alphas", type=float, nargs="+", default=[float(a) for a in range(11)],
                    help="incidences of the map in degrees (default 0 1 ... 10)")
    ap.add_argument("--speeds", type=float, nargs="+", default=[10.0, 30.0, 50.0],
                    help="speeds of the map in m/s (default 10 30 50: Re 1, 3 and 5 million with the 1 m chord)")
    ap.add_argument("--heights", type=int, default=50, help="ride heights per sweep of the map (default 50)")
    ap.add_argument("--h-range", type=float, nargs=2, default=[0.05, 0.6], help="lowest and highest h/c of the map (default 0.05 0.6)")
    ap.add_argument("--airfoil-dirs", nargs="+", default=list(unet_data.AIRFOIL_DIRS), help="folders with the airfoil files")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for torch (default: torch's choice)")
    args = ap.parse_args(argv)
    if not args.data and not args.sweeps:
        ap.error("give --data (the map), --sweeps (the comparison), or both")
    return args


# ---------------------------------------------------------------- runs, airfoils and networks

def read_runs(folders):
    """Every run in the samples.csv files of the folders, with the name of its folder as 'set'."""
    rows = []
    for folder in folders:
        name = os.path.basename(os.path.normpath(folder))
        paths = sorted(glob.glob(os.path.join(folder, "w*", "samples.csv")))
        if not paths:
            raise SystemExit(f"no w*/samples.csv in {folder}")
        for path in paths:
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    rows.append({"set": name, "airfoil": r["airfoil"], "h": float(r["h_over_c"]),
                                 "alpha": float(r["alpha_deg"]), "U": float(r["U"]), "converged": r["converged"] == "1"})
    return rows


def not_converged_share(runs):
    """Runs and the share that did not converge, per bin of log h and alpha (Fig. 2a)."""
    h = np.array([r["h"] for r in runs])
    a = np.array([r["alpha"] for r in runs])
    bad = ~np.array([r["converged"] for r in runs])
    total, _, _ = np.histogram2d(h, a, (H_EDGES, A_EDGES))
    failed, _, _ = np.histogram2d(h[bad], a[bad], (H_EDGES, A_EDGES))
    return total, failed / np.maximum(total, 1)


def bin_of(edges, v):
    return int(np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2))


def shape_descriptors(points):
    """Largest camber (signed, the one of largest size) and largest thickness in percent of the
    chord, where along the chord they lie, and the angle of the camber line at the trailing edge in
    degrees (from 90% to 99% of the chord; positive when the camber line turns down there, towards
    the side that faces the ground when the airfoil is upside down)."""
    n = SHAPE_STATIONS
    f = baseline_cl.shape_features(points, n)
    camber, thick = f[:n], f[n:]
    x = baseline_cl.stations(n)
    i, j = int(np.argmax(np.abs(camber))), int(np.argmax(thick))
    c90, c99 = np.interp([0.90, 0.99], x, camber)
    return {"camber_pct": 100 * camber[i], "x_camber": x[i], "thickness_pct": 100 * thick[j], "x_thickness": x[j],
            "te_camber_deg": math.degrees(math.atan(-(c99 - c90) / 0.09))}


def load_networks(folder):
    """The networks in folder (s<seed>/model.pt and summary.json of baseline_cl.py)."""
    nets = []
    for path in sorted(glob.glob(os.path.join(folder, "s*", "summary.json"))):
        with open(path) as f:
            summary = json.load(f)
        s = summary["settings"]
        scale = {k: np.asarray(v, dtype=np.float64) for k, v in summary["scale"].items()}
        net = baseline_cl.make_net(len(scale["x_mean"]), s["width"], s["depth"])
        net.load_state_dict(torch.load(os.path.join(os.path.dirname(path), "model.pt"), map_location="cpu", weights_only=True))
        net.eval()
        nets.append({"name": os.path.basename(os.path.dirname(path)), "net": net, "scale": scale,
                     "stations": int(s["stations"]), "target": s["target"], "folder": os.path.dirname(path)})
    if not nets:
        raise SystemExit(f"no s*/summary.json in {folder}")
    if len({n["stations"] for n in nets}) > 1 or len({n["target"] for n in nets}) > 1:
        raise SystemExit("the networks differ in their inputs or in the Cl they learnt")
    if nets[0]["target"] == "cd":
        raise SystemExit(f"{folder}: these networks learnt the drag; the map needs Cl")
    return nets


def predict(net, x):
    """Cl of one network for the unscaled inputs x (N x 68)."""
    s = net["scale"]
    xs = torch.tensor((x - s["x_mean"]) / s["x_std"], dtype=torch.float32)
    with torch.no_grad():
        return net["net"](xs)[:, 0].double().numpy() * s["y_std"] + s["y_mean"]


def inputs(shape, h, alpha, U):
    """The networks' inputs (baseline_cl.features): the shape, h/c, log h/c, alpha and U."""
    h = np.asarray(h, dtype=np.float64)
    n = h.size
    return np.column_stack([np.repeat(shape[None, :], n, axis=0), h, np.log(h),
                            np.broadcast_to(alpha, n), np.broadcast_to(U, n)])


def check_networks(nets, airfoil_dirs):
    """Each network predicts its test samples again: the largest difference from its stored
    predictions (errors_test.csv: h/c and alpha rounded to 4 decimals, U to 3)."""
    worst, shapes = [], {}
    for net in nets:
        with open(os.path.join(net["folder"], "errors_test.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        diff = 0.0
        for r in rows:
            if r["airfoil"] not in shapes:
                shapes[r["airfoil"]] = baseline_cl.shape_features(baseline_cl.airfoil_points(r["airfoil"], airfoil_dirs),
                                                                  net["stations"])
            x = inputs(shapes[r["airfoil"]], [float(r["h_over_c"])], float(r["alpha_deg"]), float(r["U"]))
            diff = max(diff, abs(predict(net, x)[0] - float(r["cl_pred"])))
        worst.append((net["name"], len(rows), diff))
    return worst


# ---------------------------------------------------------------- the peak of one sweep

def peak(h, d):
    """The peak of the downforce d along the ride heights h (ascending): its ride height, its
    downforce, and where it lies ('inside', 'lowest' or 'highest' height). Inside, the top of the
    parabola in log h through the largest value and its two neighbours."""
    i = int(np.argmax(d))
    if i == 0:
        return float(h[0]), float(d[0]), "lowest"
    if i == len(d) - 1:
        return float(h[-1]), float(d[-1]), "highest"
    x0, x1, x2 = np.log(h[i - 1:i + 2])
    y0, y1, y2 = d[i - 1:i + 2]
    # the parabola y = y1 + b (x - x1) + a (x - x1)^2 through the three points
    s0, s2 = (y0 - y1) / (x0 - x1), (y2 - y1) / (x2 - x1)
    a = (s2 - s0) / (x2 - x0)
    b = s0 - a * (x0 - x1)
    if a >= 0:  # no top between the neighbours (cannot happen at a largest value, but be safe)
        return float(h[i]), float(y1), "inside"
    dx = min(max(-b / (2 * a), x0 - x1), x2 - x1)
    return float(math.exp(x1 + dx)), float(y1 + b * dx + a * dx * dx), "inside"


def dips(d, tol=0.005):
    """Runs of a simulated sweep (downforce d in the order of h) whose downforce lies more than tol
    below that of both neighbours: none along a curve that rises to one peak and falls."""
    return sum(1 for i in range(1, len(d) - 1) if d[i] < (1 - tol) * min(d[i - 1], d[i + 1]))


SEP_STATIONS = 400  # points per side for the separation point


def separation_points(d, airfoil_dirs):
    """Where the backward flow next to the wall begins on each side of the wing of sample d (an
    open .npz), in x/c along the chord (from the leading edge, the outline's point farthest from
    the trailing edge, to the middle of the trailing edge): the start of the region of backward
    flow that reaches past 95% of the chord, looked for between 30% and 98% of the chord, as
    validation_zerihan.separation (in the last 2% a small corner eddy at the trailing edge can turn
    the flow forwards again); NaN where there is none (attached). {'ground': x, 'top': x}."""
    import surface_data
    t = surface_data.targets(d, SEP_STATIONS, airfoil_dirs)
    o = t["outline"]
    u = surface_data.stations(SEP_STATIONS)
    le = o.at(np.array([o.s_le]))[0]
    chord = 0.5 * (o.p[0] + o.p[-1]) - le
    out = {}
    for k, side in enumerate(surface_data.SIDES):
        x = (o.at(o.s_of(side, u)) - le) @ chord / (chord @ chord)
        sel = (x > 0.3) & (x < 0.98)
        xs, back = x[sel], t["cf"][k][sel] < 0
        if not back.any() or xs[np.where(back)[0][-1]] < 0.95:
            out[side] = math.nan
            continue
        i = int(np.where(back)[0][-1])
        while i > 0 and back[i - 1]:
            i -= 1
        out[side] = float(xs[i])
    return out


# ---------------------------------------------------------------- the map

def quantiles(v, q=(10, 50, 90)):
    return np.percentile(v, q) if len(v) else [math.nan] * len(q)


def map_summary(rows, args, curves, h):
    from scipy import stats
    print("\nPeak ride height h/c (median, 10-90% of the airfoils) and where the peak lies, by incidence and speed:")
    for U in args.speeds:
        print(f"  U = {U:g} m/s (Re {U / 10:g} million)")
        for a in args.alphas:
            sel = [r for r in rows if r["U"] == U and r["alpha_deg"] == a]
            inside = [r for r in sel if r["peak_at"] == "inside"]
            hp = [r["h_peak"] for r in inside]
            q = quantiles(hp)
            low = sum(r["peak_at"] == "lowest" for r in sel)
            high = sum(r["peak_at"] == "highest" for r in sel)
            gain = quantiles([r["gain_pct"] for r in sel])
            fail = np.median([r["not_converged_pct"] for r in sel])
            print(f"    alpha {a:4.1f}: peak inside {len(inside):4d}/{len(sel)}, at 0.05 {low:4d}, at 0.6 {high:3d}; "
                  f"h/c {q[1]:.3f} ({q[0]:.3f}-{q[2]:.3f}); gain over 0.6 {gain[1]:5.1f}% ({gain[0]:.1f}-{gain[2]:.1f}); "
                  f"not converged near the peak {fail:4.1f}% (median)")

    # the two networks: how far apart are their peaks?
    names = [k[len("h_peak_"):] for k in rows[0] if k.startswith("h_peak_")]
    if len(names) >= 2:
        both = [r for r in rows if r["peak_at"] == "inside"]
        ratio = np.array([abs(math.log(r["h_peak_" + names[0]] / r["h_peak_" + names[1]])) for r in both])
        print(f"\nThe networks {names[0]} and {names[1]} place the peaks (inside the range) {100 * (math.exp(np.median(ratio)) - 1):.1f}% apart "
              f"in h/c (median; 90th percentile {100 * (math.exp(np.percentile(ratio, 90)) - 1):.1f}%)")

    # Reynolds number: peaks of the same airfoil and incidence at the lowest and highest speed
    lo, hi = min(args.speeds), max(args.speeds)
    if lo != hi:
        key = {(r["airfoil"], r["alpha_deg"], r["U"]): r for r in rows}
        print(f"\nPeak at {hi:g} m/s over that at {lo:g} m/s, same airfoil and incidence (both inside the range):")
        for a in args.alphas:
            pairs = [(key[(f, a, lo)], key[(f, a, hi)]) for (f, aa, U) in key if aa == a and U == lo]
            r = [q["h_peak"] / p["h_peak"] for p, q in pairs if p["peak_at"] == q["peak_at"] == "inside"]
            g = [q["downforce_peak"] / p["downforce_peak"] for p, q in pairs]
            if r:
                print(f"    alpha {a:4.1f}: h/c ratio {np.median(r):.3f} ({np.percentile(r, 10):.3f}-{np.percentile(r, 90):.3f}), "
                      f"peak downforce ratio {np.median(g):.3f}; n = {len(r)}")

    # what does the peak follow? rank correlations at one speed, peaks inside the range
    mid = sorted(args.speeds)[len(args.speeds) // 2]
    sel = [r for r in rows if r["U"] == mid and r["peak_at"] == "inside"]
    if len(sel) > 10:
        lh = np.log([r["h_peak"] for r in sel])
        print(f"\nWhat the peak ride height follows (U = {mid:g} m/s, the {len(sel)} sweeps with the peak inside the range; "
              f"Spearman's rank correlation with log h/c of the peak; also with the peak downforce and the gain):")
        for k in ("alpha_deg", "camber_pct", "thickness_pct", "x_camber", "x_thickness", "te_camber_deg"):
            v = np.array([r[k] for r in sel])
            rho_h = stats.spearmanr(v, lh)[0]
            rho_d = stats.spearmanr(v, [r["downforce_peak"] for r in sel])[0]
            rho_g = stats.spearmanr(v, [r["gain_pct"] for r in sel])[0]
            print(f"    {k:15s} h/c of the peak {rho_h:+.2f}, peak downforce {rho_d:+.2f}, gain {rho_g:+.2f}")
        # within one incidence, the shape alone
        for a in args.alphas:
            s2 = [r for r in sel if r["alpha_deg"] == a]
            if len(s2) > 50 and a in (2.0, 5.0, 8.0):
                lh2 = np.log([r["h_peak"] for r in s2])
                text = ", ".join(f"{k} {stats.spearmanr([r[k] for r in s2], lh2)[0]:+.2f}"
                                 for k in ("camber_pct", "thickness_pct", "x_camber", "x_thickness", "te_camber_deg"))
                print(f"    alpha {a:g} only (n = {len(s2)}): {text}")
        # a straight line in log h/c over the standardised descriptors: which matter most?
        keys = ["alpha_deg", "camber_pct", "thickness_pct", "x_camber", "x_thickness", "te_camber_deg"]
        X = np.array([[r[k] for k in keys] for r in sel])
        Xs = (X - X.mean(0)) / X.std(0)
        A = np.column_stack([np.ones(len(sel)), Xs])
        coef, *_ = np.linalg.lstsq(A, lh, rcond=None)
        r2 = 1 - np.sum((A @ coef - lh) ** 2) / np.sum((lh - lh.mean()) ** 2)
        print("    straight line in log h/c over the standardised descriptors (change of h/c per standard deviation): "
              + ", ".join(f"{k} {100 * (math.exp(c) - 1):+.1f}%" for k, c in zip(keys, coef[1:])) + f"; R2 {r2:.2f}")

    # do the curves collapse when scaled by the peak? downforce / peak downforce at h / h_peak
    print("\nThe sweeps scaled by their peak (peak inside the range, U = "
          f"{mid:g} m/s): downforce / peak downforce at given h / h_peak (median, 10-90%):")
    ia = {a: k for k, a in enumerate(args.alphas)}
    iu = {U: k for k, U in enumerate(args.speeds)}
    names_idx = {f: k for k, f in enumerate(curves["airfoils"])}
    for a in args.alphas:
        sel = [r for r in rows if r["U"] == mid and r["alpha_deg"] == a and r["peak_at"] == "inside"]
        if len(sel) < 20:
            continue
        parts = []
        for x in (0.5, 0.7, 1.5, 2.0, 3.0):
            v = []
            for r in sel:
                hx = r["h_peak"] * x
                if h[0] <= hx <= h[-1]:
                    d = curves["downforce"][names_idx[r["airfoil"]], iu[mid], ia[a]]
                    v.append(np.interp(math.log(hx), np.log(h), d) / r["downforce_peak"])
            if len(v) >= 20:
                q = np.percentile(v, (10, 50, 90))
                parts.append(f"{x:g}: {q[1]:.3f} ({q[0]:.3f}-{q[2]:.3f}, n {len(v)})")
        print(f"    alpha {a:4.1f}: " + "; ".join(parts))

    # the test airfoils, to choose wings for new simulations
    tests = sorted({r["airfoil"] for r in rows if r["set"] == "test"})
    if tests:
        show = [a for a in (2.0, 5.0, 8.0) if a in args.alphas]
        print(f"\nTest airfoils at U = {mid:g} m/s: camber, thickness (% chord), peak h/c (and peak downforce) at "
              + ", ".join(f"{a:g}" for a in show) + " deg ('<' at 0.05, '>' at 0.6):")
        key = {(r["airfoil"], r["alpha_deg"], r["U"]): r for r in rows}
        for f in tests:
            r0 = key[(f, show[0], mid)]
            text = []
            for a in show:
                r = key[(f, a, mid)]
                mark = {"inside": " ", "lowest": "<", "highest": ">"}[r["peak_at"]]
                text.append(f"{mark}{r['h_peak']:.3f} ({r['downforce_peak']:.2f}, fail {r['not_converged_pct']:.0f}%)")
            print(f"    {f:12s} camber {r0['camber_pct']:5.2f} at {r0['x_camber']:.2f}, thickness {r0['thickness_pct']:5.2f} at "
                  f"{r0['x_thickness']:.2f}, TE {r0['te_camber_deg']:5.1f} deg: " + "  ".join(text))


MAP_COLUMNS = ["airfoil", "set", "alpha_deg", "U", "Re", "h_peak", "peak_at", "downforce_peak", "downforce_h_max",
               "downforce_h_min", "gain_pct", "loss_pct", "not_converged_pct", "runs_near_peak",
               "camber_pct", "x_camber", "thickness_pct", "x_thickness", "te_camber_deg"]


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items() if k in columns})


def make_map(args, nets):
    start = time.time()
    runs = read_runs(args.data)
    train_runs = [r for r in runs if r["set"] == os.path.basename(os.path.normpath(args.data[0]))]
    total, share = not_converged_share(train_runs)
    airfoils = {}  # name -> set, in the order of the folders
    for r in runs:
        if r["converged"]:
            airfoils.setdefault(r["airfoil"], r["set"])
    n_st = nets[0]["stations"]
    shapes, descr = {}, {}
    for f in airfoils:
        pts = baseline_cl.airfoil_points(f, args.airfoil_dirs)
        shapes[f] = baseline_cl.shape_features(pts, n_st)
        descr[f] = shape_descriptors(pts)
    counts = {s: sum(v == s for v in airfoils.values()) for s in dict.fromkeys(airfoils.values())}
    print(f"\nTHE MAP: {len(runs)} runs read; {len(airfoils)} airfoils with converged runs ("
          + ", ".join(f"{n} {s}" for s, n in counts.items()) + f"); {time.time() - start:.0f} s")

    h = np.geomspace(args.h_range[0], args.h_range[1], args.heights)
    names = list(airfoils)
    shape_all = np.array([shapes[f] for f in names])
    grid = (len(names), len(args.speeds), len(args.alphas), len(h))
    per_net = np.empty((len(nets),) + grid)
    cond = np.array([(U, a, hh) for U in args.speeds for a in args.alphas for hh in h])  # speeds x alphas x heights
    chunk = max(1, 200000 // len(cond))
    for i0 in range(0, len(names), chunk):
        sh = shape_all[i0:i0 + chunk]
        x = np.column_stack([np.repeat(sh, len(cond), axis=0), np.tile(cond[:, 2], len(sh)), np.log(np.tile(cond[:, 2], len(sh))),
                             np.tile(cond[:, 1], len(sh)), np.tile(cond[:, 0], len(sh))])
        for k, net in enumerate(nets):
            per_net[k, i0:i0 + len(sh)] = -predict(net, x).reshape((len(sh),) + grid[1:])
    down = per_net.mean(axis=0)
    print(f"sweeps: {len(names)} airfoils x {len(args.speeds)} speeds x {len(args.alphas)} incidences x {len(h)} heights "
          f"predicted in {time.time() - start:.0f} s")

    rows = []
    for i, f in enumerate(names):
        for j, U in enumerate(args.speeds):
            for k, a in enumerate(args.alphas):
                d = down[i, j, k]
                hp, dp, where = peak(h, d)
                bi, bj = bin_of(H_EDGES, hp), bin_of(A_EDGES, a)
                row = {"airfoil": f, "set": airfoils[f], "alpha_deg": a, "U": U, "Re": U * 1e5, "h_peak": hp,
                       "peak_at": where, "downforce_peak": dp, "downforce_h_max": d[-1], "downforce_h_min": d[0],
                       "gain_pct": 100 * (dp / d[-1] - 1), "loss_pct": 100 * (1 - d[0] / dp),
                       "not_converged_pct": 100 * share[bi, bj], "runs_near_peak": int(total[bi, bj]), **descr[f]}
                for n, net in enumerate(nets):
                    row["h_peak_" + net["name"]] = peak(h, per_net[n, i, j, k])[0]
                rows.append(row)
    write_csv(os.path.join(args.out, "peaks.csv"), rows, MAP_COLUMNS + ["h_peak_" + net["name"] for net in nets])
    curves = {"h": h, "alphas": np.array(args.alphas), "speeds": np.array(args.speeds), "airfoils": np.array(names),
              "sets": np.array([airfoils[f] for f in names]), "downforce": down.astype(np.float32),
              "network_range": (per_net.max(axis=0) - per_net.min(axis=0)).astype(np.float32)}
    np.savez_compressed(os.path.join(args.out, "curves.npz"), **curves)
    print(f"wrote {os.path.join(args.out, 'peaks.csv')} ({len(rows)} sweeps) and curves.npz")
    map_summary(rows, args, curves, h)


# ---------------------------------------------------------------- the comparison with simulated sweeps

def sweep_folders(paths):
    """The sweep folders: each folder given that holds runs (<run>/samples.csv), else each of its
    subfolders that does."""
    out = []
    for p in paths:
        if glob.glob(os.path.join(p, "*", "samples.csv")):
            out.append(os.path.normpath(p))
            continue
        subs = sorted({os.path.dirname(os.path.dirname(s)) for s in glob.glob(os.path.join(p, "*", "*", "samples.csv"))})
        if not subs:
            print(f"no runs (<run>/samples.csv) in {p} or in its subfolders yet")
        out += [os.path.normpath(s) for s in subs]
    return out


def one_sweep(folder, nets, airfoil_dirs, shapes):
    """The runs of one simulated sweep with the networks' downforce, and its row of sweeps.csv
    (None if fewer than 3 runs in ground effect became steady)."""
    import sweep_report
    name = os.path.basename(folder)
    runs = [r for r in sweep_report.load(folder) if r["h"] is not None]
    ok = [r for r in runs if r["converged"]]
    if len(ok) < 3:
        print(f"  {name}: {len(ok)} converged runs in ground effect (of {len(runs)} finished): left out for now")
        return [], None
    # every run's inputs, with h from the outline (also for the runs that did not converge, whose
    # samples are in not_converged/)
    recs = {r["run"]: baseline_cl.record(folder, r["file"], airfoil_dirs, nets[0]["stations"], shapes) for r in runs}
    conds = {(rec["airfoil"], round(rec["alpha"], 4), round(rec["U"], 3)) for rec in recs.values()}
    if len(conds) > 1:
        raise SystemExit(f"{folder}: the runs differ in airfoil, incidence or speed: {sorted(conds)}")
    first = next(iter(recs.values()))
    airfoil, alpha, U, shape = first["airfoil"], first["alpha"], first["U"], first["shape"]

    # the networks at every run; the simulated downforce of the converged ones, and where the flow
    # separates on each side
    points = []
    for r in runs:
        rec = recs[r["run"]]
        p = np.array([-predict(net, inputs(shape, [rec["h"]], alpha, U))[0] for net in nets])
        cfd = -rec["cl_openfoam"] if r["converged"] else math.nan
        with np.load(os.path.join(folder, *r["file"].split("/"))) as d:
            sep = separation_points(d, airfoil_dirs)
        points.append({"sweep": name, "run": r["run"], "h_set": r["h"], "h": rec["h"], "alpha_deg": alpha, "U": U,
                       "converged": int(r["converged"]), "reversed_flow_pct": r["reversed"], "iterations": r["iterations"],
                       "downforce_cfd": cfd, "downforce_net": p.mean(), "net_min": p.min(), "net_max": p.max(),
                       "net_vs_cfd_pct": 100 * (p.mean() / cfd - 1) if r["converged"] else math.nan,
                       "x_sep_ground": sep["ground"], "x_sep_top": sep["top"]})
    points.sort(key=lambda q: q["h"])
    conv = [q for q in points if q["converged"]]
    h_c = np.array([q["h"] for q in conv])
    d_c = np.array([q["downforce_cfd"] for q in conv])
    d_n = np.array([q["downforce_net"] for q in conv])
    hp_c, dp_c, at_c = peak(h_c, d_c)
    hp_n, dp_n, at_n = peak(h_c, d_n)
    i = int(np.argmax(d_c))
    lo = h_c[i - 1] if i > 0 else 0.0
    hi = h_c[i + 1] if i < len(h_c) - 1 else h_c[i]
    unsteady = any(not q["converged"] and lo <= q["h"] <= hi for q in points)
    h_f = np.geomspace(h_c[0], h_c[-1], FINE)
    d_f = np.mean([-predict(net, inputs(shape, h_f, alpha, U)) for net in nets], axis=0)
    hp_f, dp_f, at_f = peak(h_f, d_f)
    err = np.abs(d_n - d_c)
    # where the flow on the side facing the ground separates at the simulated peak (1: attached,
    # no backward flow reaching past 95% of the chord), between the converged runs in log h
    x_sep = np.array([1.0 if math.isnan(q["x_sep_ground"]) else q["x_sep_ground"] for q in conv])
    row = {"sweep": name, "airfoil": airfoil, "alpha_deg": alpha, "U": U, "runs": len(runs), "converged": len(conv),
           "error_pct": 100 * err.mean() / np.abs(d_c).mean(), "max_error_pct": 100 * float(np.max(err / np.abs(d_c))),
           "dips_cfd": dips(d_c), "h_peak_cfd": hp_c, "peak_at_cfd": at_c, "downforce_peak_cfd": dp_c,
           "unsteady_near_peak": int(unsteady), "h_peak_net": hp_n, "peak_at_net": at_n, "downforce_peak_net": dp_n,
           "h_peak_net_fine": hp_f, "peak_at_net_fine": at_f, "downforce_peak_net_fine": dp_f,
           "peak_h_ratio": hp_n / hp_c, "peak_downforce_error_pct": 100 * (dp_n / dp_c - 1),
           "h_top": h_c[-1], "gain_cfd_pct": 100 * (dp_c / d_c[-1] - 1), "gain_net_pct": 100 * (dp_n / d_n[-1] - 1),
           "x_sep_at_peak": float(np.interp(math.log(hp_c), np.log(h_c), x_sep))}
    return points, row


SWEEP_COLUMNS = ["sweep", "airfoil", "alpha_deg", "U", "runs", "converged", "error_pct", "max_error_pct", "dips_cfd",
                 "h_peak_cfd", "peak_at_cfd", "downforce_peak_cfd", "unsteady_near_peak", "h_peak_net", "peak_at_net",
                 "downforce_peak_net", "h_peak_net_fine", "peak_at_net_fine", "downforce_peak_net_fine", "peak_h_ratio",
                 "peak_downforce_error_pct", "h_top", "gain_cfd_pct", "gain_net_pct", "x_sep_at_peak"]
POINT_COLUMNS = ["sweep", "run", "h_set", "h", "alpha_deg", "U", "converged", "reversed_flow_pct", "iterations",
                 "downforce_cfd", "downforce_net", "net_min", "net_max", "net_vs_cfd_pct", "x_sep_ground", "x_sep_top"]


def compare_sweeps(args, nets):
    from scipy import stats
    folders = sweep_folders(args.sweeps)
    print(f"\nTHE SIMULATED SWEEPS: {len(folders)} folders")
    shapes, points, rows = {}, [], []
    for folder in folders:
        p, row = one_sweep(folder, nets, args.airfoil_dirs, shapes)
        points += p
        if row:
            rows.append(row)
    if not rows:
        return
    write_csv(os.path.join(args.out, "sweeps.csv"), rows, SWEEP_COLUMNS)
    write_csv(os.path.join(args.out, "sweep_points.csv"), points, POINT_COLUMNS)
    mark = {"inside": " ", "lowest": "<", "highest": ">"}
    flag = lambda r: ("!" if r["dips_cfd"] else "") + ("*" if r["unsteady_near_peak"] else "")
    print(f"  {'sweep':22s} {'alpha':>5s} {'U':>4s} {'steady':>7s} {'error':>6s} {'max':>5s}   {'peak h/c: CFD':>13s} "
          f"{'net':>7s} {'(fine)':>7s} {'ratio':>6s}   {'peak -Cl: CFD':>13s} {'net':>6s}   {'gain: CFD':>9s} {'net':>6s}   "
          f"{'separation at the peak':>22s}")
    for r in rows:
        print(f"  {r['sweep']:22s} {r['alpha_deg']:5.2f} {r['U']:4.0f} {r['converged']:3d}/{r['runs']:<3d} {r['error_pct']:5.2f}% "
              f"{r['max_error_pct']:4.1f}%   {mark[r['peak_at_cfd']]}{r['h_peak_cfd']:6.3f}{flag(r):2s}    "
              f"{mark[r['peak_at_net']]}{r['h_peak_net']:6.3f} {mark[r['peak_at_net_fine']]}{r['h_peak_net_fine']:6.3f} "
              f"{r['peak_h_ratio']:6.3f}   {r['downforce_peak_cfd']:13.3f} {r['downforce_peak_net']:6.3f}   "
              f"{r['gain_cfd_pct']:8.1f}% {r['gain_net_pct']:5.1f}%   {r['x_sep_at_peak']:22.3f}")
    print("  ('<': at the lowest converged run, '>': at the highest; '*': a run that did not converge next to the "
          "simulated peak; '!': the simulated downforce dips at a run, more than 0.5% below both neighbours; error: mean "
          "|difference| / mean downforce along the sweep, max: the largest at one run; separation: x/c where the backward "
          "flow begins on the side facing the ground, 1 if attached)")

    irregular = {r["sweep"] for r in rows if r["dips_cfd"]}
    for label, sweeps in (("all", rows), ("without the sweeps marked '!'", [r for r in rows if r["sweep"] not in irregular])):
        names = {r["sweep"] for r in sweeps}
        conv = [q for q in points if q["converged"] and q["sweep"] in names]
        if not conv:
            continue
        e = np.array([abs(q["downforce_net"] - q["downforce_cfd"]) for q in conv])
        print(f"\n  {label}: {len(conv)} converged runs of {len(sweeps)} sweeps: error "
              f"{100 * e.mean() / np.mean([abs(q['downforce_cfd']) for q in conv]):.2f}% (mean |difference| / mean downforce), "
              f"largest {max(abs(q['net_vs_cfd_pct']) for q in conv):.1f}% at one run")
        if not irregular:
            break
    same = sum(r["peak_at_cfd"] == r["peak_at_net"] for r in rows)
    print(f"  peak inside / at the lowest / at the highest run: the same for the simulations and the networks in {same} of {len(rows)} sweeps")
    both = [r for r in rows if r["peak_at_cfd"] == r["peak_at_net"] == "inside"]
    regular = [r for r in both if not r["unsteady_near_peak"] and r["sweep"] not in irregular]
    for label, sel in (("peak inside the sweep in both", both), ("of those, not marked '*' or '!'", regular)):
        if sel:
            dev = np.array([abs(math.log(r["peak_h_ratio"])) for r in sel])
            dd = np.array([abs(r["peak_downforce_error_pct"]) for r in sel])
            print(f"  {label} ({len(sel)} sweeps): peak h/c {100 * (math.exp(np.median(dev)) - 1):.1f}% apart (median), "
                  f"{100 * (math.exp(dev.max()) - 1):.1f}% at most; peak downforce {np.median(dd):.2f}% (median), {dd.max():.2f}% at most")
    if len(regular) >= 4:
        rho = stats.spearmanr([r["h_peak_cfd"] for r in regular], [r["h_peak_net"] for r in regular])[0]
        print(f"  rank correlation of the peak heights, simulated and predicted, over those {len(regular)} sweeps: {rho:+.2f}")
        xs = np.array([r["x_sep_at_peak"] for r in regular])
        print(f"  where the flow separates at the simulated peak, x/c (those {len(regular)} sweeps): median {np.median(xs):.3f}, "
              f"10-90% {np.percentile(xs, 10):.3f}-{np.percentile(xs, 90):.3f}, range {xs.min():.3f}-{xs.max():.3f}")
        for k in ("alpha_deg", "U", "h_peak_cfd"):
            print(f"    rank correlation with {k}: {stats.spearmanr([r[k] for r in regular], xs)[0]:+.2f}")

    # how the peak moves with the incidence (same airfoil and speed) and with the speed (same airfoil and incidence)
    groups = {}
    for r in rows:
        groups.setdefault((r["airfoil"], round(r["U"])), []).append(r)
    text = []
    for (f, U), sel in sorted(groups.items()):
        if len(sel) >= 2:
            sel.sort(key=lambda r: r["alpha_deg"])
            text.append(f"    {f:10s} U {U:2d}: " + ", ".join(
                f"{r['alpha_deg']:g} deg {mark[r['peak_at_cfd']]}{r['h_peak_cfd']:.3f}{flag(r)}"
                f" / {mark[r['peak_at_net']]}{r['h_peak_net']:.3f}" for r in sel))
    if text:
        print("\n  the peak h/c with the incidence, simulated / predicted:\n" + "\n".join(text))
    groups = {}
    for r in rows:
        groups.setdefault((r["airfoil"], round(r["alpha_deg"], 2)), []).append(r)
    text = []
    for (f, a), sel in sorted(groups.items()):
        if len({round(r["U"]) for r in sel}) >= 2:
            sel.sort(key=lambda r: r["U"])
            text.append(f"    {f:10s} {a:g} deg: " + ", ".join(
                f"U {r['U']:.0f} {mark[r['peak_at_cfd']]}{r['h_peak_cfd']:.3f}{flag(r)}"
                f" / {mark[r['peak_at_net']]}{r['h_peak_net']:.3f} (-Cl {r['downforce_peak_cfd']:.3f} / {r['downforce_peak_net']:.3f})"
                for r in sel))
    if text:
        print("\n  the peak h/c with the speed, simulated / predicted (and the peak downforce):\n" + "\n".join(text))
    print(f"\nwrote {os.path.join(args.out, 'sweeps.csv')} and sweep_points.csv")


# ---------------------------------------------------------------- main

def main(argv=None):
    args = parse(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    nets = load_networks(args.models)
    print(f"{len(nets)} networks ({', '.join(n['name'] for n in nets)}, learnt Cl {nets[0]['target']})")
    for name, n, diff in check_networks(nets, args.airfoil_dirs):
        print(f"  {name}: predicts its {n} test samples again within {diff:.1e} of its stored Cl")
        if diff > 2e-3:
            raise SystemExit(f"{name}: the inputs built here do not reproduce the stored predictions")
    if args.data:
        make_map(args, nets)
    if args.sweeps:
        compare_sweeps(args, nets)


if __name__ == "__main__":
    main()
