"""Figure and table of step 14: the U-Net against the comparison model that predicts only the
downforce (baseline_cl.py), both against OpenFOAM's force. Left: the downforce error on the test
samples by ride height. Right: the ride-height sweeps of the two wings that neither model saw.

Reads <baseline>/compare.json and <baseline>/sweep_<wing>.csv, written by
    baseline_cl.py --target openfoam ... --sweep <sweeps> --compare <U-Net runs>
and the U-Net's sweep tables (sweep_report.py --csv, the same two U-Nets as in --compare). The
U-Net's Cl comes from its pressure on the 128 x 128 grid; it is mapped to OpenFOAM's by the line in
compare.json (fitted over the training samples), as in baseline_cl.compare.

--csv writes the numbers of the figure: one row per bin of h/c ('error_by_h': the errors in % of
mean |Cl|, as in compare.json) and one per run of each sweep ('sweep_<wing>': -Cl of OpenFOAM, of
the U-Net and of the comparison model, the mean of the two models of each kind and their range).

From the ground_effect folder (Windows):
    ..\\.venv\\Scripts\\python.exe downforce_report.py --baseline data\\unet\\baseline_cl_openfoam --unet-sweeps ..\\results\\sweep_tyrrell026_distance.csv ..\\results\\sweep_naca4412_distance.csv --plot ..\\results\\downforce_model.png --csv ..\\results\\downforce_model.csv
"""
import argparse
import csv
import json
import os

import numpy as np

BLUE, ORANGE, AQUA, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#333333", "#d9d9d9"  # categorical slots 1-3
WINGS = {"tyrrell026": "Tyrrell 026 (F1 wing)", "naca4412": "NACA 4412"}
COLUMNS = ["kind", "x", "n", "cfd", "unet", "unet_min", "unet_max", "model", "model_min", "model_max"]


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", required=True, help="output folder of baseline_cl.py --target openfoam")
    ap.add_argument("--unet-sweeps", nargs="+", required=True, help="the U-Net's sweep tables (sweep_report.py --csv)")
    ap.add_argument("--plot", help="draw the figure into this image")
    ap.add_argument("--csv", help="write the figure's numbers into this file")
    return ap.parse_args(argv)


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def rows(args):
    with open(os.path.join(args.baseline, "compare.json")) as f:
        comp = json.load(f)
    if comp.get("target") != "openfoam":
        raise SystemExit(f"{args.baseline}: compare.json is not against OpenFOAM's Cl (--target openfoam)")
    a, b = comp["grid_to_openfoam_line"]  # Cl_openfoam = a Cl_grid + b; downforce = -Cl
    out = [{"kind": "error_by_h", "x": f"{r['lo']:g}-{r['hi']:g}", "n": r["n"], "unet": 100 * r["unet"],
            "model": 100 * r["net"]} for r in comp["test"]["by_h"]]
    for path in args.unet_sweeps:
        wing = next(w for w in WINGS if w in os.path.basename(path))
        model = {r["run"]: r for r in read(os.path.join(args.baseline, f"sweep_{wing}.csv"))}
        for r in read(path):
            if r["run"] not in model or r["downforce_unet"] in ("", "nan"):
                continue  # free air, or a run whose Cl did not become steady
            m = model[r["run"]]
            to_of = lambda d: a * float(d) - b
            out.append({"kind": f"sweep_{wing}", "x": float(r["h_over_c"]), "cfd": float(r["downforce"]),
                        "unet": to_of(r["downforce_unet"]), "unet_min": to_of(r["unet_min"]),
                        "unet_max": to_of(r["unet_max"]), "model": float(m["downforce_net"]),
                        "model_min": float(m["net_min"]), "model_max": float(m["net_max"])})
            assert abs(float(m["downforce"]) - float(r["downforce"])) < 1e-3, (wing, r["run"])
    return comp, out


def plot(comp, table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sweeps = [k for k in dict.fromkeys(r["kind"] for r in table) if k.startswith("sweep_")]
    fig, axes = plt.subplots(1, 1 + len(sweeps), figsize=(5.2 * (1 + len(sweeps)), 4.6))
    ax = axes[0]
    by_h = [r for r in table if r["kind"] == "error_by_h"]
    x = np.arange(len(by_h))
    test = comp["test"]
    for key, color, marker, name in (("unet", ORANGE, "s", f"U-Net, from its pressure ({100 * test['cl_error_unet']:.1f}% overall)"),
                                     ("model", AQUA, "^", f"comparison model ({100 * test['cl_error_net']:.1f}% overall)")):
        y = [r[key] for r in by_h]
        ax.plot(x, y, "-", color=color, lw=1.5, marker=marker, ms=8, mec="white", mew=1.2, label=name)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(9, -3), fontsize=8, color=INK)
    ax.set_xticks(x, [f"{r['x']}\n(n = {r['n']})" for r in by_h], fontsize=8)
    ax.set_xlabel("ride height h/c")
    ax.set_ylabel("downforce error (% of mean |Cl|)")
    ax.set_ylim(0, 1.45 * max(max(r["unet"], r["model"]) for r in by_h))  # room for the legend above the points
    ax.set_title(f"{test['n']} test samples (30 airfoils not trained on)", fontsize=10)
    ax.legend(loc="upper center", fontsize=8, frameon=False)
    for ax, kind in zip(axes[1:], sweeps):
        rs = sorted((r for r in table if r["kind"] == kind), key=lambda r: r["x"])
        h = [r["x"] for r in rs]
        for key, color, marker, name in (("unet", ORANGE, "s", "U-Net (mean and range of 2)"),
                                         ("model", AQUA, "^", "comparison model (mean and range of 2)")):
            ax.fill_between(h, [r[key + "_min"] for r in rs], [r[key + "_max"] for r in rs], color=color, alpha=0.2, lw=0)
            ax.plot(h, [r[key] for r in rs], "-", color=color, lw=1.5, marker=marker, ms=7, mec="white", mew=1.0, label=name)
        # the CFD on top, with open markers, so that it shows where a model lies on it
        ax.plot(h, [r["cfd"] for r in rs], "-", color=BLUE, lw=1.2, marker="o", ms=9, mfc="none", mew=1.6, zorder=5,
                label="CFD, OpenFOAM's force")
        ax.set_xlabel("ride height h/c")
        ax.set_ylabel("downforce coefficient, -Cl")
        ax.set_title(f"{WINGS[kind[len('sweep_'):]]}, not among the training airfoils", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()  # the CFD first in the legend
        ax.legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1], loc="upper right", fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Downforce: the U-Net against a small network that predicts only the downforce, both against "
                 "OpenFOAM's force", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print("saved", path)


def main(argv=None):
    args = parse(argv)
    comp, table = rows(args)
    for kind in dict.fromkeys(r["kind"] for r in table):
        rs = [r for r in table if r["kind"] == kind]
        if kind == "error_by_h":
            print("test samples, error by h/c (U-Net / comparison model, %): " +
                  "; ".join(f"{r['x']}: {r['unet']:.2f} / {r['model']:.2f}" for r in rs))
        else:
            dev = lambda k: np.array([r[k] / r["cfd"] - 1 for r in rs]) * 100
            print(f"{kind}: vs OpenFOAM, U-Net mean {dev('unet').mean():+.1f}% (mean |.| {np.abs(dev('unet')).mean():.1f}%), "
                  f"comparison model mean {dev('model').mean():+.1f}% (mean |.| {np.abs(dev('model')).mean():.1f}%)")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for r in table:
                w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
        print("saved", args.csv)
    if args.plot:
        plot(comp, table, args.plot)


if __name__ == "__main__":
    main()
