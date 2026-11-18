"""Pressure and friction along the wing: the simulation against the surface model (surface_model.py),
for test samples picked by the model's error of Cp (--pick: percentiles, so that typical and bad cases
are shown) and for any other samples (--samples). One row per sample. Left: -Cp along the chord on
the side facing the ground and on the side facing up. Right: Cf on both sides, negative where the
flow next to the wall runs back (separated flow). x/c is the position along the chord line of the
wing, from the leading edge (0) to the trailing edge (1).

From the ground_effect folder (Windows):
    ..\\.venv\\Scripts\\python.exe surface_report.py --model data\\unet\\surface_model\\s0 --test data\\dataset\\test --pick 10 50 90 --samples data\\sweep_tyrrell026\\h0134\\tyrrell026_h0134_a+0345_U0100.npz --plot ..\\results\\surface_model_cp.png
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

import baseline_cl
import surface_data
import unet_data

BLUE, AQUA, INK, GRID = "#2a78d6", "#1baf7a", "#333333", "#d9d9d9"


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="a model folder of surface_model.py (s<seed>, with summary.json)")
    ap.add_argument("--test", help="folder with the test samples, for --pick")
    ap.add_argument("--pick", type=float, nargs="*", default=[], help="percentiles of the model's Cp error to pick test samples at")
    ap.add_argument("--samples", nargs="*", default=[], help=".npz files of other samples to show")
    ap.add_argument("--plot", required=True, help="the image to write")
    ap.add_argument("--airfoil-dirs", nargs="+", default=list(unet_data.AIRFOIL_DIRS), help="folders with the airfoil files")
    return ap.parse_args(argv)


def curves(path, net, summary, airfoil_dirs):
    """The simulated and the predicted Cp and Cf of one sample, at the points of each side, with
    their positions along the chord, and the sample's settings."""
    st, n = summary["settings"], summary["settings"]["points"]
    sc = {k: np.asarray(v) for k, v in summary["scale"].items()}
    folder, name = os.path.split(os.path.normpath(path))
    rec = baseline_cl.record(folder, name, airfoil_dirs, 32, {})
    x = (baseline_cl.features([rec]) - sc["x_mean"]) / sc["x_std"]
    with torch.no_grad():
        y = net(torch.tensor(x, dtype=torch.float32)).double().numpy()[0] * sc["y_std"] + sc["y_mean"]
    with np.load(path) as d:
        t = surface_data.targets(d, n, airfoil_dirs)
        alpha = float(d["alpha"])
    o = t["outline"]
    chord = np.array([np.cos(np.radians(alpha)), np.sin(np.radians(alpha))])  # leading -> trailing edge
    u = surface_data.stations(n)
    xc = [0.25 + (o.at(o.s_of(side, u)) - [0.25, 0.0]) @ chord for side in surface_data.SIDES]
    pred = y[:2 * n].reshape(2, n), y[2 * n:4 * n].reshape(2, n)
    return {"xc": xc, "cp": t["cp"], "cf": t["cf"], "cp_pred": pred[0], "cf_pred": pred[1], "rec": rec}


def main(argv=None):
    args = parse(argv)
    with open(os.path.join(args.model, "summary.json")) as f:
        summary = json.load(f)
    st = summary["settings"]
    n = st["points"]
    net = baseline_cl.make_net(68, st["width"], st["depth"], 4 * n + 1)
    net.load_state_dict(torch.load(os.path.join(args.model, "model.pt"), map_location="cpu", weights_only=True))
    net.eval()
    picks = []
    if args.pick:
        with open(os.path.join(args.model, "errors_test.csv")) as f:
            rows = list(csv.DictReader(f))
        err = np.array([float(r["cp_error"]) for r in rows])
        for q in args.pick:
            i = int(np.argmin(np.abs(err - np.percentile(err, q))))
            picks.append((os.path.join(args.test, *rows[i]["file"].split("/")), f"test sample at the {q:g}th percentile of the Cp error"))
            print(f"{q:g}th percentile: {rows[i]['file']} (Cp error {err[i]:.4f})")
    picks += [(p, "") for p in args.samples]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(picks), 2, figsize=(12, 3.3 * len(picks)), squeeze=False)
    for row, (path, note) in zip(axes, picks):
        c = curves(path, net, summary, args.airfoil_dirs)
        r = c["rec"]
        for k, (side, style) in enumerate(zip(surface_data.SIDES, ("-", "--"))):
            label = "side facing the ground" if side == "ground" else "side facing up"
            row[0].plot(c["xc"][k], -c["cp"][k], style, color=BLUE, lw=1.6, label=f"CFD, {label}")
            row[0].plot(c["xc"][k], -c["cp_pred"][k], style, color=AQUA, lw=1.4, label=f"surface model, {label}")
            row[1].plot(c["xc"][k], c["cf"][k], style, color=BLUE, lw=1.6)
            row[1].plot(c["xc"][k], c["cf_pred"][k], style, color=AQUA, lw=1.4)
        row[1].axhline(0, color=INK, lw=0.8)
        row[0].set_ylabel("-Cp")
        row[1].set_ylabel("Cf")
        title = (f"{r['airfoil']}, h/c {r['h_file']:.3f}, α {r['alpha']:.1f}°, U {r['U']:.0f} m/s"
                 + (f" ({note})" if note else ""))
        row[0].set_title(title, fontsize=9, loc="left")
        for ax in row:
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("x/c along the chord")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Pressure (left, suction up) and friction (right) along the wing: CFD against the surface model", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(args.plot, dpi=110)
    print("saved", args.plot)


if __name__ == "__main__":
    main()
