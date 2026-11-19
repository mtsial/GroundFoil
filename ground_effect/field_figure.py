"""Figures of the flow fields: the simulation, the U-Net's prediction and their difference side by
side, one sample per row. One quantity per figure: the pressure (p / (1/2 rho U^2), minus its mean
over the air), Ux / U or Uy / U.

The samples: --samples (paths of .npz files of groundGen.py), and/or --pick, which takes the test
samples (--test) whose field error is closest to the given percentiles, so that typical and bad
cases are shown and not only good ones. The prediction is that of the first --model (the errors in
the docs are those of single models); the field error is the original measure ('all') with the
output set to 0 inside the wing and the ground, and -Cl comes from the simulated and the predicted
pressure on the grid, as in the docs.

In each row the simulated and the predicted field share one colour scale (white at 0, or at the
oncoming speed for Ux), and the difference has its own symmetric scale, so that its size can be
read off the colour bar.

From the ground_effect folder (Windows):
    ..\\.venv\\Scripts\\python.exe field_figure.py --model data\\unet\\distance\\f1_scratch_s0\\model.pt --test data\\dataset\\test --pick 10 50 90 --plot fields_test.png
"""
import argparse
import os

import numpy as np
import torch

import unet_data
from predict import load_model
from train_unet import predict

QUANTITIES = {"p": ("pressure p / (½ρU²), minus its mean over the air", 0.0),
              "Ux": ("velocity Ux / U", 1.0), "Uy": ("velocity Uy / U", 0.0)}  # name, the neutral middle
CHANNEL = {"p": 0, "Ux": 1, "Uy": 2}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", nargs="+", required=True, help="model.pt files; the first one predicts")
    ap.add_argument("--samples", nargs="*", default=[], help=".npz files of samples to show")
    ap.add_argument("--test", help="folder with the test samples, for --pick")
    ap.add_argument("--pick", type=float, nargs="*", default=[], help="percentiles of the field error to pick test samples at")
    ap.add_argument("--quantity", choices=tuple(QUANTITIES), default="p", help="the field to show (default p)")
    ap.add_argument("--plot", required=True, help="the image to write")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for torch (default 2)")
    return ap.parse_args(argv)


def physical(fields):
    """Normalised outputs (N x 3 x 128 x 128) as p / (1/2 rho U^2), Ux / U, Uy / U."""
    f = np.asarray(fields, np.float64) * np.array(unet_data.MAX_TARGETS)[None, :, None, None]
    f[:, 0] *= 2
    return f


def evaluate(net, kind, folder, files):
    """The samples (unet_data.load), the first model's prediction, field errors and Cl."""
    data = unet_data.load(folder, files=files, distance=kind)
    pred = unet_data.zero_solid(predict(net, torch.from_numpy(data["inputs"])), data["inputs"])
    err = unet_data.relative_errors(pred, data["targets"])["all"]
    cl_pred, cl_cfd = unet_data.lift_errors(pred, data["targets"], data["inputs"])
    return data, pred, err, cl_pred, cl_cfd


def main(argv=None):
    args = parse(argv)
    torch.set_num_threads(args.threads)
    net, kind = load_model(args.model[0])
    rows = []  # (folder, file, note, index in data, data, prediction, field errors, Cl predicted, Cl simulated)
    if args.pick:
        if not args.test:
            raise SystemExit("--pick needs --test")
        files = [os.path.relpath(f, args.test).replace(os.sep, "/") for f in unet_data.sample_files(args.test)]
        data, pred, err, cl_pred, cl_cfd = evaluate(net, kind, args.test, files)
        for q in args.pick:
            i = int(np.argmin(np.abs(err - np.percentile(err, q))))
            rows.append((args.test, data["file"][i], f"test sample at the {q:g}th percentile of the field error", i,
                         data, pred, err, cl_pred, cl_cfd))
            print(f"{q:g}th percentile: {data['file'][i]} (field error {100 * err[i]:.2f}%)")
    for path in args.samples:
        folder, name = os.path.split(os.path.normpath(path))
        rows.append((folder, name, "", 0, *evaluate(net, kind, folder, [name])))
    plot(args, rows)


def plot(args, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    name, middle = QUANTITIES[args.quantity]
    c = CHANNEL[args.quantity]
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 2.9 * len(rows)), squeeze=False)
    for r, (folder, file, note, i, data, pred, err, cl_pred, cl_cfd) in enumerate(rows):
        with np.load(os.path.join(folder, *file.split("/"))) as d:
            outline, ground_y = unet_data.wing_outline(d)
        mask = data["inputs"][i][2] > 0.5
        cfd, net_out = physical(data["targets"][i:i + 1])[0][c], physical(pred[i:i + 1])[0][c]
        lim = float(max(np.abs(cfd[~mask] - middle).max(), np.abs(net_out[~mask] - middle).max()))
        diff = net_out - cfd
        # the difference's scale: 99.5% of the air cells within it (a few cells at the wall would hide the rest)
        dlim = float(np.percentile(np.abs(diff[~mask]), 99.5))
        bottom = max(-1.0, ground_y - 0.08) if np.isfinite(ground_y) else -1.0
        top = min(1.0, outline[:, 1].max() + 0.5)
        panels = ((f"simulation: {name}", cfd, TwoSlopeNorm(middle, middle - lim, middle + lim), "neither"),
                  ("U-Net", net_out, TwoSlopeNorm(middle, middle - lim, middle + lim), "neither"),
                  ("U-Net minus simulation (scale: 99.5% of the cells)", diff, TwoSlopeNorm(0.0, -dlim, dlim), "both"))
        for ax, (title, field, norm, extend) in zip(axes[r], panels):
            # the values belong to the grid points x = -0.5 + 2i/128 (groundGen.grid_x): centre each
            # pixel on its point, half a cell below and to the left of the window [-0.5, 1.5] x [-1, 1]
            half = 1 / unet_data.RES
            ax.set_facecolor("#bbbbbb")  # the masked (solid) points, as grey as the drawn wing and ground
            im = ax.imshow(np.ma.masked_where(mask.T, field.T), origin="lower", cmap="RdBu_r", norm=norm,
                           extent=[-0.5 - half, 1.5 - half, -1 - half, 1 - half])
            ax.add_patch(patches.Polygon(outline, closed=True, facecolor="#bbbbbb", edgecolor="#333333", lw=0.8))
            if np.isfinite(ground_y) and -1 < ground_y < 1:
                ax.axhspan(-1, ground_y, color="#bbbbbb")
                ax.axhline(ground_y, color="#333333", lw=0.8)
            ax.set_ylim(bottom, top)
            ax.set_xticks([-0.5, 0, 0.5, 1, 1.5])
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="3%", pad=0.08), extend=extend)
        label = (f"{data['airfoil'][i]}, h/c {data['h'][i]:.3f}, α {data['alpha'][i]:.1f}°, U {data['U'][i]:.0f} m/s: "
                 f"−Cl {-cl_cfd[i]:.3f} simulated, {-cl_pred[i]:.3f} U-Net; field error {100 * err[i]:.2f}%")
        axes[r][0].set_ylabel("y (chords)")
        axes[r][1].text(0.5, 1.22, label + (f"\n({note})" if note else ""), transform=axes[r][1].transAxes,
                        ha="center", va="bottom", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("x (chords)")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=110)
    print("saved", args.plot)


if __name__ == "__main__":
    main()
