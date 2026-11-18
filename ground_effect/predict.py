"""Predict the flow around an airfoil in ground effect with trained U-Nets, without a simulation:
from the airfoil, the ride height, the angle of attack and the speed.

The U-Net's inputs are built as in the dataset: the airfoil is turned upside down and placed by
groundGen.py's own functions, a grid point is solid inside the wing or below the ground, and
models trained with --distance also get the signed distance (unet_data.distance_field, or
mask_distance for --distance mask). Each model's downforce coefficient -Cl comes from its
predicted pressure on the 128 x 128 grid, as everywhere in the docs (unet_data.lift); on the
simulations, this way gives about 3% less downforce than OpenFOAM's forces. With several models,
their mean is printed too.

The airfoil: a .dat file in the UIUC format, the name of one in the dataset's folders
(unet_data.AIRFOIL_DIRS), or nacaXXXX. --h is the gap between the lowest point of the wing and the
ground, in chords, with the wing at --alpha (or at --h-alpha, as in the wind tunnel: see
groundGen.place_airfoil). The models were trained on h/c 0.05-0.6, alpha 0-10 degrees and
10-50 m/s; outside that range the script warns.

--plot draws the predicted pressure and velocity (the mean of the models); --npz saves them:
'p' (pressure / density in m^2/s^2, relative to its mean over the air on the grid), 'Ux' and 'Uy'
(m/s), 'mask' and 'cl' (one per model), on the grid x -0.5..1.5, y -1..1 chords, 128 x 128 with
the x index first (0 inside the solid).

From the ground_effect folder (Windows; well under a second per model on the CPU):
    ..\\.venv\\Scripts\\python.exe predict.py --airfoil naca4412 --h 0.1 --alpha 4 --U 30 --model data\\unet\\distance\\f1_scratch_s0\\model.pt --plot flow.png
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import unet_data
from unet import TurbNetG

RANGES = {"h/c": (0.05, 0.6), "alpha": (0.0, 10.0), "U": (10.0, 50.0)}  # of the training samples


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--airfoil", required=True, help="a .dat file, an airfoil of the dataset's folders, or nacaXXXX")
    ap.add_argument("--h", type=float, required=True, help="ride height h/c: lowest point of the wing to the ground, in chords")
    ap.add_argument("--alpha", type=float, required=True, help="angle of attack in degrees (positive lifts the trailing edge)")
    ap.add_argument("--U", type=float, required=True, help="speed in m/s (chord 1 m, as in the dataset)")
    ap.add_argument("--h-alpha", type=float, help="measure h with the wing at this angle, then turn it to --alpha "
                                                   "about the quarter chord (as in the wind tunnel)")
    ap.add_argument("--model", nargs="+", required=True, help="model.pt files of train_unet.py (summary.json next to them)")
    ap.add_argument("--plot", help="draw the predicted fields into this image")
    ap.add_argument("--npz", help="save the predicted fields into this file")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads for torch (default 1)")
    ap.add_argument("--airfoil-dirs", nargs="+", default=list(unet_data.AIRFOIL_DIRS), help="folders with the airfoil files")
    return ap.parse_args(argv)


def airfoil_points(airfoil, airfoil_dirs=unet_data.AIRFOIL_DIRS):
    """The airfoil's points as the dataset's simulations used them (boundary-layer mesh)."""
    import groundGen  # numpy and the standard library only
    path = airfoil if os.path.isfile(airfoil) else next(
        (p for p in (os.path.join(k, airfoil + ".dat") for k in airfoil_dirs) if os.path.isfile(p)), airfoil)
    try:
        points = groundGen.load_airfoil(path)
    except OSError:
        raise SystemExit(f"airfoil {airfoil}: no such file, not in {list(airfoil_dirs)} and not nacaXXXX")
    return groundGen.tidy_trailing_edge(points)[0]


def build_inputs(points, h, alpha, U, h_alpha=None):
    """The U-Net's inputs (3 x 128 x 128, normalised as unet_data.load does, input 1 = 0), the
    signed distance to the wing and the ground (chords), the outline and the y of the ground."""
    import groundGen
    outline, ground_y = groundGen.place_airfoil(points, alpha, h, h_alpha)
    dist = unet_data.distance_field(outline, ground_y)
    mask = dist < 0
    inputs = np.zeros((3, unet_data.RES, unet_data.RES), np.float32)
    inputs[0] = np.where(mask, 0.0, U / unet_data.MAX_INPUTS[0])  # the freestream x velocity in the air
    inputs[2] = mask
    return inputs, dist.astype(np.float32), outline, ground_y


def load_model(path):
    """The network and the kind of distance input it was trained with (None, 'outline', 'mask')."""
    info = os.path.join(os.path.dirname(path), "summary.json")
    settings = {}
    if os.path.exists(info):
        with open(info) as f:
            settings = json.load(f)["settings"]
    net = TurbNetG(channelExponent=settings.get("expo", 5))
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net, unet_data.distance_kind(settings.get("distance"))


def run_model(net, kind, inputs, dist):
    """The model's normalised output (3 x 128 x 128) for one sample, and the seconds it took."""
    x = inputs.copy()
    if kind == "outline":
        x[1] = dist
    elif kind == "mask":
        x[1] = unet_data.mask_distance(inputs[2])
    with torch.no_grad():
        t = time.perf_counter()
        out = net(torch.from_numpy(x[None]))[0].numpy()
        return out, time.perf_counter() - t


def plot(path, fields, mask, outline, ground_y, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import TwoSlopeNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # diverging colours around a neutral middle: 0 for the pressure and Uy, the oncoming speed for Ux
    panels = (("pressure p / (½ρU²), minus its mean over the air", fields[0], 0.0),
              ("velocity Ux / U (white: the oncoming speed)", fields[1], 1.0), ("velocity Uy / U", fields[2], 0.0))
    bottom = max(-1.0, ground_y - 0.08) if np.isfinite(ground_y) else -1.0  # leave out most of the ground
    top = min(1.0, outline[:, 1].max() + 0.6)
    fig, axes = plt.subplots(1, 3, figsize=(15, 1.0 + 2.2 * (top - bottom)))
    for ax, (name, field, middle) in zip(axes, panels):
        f = np.ma.masked_where(mask.T, field.T)  # stored with the x index first; x horizontal
        lim = float(np.abs(f - middle).max())
        im = ax.imshow(f, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(middle, middle - lim, middle + lim),
                       extent=[-0.5, 1.5, -1, 1])
        ax.add_patch(patches.Polygon(outline, closed=True, facecolor="#bbbbbb", edgecolor="#333333", lw=0.8))
        if -1 < ground_y < 1:
            ax.axhspan(-1, ground_y, color="#bbbbbb")
            ax.axhline(ground_y, color="#333333", lw=0.8)
        ax.set_ylim(bottom, top)
        ax.set_xticks([-0.5, 0, 0.5, 1, 1.5])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x (chords)")
        ax.set_ylabel("y (chords)")
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="3%", pad=0.08))  # as tall as the image
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    args = parse(argv)
    torch.set_num_threads(args.threads)
    for name, value in (("h/c", args.h), ("alpha", args.alpha), ("U", args.U)):
        lo, hi = RANGES[name]
        if not lo <= value <= hi:
            print(f"warning: {name} = {value:g} is outside the training range {lo:g}-{hi:g}")
    import groundGen  # noqa: F401  loaded before the clock starts, as are the modules of distance_field
    import matplotlib.path  # noqa: F401
    import scipy.spatial  # noqa: F401
    t = time.perf_counter()
    points = airfoil_points(args.airfoil, args.airfoil_dirs)
    inputs, dist, outline, ground_y = build_inputs(points, args.h, args.alpha, args.U, args.h_alpha)
    mask = inputs[2] > 0.5
    print(f"inputs built in {1000 * (time.perf_counter() - t):.0f} ms ({int(mask.sum())} solid grid points)")
    outs, cls = [], []
    for path in args.model:
        net, kind = load_model(path)
        run_model(net, kind, inputs, dist)  # the first call of a model is slower: not timed
        out, seconds = run_model(net, kind, inputs, dist)
        cl = float(unet_data.lift(out[0].astype(np.float64) * unet_data.MAX_TARGETS[0], mask))
        outs.append(out)
        cls.append(cl)
        print(f"{path}: -Cl = {-cl:.3f} ({1000 * seconds:.0f} ms, input 1: {kind or 'not used'})")
    if len(cls) > 1:
        print(f"mean of the {len(cls)} models: -Cl = {-np.mean(cls):.3f} (range {-max(cls):.3f} to {-min(cls):.3f})")
    out = unet_data.zero_solid(np.mean(outs, axis=0)[None], inputs[None])[0]
    fields = unet_data.denormalise(out, args.U)  # p / rho in m^2/s^2, Ux and Uy in m/s
    if args.npz:
        np.savez_compressed(args.npz, p=fields[0], Ux=fields[1], Uy=fields[2], mask=mask, cl=np.array(cls))
        print("saved", args.npz)
    if args.plot:
        title = (f"{os.path.basename(args.airfoil)} upside down, h/c = {args.h:g}, alpha = {args.alpha:g} deg, "
                 f"U = {args.U:g} m/s: predicted -Cl = {-np.mean(cls):.3f} (U-Net, {len(cls)} model{'s' * (len(cls) > 1)})")
        scaled = np.array([fields[0] / (0.5 * args.U ** 2), fields[1] / args.U, fields[2] / args.U])
        plot(args.plot, scaled, mask, outline, ground_y, title)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
