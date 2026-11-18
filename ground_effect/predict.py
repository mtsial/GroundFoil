"""Predict the flow around an airfoil in ground effect with the trained models, without a simulation:
from the airfoil, the ride height, the angle of attack and the speed. Three kinds of model, any of
them or all together:
- U-Nets (--model, train_unet.py): the pressure and velocity fields around the wing;
- networks for the downforce alone (--force-model, baseline_cl.py, step 14): Cl;
- surface models (--surface-model, surface_model.py, step 15): the pressure and the friction along
  the wing, and from them Cl, Cd and Cm.

The U-Net's inputs are built as in the dataset: the airfoil is turned upside down and placed by
groundGen.py's own functions, a grid point is solid inside the wing or below the ground, and
models trained with --distance also get the signed distance (unet_data.distance_field, or
mask_distance for --distance mask). Each U-Net's downforce coefficient -Cl comes from its
predicted pressure on the 128 x 128 grid, as everywhere in the docs (unet_data.lift); on the
simulations, this way gives about 3% less downforce than OpenFOAM's forces. The two small networks
get the inputs they were trained with (baseline_cl.features): the camber line and the thickness
along the chord from the airfoil's points, the ride height measured on the placed wing, its
logarithm, the angle and the speed. A force network predicts the Cl it learnt: OpenFOAM's force
(baseline_cl.py --target openfoam) or the grid's Cl (the default target there). A surface model's
Cl, Cd and Cm (about the quarter chord) are its pressure and friction added up over the mesh's wall
faces of this wing, as OpenFOAM adds them up (surface_data.forces); 'reversed flow' is the share of
the surface where its friction is negative (separated flow). With several models of a kind, their
mean is printed too.

The airfoil: a .dat file in the UIUC format, the name of one in the dataset's folders
(unet_data.AIRFOIL_DIRS), or nacaXXXX. --h is the gap between the lowest point of the wing and the
ground, in chords, with the wing at --alpha (or at --h-alpha, as in the wind tunnel: see
groundGen.place_airfoil). The models were trained on h/c 0.05-0.6, alpha 0-10 degrees and
10-50 m/s; outside that range the script warns.

--plot draws the U-Nets' pressure and velocity (their mean); --plot-surface the surface models'
pressure and friction along the chord (their mean). --npz saves what was predicted: from U-Nets
'p' (pressure / density in m^2/s^2, relative to its mean over the air on the grid), 'Ux' and 'Uy'
(m/s) and 'mask', on the grid x -0.5..1.5, y -1..1 chords, 128 x 128 with the x index first (0
inside the solid), and 'cl' (one per U-Net); from force networks 'cl_force' (one per model); from
surface models 'cp' and 'cf' (their mean, 2 x points: the side facing the ground, the side facing
up, from the leading to the trailing edge), 'xc' (the points' positions along the chord), 'cp_base'
and 'cl_surface', 'cd_surface', 'cm_surface' (one per model).

From the ground_effect folder (Windows; well under a second per model on the CPU):
    ..\\.venv\\Scripts\\python.exe predict.py --airfoil naca4412 --h 0.1 --alpha 4 --U 30 --model data\\unet\\distance\\f1_scratch_s0\\model.pt --plot flow.png
    ..\\.venv\\Scripts\\python.exe predict.py --airfoil naca4412 --h 0.1 --alpha 4 --U 30 --force-model data\\unet\\baseline_cl_openfoam\\s0 --surface-model data\\unet\\surface_model\\s0 --plot-surface surface.png
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
BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#333333", "#d9d9d9"  # categorical slots 1 and 2, as in the reports


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--airfoil", required=True, help="a .dat file, an airfoil of the dataset's folders, or nacaXXXX")
    ap.add_argument("--h", type=float, required=True, help="ride height h/c: lowest point of the wing to the ground, in chords")
    ap.add_argument("--alpha", type=float, required=True, help="angle of attack in degrees (positive lifts the trailing edge)")
    ap.add_argument("--U", type=float, required=True, help="speed in m/s (chord 1 m, as in the dataset)")
    ap.add_argument("--h-alpha", type=float, help="measure h with the wing at this angle, then turn it to --alpha "
                                                   "about the quarter chord (as in the wind tunnel)")
    ap.add_argument("--model", nargs="*", default=[], help="U-Nets: model.pt files of train_unet.py (summary.json next to them)")
    ap.add_argument("--force-model", nargs="*", default=[],
                    help="networks for the downforce alone: s<seed> folders of baseline_cl.py (or their model.pt)")
    ap.add_argument("--surface-model", nargs="*", default=[],
                    help="surface models: s<seed> folders of surface_model.py (or their model.pt)")
    ap.add_argument("--plot", help="draw the U-Nets' predicted fields into this image")
    ap.add_argument("--plot-surface", help="draw the surface models' pressure and friction along the chord into this image")
    ap.add_argument("--npz", help="save the predictions into this file")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads for torch (default 1)")
    ap.add_argument("--airfoil-dirs", nargs="+", default=list(unet_data.AIRFOIL_DIRS), help="folders with the airfoil files")
    args = ap.parse_args(argv)
    if not (args.model or args.force_model or args.surface_model):
        ap.error("give at least one model: --model, --force-model or --surface-model")
    if args.plot and not args.model:
        ap.error("--plot draws the U-Nets' fields: give --model")
    if args.plot_surface and not args.surface_model:
        ap.error("--plot-surface draws the surface models' curves: give --surface-model")
    return args


def airfoil_points(airfoil, airfoil_dirs=unet_data.AIRFOIL_DIRS):
    """The airfoil's points as in the file (the small networks' shape inputs were made from these)
    and as the dataset's simulations used them (boundary-layer mesh: groundGen.tidy_trailing_edge)."""
    import groundGen  # numpy and the standard library only
    path = airfoil if os.path.isfile(airfoil) else next(
        (p for p in (os.path.join(k, airfoil + ".dat") for k in airfoil_dirs) if os.path.isfile(p)), airfoil)
    try:
        points = groundGen.load_airfoil(path)
    except OSError:
        raise SystemExit(f"airfoil {airfoil}: no such file, not in {list(airfoil_dirs)} and not nacaXXXX")
    return points, groundGen.tidy_trailing_edge(points)[0]


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


def load_small(path):
    """A network of baseline_cl.py or surface_model.py (a model folder s<seed>, or its model.pt):
    the network, its settings and its scales (inputs and outputs, from its training samples)."""
    import baseline_cl
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    with open(os.path.join(folder, "summary.json")) as f:
        summary = json.load(f)
    st = summary["settings"]
    sc = {k: np.asarray(v, dtype=np.float64) for k, v in summary["scale"].items()}
    net = baseline_cl.make_net(sc["x_mean"].size, st["width"], st["depth"], sc["y_mean"].size)
    net.load_state_dict(torch.load(os.path.join(folder, "model.pt"), map_location="cpu", weights_only=True))
    net.eval()
    return net, st, sc


def small_inputs(raw, outline, ground_y, alpha, U, n_in):
    """The inputs of a small network, unscaled (baseline_cl.features): the shape at the stations its
    inputs had, from the airfoil's points as in the file, and the ride height of the placed wing."""
    import baseline_cl
    shape = baseline_cl.shape_features(raw, (n_in - 4) // 2)
    rec = {"shape": shape, "h": float(outline[:, 1].min() - ground_y), "alpha": alpha, "U": U}
    return baseline_cl.features([rec])


def run_small(net, sc, x):
    """The network's outputs for one sample, unscaled, and the seconds it took (after a first call,
    which is slower and not timed)."""
    xs = torch.tensor((x - sc["x_mean"]) / sc["x_std"], dtype=torch.float32)
    with torch.no_grad():
        net(xs)
        t = time.perf_counter()
        y = net(xs).double().numpy()[0]
        seconds = time.perf_counter() - t
    return y * sc["y_std"] + sc["y_mean"], seconds


def chord_positions(o, alpha, n):
    """x/c along the chord line (leading edge 0, trailing edge 1) of the n points of each side."""
    import surface_data
    chord = np.array([np.cos(np.radians(alpha)), np.sin(np.radians(alpha))])  # leading -> trailing edge
    u = surface_data.stations(n)
    return np.array([0.25 + (o.at(o.s_of(side, u)) - [0.25, 0.0]) @ chord for side in surface_data.SIDES])


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


def plot_surface(path, xc, cp, cf, title):
    """-Cp and Cf along the chord on both sides (the mean of the surface models)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import surface_data
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for k, (side, color) in enumerate(zip(surface_data.SIDES, (BLUE, ORANGE))):
        label = "side facing the ground" if side == "ground" else "side facing up"
        axes[0].plot(xc[k], -cp[k], color=color, lw=1.6, label=label)
        axes[1].plot(xc[k], cf[k], color=color, lw=1.6, label=label)
    axes[1].axhline(0, color=INK, lw=0.8)
    axes[0].set(ylabel="-Cp (suction up)", title="Pressure")
    axes[1].set(ylabel="Cf (negative: flow running back)", title="Friction")
    for ax in axes:
        ax.set_xlabel("x/c along the chord")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
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
    if args.force_model or args.surface_model:
        import baseline_cl  # noqa: F401
        import surface_data
    t = time.perf_counter()
    raw, points = airfoil_points(args.airfoil, args.airfoil_dirs)
    inputs, dist, outline, ground_y = build_inputs(points, args.h, args.alpha, args.U, args.h_alpha)
    mask = inputs[2] > 0.5
    print(f"inputs built in {1000 * (time.perf_counter() - t):.0f} ms ({int(mask.sum())} solid grid points)")
    saved = {}
    name = os.path.basename(args.airfoil)

    if args.model:
        outs, cls = [], []
        for path in args.model:
            net, kind = load_model(path)
            run_model(net, kind, inputs, dist)  # the first call of a model is slower: not timed
            out, seconds = run_model(net, kind, inputs, dist)
            cl = float(unet_data.lift(out[0].astype(np.float64) * unet_data.MAX_TARGETS[0], mask))
            outs.append(out)
            cls.append(cl)
            print(f"U-Net {path}: -Cl = {-cl:.3f} from the grid ({1000 * seconds:.0f} ms, input 1: {kind or 'not used'})")
        if len(cls) > 1:
            print(f"mean of the {len(cls)} U-Nets: -Cl = {-np.mean(cls):.3f} (range {-max(cls):.3f} to {-min(cls):.3f})")
        out = unet_data.zero_solid(np.mean(outs, axis=0)[None], inputs[None])[0]
        fields = unet_data.denormalise(out, args.U)  # p / rho in m^2/s^2, Ux and Uy in m/s
        saved.update(p=fields[0], Ux=fields[1], Uy=fields[2], mask=mask, cl=np.array(cls))
        if args.plot:
            title = (f"{name} upside down, h/c = {args.h:g}, alpha = {args.alpha:g} deg, U = {args.U:g} m/s: "
                     f"predicted -Cl = {-np.mean(cls):.3f} (U-Net, {len(cls)} model{'s' * (len(cls) > 1)})")
            scaled = np.array([fields[0] / (0.5 * args.U ** 2), fields[1] / args.U, fields[2] / args.U])
            plot(args.plot, scaled, mask, outline, ground_y, title)
            print("saved", args.plot)

    if args.force_model:
        cls = []
        for path in args.force_model:
            net, st, sc = load_small(path)
            x = small_inputs(raw, outline, ground_y, args.alpha, args.U, sc["x_mean"].size)
            y, seconds = run_small(net, sc, x)
            cls.append(float(y[0]))
            which = "OpenFOAM's force" if st.get("target") == "openfoam" else "the grid's Cl"
            print(f"force network {path}: -Cl = {-cls[-1]:.3f} ({which}; {1000 * seconds:.2f} ms)")
        if len(cls) > 1:
            print(f"mean of the {len(cls)} force networks: -Cl = {-np.mean(cls):.3f} "
                  f"(range {-max(cls):.3f} to {-min(cls):.3f})")
        saved["cl_force"] = np.array(cls)

    if args.surface_model:
        o = surface_data.Outline(outline)  # the wing's surface as meshed, for adding up the forces
        ys, forces, n = [], [], None
        for path in args.surface_model:
            net, st, sc = load_small(path)
            if n is not None and st["points"] != n:
                raise SystemExit("the surface models have different numbers of points per side")
            n = st["points"]
            x = small_inputs(raw, outline, ground_y, args.alpha, args.U, sc["x_mean"].size)
            y, seconds = run_small(net, sc, x)
            cp, cf, base = y[:2 * n].reshape(2, n), y[2 * n:4 * n].reshape(2, n), float(y[4 * n])
            t = time.perf_counter()
            cl, cd, cm = surface_data.forces(cp, cf, base, o, groundGen.WALL_SIZE)
            seconds_forces = time.perf_counter() - t
            ys.append(y)
            forces.append((cl, cd, cm))
            back = 100 * reversed_share(cf)
            print(f"surface model {path}: -Cl = {-cl:.3f}, Cd = {cd:.4f}, Cm = {cm:+.4f}; reversed flow on {back:.0f}% "
                  f"of the surface ({1000 * seconds:.2f} ms, forces {1000 * seconds_forces:.1f} ms)")
        f = np.array(forces)
        if len(f) > 1:
            print(f"mean of the {len(f)} surface models: -Cl = {-f[:, 0].mean():.3f}, Cd = {f[:, 1].mean():.4f}, "
                  f"Cm = {f[:, 2].mean():+.4f}")
        y = np.mean(ys, axis=0)
        cp, cf = y[:2 * n].reshape(2, n), y[2 * n:4 * n].reshape(2, n)
        xc = chord_positions(o, args.alpha, n)
        saved.update(cp=cp, cf=cf, xc=xc, cp_base=float(y[4 * n]), cl_surface=f[:, 0], cd_surface=f[:, 1],
                     cm_surface=f[:, 2])
        if args.plot_surface:
            title = (f"{name} upside down, h/c = {args.h:g}, alpha = {args.alpha:g} deg, U = {args.U:g} m/s: "
                     f"predicted -Cl = {-f[:, 0].mean():.3f}, Cd = {f[:, 1].mean():.4f} "
                     f"(surface model, {len(f)} model{'s' * (len(f) > 1)})")
            plot_surface(args.plot_surface, xc, cp, cf, title)
            print("saved", args.plot_surface)

    if args.npz:
        np.savez_compressed(args.npz, **saved)
        print("saved", args.npz)


def reversed_share(cf):
    """The share of the surface where the friction is negative (as surface_model.reversed_share)."""
    import surface_model
    return surface_model.reversed_share(cf)


if __name__ == "__main__":
    main()
