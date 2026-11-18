"""The pressure and the friction along the wing's surface, at fixed points: the targets of the
surface model (surface_model.py), and the sums that turn them into Cl, Cd and Cm.

Each sample stores OpenFOAM's pressure and wall shear stress at the centres of the wall faces
('surf_xy_p', 'surf_p', 'surf_tau'; groundGen.read_surface), in the order of the mesh. Here they are
put in order along the wing's surface as it was meshed: the Catmull-Rom spline through the
outline's points (gmsh's built-in 'Spline'; unet_data.wing_outline gives the points), closed by a
straight line at the trailing edge (Outline). gmsh cuts the spline into equal faces of at most
groundGen.WALL_SIZE (0.005 chords), which Outline.faces rebuilds: on the test samples every surface
point lies at the middle of a rebuilt face, and gets its exact position along the surface. The
surface is split at the leading edge (its point farthest from the trailing edge) into two sides:
the side facing the ground and the side facing up. Each runs from the leading edge (u = 0) to the
trailing edge (u = 1), u being the arc length along that side divided by its length. A blunt
trailing edge leaves a short base between the two sides; its pressure is kept as one mean.

At n points of u per side (stations(n)), closer together at both ends, a sample gets
- Cp = p / (1/2 U^2), the pressure (OpenFOAM's, relative to the outlet's), and
- Cf = the wall shear stress on the wing along the side, towards the trailing edge, / (1/2 U^2):
  positive where the flow next to the wall runs towards the trailing edge, negative where it runs
  back (separated flow).
Cl, Cd and Cm (about the quarter chord, positive anticlockwise in the sample's frame, in which the
wing is upside down) are the sums of -Cp n + Cf t over the rebuilt faces (n the outward normal, t
the direction towards the trailing edge), each face with Cp and Cf read from the points at its
middle: the sum OpenFOAM makes (forces).

From the ground_effect folder, the check that the stations lose little against OpenFOAM's forces:
    ..\\.venv\\Scripts\\python.exe surface_data.py --folder data\\dataset\\test --stations 64 128 200
On the 102 test samples, 128 points per side give Cl within 0.015% and Cd within 0.54% of OpenFOAM's
on average (the samples' own surface data, face by face: 0.005% and 0.31%).
"""
import argparse
import os

import numpy as np

import unet_data

SIDES = ("ground", "top")  # the side facing the ground (the suction side), the side facing up


def stations(n):
    """n values of u from 0 (leading edge) to 1 (trailing edge), closer together at both ends."""
    return 0.5 * (1 - np.cos(np.pi * np.arange(n) / (n - 1)))


def catmull_rom(points, per_segment=32):
    """Points along the Catmull-Rom spline through points (the curve of gmsh's built-in 'Spline',
    which meshes the wing; the ends continue straight), per_segment between each pair."""
    p = np.asarray(points, float)
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    t = np.linspace(0, 1, per_segment, endpoint=False)[None, :, None]
    p0, p1, p2, p3 = (ext[i:i + len(p) - 1][:, None, :] for i in range(4))
    c = 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
    return np.vstack([c.reshape(-1, 2), p[-1:]])


class Outline:
    """The wing's surface as meshed (sample frame): the Catmull-Rom spline through the outline's
    points, closed by a straight line from the last point back to the first (the base of a blunt
    trailing edge), with arc lengths every step chords; its two sides and its base."""

    def __init__(self, outline, step=0.0002):
        self.p = np.asarray(outline, float)
        fine = np.vstack([catmull_rom(self.p), self.p[:1]])
        s_fine = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(fine, axis=0).T))])
        self.s_nodes, self.closed = s_fine, fine
        self.total = s_fine[-1]
        self.s_end = s_fine[-2]  # the spline ends at the last point; then the base
        t = np.append(np.arange(0, self.total, step), self.total)
        self.dense_s = t
        self.dense = self.at(t)
        te = 0.5 * (self.p[0] + self.p[-1])
        on_spline = t <= self.s_end
        k = int(np.argmax(np.where(on_spline, np.hypot(*(self.dense - te).T), -1)))
        self.s_le = float(t[k])  # the leading edge: the point farthest from the trailing edge
        d = self.dense
        area = 0.5 * np.sum(d[:, 0] * np.roll(d[:, 1], -1) - np.roll(d[:, 0], -1) * d[:, 1])
        self.outward = 1.0 if area > 0 else -1.0  # anticlockwise: the outward normal is (t_y, -t_x)
        # which side faces the ground: the one that lies lower on average
        self.ground_is_a = d[t <= self.s_le, 1].mean() < d[(t > self.s_le) & on_spline, 1].mean()

    def faces(self, wall=None):
        """The mesh's wall faces, rebuilt: the spline cut into equal pieces of about wall chords
        (groundGen.WALL_SIZE, as gmsh does) and the base into pieces of at most that. Returns the
        arc lengths of their ends (n+1) and of their middles (n), their unit directions (towards
        growing s) and their lengths: straight segments between the ends, as OpenFOAM sees them."""
        if wall is None:
            import groundGen
            wall = groundGen.WALL_SIZE
        n_spline = max(1, int(np.ceil(self.s_end / wall - 1e-9)))  # gmsh rounds up
        base = self.total - self.s_end
        n_base = int(np.ceil(base / wall - 1e-9)) if base > 1e-9 else 0
        ends = np.concatenate([np.linspace(0, self.s_end, n_spline + 1),
                               np.linspace(self.s_end, self.total, n_base + 1)[1:] if n_base else []])
        pts = self.at(ends)
        seg = np.diff(pts, axis=0)
        length = np.hypot(*seg.T)
        return ends, 0.5 * (ends[1:] + ends[:-1]), seg / length[:, None], length

    def at(self, s):
        """Points of the outline at arc lengths s."""
        return np.column_stack([np.interp(s, self.s_nodes, self.closed[:, 0]), np.interp(s, self.s_nodes, self.closed[:, 1])])

    def tangent(self, s, ds=1e-4):
        """Unit tangents in the direction of increasing s."""
        d = self.at(np.clip(s + ds, 0, self.total)) - self.at(np.clip(s - ds, 0, self.total))
        return d / np.hypot(*d.T)[:, None]

    def s_of(self, side, u):
        """Arc length of the points u of a side ('ground' or 'top')."""
        a = (side == "ground") == self.ground_is_a
        return self.s_le - u * self.s_le if a else self.s_le + u * (self.s_end - self.s_le)

    def to_te(self, side):
        """+1 if the side runs towards the trailing edge as s grows, else -1."""
        a = (side == "ground") == self.ground_is_a
        return -1.0 if a else 1.0

    def locate(self, pts, wall=None):
        """For points on the surface: side index (0 ground, 1 top, 2 base) and u along the side.
        Points at the middle of a face (Outline.faces; within 1e-5 chords) get that face's exact
        arc length, others that of the nearest point of the dense outline (to 0.0001 chords): around
        the leading edge Cp changes by several units from one face to the next, so a shift of 0.0001
        chords there changes the pressure drag by a percent."""
        from scipy.spatial import cKDTree
        dist, k = cKDTree(self.dense).query(pts)
        s = self.dense_s[k]
        ends, s_mid, _, _ = self.faces(wall)
        mids = 0.5 * (self.at(ends[1:]) + self.at(ends[:-1]))
        dist_f, kf = cKDTree(mids).query(pts)
        exact = dist_f < 1e-5
        s = np.where(exact, s_mid[kf], s)
        self.exact_share = float(exact.mean())
        in_a = s <= self.s_le
        in_b = (s > self.s_le) & (s <= self.s_end)
        u = np.where(in_a, (self.s_le - s) / self.s_le, (s - self.s_le) / (self.s_end - self.s_le))
        side = np.where(in_a, 0, np.where(in_b, 1, 2))
        if not self.ground_is_a:  # side a is the top
            side = np.where(side == 2, 2, 1 - side)
        return side, np.clip(u, 0, 1)


def wall_size(d):
    """The mesh's face size along the wing of sample d (groundGen.WALL_SIZE / refine)."""
    import groundGen
    return groundGen.WALL_SIZE / (float(d["refine"]) if "refine" in d.files else 1.0)


def targets(d, n, airfoil_dirs=unet_data.AIRFOIL_DIRS):
    """Cp and Cf of one sample (an open .npz) at n stations per side: dict with 'cp' and 'cf'
    (2 x n: ground side, top side), 'cp_base' (the mean pressure on the base, or NaN without a base),
    the Outline and 'exact' (the share of the surface points matched to a rebuilt face)."""
    outline, _ = unet_data.wing_outline(d, airfoil_dirs)
    o = Outline(outline)
    q = 0.5 * float(d["U"]) ** 2
    xy, p, tau = d["surf_xy_p"], d["surf_p"] / q, -d["surf_tau"][:, :2] / q  # the stress on the wing
    side, u = o.locate(xy, wall_size(d))
    u_st = stations(n)
    cp, cf = np.empty((2, n)), np.empty((2, n))
    for k, name in enumerate(SIDES):
        sel = side == k
        order = np.argsort(u[sel])
        uu = u[sel][order]
        s = o.s_of(name, uu)
        t = o.tangent(s) * o.to_te(name)  # towards the trailing edge
        cp[k] = np.interp(u_st, uu, p[sel][order])
        cf[k] = np.interp(u_st, uu, np.sum(tau[sel][order] * t, axis=1))
    base = side == 2
    return {"cp": cp, "cf": cf, "cp_base": float(p[base].mean()) if base.any() else float("nan"), "outline": o,
            "exact": o.exact_share}


def forces(cp, cf, cp_base, o, wall=None):
    """Cl, Cd and Cm (about the quarter chord at (0.25, 0)) from Cp and Cf at the stations of each side
    (2 x n arrays) and the base's mean Cp, summed as OpenFOAM sums them: over the mesh's wall faces
    (Outline.faces), each a straight piece with one Cp and one Cf, read from the stations at its
    middle. Around the leading edge the faces (0.005 chords) are long against the change of the
    pressure, so this sum differs from that of a smooth curve, by several percent of the drag (the
    pressure drag is the small difference of large forces there). Without base data (NaN), the base
    gets the mean Cp of the two sides' last stations."""
    u_st = stations(cp.shape[-1])
    ends, s_mid, t_s, length = o.faces(wall)
    mid = 0.5 * (o.at(ends[1:]) + o.at(ends[:-1]))
    normal = o.outward * np.column_stack([t_s[:, 1], -t_s[:, 0]])
    on_a, on_b = s_mid <= o.s_le, (s_mid > o.s_le) & (s_mid <= o.s_end)
    traction = np.zeros_like(mid)
    for k, name in enumerate(SIDES):
        a = (name == "ground") == o.ground_is_a
        sel = on_a if a else on_b
        u = (o.s_le - s_mid[sel]) / o.s_le if a else (s_mid[sel] - o.s_le) / (o.s_end - o.s_le)
        traction[sel] = (-np.interp(u, u_st, cp[k])[:, None] * normal[sel]
                         + np.interp(u, u_st, cf[k])[:, None] * t_s[sel] * o.to_te(name))
    base = ~(on_a | on_b)
    if base.any():
        cpb = cp_base if np.isfinite(cp_base) else 0.5 * (cp[0, -1] + cp[1, -1])
        traction[base] = -cpb * normal[base]
    f = (traction * length[:, None]).sum(axis=0)
    r = mid - [0.25, 0.0]
    m = float(((r[:, 0] * traction[:, 1] - r[:, 1] * traction[:, 0]) * length).sum())
    return float(f[1]), float(f[0]), m


def full_forces(d, o):
    """Cl, Cd and Cm from the sample's own surface data, each point on its rebuilt face (the check
    of Outline.faces: on the test samples within 0.005% of OpenFOAM's Cl and 0.3% of its Cd)."""
    from scipy.spatial import cKDTree
    q = 0.5 * float(d["U"]) ** 2
    ends, _, t, length = o.faces(wall_size(d))
    mid = 0.5 * (o.at(ends[1:]) + o.at(ends[:-1]))
    _, k = cKDTree(d["surf_xy_p"]).query(mid)
    p, tau = d["surf_p"][k] / q, -d["surf_tau"][k, :2] / q
    normal = o.outward * np.column_stack([t[:, 1], -t[:, 0]])
    traction = -p[:, None] * normal + tau
    f = (traction * length[:, None]).sum(axis=0)
    r = mid - [0.25, 0.0]
    m = float(((r[:, 0] * traction[:, 1] - r[:, 1] * traction[:, 0]) * length).sum())
    return float(f[1]), float(f[0]), m


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check the stations against OpenFOAM's forces")
    ap.add_argument("--folder", required=True, help="folder of samples")
    ap.add_argument("--stations", type=int, nargs="+", default=[64, 128, 200])
    ap.add_argument("--max", type=int, default=0, help="only the first this many samples")
    args = ap.parse_args(argv)
    files = unet_data.sample_files(args.folder)[:args.max or None]
    rows = {n: [] for n in args.stations}
    full = []
    for path in files:
        with np.load(path) as d:
            cl_of, cd_of = float(d["Cl"]), float(d["Cd"])
            for n in args.stations:
                t = targets(d, n)
                rows[n].append((cl_of, cd_of, *forces(t["cp"], t["cf"], t["cp_base"], t["outline"], wall_size(d)), t["exact"]))
            full.append((cl_of, cd_of, *full_forces(d, t["outline"])))
    full = np.array(full)
    print(f"{len(files)} samples of {args.folder}; surface points matched to a rebuilt face: "
          f"{100 * np.mean([r[-1] for r in rows[args.stations[0]]]):.1f}%")
    print(f"  the samples' own surface data on the rebuilt faces: Cl {100 * np.abs(full[:, 2] - full[:, 0]).mean() / np.abs(full[:, 0]).mean():.3f}%, "
          f"Cd {100 * np.abs(full[:, 3] - full[:, 1]).mean() / full[:, 1].mean():.2f}% off OpenFOAM's on average")
    for n in args.stations:
        r = np.array(rows[n])
        print(f"  {n} stations per side: Cl {100 * np.abs(r[:, 2] - r[:, 0]).mean() / np.abs(r[:, 0]).mean():.3f}% "
              f"(max {100 * np.max(np.abs(r[:, 2] - r[:, 0]) / np.abs(r[:, 0])):.2f}%), Cd "
              f"{100 * np.abs(r[:, 3] - r[:, 1]).mean() / r[:, 1].mean():.2f}% off OpenFOAM's; Cm "
              f"{np.abs(r[:, 4] - full[:, 4]).mean():.4f} off that from all the points (mean |Cm| {np.abs(full[:, 4]).mean():.3f})")


if __name__ == "__main__":
    main()
