"""The network for the drag alone (baseline_cl.py --target cd) against the drag that the surface
model (surface_model.py) adds up from its pressure and friction along the wing, sample by sample:

- the Cd error on the test and validation samples: each sample's |dCd| as the mean over the seeds,
  their mean divided by the mean |Cd| (as for Cl in baseline_cl.py and surface_model.py);
- the samples, and the test airfoils (their samples' mean |dCd|), on which each model is closer,
  with the Wilcoxon signed-rank test;
- the test error by ride height (the bins of baseline_cl.py);
- the two ride-height sweeps: the mean and the largest difference from OpenFOAM's Cd.

Both folders hold s0, s1, ... with errors_test.csv and errors_val.csv, and sweep_<wing>.csv from
--sweep. Writes --csv: kind (error, error_by_h or sweep_<wing>), x (the set, the bin of h/c or h/c),
n, openfoam (Cd, sweeps only), network, network_min, network_max, surface, surface_min,
surface_max (errors in %, or Cd on the sweeps with the range of the seeds).

From the ground_effect folder (see docs/howto.md, "The drag alone"):
    ..\\.venv\\Scripts\\python.exe drag_report.py --drag data\\unet\\final\\baseline_cd --surface data\\unet\\final\\surface_model --csv ..\\results\\drag_model_final.csv
"""
import argparse
import csv
import glob
import os

import numpy as np
from scipy import stats

H_BINS = (0.05, 0.08, 0.135, 0.22, 0.365, 0.601)  # bins of h/c, as in baseline_cl.py
WINGS = ("naca4412", "tyrrell026")


def per_sample(folder, split):
    """{file: (airfoil, h/c, OpenFOAM's Cd, mean over the seeds of |predicted Cd - Cd|)}"""
    runs = sorted(glob.glob(os.path.join(folder, "s*", f"errors_{split}.csv")))
    if not runs:
        raise SystemExit(f"no s*/errors_{split}.csv in {folder}")
    out = {}
    for path in runs:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                cd = float(r["cd_openfoam"])
                out.setdefault(r["file"], [r["airfoil"], float(r["h_over_c"]), cd, []])[3].append(abs(float(r["cd_pred"]) - cd))
    if len({len(v[3]) for v in out.values()}) != 1:
        raise SystemExit(f"{folder}: the seeds' {split} files hold different samples")
    return {k: (a, h, cd, float(np.mean(e))) for k, (a, h, cd, e) in out.items()}


def read_sweep(path, h, cd, mean, lo, hi):
    with open(path, newline="") as f:
        return {round(float(r[h]), 3): [float(r[k]) for k in (cd, mean, lo, hi)] for r in csv.DictReader(f)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--drag", required=True, help="folder of baseline_cl.py --target cd")
    ap.add_argument("--surface", required=True, help="folder of surface_model.py")
    ap.add_argument("--csv", help="table of the results")
    args = ap.parse_args(argv)
    table = []
    for split in ("test", "val"):
        net, surf = per_sample(args.drag, split), per_sample(args.surface, split)
        if sorted(net) != sorted(surf):
            raise SystemExit(f"{split}: the two models' files hold different samples")
        files = sorted(net)
        off = max(abs(net[k][2] - surf[k][2]) / net[k][2] for k in files)
        if off > 1e-3:  # the csv files keep 6 digits
            raise SystemExit(f"{split}: the two files' OpenFOAM Cd differ by up to {off:.1e}")
        cd = np.array([net[k][2] for k in files])
        en, es = np.array([net[k][3] for k in files]), np.array([surf[k][3] for k in files])
        m = np.abs(cd).mean()
        w = stats.wilcoxon(en, es)
        print(f"{split} ({len(files)} samples): Cd error, drag network {100 * en.mean() / m:.2f}%, surface model "
              f"{100 * es.mean() / m:.2f}%; the drag network is closer on {(en < es).sum()} of {len(files)}, "
              f"Wilcoxon p = {w.pvalue:.1e}")
        table.append({"kind": "error", "x": split, "n": len(files), "network": 100 * en.mean() / m,
                      "surface": 100 * es.mean() / m})
        if split != "test":
            continue
        airfoils = sorted({net[k][0] for k in files})
        gn = np.array([np.mean([net[k][3] for k in files if net[k][0] == a]) for a in airfoils])
        gs = np.array([np.mean([surf[k][3] for k in files if net[k][0] == a]) for a in airfoils])
        print(f"  by airfoil (the mean |dCd| of its samples): the drag network is closer on {(gn < gs).sum()} of "
              f"{len(airfoils)}, Wilcoxon p = {stats.wilcoxon(gn, gs).pvalue:.1e}")
        h = np.array([net[k][1] for k in files])
        for lo, hi in zip(H_BINS[:-1], H_BINS[1:]):
            sel = (h >= lo) & (h < hi)
            if sel.any():
                mm = np.abs(cd[sel]).mean()
                row = {"kind": "error_by_h", "x": f"{lo:g}-{hi:g}", "n": int(sel.sum()),
                       "network": 100 * en[sel].mean() / mm, "surface": 100 * es[sel].mean() / mm}
                table.append(row)
                print(f"  h/c {lo:5.3g}-{hi:<5.3g} n={row['n']:3d}: drag network {row['network']:5.2f}%, "
                      f"surface model {row['surface']:5.2f}% (mean Cd {mm:.4f})")
    for wing in WINGS:
        net = read_sweep(os.path.join(args.drag, f"sweep_{wing}.csv"), "h_over_c", "drag", "drag_net", "net_min", "net_max")
        surf = read_sweep(os.path.join(args.surface, f"sweep_{wing}.csv"), "h_over_c", "cd", "cd_model",
                          "cd_model_min", "cd_model_max")
        if sorted(net) != sorted(surf) or max(abs(net[x][0] - surf[x][0]) for x in net) > 1e-6:
            raise SystemExit(f"{wing}: the two sweep tables hold other runs or another OpenFOAM Cd")
        dn = [100 * abs(net[x][1] / net[x][0] - 1) for x in sorted(net)]
        ds = [100 * abs(surf[x][1] / surf[x][0] - 1) for x in sorted(net)]
        print(f"sweep {wing} ({len(net)} ride heights): difference from OpenFOAM's Cd, mean (largest): drag network "
              f"{np.mean(dn):.2f}% ({max(dn):.2f}%), surface model {np.mean(ds):.2f}% ({max(ds):.2f}%)")
        for x in sorted(net):
            table.append({"kind": f"sweep_{wing}", "x": f"{x:.3f}", "n": "", "openfoam": net[x][0],
                          "network": net[x][1], "network_min": net[x][2], "network_max": net[x][3],
                          "surface": surf[x][1], "surface_min": surf[x][2], "surface_max": surf[x][3]})
    if args.csv:
        cols = ["kind", "x", "n", "openfoam", "network", "network_min", "network_max", "surface", "surface_min",
                "surface_max"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in table:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
