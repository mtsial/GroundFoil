# Files

[README](../README.md) · [Workflow](../README.md#workflow)

Every file of the repository and what it does. The commands that use them
are in [How to run it](howto.md).

*Words used here: [glossary](glossary.md).*

---

```
README.md                   the front page: summary, the steps, main results, quick start
CITATION.cff                how to cite this repository (GitHub's "Cite this repository" button)
LICENSE                     Apache License 2.0
docs/                       one page per step (01-reproduction.md to 19-classical-baselines.md), results.md (all results), layout.md (this page), howto.md, glossary.md, references.md
Dockerfile                  OpenFOAM v2606 + gmsh + Python environment
patches/openfoam-v2606.patch  fixes to the original code for OpenFOAM v2606
scripts/view_sample.py      plots all six channels of a training sample
scripts/plot_learning_curve.py  plots training/validation loss
scripts/run_seeds.ps1       trains + tests with several seeds, unattended
scripts/summarize_seeds.py  mean ± std table over the seed runs
ground_effect/groundGen.py        data generation: inverted airfoil above a moving ground
ground_effect/case_template/      OpenFOAM case and gmsh template for ground effect
ground_effect/plot_convergence.py Cl and Cd histories of runs
ground_effect/plot_mesh.py        draws the mesh around the wing
ground_effect/mesh_quality.py     y+ of the first cell, and cell sizes in the sampled window
ground_effect/validate_les.py     free-air runs vs LES surface pressure and skin friction
ground_effect/mesh_study.sh       runs the mesh study, round 2 (inside the container)
ground_effect/mesh_study_report.py  figures and tables of round 2
ground_effect/verification.sh     round 3: nuTilda scheme, grid convergence, domain (inside the container)
ground_effect/verification_report.py  tables and figures of round 3, including the GCI
ground_effect/sampling_test.py    checks the row-by-row sampling on finished runs (inside the container)
ground_effect/make_tyrrell026.py  Tyrrell 026 wing coordinates (Zerihan 2001, Table 2) -> airfoil files (smoothed, and as tabulated)
ground_effect/panel.py            inviscid surface pressure on the curve gmsh builds through airfoil points (panel method)
ground_effect/shape_check.py      kinks in that curve, before and after smoothing, for many airfoil files
ground_effect/smooth_airfoils.py  smooths the airfoil files for the dataset (quintic spline in sqrt(x)); leaves out files with errors
ground_effect/smoothing_test.py   tests the smoothing against stand-in airfoils whose shape is known exactly
ground_effect/validation_zerihan.sh   runs the ground-effect validation (inside the container)
ground_effect/validation_zerihan.py   compares those runs with Zerihan's experiments and CFD
ground_effect/param_scan.sh       parameter scan: 200 random runs drawn like the dataset (inside the container)
ground_effect/param_scan_report.py  sorts the runs (steady, clipped, settling, oscillating, failed) by ride height, angle and speed
ground_effect/clip_test.sh        what the nuTilda clipping changes, and whether settling runs converge (inside the container)
ground_effect/clip_test_report.py tables and figure of that test
ground_effect/clip_cells.py       finds the cells where OpenFOAM clipped nuTilda in a finished run
ground_effect/dataset.sh          makes the dataset with several runs at a time (inside the container)
ground_effect/make_dataset_zip.py zips the samples for a Kaggle dataset, after checking them against the lists of runs
ground_effect/unet.py             the U-Net of the original code (copied from DfpNet.py, unchanged)
ground_effect/unet_data.py        loads and normalises the ground-effect samples (optionally with the signed distance); the original error measure and the downforce from the pressure
ground_effect/train_unet.py       trains one U-Net (from random weights or from the free-air model) and tests it
ground_effect/learning_curve.py   trains and tests U-Nets on parts of the data, then the table and figure
ground_effect/loss_study.py       compares training losses and inputs for the downforce: table and figure
ground_effect/sweep_naca4412.sh   ride-height sweep of the NACA 4412: 12 heights and free air (inside the container)
ground_effect/sweep_tyrrell026.sh ride-height sweep of the Tyrrell 026 at the tunnel's heights and 10 m/s (inside the container)
ground_effect/sweep_report.py     table and figure of a sweep, with the U-Net's predictions
ground_effect/ride_height_map.py  where the downforce peaks: the downforce network's sweeps of every airfoil, and the network against simulated sweeps
ground_effect/sweep_peaks.sh      ride-height sweeps of six test airfoils at three angles and three speeds, to check the map (inside the container)
ground_effect/sweep_goe775_check.sh  goe775 at 10 m/s on the fine mesh and at ride heights in between: the one irregular sweep (inside the container)
ground_effect/baseline_cl.py      comparison model: a small network that predicts only the downforce (or, with --target cd, the drag), from the shape, ride height, angle and speed
ground_effect/predict.py          the trained models for any airfoil, ride height, angle and speed, without a simulation: the U-Net's flow, the downforce network's Cl, the surface model's pressure, friction and forces
ground_effect/field_figure.py     simulated and predicted flow fields side by side, with their difference
ground_effect/downforce_report.py figure and table of the U-Net against the comparison model (step 14)
ground_effect/cl_learning_curve.py  downforce error against the amount of data: small network, Gaussian process and U-Net
ground_effect/gp_sweep.py         the Gaussian process fitted on all the training samples, on the ride-height sweeps, with its standard deviation
ground_effect/drag_report.py      the drag network against the drag added up from the surface model, sample by sample (step 17)
ground_effect/classical_baselines.py  classical regressions for the forces (polynomial, random forest, support vector), and every model scored alike (step 19)
ground_effect/surface_data.py     the pressure and friction along the wing at fixed points, on the mesh's own faces; Cl, Cd, Cm from them
ground_effect/surface_model.py    a small network for the pressure and the friction along the wing (step 15)
ground_effect/surface_report.py   pressure and friction along the wing: CFD against the surface model
results/                    figures and tables of the results
Deep-Flow-Prediction/       the original code, which you clone here (see the README's Quick start); not part of this repo
```

---

[README](../README.md) · [Workflow](../README.md#workflow)
