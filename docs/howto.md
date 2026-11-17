# How to run it

[README](../README.md) · [Workflow](../README.md#workflow)

Every command behind the results on these pages: the Docker image, the original
free-air pipeline, the ground-effect runs and checks, the airfoil smoothing, the
dataset, and the training on Kaggle.

*Words used here: [glossary](glossary.md). References: [list](references.md).*

**On this page:**

- [Setup](#setup)
- [Free-air samples (the original code)](#free-air-samples-the-original-code)
- [Ground-effect samples and the CFD checks](#ground-effect-samples-and-the-cfd-checks)
- [Airfoil smoothing (on Windows)](#airfoil-smoothing-on-windows)
- [Parameter scan and dataset](#parameter-scan-and-dataset)
- [Training the U-Net](#training-the-u-net)
- [Ride-height sweeps of new wings](#ride-height-sweeps-of-new-wings)
- [Notes](#notes)

---

## Setup

**Requirements:** Docker Desktop. No local OpenFOAM install is needed.

```powershell
# 1. Build the image: OpenFOAM v2606 + gmsh + Python
docker build -t openfoam-ml .

# 2. Get the original code with Linux line endings, then apply the v2606 fixes
git clone --config core.autocrlf=false https://github.com/thunil/Deep-Flow-Prediction.git
cd Deep-Flow-Prediction
git apply ../patches/openfoam-v2606.patch
cd ..

# 3. Start the container with this folder shared as /work
docker run -it --rm -v ${PWD}:/work openfoam-ml
```

## Free-air samples (the original code)

Inside the container:
```bash
cd /work/Deep-Flow-Prediction/data
bash download_airfoils.sh        # about 1,500 training and 30 test airfoils (UIUC database)
python3 dataGen.py               # number of simulations set by `samples` in dataGen.py
python3 /work/scripts/view_sample.py train/<file>.npz /work/results/sample.png
```

## Ground-effect samples and the CFD checks

Ground-effect samples, also inside the container (`python3 groundGen.py --help`
lists all options):
```bash
cd /work/ground_effect
python3 groundGen.py --airfoil naca23012.dat --alpha 4 --h 0.2 --U 30   # one case
python3 groundGen.py --free-air --airfoil naca4412 --alpha 5 --U 10      # no ground
python3 groundGen.py --mesh-only --airfoil naca4412 --alpha 5 --h 0.2   # check a mesh
python3 /work/scripts/view_sample.py data/train/<file>.npz /work/results/<name>.png
bash mesh_study.sh       # mesh study, round 2
bash verification.sh     # round 3: nuTilda scheme, grid convergence, domain (4 runs at a time)
python3 sampling_test.py <work folder> <its sample .npz>   # check the row-by-row sampling on a finished run
bash validation_zerihan.sh   # Tyrrell 026 wing vs the wind tunnel tests of Zerihan (2001): 33 runs, 4 at a time
```

The validation needs the wing coordinates, which are not in this repository.
Copy the 44 rows of Table 2 of Zerihan's thesis into
`data/airfoils/tyrrell026_table2.txt`, then run
`.venv\Scripts\python.exe ground_effect\make_tyrrell026.py` on Windows. It
writes `data/airfoils/tyrrell026.dat` (smoothed, used by the runs) and
`tyrrell026_raw.dat` (the table points as they are). After the runs,
`ground_effect\validation_zerihan.py` makes the comparison.

## Airfoil smoothing (on Windows)

The ground-effect dataset uses smoothed airfoil files (see
[Airfoil shapes](05-airfoil-shapes.md)). They are made on Windows,
because the smoothing needs scipy, which the container does not have:
```powershell
# smooth the airfoil files (a few minutes with 8 cores; the files left out are listed with the reason)
.venv\Scripts\python.exe ground_effect\smooth_airfoils.py Deep-Flow-Prediction\data\airfoil_database --out data\airfoil_database_smooth --jobs 8
.venv\Scripts\python.exe ground_effect\smooth_airfoils.py Deep-Flow-Prediction\data\airfoil_database_test --out data\airfoil_database_test_smooth --jobs 8
# optional checks: kinks before and after smoothing (about 13 minutes with 8 cores),
# and the test against stand-in shapes (about 22 minutes)
.venv\Scripts\python.exe ground_effect\shape_check.py Deep-Flow-Prediction\data\airfoil_database --jobs 8 --csv results\airfoil_shape_scan.csv
.venv\Scripts\python.exe ground_effect\smoothing_test.py Deep-Flow-Prediction\data\airfoil_database --jobs 8 --csv results\smoothing_test.csv
# the two figures, from those tables
.venv\Scripts\python.exe ground_effect\shape_check.py --from-csv results\airfoil_shape_scan.csv --wiggle-plot results\airfoil_wiggles.png
.venv\Scripts\python.exe ground_effect\smoothing_test.py --from-csv results\smoothing_test.csv --plot results\airfoil_smoothing.png --example Deep-Flow-Prediction\data\airfoil_database\goe387.dat
```
Then, in the container, `groundGen.py` draws its airfoils from the smoothed
folder: `python3 groundGen.py --airfoil-dir /work/data/airfoil_database_smooth ...`.
Without `--airfoil-dir` it uses the original files, as in all runs before.

## Parameter scan and dataset

The parameter scan, its tests and the dataset (inside the container; the
reports run on Windows):
```bash
cd /work/ground_effect
bash param_scan.sh     # 200 random runs, 4 at a time (4 hours), with the scan's settings
bash clip_test.sh      # nuTilda clipping and continued runs: 20 runs (45 minutes); skips runs already done
# the training set: 4 x 1000 runs, about 23 hours each (8 at a time is no faster here than 4)
JOBS=8 bash dataset.sh                                  # runs 1-1000: seeds 2000-2007, 8 at a time
SEED=3000 bash dataset.sh                               # runs 1001-2000: seeds 3000-3003, 4 at a time
SEED=4000 bash dataset.sh && SEED=5000 bash dataset.sh  # runs 2001-4000, one 1000 after the other
# the test set: 120 runs on the 30 test airfoils, seeds 9000-9003 (3 hours)
OUT=data/dataset/test AIRFOILS=../data/airfoil_database_test_smooth RUNS=120 SEED=9000 WORKER=300 bash dataset.sh
```
```powershell
.venv\Scripts\python.exe ground_effect\param_scan_report.py --plot results\param_scan.png --csv results\param_scan.csv
.venv\Scripts\python.exe ground_effect\clip_test_report.py --plot results\clip_test.png
.venv\Scripts\python.exe ground_effect\clip_cells.py ground_effect\work\worker102   # where nuTilda was clipped in a finished run
.venv\Scripts\python.exe ground_effect\param_scan_report.py ground_effect\data\dataset\train --logs ground_effect\logs\dataset_train
```
The last line reports on the dataset the same way as on the scan.

## Training the U-Net

Training the U-Net on the ground-effect samples (`--help` lists all options).
A short check on the laptop's CPU:
```powershell
cd ground_effect
..\.venv\Scripts\python.exe train_unet.py --data data\dataset\train --fraction 0.1 --iterations 100 --threads 2 --out data\unet\check
```
The learning curve needs a GPU, for example a Kaggle notebook. On a Tesla T4,
one run of 80,000 iterations takes about 16 minutes alone (12 ms per
iteration) and about 21 minutes when Kaggle's two T4s run one each
(`--gpus 2`), so the 16 runs take about 3 hours:
1. Make a private Kaggle dataset from a zip file with the folders `train` and
   `test` (from `ground_effect/data/dataset/`) and the file `long80k_modelG`
   (the Run 3 model, from `Deep-Flow-Prediction/train/`).
2. Make a notebook with that dataset, choose the accelerator "GPU T4 x2" and
   turn on internet access. Put this in a cell:
   ```python
   !git clone https://github.com/taimooraslam22g/GroundFoil.git
   %cd GroundFoil/ground_effect
   import glob, os
   model = glob.glob("/kaggle/input/**/long80k_modelG", recursive=True)[0]
   data = os.path.dirname(model)
   !python learning_curve.py --data {data}/train --test {data}/test --pretrained {model} --gpus 2 --out /kaggle/working/learning_curve --plot /kaggle/working/learning_curve.png
   ```
   Then click "Save Version" and choose "Save & Run All (Commit)": the
   notebook runs in the background (the browser can be closed) and its
   output is kept.
3. Download the notebook's output (its Output tab) and put the folder
   `learning_curve` from it into `ground_effect/data/unet/`, then on Windows:
   ```powershell
   .venv\Scripts\python.exe ground_effect\learning_curve.py --out ground_effect\data\unet\learning_curve --report-only --plot results\ground_learning_curve.png
   ```
   For more data later, make a new version of the Kaggle dataset from a new
   zip file; a notebook made from the dataset's page uses its newest version.
If the notebook stops before the end, run the same command again: finished
runs are skipped.

The loss comparison ([Downforce](09-downforce.md))
ran in the same notebook, one `learning_curve.py` per loss, for example:
```python
for name, opt in [("cl0.001", "--cl-weight 0.001"), ("wall10", "--wall-weight 10"), ("wall10c1", "--wall-weight 10 --wall-cells 1")]:
    !python learning_curve.py --data {data}/train --test {data}/test --pretrained {model} --fractions 1 --starts pretrained --gpus 2 {opt} --out /kaggle/working/loss/{name}
```
Put the folder `loss` from its output into `ground_effect/data/unet/`, next
to `learning_curve` (its runs with all the samples are the plain loss). Then
the table and the figure:
```powershell
cd ground_effect
..\.venv\Scripts\python.exe loss_study.py --data data\dataset\train --test data\dataset\test --plain data\unet\learning_curve --loss data\unet\loss --plot ..\results\ground_loss_study.png
```

The learning curve with all the data ([All the data](10-all-data.md)) ran
in one notebook on the dataset version with all 4000 runs: the weighted loss
on 3.125–100% of the samples, then the plain loss on all of them (about 5
hours):
```python
!python learning_curve.py --data {data}/train --test {data}/test --pretrained {model} --gpus 2 --wall-weight 10 --fractions 0.03125 0.0625 0.125 0.25 0.5 1 --out /kaggle/working/learning_curve_wall --plot /kaggle/working/learning_curve_wall.png
!python learning_curve.py --data {data}/train --test {data}/test --pretrained {model} --gpus 2 --fractions 1 --starts pretrained --out /kaggle/working/plain_all
```
Put the folders `learning_curve_wall` and `plain_all` from its output into
`ground_effect/data/unet/`. Then make the figure, and compare the two losses
again. `loss_study.py` wants one folder per loss, so first copy the two
weighted runs with all the samples into `loss_all/wall10/`:
```powershell
cd ground_effect
..\.venv\Scripts\python.exe learning_curve.py --out data\unet\learning_curve_wall --report-only --measures all_zeroed p_near_wing cl --plot ..\results\ground_learning_curve_all.png --title "U-Net error in ground effect against the amount of training data, loss weighted 11 times next to the wing"
foreach ($s in 0, 1) { $d = "data\unet\loss_all\wall10\f1_pretrained_s$s"; New-Item -ItemType Directory -Force $d | Out-Null; Copy-Item data\unet\learning_curve_wall\f1_pretrained_s$s\model.pt, data\unet\learning_curve_wall\f1_pretrained_s$s\summary.json $d }
..\.venv\Scripts\python.exe loss_study.py --data data\dataset\train --test data\dataset\test --plain data\unet\plain_all --loss data\unet\loss_all
```

The distance input ([The exact wing shape](13-distance-input.md)) is built
from the airfoil files, so Kaggle needs them too. Zip the smoothed files, in
the project folder (`tar` comes with Windows):
```powershell
tar -a -cf groundfoil_airfoils.zip -C data airfoil_database_smooth airfoil_database_test_smooth
```
Make a private Kaggle dataset from `groundfoil_airfoils.zip` and add it to
the notebook next to the dataset of samples ("Add Input"). Then, with
`model` and `data` found as above, the 4 runs with all the samples and the
wall weighting (under an hour):
```python
smooth = glob.glob("/kaggle/input/**/airfoil_database_smooth", recursive=True)[0]
test_smooth = glob.glob("/kaggle/input/**/airfoil_database_test_smooth", recursive=True)[0]
!python learning_curve.py --data {data}/train --test {data}/test --pretrained {model} --gpus 2 --fractions 1 --wall-weight 10 --distance --airfoil-dirs {smooth} {test_smooth} --out /kaggle/working/distance
```
The same with the distance computed from the mask instead (the 2 runs from
random weights, about 25 minutes; no airfoil files needed):
```python
!python learning_curve.py --data {data}/train --test {data}/test --gpus 2 --fractions 1 --starts scratch --wall-weight 10 --distance mask --out /kaggle/working/distance_mask
```
Put the folders `distance` and `distance_mask` from the outputs into
`ground_effect/data/unet/`. Then the table and the figure, which compare
their runs with those of `learning_curve_wall` (both starts):
```powershell
cd ground_effect
..\.venv\Scripts\python.exe loss_study.py --data data\dataset\train --test data\dataset\test --plain data\unet\plain_all --loss data\unet\learning_curve_wall data\unet\distance data\unet\distance_mask --starts pretrained scratch --csv data\unet\distance\loss_study.csv --title "The distance input" --plot ..\results\ground_distance_input.png
```

## Ride-height sweeps of new wings

The NACA 4412 at 12 ride heights and in free air
([A new wing](11-naca4412-sweep.md)). It runs inside the container, 4 runs at
a time, in about 20 minutes. Runs that already have a sample are skipped:
```bash
cd /work/ground_effect
bash sweep_naca4412.sh
```
Then, on Windows, the table and the figure, with the two U-Nets trained on
all the samples from the free-air model (from the Kaggle output above):
```powershell
cd ground_effect
..\.venv\Scripts\python.exe sweep_report.py --model data\unet\learning_curve_wall\f1_pretrained_s0\model.pt data\unet\learning_curve_wall\f1_pretrained_s1\model.pt --plot ..\results\sweep_naca4412.png --csv ..\results\sweep_naca4412.csv
```
The Tyrrell 026 ([The F1 wing](12-tyrrell026-sweep.md)) goes the same way, in
about 30 minutes. It needs the wing's file, as the validation does (see
above):
```bash
bash sweep_tyrrell026.sh
```
```powershell
..\.venv\Scripts\python.exe sweep_report.py --sweep data\sweep_tyrrell026 --model data\unet\learning_curve_wall\f1_pretrained_s0\model.pt data\unet\learning_curve_wall\f1_pretrained_s1\model.pt --plot ..\results\sweep_tyrrell026.png --csv ..\results\sweep_tyrrell026.csv
```
Both sweeps with the two models trained from random weights with the
distance input ([The exact wing shape](13-distance-input.md)); for the
Tyrrell 026, the distance is built from the same wing file:
```powershell
..\.venv\Scripts\python.exe sweep_report.py --model data\unet\distance\f1_scratch_s0\model.pt data\unet\distance\f1_scratch_s1\model.pt --label "U-Net with the distance input, from random weights" --plot ..\results\sweep_naca4412_distance.png --csv ..\results\sweep_naca4412_distance.csv
..\.venv\Scripts\python.exe sweep_report.py --sweep data\sweep_tyrrell026 --model data\unet\distance\f1_scratch_s0\model.pt data\unet\distance\f1_scratch_s1\model.pt --label "U-Net with the distance input, from random weights" --plot ..\results\sweep_tyrrell026_distance.png --csv ..\results\sweep_tyrrell026_distance.csv
```
And the tables only, with the two models trained with the distance from the
mask:
```powershell
..\.venv\Scripts\python.exe sweep_report.py --model data\unet\distance_mask\f1_scratch_s0\model.pt data\unet\distance_mask\f1_scratch_s1\model.pt --csv ..\results\sweep_naca4412_distance_mask.csv
..\.venv\Scripts\python.exe sweep_report.py --sweep data\sweep_tyrrell026 --model data\unet\distance_mask\f1_scratch_s0\model.pt data\unet\distance_mask\f1_scratch_s1\model.pt --csv ..\results\sweep_tyrrell026_distance_mask.csv
```

## Notes

The Windows scripts (training, `make_tyrrell026.py` and the airfoil scripts)
run in a local Python environment, `.venv`: Python 3.14 with numpy 2.5,
scipy 1.18, matplotlib 3.11 and PyTorch 2.15 (CPU).

Runs that do not converge (Cl not steady, even after continuing while it
settled) are saved in `not_converged/` inside the output folder, not with the
training samples. `samples.csv` lists every run, with a `converged` column
(before the parameter scan, it also required nuTilda to be no longer clipped
at the end), and with `clipped` and `clip_min`.
