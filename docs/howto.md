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
- [The published data (Zenodo)](#the-published-data-zenodo)
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
   !git clone https://github.com/mtsial/GroundFoil.git
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
smooth = glob.glob("/kaggle/input/**/airfoil_database_smooth", recursive=True)
test_smooth = glob.glob("/kaggle/input/**/airfoil_database_test_smooth", recursive=True)
assert smooth and test_smooth, "add the dataset of airfoil files (groundfoil_airfoils.zip) to the notebook"
smooth, test_smooth = smooth[0], test_smooth[0]
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

## A model for the downforce alone

The comparison model of [The downforce alone](14-downforce-model.md) trains on
Kaggle's GPUs (on one CPU thread a model takes 2–15 minutes; the two tunings
are 216 models). Both Kaggle datasets are needed, the samples and the airfoil
files, and "GPU T4 x2". With `data`, `smooth` and `test_smooth` found as in the
cells above:
```python
grid = "--widths 128 256 512 --depths 2 3 4 --weight-decays 1e-5 1e-4 1e-3 --lrs 1e-3 3e-3"
!python baseline_cl.py --data {data}/train --test {data}/test --airfoil-dirs {smooth} {test_smooth} --search {grid} --jobs 4 --out /kaggle/working/baseline_cl
!python baseline_cl.py --data {data}/train --test {data}/test --airfoil-dirs {smooth} {test_smooth} --search {grid} --jobs 4 --target openfoam --out /kaggle/working/baseline_cl_openfoam
```
Each model's result is saved as soon as it is trained (`search_runs.csv`), and
a run started again skips the models already there. Put the folders
`baseline_cl` and `baseline_cl_openfoam` from the output into
`ground_effect/data/unet/`. The comparison with the U-Net and the sweeps need
files that stay on Windows (the Tyrrell 026's shape), so they run there, with
the models from Kaggle (`--load`; it first checks that they give the same Cl
here). The training samples must be those of the Kaggle dataset: since the
local folder holds more by now, `train_files.txt` lists them (the training
samples in `ground_effect_4000runs.zip`). Then the figure:
```powershell
cd ground_effect
foreach ($d in "baseline_cl", "baseline_cl_openfoam") { ..\.venv\Scripts\python.exe baseline_cl.py --data data\dataset\train --files data\unet\baseline_cl\train_files.txt --test data\dataset\test --out data\unet\$d --load --sweep data\sweep_naca4412 data\sweep_tyrrell026 --compare data\unet\distance\f1_scratch_s0 data\unet\distance\f1_scratch_s1 }
..\.venv\Scripts\python.exe downforce_report.py --baseline data\unet\baseline_cl_openfoam --unet-sweeps ..\results\sweep_tyrrell026_distance.csv ..\results\sweep_naca4412_distance.csv --plot ..\results\downforce_model.png --csv ..\results\downforce_model.csv
```
`train_files.txt` was written from the zip with:
```powershell
..\.venv\Scripts\python.exe -c "import zipfile; z = zipfile.ZipFile('../data/kaggle/ground_effect_4000runs.zip'); n = sorted(x[6:] for x in z.namelist() if x.startswith('train/') and x.endswith('.npz') and 'not_converged' not in x); open('data/unet/baseline_cl/train_files.txt', 'w').write(chr(10).join(n) + chr(10))"
```

### Learning curves for the downforce

`cl_learning_curve.py` trains the small network and a Gaussian process on
97 to 3,096 samples (the same samples as the U-Net runs of
`learning_curve.py` at each size) and adds the U-Net's rows. The Gaussian
process's memory and time grow with the cube of the number of samples, so it
runs on Kaggle's GPU (both datasets, "GPU T4 x2"), with `data`, `smooth` and
`test_smooth` found as above:
```python
!python cl_learning_curve.py --data {data}/train --test {data}/test --airfoil-dirs {smooth} {test_smooth} --csv /kaggle/working/cl_learning_curve.csv
!python cl_learning_curve.py --data {data}/train --test {data}/test --airfoil-dirs {smooth} {test_smooth} --models gp --gp-components 8 --csv /kaggle/working/cl_learning_curve.csv
```
Each row of the table is written as soon as it is done, and a run started
again skips the rows already there. Then, on Windows, with the table from
the output, the U-Net's rows and the figure:
```powershell
cd ground_effect
..\.venv\Scripts\python.exe cl_learning_curve.py --data data\dataset\train --files data\unet\baseline_cl\train_files.txt --test data\dataset\test --models --unet data\unet\learning_curve_wall --unet-extra data\unet\distance\f1_scratch_s0 data\unet\distance\f1_scratch_s1 --csv ..\results\downforce_learning_curve.csv --plot ..\results\downforce_learning_curve.png
```

## The pressure along the wing

The surface model of [The pressure along the wing](15-surface-model.md). First
the check that the surface data, put on the mesh's own faces, give OpenFOAM's
forces, and how much is lost with 64, 128 and 200 points per side (a few
seconds):
```powershell
cd ground_effect
..\.venv\Scripts\python.exe surface_data.py --folder data\dataset\test --stations 64 128 200
```
Then the two models (about 6 minutes each on the CPU, plus 2 minutes to read
the samples), with both sweeps and the other two models' downforce, and the
figure of the pressure curves:
```powershell
..\.venv\Scripts\python.exe surface_model.py --data data\dataset\train --files data\unet\baseline_cl\train_files.txt --test data\dataset\test --out data\unet\surface_model --sweep data\sweep_naca4412 data\sweep_tyrrell026 --compare-net data\unet\baseline_cl_openfoam --compare-unet data\unet\distance\f1_scratch_s0 data\unet\distance\f1_scratch_s1
..\.venv\Scripts\python.exe surface_report.py --model data\unet\surface_model\s0 --test data\dataset\test --pick 50 90 --samples data\sweep_tyrrell026\h0134\tyrrell026_h0134_a+0345_U0100.npz data\sweep_tyrrell026\h0067\tyrrell026_h0067_a+0345_U0100.npz --plot ..\results\surface_model_cp.png
copy data\unet\surface_model\sweep_naca4412.csv ..\results\sweep_naca4412_surface_model.csv
copy data\unet\surface_model\sweep_tyrrell026.csv ..\results\sweep_tyrrell026_surface_model.csv
```
`--search` chooses the width, the depth and the weight decay on the validation
samples (8 combinations, 2 seeds each; better on a GPU).

## The final runs, on all the data

When every batch has finished, all the models are trained again on all the
samples, in one Kaggle notebook. First the zip for the Kaggle dataset, in the
project folder. It checks that every sample is in its folder's
`samples.csv` and every line there has its sample, so it refuses to zip while
a batch is still running (`--check` does only the check):
```powershell
.venv\Scripts\python.exe ground_effect\make_dataset_zip.py data\kaggle\ground_effect_final.zip
```
Upload the zip as a new version of the Kaggle dataset of samples, with the
old file removed from that version (the cell below stops if it finds two
datasets of samples). Then a notebook with both datasets (the samples and the
airfoil files), "GPU T4 x2" and internet on, and this cell. It takes about
6–7 hours, so start it with "Save & Run All (Commit)":
```python
!git clone https://github.com/mtsial/GroundFoil.git
%cd GroundFoil/ground_effect
!git log -1 --format="code version %h: %s"
import glob, os, subprocess
w0 = glob.glob("/kaggle/input/**/train/w0", recursive=True)
assert len(w0) == 1, w0  # exactly one dataset of samples
data = os.path.dirname(os.path.dirname(w0[0]))
smooth = glob.glob("/kaggle/input/**/airfoil_database_smooth", recursive=True)
test_smooth = glob.glob("/kaggle/input/**/airfoil_database_test_smooth", recursive=True)
assert smooth and test_smooth, "add the dataset of airfoil files (groundfoil_airfoils.zip) to the notebook"
smooth, test_smooth = smooth[0], test_smooth[0]
n = len([f for f in glob.glob(f"{data}/train/**/*.npz", recursive=True) if "not_converged" not in f])
print(f"{n} training samples (the validation samples among them)")
assert n > 5000, "an older version of the dataset?"
out = "/kaggle/working"
common = f"--data {data}/train --test {data}/test --airfoil-dirs {smooth} {test_smooth}"

# 1. The U-Net (distance input, random start, loss weighted next to the wing): the two
#    models with all the samples first, then the learning curve down to 1/64 (about 2.5 h)
unet = f"{common} --gpus 2 --starts scratch --wall-weight 10 --distance"
!python learning_curve.py {unet} --fractions 1 --out {out}/final_unet
!python learning_curve.py {unet} --fractions 0.5 0.25 0.125 0.0625 0.03125 0.015625 --out {out}/final_unet --plot {out}/final_unet.png

# 2. The network for the downforce alone, its settings chosen on the validation samples
#    as in the first search, for the grid's Cl and for OpenFOAM's (about 2 h)
grid = "--widths 128 256 512 --depths 2 3 4 --weight-decays 1e-5 1e-4 1e-3 --lrs 1e-3 3e-3"
for target, name in [("grid", "baseline_cl"), ("openfoam", "baseline_cl_openfoam")]:
    !python baseline_cl.py {common} --search {grid} --jobs 4 --target {target} --out {out}/{name}

# 3. One GPU each, side by side: the surface model with its search, and the learning
#    curves of the downforce with the Gaussian process (about 1 h)
fractions = "1 0.5 0.25 0.125 0.0625 0.03125 0.015625"
jobs = [f"python surface_model.py {common} --search --out {out}/surface_model",
        f"python cl_learning_curve.py {common} --fractions {fractions} --csv {out}/cl_learning_curve.csv && "
        f"python cl_learning_curve.py {common} --fractions {fractions} --models gp --gp-components 8 --csv {out}/cl_learning_curve.csv"]
procs = [subprocess.Popen(cmd, shell=True, stdout=open(f"{out}/side_by_side_gpu{i}.log", "w"),
                          stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)))
         for i, cmd in enumerate(jobs)]
print("exit codes:", [p.wait() for p in procs])
!tail -n 4 {out}/side_by_side_gpu0.log {out}/side_by_side_gpu1.log

# 4. A check: does the U-Net with all the samples gain from twice the training? (about 45 min)
!python learning_curve.py {unet} --fractions 1 --iterations 160000 --out {out}/final_unet_160k
```
Each part writes its results as it goes and skips the finished ones when it
is run again. A new Kaggle session starts with an empty `/kaggle/working`,
though: to go on after an interruption, add the stopped version's output to
the notebook as an input and copy it into `/kaggle/working` before running
the cell again. (Our run took 8 h 15 min: the Gaussian processes of part 3
took 2.5 h, not 1 h.)

Then, on Windows: unzip the output into `ground_effect\data\unet\final\`,
without the folder `GroundFoil` (the code). The training samples are those
of the dataset zip:
```powershell
cd ground_effect
..\.venv\Scripts\python.exe -c "import zipfile; z = zipfile.ZipFile('../data/kaggle/ground_effect_final.zip'); n = sorted(x[6:] for x in z.namelist() if x.startswith('train/') and x.endswith('.npz') and 'not_converged' not in x); open('data/unet/final/train_files.txt', 'w').write(chr(10).join(n) + chr(10))"
```
The sweeps with the final U-Nets; the small networks and the surface model
loaded again (`--load`; each first checks that its models give here what
they gave on Kaggle), with their sweeps and comparisons; and the figures
(about 5 minutes in all):
```powershell
..\.venv\Scripts\python.exe sweep_report.py --model data\unet\final\final_unet\f1_scratch_s0\model.pt data\unet\final\final_unet\f1_scratch_s1\model.pt --label "U-Net with the distance input, all the samples" --plot ..\results\sweep_naca4412_final.png --csv ..\results\sweep_naca4412_final.csv
..\.venv\Scripts\python.exe sweep_report.py --sweep data\sweep_tyrrell026 --model data\unet\final\final_unet\f1_scratch_s0\model.pt data\unet\final\final_unet\f1_scratch_s1\model.pt --label "U-Net with the distance input, all the samples" --plot ..\results\sweep_tyrrell026_final.png --csv ..\results\sweep_tyrrell026_final.csv
foreach ($d in "baseline_cl", "baseline_cl_openfoam") { ..\.venv\Scripts\python.exe baseline_cl.py --data data\dataset\train --files data\unet\final\train_files.txt --test data\dataset\test --out data\unet\final\$d --load --sweep data\sweep_naca4412 data\sweep_tyrrell026 --compare data\unet\final\final_unet\f1_scratch_s0 data\unet\final\final_unet\f1_scratch_s1 }
..\.venv\Scripts\python.exe downforce_report.py --baseline data\unet\final\baseline_cl_openfoam --unet-sweeps ..\results\sweep_tyrrell026_final.csv ..\results\sweep_naca4412_final.csv --plot ..\results\downforce_model_final.png --csv ..\results\downforce_model_final.csv
..\.venv\Scripts\python.exe surface_model.py --data data\dataset\train --files data\unet\final\train_files.txt --test data\dataset\test --out data\unet\final\surface_model --load --sweep data\sweep_naca4412 data\sweep_tyrrell026 --compare-net data\unet\final\baseline_cl_openfoam --compare-unet data\unet\final\final_unet\f1_scratch_s0 data\unet\final\final_unet\f1_scratch_s1
..\.venv\Scripts\python.exe surface_report.py --model data\unet\final\surface_model\s0 --test data\dataset\test --pick 50 90 --samples data\sweep_tyrrell026\h0134\tyrrell026_h0134_a+0345_U0100.npz data\sweep_tyrrell026\h0067\tyrrell026_h0067_a+0345_U0100.npz --plot ..\results\surface_model_cp_final.png
copy data\unet\final\surface_model\sweep_naca4412.csv ..\results\sweep_naca4412_surface_model_final.csv
copy data\unet\final\surface_model\sweep_tyrrell026.csv ..\results\sweep_tyrrell026_surface_model_final.csv
..\.venv\Scripts\python.exe learning_curve.py --out data\unet\final\final_unet --report-only --measures all_zeroed p_near_wing cl --plot ..\results\ground_learning_curve_final.png --title "U-Net error in ground effect against the amount of training data: the final models (distance input, random starting weights)"
```
The U-Net's rows of the downforce learning curve go into the table from
Kaggle. Give the seven sizes of the notebook: without `--fractions`,
`cl_learning_curve.py` takes the six of step 15 and leaves out 97 samples.
```powershell
..\.venv\Scripts\python.exe cl_learning_curve.py --data data\dataset\train --files data\unet\final\train_files.txt --test data\dataset\test --models --unet data\unet\final\final_unet --fractions 1 0.5 0.25 0.125 0.0625 0.03125 0.015625 --csv data\unet\final\cl_learning_curve.csv --plot ..\results\downforce_learning_curve_final.png --unet-label "U-Net with the distance input, from its pressure"
copy data\unet\final\cl_learning_curve.csv ..\results\downforce_learning_curve_final.csv
```
The results are in [All the data: the final models](16-final-models.md).

## The published data (Zenodo)

The samples, the final models and the NACA 4412 sweep are on Zenodo,
https://doi.org/10.5281/zenodo.24424329; its README describes every file.
Download the files into the project folder, check them (in Git Bash; every
line should say OK), and unzip them where the commands on this page expect
them:
```bash
sha256sum -c SHA256SUMS.txt
```
```powershell
.venv\Scripts\python.exe -m zipfile -e groundfoil_samples.zip ground_effect\data\dataset
.venv\Scripts\python.exe -m zipfile -e groundfoil_models.zip ground_effect\data\unet
.venv\Scripts\python.exe -m zipfile -e sweep_naca4412.zip ground_effect\data
```
The airfoil files are not on Zenodo, as the UIUC database states no terms for
passing them on: make them as in [Airfoil smoothing](#airfoil-smoothing-on-windows),
then check them against the list from Zenodo (in Git Bash, in the folder
`data`; every line should say OK):
```bash
sha256sum -c ../airfoil_checksums.txt
```
The commands after the final run (above) and those below then work as they
are, without the Tyrrell 026 wing: its coordinates and its sweep are not
public, so leave out `data\sweep_tyrrell026` and the commands that need it.

## Predictions for any wing, and pictures of the flow

`predict.py` runs the U-Net for an airfoil file (or `nacaXXXX`, or an airfoil
of the dataset's folders), a ride height, an angle and a speed, without a
simulation: it prints -Cl from each model and draws the flow. It builds the
inputs as the dataset has them; on the 102 test samples, the final U-Nets'
downforce is within 0.10% on average of that from the simulation's own
inputs. On one CPU thread with nothing else running, a prediction takes
15 ms, plus 12 ms to build the inputs (a simulation: a median of 318 s on
one core; the timings are in [step 16](16-final-models.md)):
```powershell
cd ground_effect
..\.venv\Scripts\python.exe predict.py --airfoil naca4412 --h 0.1 --alpha 4 --U 30 --model data\unet\final\final_unet\f1_scratch_s0\model.pt data\unet\final\final_unet\f1_scratch_s1\model.pt --plot flow.png
```
It also runs the two small networks, alone or with the U-Net: the network
for the downforce alone ([step 14](14-downforce-model.md); `--force-model`,
its model folders) and the surface model ([step 15](15-surface-model.md);
`--surface-model`), which gives Cl, Cd and Cm and, with `--plot-surface`,
draws its pressure and friction along the chord. From the geometry alone,
both give the predictions they stored for the 102 test samples within 5e-6
(the rounding of those tables). From the airfoil file, the network for the
downforce takes 0.65 ms, and the surface model 0.72 ms, or 2.5 ms with its
forces added up:
```powershell
..\.venv\Scripts\python.exe predict.py --airfoil naca4412 --h 0.1 --alpha 4 --U 30 --force-model data\unet\final\baseline_cl_openfoam\s0 data\unet\final\baseline_cl_openfoam\s1 --surface-model data\unet\final\surface_model\s0 data\unet\final\surface_model\s1 --plot-surface surface.png
```
`field_figure.py` shows simulated and predicted fields side by side, with their
difference, for test samples chosen by their error (here at the 10th, 50th
and 90th percentile) and for any other sample:
```powershell
..\.venv\Scripts\python.exe field_figure.py --model data\unet\final\final_unet\f1_scratch_s0\model.pt --test data\dataset\test --pick 10 50 90 --samples data\sweep_tyrrell026\h0134\tyrrell026_h0134_a+0345_U0100.npz --quantity p --plot fields_p.png
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
