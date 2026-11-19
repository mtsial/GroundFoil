# 7. The dataset

[README](../README.md) · ← [6. Parameter scan](06-parameter-scan.md) · [8. U-Net learning curve](08-unet-learning-curve.md) → · [Workflow](../README.md#workflow)

The training data: random airfoils at random angles of attack (0–10°), ride
heights (h/c 0.05–0.6) and speeds (10–50 m/s), and a test set from 30
airfoils that are never used for training (its 102 samples come from 28 of
them). 4000 runs gave 3,436 samples on 1,286 different airfoils.

*Words used here: [glossary](glossary.md). References: [list](references.md).*

---

**The dataset** is made by `dataset.sh`, 1000 runs at a time (about 23 hours),
and a test set from the 30 test airfoils. Running 8 at a time instead of 4 does not
help on this laptop (Ryzen 7 7730U, 8 cores): each run then takes twice as
long (9.4 instead of 4.7 minutes for 1000 iterations, with the same mesh
sizes), so the laptop finishes the same number of runs per hour.

Each new 1000 runs use new random seeds, so no case is repeated. Every run is
listed in `samples.csv`. Runs whose Cl does not become steady go to
`not_converged/` and are not used for training.

| | Seeds | Runs | Samples (Cl steady) | Not converged | Failed before a sample |
|---|---|---|---|---|---|
| Training, first 1000 runs (8 at a time) | 2000–2007 | 1000 | 867 | 120 | 13 |
| Training, second 1000 runs | 3000–3003 | 1000 | 849 | 136 | 15 |
| Training, third 1000 runs | 4000–4003 | 1000 | 861 | 122 | 17 |
| Training, fourth 1000 runs | 5000–5003 | 1000 | 859 | 118 | 23 |
| **Training, all** | | **4000** | **3,436** | **496** | **68** |
| Test set (the 30 test airfoils) | 9000–9003 | 120 | 102 | 13 | 5 |

**The test set.** Its 102 samples come from 28 of the 30 test airfoils: the
one run of ah63k127 and the one of mue139 did not converge. Two test airfoils
have almost the same shape as a training airfoil: goe439 (like goe428) and
kc135d (like b707d) differ from it by at most 0.09% of the chord in camber and
thickness, where only 3% of the training airfoils are as close to another one.
They have 7 of the 102 test samples. Without them, the errors of step 13 and
later are 3–5% higher (the U-Net's field error 0.89% instead of 0.87%, its
downforce 4.0% instead of 3.9%; the small networks of steps 14 and 15 1.52%
and 1.40% instead of 1.46% and 1.34%), so they flatter the test results only
a little.

So 86% of the runs give a sample (85–87% in each 1000), more than the 75–80%
expected from the parameter scan. Of the 68 training runs that failed, 56
failed in the mesher (gmsh) and 12 in the solver. About 10% of the samples are
kept for validation, chosen by a checksum of the file name: 340 of the 3,436,
which leaves 3,096 for training.

**Which airfoils.** Each run picks its airfoil at random from the 1,462
smoothed training airfoils ([step 5](05-airfoil-shapes.md)), as the original
code does, so some airfoils have several samples and some none. The 3,436
samples cover 1,286 airfoils. Of the other 176, 99 were never picked (4000
random picks are expected to miss about 95) and 77 were picked but gave no
sample: 61 because Cl did not become steady at the conditions drawn, 12
because the mesh failed, 2 because the solver crashed and 2 for a mix of
these. Seven airfoils failed in the mesher on every try (2–4 each): ah83150q,
fx60177, fx61147, fx80080, kenmar, marsden and sd7043.

---

[README](../README.md) · ← [6. Parameter scan](06-parameter-scan.md) · [8. U-Net learning curve](08-unet-learning-curve.md) → · [Workflow](../README.md#workflow)
