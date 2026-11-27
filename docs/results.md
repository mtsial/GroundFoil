# All results

[README](../README.md) · [Workflow](../README.md#workflow)

Every main result of the study, with the numbers behind it; each row links
to the step where it was found.

*Words used here: [glossary](glossary.md). References: [list](references.md).*

---

| | Result | Details |
|---|---|---|
| Reproduction, free air | 2.96% test error with the paper's training length (paper: 2.6%); the pressure error as in the paper's best model (14.9% against 14.8%), the velocity error close to it | [Step 1](01-reproduction.md) |
| CFD against high-resolution LES | NACA 4412 in free air, with the settings of the dataset: skin friction within 7.6% (suction side) and 7.3% (pressure side) on average; surface pressure also close (mean \|ΔCp\| 0.009–0.014) | [Step 3](03-mesh-study.md) |
| CFD against the wind tunnel | Tyrrell 026 F1 wing: the downforce gained near the ground within 4% | [Step 4](04-wind-tunnel.md) |
| Mesh study | numerical uncertainty of the downforce 0.04–3%, largest closest to the ground | [Steps 3 and 4](03-mesh-study.md) |
| Airfoil shapes | 1,462 of the 1,497 UIUC files smoothed: pressure wiggles above 0.1 in 453 files before, in none after | [Step 5](05-airfoil-shapes.md) |
| Where steady RANS converges | Cl becomes steady in 86% of the dataset's runs; close to the ground the flow separates, and part of the range has no steady solution | [Step 6](06-parameter-scan.md) |
| U-Net in ground effect | 0.8% error on 28 unseen airfoils with all 6,177 training samples and the distance input (0.9% with 3,096 samples, 1.4% without the distance input; the free-air model as it is: 20%), with the output set to zero inside the wing; with Thuerey et al.'s error over all cells, 0.99% (free-air model: 24%) | [Steps 8, 10, 13 and 16](16-final-models.md) |
| Downforce from the U-Net | 3.3% error with all the samples (3.9% with 3,096), the loss weighted next to the wing, the distance input and random starting weights (with 3,096 samples, from the free-air model: 6.2% with the weighting alone, 9.9% with the plain loss; from random weights without the distance: 6.8%); twice the training does not lower it | [Steps 9, 10, 13 and 16](16-final-models.md) |
| Downforce from a model of its own | a small network on the shape, ride height, angle and speed: 1.5% error on the 28 test airfoils (1.2% against OpenFOAM's force, where the U-Net is 4.0%); 0.65 ms per prediction from the airfoil file, against 28 ms for the U-Net and 318 s for a simulation | [Steps 14 and 16](16-final-models.md) |
| Pressure along the wing | a small network for the pressure and the friction along the surface: downforce 1.2%, drag 7.5% on the test airfoils (4–5% on the two wings), moment 2.2%; the separated part of the surface, 10% of it on average, is off by 0.2% of the surface | [Steps 15 and 16](16-final-models.md) |
| Drag from a model of its own | the small network of step 14 trained on the drag: 1.6% error on the test airfoils, against 7.5% for the drag added up from the surface model; 0.3% and 1.3% on the two wings (4.0% and 4.7%) | [Step 17](17-drag-model.md) |
| Where the downforce peaks | over 1,444 airfoils, the ride height of the largest downforce rises with the angle (median h/c 0.107, 0.146 and 0.204 at 2°, 5° and 8°), the camber and the thickness, and falls slightly with the speed; checked against 264 new simulations of six unseen airfoils: within 1.4% (median); at the peak the flow under the wing has separated from x/c ≈ 0.78 | [Step 18](18-ride-height-map.md) |
| Downforce from little data | a Gaussian process is the most accurate model of the downforce with up to about 200 training samples (7.4% with 97, where the small network is 12.4%), and says how sure it is; from 386 samples on, the small network is better (with all the samples by a margin not significant over the 28 test airfoils) | [Steps 16 and 19](16-final-models.md) |
| Classical regressions | polynomial regression, a random forest and support vector regression, tuned on the validation samples: downforce 3.8–6.9% and drag 5.4–8.7% off on the test airfoils, three to five times the small networks' errors, at every amount of data | [Step 19](19-classical-baselines.md) |
| Unseen wing: NACA 4412 | at 11 ride heights: downforce within 2.0% with all the samples (5% and 4% with 3,096, without and with the distance input)¹, peak at the same height as the CFD | [Steps 11, 13 and 16](16-final-models.md) |
| Unseen wing: Tyrrell 026 (F1) | at the tunnel's 9 ride heights: downforce within 2.6% with all the samples (1.1% with 3,096 and the distance input, 10–19% low without it)¹; the suction peak still too weak | [Steps 12, 13 and 16](16-final-models.md) |

¹ Against the downforce that the simulated pressure gives on the same
128 × 128 grid. Against OpenFOAM's own force, with the straight-line
correction of [step 14](14-downforce-model.md), the final U-Net is
within 1.9% (NACA 4412) and 3.0% (Tyrrell 026), and the two small models
within 0.9% on both wings.

The test errors are those of all 102 test samples. Two test airfoils are
near copies of training airfoils (see [step 7](07-dataset.md)); without
their 7 samples the errors are 3–5% higher (0.8% stays 0.8%, 3.3% becomes
3.4%).

---

[README](../README.md) · [Workflow](../README.md#workflow)
