# 17. The drag alone: a network for the drag

[README](../README.md) · ← [16. All the data](16-final-models.md) · [18. Where the downforce peaks](18-ride-height-map.md) → · [Workflow](../README.md#workflow)

The surface model of [step 15](15-surface-model.md) gets its drag by adding
up its pressure and friction along the wing, and on the test airfoils that
drag is 7.5% off OpenFOAM's. Would a network that predicts the drag
directly do better, as the network of [step 14](14-downforce-model.md) does
for the downforce? We trained step 14's network on OpenFOAM's drag instead
of the downforce. It is almost five times closer: 1.6% off on the 28 test
airfoils, and 0.3% and 1.3% on the two wings of steps 11 and 12, where the
surface model is 4.0% and 4.7% off.

*Words used here: [glossary](glossary.md). References: [list](references.md).*

---

**The network.** The same as step 14's: the shape as the camber and the
thickness at 32 points along the chord, the ride height (and its
logarithm), the angle and the speed; four layers of 256 neurons. Its
settings are those chosen for the downforce in [step 16](16-final-models.md),
not tuned again for the drag. Only the target changes, to OpenFOAM's Cd
(`baseline_cl.py --target cd`). The same 6,177 training and 709 validation
samples as in step 16; 2 seeds, about 2.5 minutes each on the laptop's CPU.
The commands are in [How to run it](howto.md#the-drag-alone).

## Test samples

Against OpenFOAM's drag, mean of the 2 seeds
([table](../results/drag_model_final.csv)):

| | Drag network | Surface model (drag added up) |
|---|---|---|
| Test samples (28 airfoils) | 1.6% | 7.5% |
| Validation samples | 1.3% | 4.9% |
| Test, h/c 0.05–0.08 (14 samples) | 1.3% | 2.6% |
| Test, h/c 0.08–0.135 (15) | 1.3% | 5.5% |
| Test, h/c 0.135–0.22 (22) | 0.9% | 6.8% |
| Test, h/c 0.22–0.365 (26) | 3.0% | 12.9% |
| Test, h/c 0.365–0.6 (25) | 1.5% | 14.2% |

- **Closer almost everywhere:** the drag network is closer than the surface
  model on 97 of the 102 test samples (Wilcoxon p = 3 × 10⁻¹⁷) and on 27 of
  the 28 test airfoils.
- **At every ride height,** and most of all far from the ground, where the
  drag is smallest: there the surface model is 14% off, the drag network
  1.5%.
- **The thick strut again:** the drag network's largest error, 3.0% at
  h/c 0.22–0.365, comes mostly from the two samples of e864, the 39% thick
  test airfoil (1.4% without them). Without them, the drag network is 1.2%
  off over all the test samples.

## The two wings

The mean (and the largest) difference from OpenFOAM's drag over the ride
heights:

| | NACA 4412 | Tyrrell 026 |
|---|---|---|
| Drag network | 0.3% (1.1%) | 1.3% (2.4%) |
| Surface model | 4.0% (12%) | 4.7% (10%) |

## Why

The drag is a small difference of large forces. At the nose, the
stagnation pressure pushes the wing back and the suction around the nose
pulls it forward, and the two nearly cancel (this is why step 15 rebuilds
the wing's surface exactly). Small errors of the surface model's pressure
there change the drag by several percent, while the downforce, the pressure
difference summed over the whole chord, hardly notices them. The drag
network learns the drag itself. Field models in free air meet the same
problem: in the AirfRANS benchmark (Bonnet et al., 2022), the drag computed
from four models of the flow did not even put the airfoils in the right
order (rank correlations between −0.3 and 0 with the true drag), while
their lift did (above 0.9).

## What it means

For each force, a small network of its own is the most accurate model we
have: 1.2% for the downforce ([step 16](16-final-models.md)) and 1.6% for
the drag. The drag network's error is well below the numerical uncertainty
of the data's drag, about 10% ([step 3](03-mesh-study.md)). The surface
model stays the model for the moment, for where the flow separates and for
the pressure along the wing; the U-Net for the whole flow.

The drag network came after the Zenodo record was published, so it is not
in it; the commands above train it again in a few minutes.

---

[README](../README.md) · ← [16. All the data](16-final-models.md) · [18. Where the downforce peaks](18-ride-height-map.md) → · [Workflow](../README.md#workflow)
