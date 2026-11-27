# 19. Classical regressions: are the networks needed?

[README](../README.md) · ← [18. Where the downforce peaks](18-ride-height-map.md) · [Workflow](../README.md#workflow)

Would a classical regression predict the forces as well as the small
networks of steps [14](14-downforce-model.md) and [17](17-drag-model.md)? We
trained three common ones on the same inputs and samples: polynomial
regression, a random forest and support vector regression, each with its
settings tuned on the validation samples. None comes close. On the 28 test
airfoils they are 3.8–6.9% off for the downforce and 5.4–8.7% for the drag,
where the downforce network is 1.2% off and the drag network 1.6%: three to
five times more accurate, and closer on 25 to 28 of the 28 airfoils. We also
scored every model of [step 16](16-final-models.md) with more measures: the
root-mean-square error, R² and the largest error.

*Words used here: [glossary](glossary.md). References: [list](references.md).*

---

**The methods.** All three from scikit-learn 1.9, on the inputs of the
downforce network (the camber and the thickness at 32 points along the
chord, the ride height and its logarithm, the angle and the speed), with
the same 6,177 training, 709 validation and 102 test samples as in step 16:

- **Polynomial regression:** the downforce as a sum of the inputs, their
  squares and their products (degree 1 or 2), with a ridge penalty that
  keeps the terms small.
- **Random forest:** the average of 300 decision trees, each grown on a
  random draw of the samples.
- **Support vector regression:** a smooth curve through the samples with a
  Gaussian kernel, which ignores errors smaller than a tolerance.

Each method's settings were chosen on the validation samples, for every
amount of training data and for the drag separately. The search of the
support vector regression grows its range where the best setting lies at
an end of it and the error still falls there, except for its strongest
setting, C, which stops at 3,000: there one fit already takes minutes with
3,088 samples, and over 20 minutes at C = 10,000. About 2.5 hours on the
laptop in all, nearly all of it the two searches of the support vector
regression with all the samples. The commands are in
[How to run it](howto.md#classical-regressions).

## Test samples

Against OpenFOAM's forces, mean of the 2 seeds
([table](../results/classical_baselines.csv)). "Network closer": on how many
of the 102 test samples, and of the 28 test airfoils (each by the mean error
of its samples), the downforce (drag) network is closer to OpenFOAM:

| Downforce | Error | RMSE | R² | Largest error | Network closer (samples) | Network closer (airfoils) |
|---|---|---|---|---|---|---|
| Surface model (step 15) | 1.18% | 0.041 | 0.991 | 0.34 | 50 | 12 |
| Downforce network (step 14) | 1.24% | 0.034 | 0.994 | 0.22 | | |
| Gaussian process (step 16) | 1.50% | 0.041 | 0.992 | 0.28 | 60 | 17 |
| Support vector regression | 3.79% | 0.064 | 0.979 | 0.30 | 90 | 26 |
| U-Net, from its pressure (step 16) | 3.97% | 0.088 | 0.961 | 0.59 | 90 | 28 |
| Polynomial regression | 6.26% | 0.111 | 0.938 | 0.36 | 93 | 28 |
| Random forest | 6.88% | 0.121 | 0.926 | 0.57 | 98 | 28 |

| Drag | Error | RMSE | R² | Largest error | Network closer (samples) | Network closer (airfoils) |
|---|---|---|---|---|---|---|
| Drag network (step 17) | 1.60% | 0.0015 | 0.997 | 0.0091 | | |
| Support vector regression | 5.37% | 0.0042 | 0.980 | 0.031 | 87 | 26 |
| Surface model (step 15) | 7.50% | 0.0051 | 0.970 | 0.023 | 97 | 27 |
| Polynomial regression | 8.01% | 0.0049 | 0.971 | 0.026 | 89 | 25 |
| Random forest | 8.68% | 0.0065 | 0.951 | 0.045 | 92 | 27 |

- **The networks win on almost every airfoil:** against each classical
  regression, for both forces, the network is closer on 25 to 28 of the 28
  test airfoils (Wilcoxon p ≤ 10⁻⁶ over the airfoils, not only over the
  samples, since the samples of one airfoil are not independent).
- **Support vector regression is the best of the three,** and its downforce
  is a little closer than the U-Net's. But the U-Net predicts the whole flow;
  its downforce is a by-product.
- **The root-mean-square error shows what the mean hides:** the surface
  model's mean error is the smallest, but its RMSE and its largest error are
  larger than the downforce network's (0.041 against 0.034, and 0.34
  against 0.22): a few of its samples are further off.
- **R² hardly tells the best models apart:** 0.991–0.994 for the three best,
  whose mean errors are 1.18–1.50%. The mean error stays our main measure.
- **The Gaussian process against the downforce network:** with all the
  samples the network is closer on 60 of the 102 test samples (p = 0.03),
  but on only 17 of the 28 test airfoils (p = 0.14): too small a margin to
  be sure of with 28 airfoils.
- **Trained on OpenFOAM's downforce, not the grid's:** the downforce network
  trained on the downforce summed on the U-Net's grid (step 14's first
  version) is 2.18% off OpenFOAM's force, against 1.24% (in the table of
  results; its Cl corrected by step 14's straight line, as the U-Net's).

## With little data

The downforce on the test samples against the number of training samples,
on the same samples at each size as in step 16
([table](../results/classical_learning_curve.csv)):

| Training samples | 97 | 193 | 386 | 772 | 1,544 | 3,088 | 6,177 |
|---|---|---|---|---|---|---|---|
| Gaussian process (step 16) | 7.4% | 5.3% | 3.7% | 2.7% | 2.5% | 2.1% | 1.5% |
| Downforce network (step 16) | 12.4% | 6.0% | 3.1% | 2.2% | 1.8% | 1.4% | 1.2% |
| Support vector regression | 13.8% | 9.6% | 7.4% | 6.3% | 5.9% | 5.4% | 3.8% |
| Polynomial regression | 15.6% | 10.6% | 8.3% | 7.1% | 6.7% | 6.2% | 6.3% |
| Random forest | 15.2% | 13.7% | 11.4% | 9.9% | 8.1% | 7.5% | 6.9% |

- **Not even with little data:** the Gaussian process stays the best model
  with up to about 200 samples, the network from about 400 on; the
  classical regressions are behind both at every size, even though their
  settings were tuned again at every size and the network's were not.
- **The polynomial stops learning:** 6.2% with 3,088 samples and 6.3% with
  all of them. A sum of squares and products cannot follow how the downforce
  changes close to the ground.

## Is the comparison fair?

The classical regressions had the advantage. Their settings were tuned on
the validation samples at every size and for the drag separately (16
settings for the polynomial, 9 for the random forest, at least 160 for the
support vector regression). The downforce network was tuned once, on all
the samples (54 settings); the drag network not at all, as it keeps the
downforce network's settings (step 17); and the Gaussian process uses no
validation samples. All models saw the same inputs and samples, and none
saw the test samples. So the networks' lead cannot come from more tuning.

Two limits:

- **The support vector regression has one kernel width for all 68 inputs,**
  as usual, while the Gaussian process learns one width for each input.
  With 64 of the inputs for the shape, the ride height, the angle and the
  speed count for little in its distances, which may be why it is behind
  the Gaussian process.
- **C stopped at 3,000** in 3 of the 48 searches: the drag with all the
  samples (0.09 points better than at C = 1,000) and the downforce with
  3,088 samples and seed 1 (0.16 points). A larger C might gain about as
  much again, far from the gap to the networks.

## What it means

For the forces, the small networks are the models to use: 1.2% for the
downforce and 1.6% for the drag, three to five times more accurate than any
classical regression; with very little data (up to about 200 samples) the
Gaussian process. The new measures keep the ranking, and the
root-mean-square error adds one thing: the surface model, as accurate as
the downforce network on average, has a few larger misses.

These models came after the Zenodo record was published, so they are not
in it; the script trains them again and scores the record's models (the
Gaussian process needs its fit, `gp.json`, made as in
[How to run it](howto.md#the-gaussian-process-on-the-sweeps)).

---

[README](../README.md) · ← [18. Where the downforce peaks](18-ride-height-map.md) · [Workflow](../README.md#workflow)
