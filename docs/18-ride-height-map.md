# 18. Where the downforce peaks: a map of 1,444 airfoils

[README](../README.md) · ← [17. The drag alone](17-drag-model.md) · [19. Classical regressions](19-classical-baselines.md) → · [Workflow](../README.md#workflow)

Every inverted airfoil has a ride height at which its downforce is largest;
closer to the ground the downforce falls again. Where is that peak, and what
moves it? The downforce network of [step 16](16-final-models.md) predicts a
whole ride-height sweep in a fraction of a second, so we used it to map the
peak for all 1,444 airfoils of the data, at angles from 0° to 10° and at
three speeds, and checked the map with 264 new simulations of six test
airfoils. The network finds the simulated peak within 1.4% (median). The
peak moves away from the ground as the angle, the camber and the thickness
grow, and slightly towards it at higher speed. At the peak, the flow under
the wing has separated from about 78% of the chord onwards.

*Words used here: [glossary](glossary.md). References: [list](references.md).*

---

**The map.** The two downforce networks of the final runs (their mean)
predict the downforce of every airfoil with a converged sample, training and
test airfoils alike, at 50 ride heights from h/c = 0.05 to 0.6 (evenly spaced
on a log scale), at angles from 0° to 10° in steps of 1°, and at 10, 30 and
50 m/s (Reynolds numbers of 1, 3 and 5 million): 2.4 million predictions in
about half a minute on the laptop. Along each sweep, the peak is the largest
of the 50 values, placed between the ride heights by a parabola (in log h/c)
through it and its two neighbours. Five numbers describe each airfoil's
shape: its largest camber and largest thickness, where along the chord they
lie, and the angle of the camber line at the trailing edge (from 90% to 99%
of the chord; positive when it turns towards the ground, as a flap would).

**The check.** New simulations, with the settings of the data, of six test
airfoils that no model was trained on, chosen to span the thickness at
little and at much camber:

| Airfoil | Thickness | Camber |
|---|---|---|
| ag09 | 4.9% | 1.9% |
| n64012a | 12% | 0 (symmetric) |
| goe775 | 21% | 0.2% |
| goe265 | 7.7% | 7.3% |
| goe07k | 11% | 4.6% |
| goe626 | 16.7% | 5.1% |

Each at 2°, 5° and 8° and 30 m/s, at the 12 ride heights of the NACA 4412
sweep ([step 11](11-naca4412-sweep.md)); goe265 and goe775 also at 5° and 10
and 50 m/s. That is 22 sweeps, 264 runs, of which 242 converged (5 hours on
the laptop, 4 runs at a time). `ride_height_map.py` compares them, with the
NACA 4412 and Tyrrell 026 sweeps, with the network at the same ride heights,
finding the peak in the same way for both. The commands are in
[How to run it](howto.md#where-the-downforce-peaks).

## The check against the simulations

24 sweeps of 8 wings ([table](../results/ride_height_sweeps.csv), every run:
[table](../results/ride_height_points.csv)):

| | Network against the simulations |
|---|---|
| Downforce, 262 converged runs | 0.83% off on average |
| Where the peak lies (inside the range, or at its lowest height) | the same in all 24 sweeps |
| Peak ride height, 18 sweeps with a regular peak inside the range | 1.4% apart (median), 4.9% at most |
| Peak downforce, the same 18 sweeps | 0.36% apart (median), 1.1% at most |

The peak ride height, simulated / predicted, at 30 m/s:

| Airfoil | 2° | 5° | 8° |
|---|---|---|---|
| goe265 | 0.118 / 0.118 | 0.171 / 0.173 | 0.248 / 0.250 |
| goe626 | 0.149 / 0.147 | 0.207 / 0.202 | 0.271 / 0.284 |
| goe775 | 0.118 / 0.119 | 0.164 / 0.168 | 0.232 / 0.239 |
| n64012a | 0.059 / 0.059 | 0.092 / 0.095 | 0.147 / 0.141 |
| goe07k | 0.144 / 0.149 | no steady flow below 0.123 | no steady flow below 0.194 |
| ag09 | still rising at 0.05 | still rising at 0.05 | still rising at 0.05 |

The NACA 4412 (5°, 30 m/s) peaks at 0.158 in both, the Tyrrell 026 (3.45°,
10 m/s) at 0.123 in both.

- **The angle:** for every airfoil with a peak at all three angles, the peak
  moves away from the ground as the angle grows, in the simulations as in
  the network.
- **The speed:** goe265 at 5° peaks at 0.188, 0.171 and 0.165 at 10, 30 and
  50 m/s (network: 0.188, 0.173, 0.166), with a downforce of 2.030, 2.190
  and 2.280 (network: 2.031, 2.193, 2.274).
- **ag09:** its downforce still grows at h/c = 0.05 at all three angles,
  in both.
- **goe07k:** at 5° and 8°, the 4 and 6 runs closest to the ground found no
  steady flow. The network puts the peak at about 0.12 and 0.09, at the edge
  of and inside that region, where the simulations cannot check it.

## One irregular sweep: goe775 at 10 m/s

One sweep, goe775 at 5° and 10 m/s (Re = 1 million), jumped up and down by
about 15% from one ride height to the next, although every run converged.
Ten more runs (`sweep_goe775_check.sh`): five on the fine mesh of the mesh
study ([step 3](03-mesh-study.md), cells √2 times smaller) at the ride
heights of the jumps, and five on the normal mesh at ride heights in between
([table](../results/ride_height_goe775_check.csv)). Downforce −Cl, the low
values in bold:

| h/c | Normal mesh | Fine mesh | Network |
|---|---|---|---|
| 0.155 | 0.762 | 0.802 | 0.782 |
| 0.170 | 0.771 | | 0.792 |
| 0.180 | **0.652** | | 0.794 |
| 0.194 | **0.648** | 0.809 | 0.794 |
| 0.200 | 0.772 | | 0.792 |
| 0.210 | 0.768 | | 0.788 |
| 0.220 | **0.642** | | 0.782 |
| 0.243 | 0.747 | 0.773 | 0.765 |
| 0.305 | 0.697 | 0.721 | 0.712 |
| 0.382 | **0.572** | 0.667 | 0.659 |

- **On the normal mesh**, the downforce flips between two levels, in no
  order, even between ride heights 0.006 chords apart: this wing has two
  steady solutions on that mesh, and the solver settles into one of them
  within its first 200 iterations. The lower one has 11–16% less downforce.
- **What differs** is how the air leaves the wing's blunt trailing edge: on
  the lower solution the air on the top side speeds round the trailing-edge
  corner, the pressure there differs between the two sides by 0.32–0.35
  times the dynamic pressure instead of 0.07, and the suction under the wing
  drops along the whole chord. Where the flow separates under the wing does
  not change.
- **On the fine mesh** the sweep is smooth. Its peak, at h/c = 0.180, lies
  2.9% from the network's (0.185), and the network is within 2.5% of it at
  every ride height.
- **The mesh matters more here than usual:** the fine mesh gives 3.5–5% more
  downforce than the normal mesh's upper level, where for the Tyrrell 026 the
  two meshes differ by 0.1–2.4% ([step 4](04-wind-tunnel.md)).

This sweep is left out of the peak statistics above. We had first read the
network's larger errors on thick airfoils at low speed as a weakness of the
network; that came from a single validation sample (s1223 at h/c 0.071).
Without it those samples are 0.88% off, like the rest.

## The map

Median peak ride height of the airfoils whose peak lies inside the range:

| Angle | 10 m/s | 30 m/s | 50 m/s | Peak inside the range (30 m/s) |
|---|---|---|---|---|
| 2° | 0.113 | 0.107 | 0.104 | 1,256 of 1,444 |
| 5° | 0.157 | 0.146 | 0.141 | 1,347 |
| 8° | 0.219 | 0.204 | 0.196 | 1,267 |

At 30 m/s the median is 0.092 at 0° and 0.248 at 10°. Most of the other
airfoils still gain downforce at h/c = 0.05. The airfoils differ widely: at
5° and 30 m/s, a tenth of them peak below 0.076 and a tenth above 0.259.
From 10 to 50 m/s, the peak moves 11–13% closer to the ground and its
downforce grows by 12–17% (medians at 2°, 5° and 8°).

**The shape**, at 5° and 30 m/s (1,347 airfoils). How much the peak changes
for one standard deviation more of each shape number, in a straight-line
fit on all five together (the ride height fitted on a log scale):

| One standard deviation more | Peak ride height | Peak downforce |
|---|---|---|
| Largest camber (2.2% of the chord) | +36% | −3.2% |
| Largest thickness (4.2% of the chord) | +38% | −9.3% |
| Position of the largest camber (0.19 chords further back) | +1% | −0.9% |
| Position of the largest thickness (0.07 chords further back) | +3% | −6.1% |
| Angle at the trailing edge (4.3°) | −4% | +13.5% |
| Share of the differences between airfoils explained | 84% | 82% |

- **The peak ride height** follows the thickness and the camber: thicker and
  more cambered airfoils peak further from the ground (rank correlations
  +0.74 and +0.58).
- **The peak downforce** follows the angle at the trailing edge most
  (rank correlation +0.61), and falls with the thickness (−0.50). The camber
  goes with more downforce on its own (+0.48), but only because cambered
  airfoils also turn down at the trailing edge (the two go together, +0.71):
  at the same angle there, more camber adds none. This is what thin-airfoil
  theory expects, where the slope of the camber line counts most near the
  trailing edge.
- **Chowdhury and Shukla (2026)** found, for NACA four-digit airfoils, more
  downforce near the ground with more camber, less thickness and the camber
  further forward. Over the 1,444 airfoils, the first two hold, but the
  position of the largest camber hardly matters (rank correlation +0.13).

## Separation at the peak

In the 18 regular simulated sweeps, the flow on the side facing the ground
separates further forward as the ride height falls. At the peak it separates
at x/c = 0.78 (median; 0.65 to 0.85): 0.77–0.85 at 2°, 0.70–0.79 at 5° and
0.65–0.75 at 8°, with no trend with the speed. The Tyrrell 026 peaks with the
separation at 0.80, the NACA 4412 at 0.79.

## What it means

- **One picture:** near the ground, the air speeds up through the gap under
  the wing and the suction grows, but so does the rise in pressure the air
  must climb towards the trailing edge. The separation that this causes
  creeps forward, and once roughly the last fifth of the chord has separated
  the downforce stops growing. A higher angle, more camber or more thickness
  steepen that rise, so the peak comes further from the ground; at higher
  speed the boundary layer resists it better, so the peak comes closer and
  grows.
- **For design:** the peak downforce depends most on how far the camber line
  turns at the trailing edge (and less thickness helps); where the peak lies
  depends on the angle, the camber and the thickness.
- **Limits:** close to the ground many runs had no steady flow: at 5°, 38%
  of the predicted peaks lie where more than a quarter of the training runs
  failed (most of them below h/c 0.13), and for goe07k the network puts the
  peak where the simulations found no steady flow. The turbulence model
  separates the flow 0.05–0.12 chords too early ([step 4](04-wind-tunnel.md)),
  which may put the peaks a little too far from the ground. Thick airfoils
  with a blunt trailing edge at the lowest speed can have two steady
  solutions on the normal mesh, as goe775 had; how often this happens in the
  data was not checked.

The map itself (`peaks.csv`, 7 MB, and `curves.npz`) is not in the
repository: `ride_height_map.py` makes it again in half a minute from the
published data and models.

---

[README](../README.md) · ← [17. The drag alone](17-drag-model.md) · [19. Classical regressions](19-classical-baselines.md) → · [Workflow](../README.md#workflow)
