# Glossary

[README](../README.md) · [Workflow](../README.md#workflow)

Words used on these pages.

---

| Word | Meaning |
|---|---|
| CFD | Computational fluid dynamics: computing airflow on a computer |
| RANS | Reynolds-averaged Navier–Stokes: computing the average flow, with a turbulence model for the turbulent eddies |
| OpenFOAM | The free CFD software used here (version v2606) |
| Mesh, cells | The grid of small cells on which the flow is computed |
| Boundary layer | The thin layer of slow air next to the wing's surface |
| y+ | How thin the first cell at the wall is, in the boundary layer's own units. y+ ≈ 1 resolves the layer right down to the wall |
| Spalart–Allmaras (SA) | The turbulence model used: one extra equation for the turbulent mixing |
| nuTilda | The quantity that equation solves for. "Clipping" means OpenFOAM resets values that come out negative |
| Iterations, converged | The solver improves its answer step by step (iterations); a run has converged when the answer stops changing |
| GCI | Grid convergence index: how much a result could still change on an infinitely fine mesh |
| LES | Large-eddy simulation: a far more detailed and costly simulation. Our reference is a high-resolution LES, close to DNS (direct numerical simulation, which computes all the turbulent eddies) |
| Chord (c) | The length of the airfoil from front to back |
| Ride height (h/c) | The gap between the wing's lowest point and the ground, divided by the chord |
| Angle of attack (α) | How far the wing is tilted against the oncoming air |
| Cl, Cd | Downforce (lift) and drag, divided by ½ × air density × speed² × chord, so that cases can be compared. Negative Cl means downforce |
| Cp | The pressure at a point of the surface, minus that of the oncoming air, divided by ½ × air density × speed². Below 0 means suction; plots show −Cp, suction up |
| Cf | The friction of the air on the surface, divided by the same. Negative where the air next to the wall runs back, towards the leading edge: separated flow |
| Cm | The turning effect of the forces about the quarter chord: it says where on the wing the force acts |
| Pressure distribution | Cp along the wing's surface, from the leading to the trailing edge: what wind tunnels measure with small pressure holes in the model |
| Reynolds number (Re) | Speed × chord ÷ the air's viscosity; here 1–5 million |
| Ground effect | The extra downforce a wing makes close to the ground |
| Moving ground | The ground moves at the air's speed, as the road does under a car |
| Separation | Where the air stops following the wing's surface; behind it the flow turns back and becomes unsteady |
| Sample | One simulation, stored as six 128 × 128 images: 3 inputs (the air's speed, and where the wing and the ground are) and 3 outputs (pressure and the two velocity components) |
| Mask | The input that shows where the wing and the ground are: each cell of the 128 × 128 grid is solid or air |
| Signed distance | For each cell, the distance to the nearest surface (the wing or the ground), negative inside the wing and below the ground. Since step 13, an input of the U-Net: it gives the exact shape, finer than the cells. Computed from the mask instead, it shows no more of the shape than the mask |
| U-Net | The type of neural network used: it turns input images into output images |
| Fully connected network (MLP) | A neural network whose input is a list of numbers, not an image; each layer connects every number to every neuron. Step 14's comparison model is one: the wing's shape at 32 points, the ride height, the angle and the speed in, the downforce out |
| Surrogate | A fast model that replaces the slow simulation |
| Loss (L1) | What training makes small: how far the network's output is from the simulation. L1 is the mean absolute difference |
| Random weights, free-air model | The two ways a training run starts: from random numbers in the network, or from the U-Net trained on the free-air data of step 1 |
| Wilcoxon test, p | A test of whether one model is closer to the simulation than another on the same samples. p is the chance of a difference this consistent if the two were equally good: small p, a real difference |
