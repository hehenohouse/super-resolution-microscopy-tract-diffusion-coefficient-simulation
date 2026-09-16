# SRM Tracking Simulation Framework

A modular framework for constructing controlled super-resolution microscopy (SRM) particle-tracking simulations.

The project is not a single physical model. Each simulation **scenario** combines reusable components. Standard scenarios use four components, while coupled scenarios may also include a shared scene model:

```text
motion model
    + boundary model
    + photophysics model
    + observation model
    + optional scene model
    = simulation scenario
```

Five scenarios are currently included:

1. A default 3D free-diffusion benchmark observed through a finite axial slab and projected to 2D.
2. A reflecting 2D baseline for confinement and tracking experiments.
3. A free-diffusion 2D benchmark for testing diffusion-coefficient recovery without a reflecting wall.
4. A single-chromosome-bound 2D simulation that combines protein diffusion in one chromosome-local effective ellipse with shared chromosome translation and rotation.
5. A multi-chromosome-bound 2D simulation with six independently moving effective chromosome ellipses and permanent protein-to-chromosome assignments.

The repository does not yet contain a tracking algorithm or tracking evaluation metrics.

## Quick start

The project requires Python 3.11 or newer.

Install it in Windows PowerShell:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -e .
```

Run the default 3D-to-2D benchmark:

```powershell
& ".\.venv\Scripts\python.exe" main.py
```

Run either retained 2D benchmark or the chromosome simulation explicitly:

```powershell
& ".\.venv\Scripts\python.exe" main.py --scenario reflecting
& ".\.venv\Scripts\python.exe" main.py --scenario free
& ".\.venv\Scripts\python.exe" main.py --scenario chromosome
& ".\.venv\Scripts\python.exe" main.py --scenario chromosomes
```

Skip GIF generation during faster numerical runs:

```powershell
& ".\.venv\Scripts\python.exe" main.py --no-animation
```

Running without `--scenario` selects the 3D-to-2D benchmark.

Each scenario writes to its own directory:

```text
outputs/<scenario_id>/observations.npz
outputs/<scenario_id>/ground_truth.npz
outputs/<scenario_id>/simulation_preview.gif
```

## Scenario summary

| Property | Default projected 3D | Reflecting 2D | Free 2D | Single chromosome | Multiple chromosomes |
|---|---|---|---|---|---|
| Scenario ID | `free_diffusion_benchmark_3d_to_2d_axial_slab_markov` | `reflecting_baseline_2d_brownian_markov` | `free_diffusion_benchmark_2d_brownian_markov` | `chromosome_bound_dynamic_2d` | `multi_chromosome_bound_dynamic_2d` |
| Physical motion | 3D Brownian | 2D Brownian | 2D Brownian | Local Brownian plus one rigid pose | Local Brownian plus independent rigid poses |
| Physical boundary | None | Reflecting square | None | One effective local ellipse | One effective local ellipse per chromosome |
| Observation region | Central 10 x 10 x 2 um | 10 x 10 um | Central 10 x 10 um | 10 x 10 um lab frame | 10 x 10 um lab frame |
| Public coordinates | 2D XY projection | 2D XY | 2D XY | Noisy 2D XY | Noisy 2D XY |
| Number of particles | 5,000 | 100 | 900 | 100 | 120 total across 6 chromosomes |
| Protein diffusion coefficient | 0.5 um^2/s | 0.5 um^2/s | 0.5 um^2/s | 0.005 um^2/s | 0.1 um^2/s |
| Intended use | Projected tracking | Wall-sensitive baseline | Free-diffusion benchmark | Single-chromosome control | Multi-chromosome coupled simulation |

All five scenarios use 200 frames, a frame interval of 0.05 s, identical three-state Markov photophysics, and reproducible random seeds. The first three use ideal localization; both chromosome scenarios add configurable Gaussian localization noise.

## Default projected 3D benchmark

Stable scenario ID:

```text
free_diffusion_benchmark_3d_to_2d_axial_slab_markov
```

Composition:

```text
Motion:       BrownianMotion3D
Boundary:     UnboundedSpace3D
Photophysics: ThreeStateMarkovBlinking
Observation:  IdealAxialSlabProjectionObservation
```

The private physical state is three-dimensional. Five thousand particles are initialized uniformly in

$$
0\leq x\leq30,
\qquad
0\leq y\leq30,
\qquad
0\leq z\leq22
$$

and then diffuse without reflection, wrapping, or clipping. The initialization cuboid is not a physical boundary.

A molecule can be detected only while it is ON and inside the central observation volume

$$
10\leq x\leq20,
\qquad
10\leq y\leq20,
\qquad
10\leq z\leq12.
$$

The observation model discards the axial coordinate and publishes only

$$
(x,y,z)\longmapsto(x,y).
$$

For isotropic 3D Brownian motion,

$$
\Delta x,\Delta y,\Delta z
\sim\mathcal{N}\!\left(0,2D\Delta t\right),
$$

so the full physical displacement obeys

$$
\mathbb{E}\!\left[\Delta x^2+\Delta y^2+\Delta z^2\right]
=6D\Delta t.
$$

After ideal XY projection, the public displacement still obeys

$$
\mathbb{E}\!\left[\Delta x^2+\Delta y^2\right]
=4D\Delta t.
$$

Therefore an XY estimator can still recover the same isotropic diffusion coefficient $D$. The practical difference from native 2D motion is trajectory censoring: an emitting molecule disappears when it leaves the axial slab and can reappear after returning, even if its photophysical state never changed.

The observation volume has 10 um of initialization padding on every face. For the default experiment, the one-coordinate 10 s displacement scale is

$$
\sqrt{2Dt}=\sqrt{10}\approx3.16\;\mu\mathrm{m},
$$

so the padding limits finite-reservoir depletion near the observation volume without introducing a wall.

## Reflecting baseline

Stable scenario ID:

```text
reflecting_baseline_2d_brownian_markov
```

Full name:

**Reflecting Baseline - Ideal 2D Brownian Motion in a Reflecting Square with Three-State Markov Photophysics**

Composition:

```text
Motion:       BrownianMotion2D
Boundary:     ReflectingSquareBoundary
Photophysics: ThreeStateMarkovBlinking
Observation:  IdealOnStateObservation over [0, 10] x [0, 10] um
```

All particles begin inside the 10 x 10 um square. A coordinate that crosses a wall is reflected back into the square. For a one-dimensional interval from 0 to $L$, the mapping is

$$
x_{\mathrm{reflected}}
= L-\left|\left(x\bmod 2L\right)-L\right|.
$$

This scenario is useful when wall interactions or confined trajectories are part of the benchmark. It should not be treated as unlimited free diffusion at long lag times.

For two independent uniformly distributed positions in a reflecting square, the long-lag two-dimensional MSD approaches

$$
\mathrm{MSD}_{\infty}=\frac{L^2}{3}.
$$

For $L=10\;\mu\mathrm{m}$, this plateau is approximately

$$
\mathrm{MSD}_{\infty}\approx33.3\;\mu\mathrm{m}^2.
$$

Fitting the free-diffusion relation over lags affected by this plateau will generally underestimate $D$.

## Free-diffusion benchmark

Stable scenario ID:

```text
free_diffusion_benchmark_2d_brownian_markov
```

Full name:

**Free-Diffusion Benchmark - Unbounded 2D Brownian Motion with a Finite Observation Window and Three-State Markov Photophysics**

Composition:

```text
Motion:       BrownianMotion2D
Boundary:     UnboundedPlane2D
Photophysics: ThreeStateMarkovBlinking
Observation:  IdealOnStateObservation over [10, 20] x [10, 20] um
```

The benchmark initializes 900 particles uniformly in a 30 x 30 um reservoir. After initialization, particle positions are never reflected, wrapped, clipped, or otherwise corrected. A particle can move beyond the initial reservoir and continues on the unbounded plane.

Only the central window is observed:

$$
10\leq x\leq20,
\qquad
10\leq y\leq20.
$$

A molecule produces a detection only when it is both ON and inside this field of view. Molecules can naturally enter and leave the observation window.

The initial reservoir has 10 um of padding between each side of the observation window and the reservoir edge. This reduces reservoir-edge depletion during the 10 s default experiment. It does not create a physical wall.

The free scenario uses 900 particles because

$$
\frac{900}{30\times30}
=
\frac{100}{10\times10}
=1\;\text{molecule}/\mu\mathrm{m}^2.
$$

Thus, the two 2D scenarios begin with the same areal density.

### Why this benchmark is preferred for recovering D

For homogeneous free Brownian motion in two dimensions,

$$
\Delta x,\Delta y
\sim
\mathcal{N}\!\left(0,2D\Delta t\right),
$$

and

$$
\mathbb{E}\!\left[\lVert\Delta\mathbf{r}\rVert^2\right]
=4D\Delta t.
$$

A one-frame ground-truth estimator is therefore

$$
\widehat{D}
=
\frac{1}{4N\Delta t}
\sum_{j=1}^{N}
\left\lVert\Delta\mathbf{r}_j\right\rVert^2.
$$

Unlike reflection, crossing the finite field-of-view boundary does not alter a physical displacement. The finite window still introduces trajectory censoring: tracks begin when a molecule enters the window and end when it leaves. Tracking and diffusion estimators should account for that selection effect, but there is no artificial bounce at the image edge.

## Chromosome-bound dynamic 2D simulation

Stable scenario ID:

```text
chromosome_bound_dynamic_2d
```

Composition:

```text
Local motion:  BrownianMotion2D
Local boundary: ReflectingEllipseBoundary
Scene:         DynamicChromosomeScene2D
Photophysics:  ThreeStateMarkovBlinking
Observation:   GaussianLocalizationObservation
```

This scenario generates proteins bound to one moving chromosome. Protein positions are evolved in a chromosome-local coordinate system and then transformed into the laboratory coordinate system:

$$
\mathbf{x}_i(t)
=
\mathbf{C}(t)
+
R(\theta(t))\mathbf{u}_i(t),
$$

where $\mathbf{u}_i(t)$ is protein $i$ in chromosome-local coordinates, $\mathbf{C}(t)$ is the shared chromosome center, and $\theta(t)$ is the shared chromosome orientation. The rotation matrix is

$$
R(\theta)
=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}.
$$

The local protein displacement follows

$$
\Delta\mathbf{u}_i
\sim
\mathcal{N}\!\left(\mathbf{0},2D_{\mathrm p}\Delta t\,I\right).
$$

The chromosome center follows an Ornstein--Uhlenbeck process around $\boldsymbol{\mu}$,

$$
d\mathbf{C}
=
-\lambda(\mathbf{C}-\boldsymbol{\mu})dt
+
\sqrt{2D_{\mathrm c}}\,d\mathbf{W},
$$

implemented with the exact finite-time update for $\lambda>0$:

$$
\mathbf{C}_{t+\Delta t}
=
\boldsymbol{\mu}
+
e^{-\lambda\Delta t}(\mathbf{C}_t-\boldsymbol{\mu})
+
\sqrt{\frac{D_{\mathrm c}}{\lambda}
\left(1-e^{-2\lambda\Delta t}\right)}\,\boldsymbol{\xi}.
$$

When $\lambda=0$, translation uses the Brownian limit. Orientation follows rotational diffusion:

$$
\theta_{t+\Delta t}
=
\theta_t
+
\sqrt{2D_{\mathrm r}\Delta t}\,\eta.
$$

Setting both `chromosome_diffusion_coefficient_um2_s=0.0` and `rotational_diffusion_rad2_s=0.0` freezes the shared chromosome pose, providing a static-chromosome simulation control. This remains static even if the initial center differs from the OU equilibrium center.

### Effective chromosome-local boundary

Local coordinates are confined by

$$
\frac{u_x^2}{a^2}
+
\frac{u_y^2}{b^2}
\leq1.
$$

The ellipse is only an **effective computational confinement mask**. It is not a literal claim that a chromosome has an elliptical shape, and it is not a cylinder-plus-hemispheres model. A long, narrow mask allows local motion to become strongly anisotropic across time scales, but this simulator does not calculate an effective dimension, RMS/MSD curve, or diffusion coefficient.

A finite step that crosses the ellipse is reflected about the local ellipse normal. If one step crosses the boundary more than once, segment intersection and residual-displacement reflection are repeated until the final point is inside. An exactly grazing step that starts on the boundary is a degenerate case with no inward specular segment; it is projected just inside the ellipse instead. The method is therefore a finite-step no-flux approximation, with specular reflection for ordinary crossings and a documented projection fallback for grazing contact.

### Localization model and coordinate layers

Visibility is determined from the noiseless laboratory position and photophysical state. Gaussian localization noise is then added only to visible detections:

$$
\mathbf{y}_{i,t}
=
\mathbf{x}_{i,t}
+
\boldsymbol{\epsilon}_{i,t},
\qquad
\boldsymbol{\epsilon}_{i,t}
\sim
\mathcal{N}(\mathbf{0},\sigma_{\mathrm{loc}}^2I).
$$

Noisy positions are not clipped to the field of view. Clipping would bias the localization-error distribution. The public file contains $\mathbf{y}_{i,t}$ only; chromosome-local positions, noiseless laboratory positions, chromosome center, orientation, and true identities remain private.

The default values below are **simulation settings, not universal biological constants**:

| Parameter | Default |
|---|---:|
| Protein diffusion coefficient $D_{\mathrm p}$ | 0.005 um^2/s |
| Chromosome diffusion coefficient $D_{\mathrm c}$ | 0.001 um^2/s |
| Chromosome relaxation rate $\lambda$ | 1/30 1/s |
| Rotational diffusion $D_{\mathrm r}$ | 0.001 rad^2/s |
| Ellipse semi-major axis $a$ | 2.0 um |
| Ellipse semi-minor axis $b$ | 0.35 um |
| Frame interval $\Delta t$ | 0.05 s |
| Localization standard deviation $\sigma_{\mathrm{loc}}$ | 0.03 um |
| Initial/equilibrium chromosome center | (5.0, 5.0) um |
| Initial orientation | 0 rad |

For the default slow protein diffusion, the one-frame physical displacement standard deviation per coordinate is

$$
\sqrt{2D_{\mathrm p}\Delta t}
=
\sqrt{0.0005}
\approx0.0224\;\mu\mathrm{m}
=
22.4\;\mathrm{nm}.
$$

The default 30 nm localization uncertainty is therefore comparable to, and slightly larger than, a one-frame protein displacement in one coordinate. Any later diffusion-recovery study should evaluate measurement noise explicitly. This repository currently generates the simulation and ground truth only; it does not perform that recovery or calculate RMS/MSD.

## Multi-chromosome-bound dynamic 2D simulation

Stable scenario ID:

```text
multi_chromosome_bound_dynamic_2d
```

Run the default six-chromosome preset with:

```powershell
& ".\.venv\Scripts\python.exe" main.py --scenario chromosomes
```

This is a separate scenario rather than a replacement for the one-chromosome control. If protein $i$ is permanently assigned to chromosome $k(i)$, its laboratory position is

$$
\mathbf{x}_i(t)
=
\mathbf{C}_{k(i)}(t)
+
R\!\left(\theta_{k(i)}(t)\right)\mathbf{u}_i(t).
$$

Each chromosome has its own OU center process, rotational-diffusion process, local effective ellipse, and bound-protein subset. `SimulationConfig.n_particles` remains the **total** protein count. The simulator divides those proteins deterministically and as evenly as possible among chromosomes; proteins do not switch chromosomes during a run.

The default CLI preset uses six chromosomes and 120 proteins, giving 20 proteins per chromosome. Initial centers and orientations are sampled reproducibly from `SimulationConfig.random_seed`. Each initial oriented ellipse is kept fully inside the observation window, and a modest minimum center separation prevents nearly coincident centers. Projected ellipses are still allowed to overlap, both initially and later. Chromosomes do not collide, exclude one another, deform, divide, or interact.

The multi-chromosome CLI preset uses these default physical simulation settings:

| Parameter | Default |
|---|---:|
| Chromosome count $K$ | 6 |
| Total proteins | 120 |
| Protein diffusion coefficient $D_{\mathrm p}$ | 0.1 um^2/s |
| Chromosome diffusion coefficient $D_{\mathrm c}$ | 0.001 um^2/s |
| Chromosome relaxation rate $\lambda$ | 1/30 1/s |
| Rotational diffusion $D_{\mathrm r}$ | 0.001 rad^2/s |
| Ellipse semi-axes $(a,b)$ | (2.0, 0.35) um |
| Minimum initial center separation | 1.5 um |
| Localization standard deviation | 0.03 um |

These are configurable simulation settings, not universal biological constants. The multi-chromosome protein diffusion is intentionally stronger than the `0.005 um^2/s` single-chromosome slow control. At the default frame interval, its one-frame displacement standard deviation per coordinate is

$$
\sqrt{2D_{\mathrm p}\Delta t}
=
\sqrt{2(0.1)(0.05)}
=0.1\;\mu\mathrm{m}
=100\;\mathrm{nm}.
$$

$D_{\mathrm p}$ controls protein motion **relative to its assigned chromosome**. In contrast, $D_{\mathrm c}$ and $D_{\mathrm r}$ translate and rotate the entire chromosome-local coordinate system. As in the one-chromosome model, every ellipse is only an effective 2D no-flux confinement mask and is not a literal chromosome shape. The reflecting boundary keeps proteins inside it. Values much larger than `0.1 um^2/s` can cause frequent finite-step reflections across the narrow `0.35 um` semi-minor axis; such settings should be checked with a smaller simulation time step.

### Multi-chromosome preview

The private ground-truth panel draws every effective ellipse, its center, major-axis direction, and the preceding two seconds of its **actual sampled center trajectory**. It also draws the preceding one second of true motion for up to two deterministically selected proteins per chromosome. These overlays make chromosome and local protein motion easier to see without multiplying or otherwise exaggerating displacement. The public observation panel remains an unlabeled set of noisy visible detections and does not reveal chromosome membership, pose, geometry, or any trail.

The multi-chromosome private bundle uses ground-truth format `1.3` and includes permanent particle-to-chromosome assignments, local positions, all chromosome centers and orientations, repeated ellipse semi-axes, and per-chromosome initialization bounds. Public observations remain format `1.0` with the same key set as every other scenario.

This scenario still performs simulation only. The center and selected-protein trails are visualization aids based on private truth; they are not inferred tracks and are not an RMS/MSD or diffusion analysis.

## Shared photophysics

All five scenarios use the same three-state model:

```text
OFF  --k_on--> ON
ON   --k_off--> OFF
ON   --k_bleach--> BLEACHED
BLEACHED --no transition--> BLEACHED
```

For a transition with constant rate $k$, the waiting time is exponentially distributed:

$$
T\sim\mathrm{Exp}(k),
\qquad
P(T>t)=e^{-kt}.
$$

The event probability during one frame is

$$
p(\Delta t)=1-e^{-k\Delta t}.
$$

For an OFF molecule,

$$
T_{\mathrm{off}}\sim\mathrm{Exp}(k_{\mathrm{on}}),
\qquad
\mathbb{E}[T_{\mathrm{off}}]=\frac{1}{k_{\mathrm{on}}}.
$$

For an ON molecule, temporary darkening and irreversible bleaching are competing events:

$$
T_{\mathrm{on}}
\sim
\mathrm{Exp}(k_{\mathrm{off}}+k_{\mathrm{bleach}}),
$$

$$
\mathbb{E}[T_{\mathrm{on}}]
=
\frac{1}{k_{\mathrm{off}}+k_{\mathrm{bleach}}}.
$$

When an ON period ends,

$$
P(\mathrm{ON}\rightarrow\mathrm{OFF})
=
\frac{k_{\mathrm{off}}}
{k_{\mathrm{off}}+k_{\mathrm{bleach}}},
$$

and

$$
P(\mathrm{ON}\rightarrow\mathrm{BLEACHED})
=
\frac{k_{\mathrm{bleach}}}
{k_{\mathrm{off}}+k_{\mathrm{bleach}}}.
$$

The default parameters are

| Parameter | Value |
|---|---:|
| Initial ON fraction | 0.2 |
| $k_{\mathrm{on}}$ | 1.0 1/s |
| $k_{\mathrm{off}}$ | 2.0 1/s |
| $k_{\mathrm{bleach}}$ | 0.08 1/s |

BLEACHED is an absorbing state:

$$
P(S_{t+\Delta t}=\mathrm{BLEACHED}\mid S_t=\mathrm{BLEACHED})=1.
$$

## Observation models

The three benchmark scenarios report an exact localization only when a molecule is ON and inside the configured field of view.

For a molecule inside the field of view,

$$
\mathbf{y}_{i,t}=\mathbf{x}_{i,t}
\quad\text{when}\quad
S_{i,t}=\mathrm{ON}.
$$

Otherwise, no detection is emitted:

$$
\mathbf{y}_{i,t}=\varnothing.
$$

The chromosome scenarios instead use the visible-only Gaussian localization model described above. In every scenario the public output is a localization table, not a camera image. It does not yet contain a point-spread function, pixels, photon noise, background fluorescence, or motion blur.

## Output trust boundary

Each scenario separates tracking input from evaluation-only data.

### Public tracking input: `observations.npz`

A tracking method may read this file. It contains only visible detections:

| Field | Shape | Meaning |
|---|---|---|
| `scenario_id` | scalar string | Stable scenario identifier |
| `observation_model` | scalar string | Observation component identifier |
| `n_frames` | scalar integer | Total number of frames |
| `frame_interval_s` | scalar float | Time between frames |
| `frame_times_s` | `(n_frames,)` | Complete frame time axis |
| `bounds_um` | `(observation_dimensions, 2)` | Public observation-coordinate bounds |
| `detection_ids` | `(n_detections,)` | Unique arbitrary detection IDs |
| `frame_indices` | `(n_detections,)` | Frame containing each detection |
| `times_s` | `(n_detections,)` | Detection times |
| `positions_um` | `(n_detections, observation_dimensions)` | Observed bright-spot coordinates |

It does not contain true particle IDs, hidden states, invisible positions, complete trajectories, motion parameters, axial coordinates, chromosome-local coordinates, chromosome assignments, chromosome centers or orientations, or a persistent particle-slot axis. Detection order is shuffled independently within every frame. For both chromosome scenarios, the published `detection_order_seed` is combined with private noiseless truth before constructing the permutation, so the public seed alone cannot replay the hidden particle ordering. The projected 3D scenario still exposes only two columns: x and y.

### Private evaluation data: `ground_truth.npz`

Do not give this file to a tracking method. It contains:

- Complete true trajectories and stable particle IDs, including XYZ coordinates for the projected 3D scenario.
- For the single-chromosome scenario (private format `1.2`): chromosome-local trajectories, noisy dense observations, visibility, center and orientation per frame, and effective ellipse semi-axes.
- For the multi-chromosome scenario (private format `1.3`): all of the above plus permanent particle-to-chromosome assignments, every chromosome pose, repeated ellipse axes, and per-chromosome initialization bounds.
- Internal photophysics states.
- Emitting, active, and in-observation-region masks.
- Complete scenario and configuration metadata.
- The hidden `detection_id -> true_particle_id` mapping.

A future evaluator can compare a tracker's predicted identities with this hidden mapping.

### Visualization: `simulation_preview.gif`

The GIF is for human inspection only. The left panel shows private ground truth inside the complete observation region; 3D truth is projected onto XY for display. Chromosome previews also show true effective ellipse outlines, centers, major-axis directions, and two-second center trails. The right panel shows only the XY detections available to tracking and never displays chromosome membership or pose.

## Consuming observations from another Python program

Use `load_observations` and open only the public file:

```python
from srm_sim import load_observations

observations = load_observations(
    "outputs/free_diffusion_benchmark_3d_to_2d_axial_slab_markov/observations.npz"
)

for frame_index in range(observations.n_frames):
    detection_ids, bright_spots_um = observations.for_frame(frame_index)

    # bright_spots_um has shape (n_visible_in_frame, 2).
    # Pass only these IDs and coordinates to the tracking method.
    print(frame_index, detection_ids.shape, bright_spots_um.shape)
```

A tracker should return an assignment such as

```text
detection_id, predicted_track_id
0,            12
1,            31
2,            7
3,            12
```

The tracker creates `predicted_track_id`. It must never receive the true `particle_id`.

## Framework layout

- `src/srm_sim/models/motion.py`: motion components.
- `src/srm_sim/models/boundary.py`: reflecting and unbounded physical domains.
- `src/srm_sim/models/photophysics.py`: blinking and bleaching.
- `src/srm_sim/models/observation.py`: finite observation windows and localization output.
- `src/srm_sim/models/scene.py`: pure single- and multi-chromosome rigid-pose dynamics for coupled scenes.
- `src/srm_sim/scenarios/`: scientifically meaningful component combinations.
- `src/srm_sim/simulator.py`: generic scenario orchestration.
- `src/srm_sim/export.py`: tracking-safe public export and private truth export.
- `src/srm_sim/visualization.py`: human-facing animation.

## Adding future models

Prefer adding reusable components and composing them into scenarios instead of copying the simulator. Examples include:

```text
BrownianMotion2D
+ UnboundedPlane2D
+ ThreeStateMarkovBlinking
+ GaussianLocalizationObservation
```

```text
ConfinedDiffusion2D
+ CircularBoundary
+ PowerLawBlinking
+ PhotonCameraObservation
```

## Not included yet

- Tracking algorithms and evaluation metrics
- Point-spread function and camera pixels
- Photon and camera noise
- Background fluorescence
- Localization uncertainty for scenarios other than the chromosome-bound simulation
- Motion blur
- Missed detections and false positives
- Multiple diffusion populations
- Directed motion, anomalous diffusion, or direct 3D localization
