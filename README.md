# SRM Tracking Simulation Framework

A modular framework for constructing controlled super-resolution microscopy (SRM) particle-tracking simulations.

The project is not a single physical model. Each simulation **scenario** combines four independent components:

```text
motion model
    + boundary model
    + photophysics model
    + observation model
    = simulation scenario
```

Three scenarios are currently included:

1. A default 3D free-diffusion benchmark observed through a finite axial slab and projected to 2D.
2. A reflecting 2D baseline for confinement and tracking experiments.
3. A free-diffusion 2D benchmark for testing diffusion-coefficient recovery without a reflecting wall.

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

Run either retained 2D scenario explicitly:

```powershell
& ".\.venv\Scripts\python.exe" main.py --scenario reflecting
& ".\.venv\Scripts\python.exe" main.py --scenario free
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

| Property | Default projected 3D benchmark | Reflecting 2D baseline | Free-diffusion 2D benchmark |
|---|---|---|---|
| Scenario ID | `free_diffusion_benchmark_3d_to_2d_axial_slab_markov` | `reflecting_baseline_2d_brownian_markov` | `free_diffusion_benchmark_2d_brownian_markov` |
| Physical motion | 3D Brownian motion | 2D Brownian motion | 2D Brownian motion |
| Physical boundary | None; unbounded space | Reflecting 10 x 10 um square | None; unbounded plane |
| Initial reservoir | 30 x 30 x 22 um | 10 x 10 um | 30 x 30 um |
| Observation region | Central 10 x 10 x 2 um volume | 10 x 10 um window | Central 10 x 10 um window |
| Public coordinates | 2D XY projection | 2D XY | 2D XY |
| Number of particles | 5,000 | 100 | 900 |
| Diffusion coefficient | 0.5 um^2/s | 0.5 um^2/s | 0.5 um^2/s |
| Intended use | Axial entry/exit and projected tracking | Confinement and wall-sensitive tracking | Free-diffusion coefficient recovery |

All three scenarios use 200 frames, a frame interval of 0.05 s, identical three-state Markov photophysics, ideal localization, and reproducible random seeds.

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

## Shared photophysics

All three scenarios use the same three-state model:

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

## Ideal observation model

The current observation component reports an exact localization only when a molecule is ON and inside the configured field of view.

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

The public output is therefore a localization table, not a camera image. It does not yet contain a point-spread function, pixels, photon noise, background fluorescence, localization uncertainty, or motion blur.

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

It does not contain true particle IDs, hidden states, invisible positions, complete trajectories, motion parameters, axial coordinates, or a persistent particle-slot axis. Detection order is shuffled independently within every frame. The projected 3D scenario therefore still exposes only two columns: x and y.

### Private evaluation data: `ground_truth.npz`

Do not give this file to a tracking method. It contains:

- Complete true trajectories and stable particle IDs, including XYZ coordinates for the projected 3D scenario.
- Internal photophysics states.
- Emitting, active, and in-observation-region masks.
- Complete scenario and configuration metadata.
- The hidden `detection_id -> true_particle_id` mapping.

A future evaluator can compare a tracker's predicted identities with this hidden mapping.

### Visualization: `simulation_preview.gif`

The GIF is for human inspection only. The left panel shows ground truth inside the complete observation region; 3D truth is projected onto XY for display. The right panel shows only the XY detections available to tracking.

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
- Localization uncertainty
- Motion blur
- Missed detections and false positives
- Multiple diffusion populations
- Directed motion, anomalous diffusion, or direct 3D localization
