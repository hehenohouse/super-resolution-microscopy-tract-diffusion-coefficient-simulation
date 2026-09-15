# SRM Tracking Simulation Framework

A modular simulation framework for constructing controlled super-resolution microscopy (SRM) particle-tracking experiments.

The project is not a single physical model. Each simulation **scenario** is composed from four independent model types:

```text
motion model
    + boundary model
    + photophysics model
    + observation model
    = simulation scenario
```

The repository currently contains one minimal baseline scenario. It does not yet contain a tracking algorithm or tracking evaluation metrics.

## Quick start

The project requires Python 3.11 or newer. A local virtual environment already exists at `.venv`.

Install the project in Windows PowerShell:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -e .
```

Run the baseline simulation:

```powershell
& ".\.venv\Scripts\python.exe" main.py
```

The command generates three files:

```text
outputs/observations.npz
outputs/ground_truth.npz
outputs/simulation_preview.gif
```

## Output trust boundary

The output is deliberately separated into public tracking input and private evaluation data.

### Tracking input: `observations.npz`

A tracking method may read this file. It contains only visible localization detections:

```text
detection_id
frame_index
time_s
observed position
```

It does not contain true particle IDs, hidden states, dark-molecule positions, complete trajectories, or a persistent particle-slot axis. Detection order is shuffled independently within each frame, so array order cannot be used as an identity shortcut.

### Private evaluation data: `ground_truth.npz`

Do not give this file to a tracking method. It contains the complete simulated trajectories, photophysical states, and the hidden mapping from each `detection_id` to its true `particle_id`.

A future evaluator can use this file after tracking has finished to compare predicted tracks with the hidden identities.

### Visualization: `simulation_preview.gif`

The GIF is for human inspection only. It is not intended as machine input. The left panel shows ground truth, while the right panel shows the observations available to a tracker.

## Consuming observations from another Python program

The safest interface is `load_observations`. It opens only the public observation file.

```python
from srm_sim import load_observations

observations = load_observations("outputs/observations.npz")

for frame_index in range(observations.n_frames):
    detection_ids, bright_spots_um = observations.for_frame(frame_index)

    # bright_spots_um has shape (n_visible_in_frame, dimensions).
    # Pass detection_ids and bright_spots_um to a tracking method.
    print(frame_index, detection_ids.shape, bright_spots_um.shape)
```

For the current two-dimensional baseline, each row of `bright_spots_um` is:

```text
[x_um, y_um]
```

A tracker can return an assignment such as:

```text
detection_id, predicted_track_id
0,            12
1,            31
2,            7
3,            12
```

The tracker creates `predicted_track_id`. It must not have access to the true `particle_id`.

### Loading without importing this package

A separate program can also use NumPy directly:

```python
import numpy as np

with np.load("outputs/observations.npz", allow_pickle=False) as data:
    detection_ids = data["detection_ids"]
    frame_indices = data["frame_indices"]
    times_s = data["times_s"]
    positions_um = data["positions_um"]
    n_frames = int(data["n_frames"])

for frame_index in range(n_frames):
    in_frame = frame_indices == frame_index
    frame_detection_ids = detection_ids[in_frame]
    frame_bright_spots_um = positions_um[in_frame]
```

## Observation file schema

`observations.npz` contains the following tracking-safe fields:

| Field | Shape | Meaning |
|---|---|---|
| `format_version` | scalar string | Observation-file format version |
| `scenario_id` | scalar string | Stable scenario identifier |
| `observation_model` | scalar string | Observation component identifier |
| `position_unit` | scalar string | `um` |
| `time_unit` | scalar string | `s` |
| `n_frames` | scalar integer | Total number of frames, including empty frames |
| `frame_interval_s` | scalar float | Time between frames |
| `frame_times_s` | `(n_frames,)` | Complete frame time axis |
| `bounds_um` | `(dimensions, 2)` | Lower and upper spatial bounds |
| `detection_order_seed` | scalar integer | Seed used only to randomize within-frame order |
| `detection_ids` | `(n_detections,)` | Unique arbitrary detection identifiers |
| `frame_indices` | `(n_detections,)` | Frame containing each detection |
| `times_s` | `(n_detections,)` | Time of each detection |
| `positions_um` | `(n_detections, dimensions)` | Observed bright-spot coordinates |

The table is flat rather than `(frame, particle, xy)`. Consequently, no column remains associated with the same true molecule across frames.

## Private ground-truth schema

`ground_truth.npz` contains evaluation-only data:

| Field | Meaning |
|---|---|
| `scenario_json` | All component names and physical parameters |
| `config_json` | Particle count, frame count, time step, and random seed |
| `particle_ids` | Stable true molecule identities |
| `positions_um` | Complete true trajectories, including invisible molecules |
| `states` | Internal photophysics states |
| `emitting` | True emitting state |
| `active` | Whether a molecule has not permanently deactivated |
| `detection_ids` | The same arbitrary detection IDs used publicly |
| `detection_frame_indices` | Frame of each detection |
| `detection_particle_ids` | Hidden true identity for each detection |
| `detection_true_positions_um` | Hidden true position associated with each detection |

This separation prevents accidental identity leakage while preserving everything required for future evaluation.

## Framework components

### Motion model

Controls true particle motion, such as Brownian, directed, confined, or anomalous diffusion.

Current component:

```text
homogeneous_brownian_motion_2d
```

### Boundary model

Controls what happens when a particle reaches the simulated domain boundary, such as reflecting, periodic, or absorbing behavior.

Current component:

```text
reflecting_square_boundary_2d
```

### Photophysics model

Controls emitting, non-emitting, and permanent deactivation states.

Current component:

```text
three_state_markov_blinking_with_bleaching
```

### Observation model

Transforms hidden ground truth into the detections available to a tracking method. Future observation models can add localization error, missed detections, PSF rendering, photon noise, or camera noise without changing the underlying motion.

Current component:

```text
ideal_emitter_localization
```

## Baseline A

Stable scenario ID:

```text
baseline_ideal_2d_brownian_markov
```

Full name:

**Baseline A - Ideal 2D Reflecting Brownian Motion with Three-State Markov Photophysics**

Composition:

```text
Motion:       BrownianMotion2D
Boundary:     ReflectingSquareBoundary
Photophysics: ThreeStateMarkovBlinking
Observation:  IdealOnStateObservation
```

### Default conditions

Frequently adjusted values are kept together in `main.py`.

| Parameter | Default | Unit or meaning |
|---|---:|---|
| `n_particles` | 100 | Molecules |
| `n_frames` | 200 | Frames |
| `frame_interval_s` | 0.05 | Seconds per frame (20 fps) |
| `field_size_um` | 10.0 | Square field side length in um |
| `diffusion_coefficient_um2_s` | 0.5 | um^2/s |
| `initial_on_fraction` | 0.2 | Initially emitting fraction |
| `k_on_s` | 1.0 | `OFF -> ON`, 1/s |
| `k_off_s` | 2.0 | `ON -> OFF`, 1/s |
| `k_bleach_s` | 0.08 | `ON -> BLEACHED`, 1/s |
| `random_seed` | 7 | Reproducible simulation seed |
| `detection_order_seed` | 17 | Reproducible within-frame shuffle seed |

These values are transparent baseline conditions, not parameters fitted to a particular fluorophore.

### Brownian-motion assumption

For a homogeneous two-dimensional Brownian process with diffusion coefficient $D$ and frame interval $\Delta t$, each coordinate increment is sampled independently:

$$
\Delta x, \Delta y \sim \mathcal{N}\!\left(0, 2D\Delta t\right).
$$

Equivalently, the standard deviation of one coordinate step is

$$
\sigma_{\mathrm{step}} = \sqrt{2D\Delta t}.
$$

The expected squared two-dimensional displacement over one frame is

$$
\mathbb{E}\!\left[\lVert\Delta\mathbf{r}\rVert^2\right]
= 4D\Delta t.
$$

For the current defaults, $D=0.5\;\mu\mathrm{m}^2/\mathrm{s}$ and $\Delta t=0.05\;\mathrm{s}$, giving

$$
\sigma_{\mathrm{step}} \approx 0.224\;\mu\mathrm{m}
$$

per coordinate and a two-dimensional RMS displacement of approximately $0.316\;\mu\mathrm{m}$ per frame.

All particles move independently and share one diffusion coefficient. The boundary is reflecting. For a square domain $[0,L]$, each coordinate is mapped back into the field using

$$
x_{\mathrm{reflected}}
= L - \left|\left(x \bmod 2L\right)-L\right|.
$$

OFF and BLEACHED molecules continue to have simulated true positions, but the observation model cannot see them.

### Photophysics assumption

```text
OFF  --k_on--> ON
ON   --k_off--> OFF
ON   --k_bleach--> BLEACHED
BLEACHED --no transition--> BLEACHED
```

For a transition with constant rate $k$, the waiting time is exponentially distributed:

$$
T \sim \mathrm{Exp}(k),
\qquad
P(T>t)=e^{-kt}.
$$

The corresponding transition probability during one frame is

$$
p(\Delta t)=1-e^{-k\Delta t}.
$$

For an OFF molecule,

$$
T_{\mathrm{off}} \sim \mathrm{Exp}(k_{\mathrm{on}}),
\qquad
\mathbb{E}[T_{\mathrm{off}}]=\frac{1}{k_{\mathrm{on}}}.
$$

Temporary darkening and permanent bleaching are competing events while a molecule is ON. Therefore,

$$
T_{\mathrm{on}}
\sim \mathrm{Exp}(k_{\mathrm{off}}+k_{\mathrm{bleach}}),
$$

with mean duration

$$
\mathbb{E}[T_{\mathrm{on}}]
= \frac{1}{k_{\mathrm{off}}+k_{\mathrm{bleach}}}.
$$

When an ON period ends, the event probabilities are

$$
P(\mathrm{ON}\rightarrow\mathrm{OFF})
= \frac{k_{\mathrm{off}}}
{k_{\mathrm{off}}+k_{\mathrm{bleach}}},
$$

and

$$
P(\mathrm{ON}\rightarrow\mathrm{BLEACHED})
= \frac{k_{\mathrm{bleach}}}
{k_{\mathrm{off}}+k_{\mathrm{bleach}}}.
$$

BLEACHED is an absorbing state, so

$$
P(S_{t+\Delta t}=\mathrm{BLEACHED}\mid S_t=\mathrm{BLEACHED})=1.
$$

### Ideal-observation assumption

The ideal observation rule is

$$
\mathbf{y}_{i,t}=
\begin{cases}
\mathbf{x}_{i,t}, & S_{i,t}=\mathrm{ON},\
\varnothing, & S_{i,t}\neq\mathrm{ON},
\end{cases}
$$

where $\mathbf{x}_{i,t}$ is the true position and $\mathbf{y}_{i,t}$ is the reported detection. The current public output is therefore a localization table, not a realistic camera image. It does not contain a PSF, pixels, photon noise, background fluorescence, localization error, or motion blur.

## Localization tracking versus image tracking

The current `observations.npz` is appropriate for tracking methods that accept per-frame localization coordinates, including nearest-neighbor linking, assignment methods, Kalman filters, probabilistic association, gap closing, and graph-based tracking.

It is not yet suitable for methods that begin with raw microscopy images. Those methods require a future image-formation pipeline:

```text
emitter coordinates
    -> PSF rendering
    -> camera pixels
    -> photon and background noise
    -> image stack
```

## Adding future models

Prefer adding one reusable component and composing it with existing components rather than copying the entire simulator. For example:

```text
BrownianMotion2D
+ ReflectingSquareBoundary
+ ThreeStateMarkovBlinking
+ GaussianLocalizationObservation
```

or:

```text
ConfinedDiffusion2D
+ CircularBoundary
+ PowerLawBlinking
+ PhotonCameraObservation
```

Register a new scenario in `src/srm_sim/scenarios/` when a scientifically meaningful combination needs to be run repeatedly.

## Not included yet

- Tracking algorithms and evaluation metrics
- Microscope PSF and camera pixels
- Photon and camera noise
- Background fluorescence
- Localization uncertainty
- Motion blur
- Missed detections and false positives
- Multiple diffusion populations
- Confinement, directed motion, anomalous diffusion, or 3D motion
