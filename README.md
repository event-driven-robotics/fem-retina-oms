# Fixational retina experiments

Minimal modules for loading DOVES, filtering saccades, generating moving scenes,
simulating events, running the OMS retina, and measuring object selectivity.
Run each stage below in a notebook or Python session. There is no pipeline runner
or experiment configuration format.

## 1. Download DOVES and install the code

Obtain the dataset from the official [DOVES page at UT Austin](https://live.ece.utexas.edu/research/doves/).
Use its **Download entire database** link. The site says the archive is password
protected and gives instructions for requesting the password from Ian van der
Linde. Extract it outside this repository. The archive contains `Images/`,
`Fixations/`, `RawData/`, and `Code/`; this code uses images and raw gaze samples
from the MATLAB files in `Fixations/`.

Follow the dataset's citation instructions, including the DOVES paper and the
van Hateren image acknowledgement:

> van der Linde, I., Rajashekar, U., Bovik, A. C., and Cormack, L. K. (2009).
> DOVES: A database of visual eye movements. Spatial Vision, 22(2), 161–177.

Install this repository with Python 3.10 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The Python blocks in sections 2–7 share variables; execute them in order in the
same session. Change the input paths and experimental settings to your own.
Examples are starting settings, not a declaration of the paper's final protocol.

## 2. Load an image and a subject's recording

Supply the actual filenames from the extracted archive explicitly. This avoids
assuming whether your image or recording filenames include an `.iml` suffix.
The image loader reads 768×1024 big-endian uint16 pixels and converts them to
8-bit grayscale, preserving the source code's normalization. The recording loader
reads x/y samples from `eye_data[0, subject]`. `subject` is a **zero-based array
index**, not necessarily the participant identifier in your experiment notes.

```python
from pathlib import Path
import numpy as np
from PIL import Image
from eye_movements.doves import load_doves, render_background
from common_utils.io import save_json

image_path = Path('/path/to/Doves/Images/imk01366.iml')
recording_path = Path('/path/to/Doves/Fixations/imk01366.iml.mat')
object_path = Path('/path/to/objects/object.png')  # RGBA with a nonempty alpha mask
subject = 0
condition = 'doves'  # 'doves', 'circular', or 'wobbly'
with_object = True  # False for a background-only stimulus
out = Path('results/example_doves_subject0')  # use a new directory per run
out.mkdir(parents=True, exist_ok=False)

image, eye = load_doves(image_path, recording_path, subject, contrast_boost=False)
Image.fromarray(image).save(out / 'background.png')
fps = 200
dt = 1 / fps
crop_shape = (720, 960)
```

Non-finite gaze samples raise an error: apply the study's missing-data policy
explicitly rather than silently interpolating them. Check the DOVES documentation
for tracker-loss artifacts when selecting usable recordings.

## 3. Filter saccades and choose the camera motion

For DOVES motion, the retained code detects saccades and stitches the increments
of the remaining segments. It does not resample the recording. Save the detector
mask and retained trace so that filtering can be inspected later.

```python
from eye_movements.utils import (
    detect_microsaccades_engbert, stitch_by_increments,
    circular, sine_velocity_trajectory,
)

n_frames = 100  # use the duration from your experiment
if condition == 'doves':
    detected = detect_microsaccades_engbert(
        eye[0], eye[1], fs=fps, lam=6, min_dur_ms=6, merge_gap_ms=10,
    )
    x, y, times = stitch_by_increments(
        eye[0], eye[1], detected['jitter_mask'], fs=fps,
        carry_over=True, detrend_segment=False,
    )
    np.save(out / 'saccade_mask.npy', detected['saccade_mask'])
    save_json(out / 'saccades.json', detected['saccades'])
    np.save(out / 'retained_trace.npy', np.column_stack((x, y)))
    if len(x) < n_frames:
        raise ValueError('Choose fewer frames: insufficient retained DOVES samples')
    trace = np.column_stack((x[:n_frames], y[:n_frames]))
elif condition == 'circular':
    trace = np.column_stack(circular(n_frames, radius=5, pixels_per_frame=0.5))
elif condition == 'wobbly':
    trace = np.column_stack(sine_velocity_trajectory(
        n_frames, v_base=0.5, v_amp=0.3, velocity_cycles=1,
        orientation_cycles=1, theta0=np.pi / 2,
    ))
else:
    raise ValueError('Unknown motion condition')
np.save(out / 'eye_trace.npy', trace)
```

Circular speed is arc length in pixels/frame. Wobbly speed parameters are also
pixels/frame, and cycle counts refer to the requested sequence duration. Use the
same intended duration across conditions. No forward/reverse trace extension is
applied automatically.

## 4. Render frames, optionally with moving objects

The object generator applies the same camera translation and crop to frames,
individual masks, and union masks. Object area fractions refer to the output
image area. It uses seeded wandering trajectories with collision avoidance;
confirm that this is the object-motion protocol intended for your experiment.
Add more `MovingObjectSpec` entries for multiple objects.

```python
from moving_objects.add_multi_moving_objects import (
    MovingObjectSpec, generate_multi_object_sequence,
)

if with_object:
    objects = [MovingObjectSpec(
        object_rgba_path=object_path, area_fraction=0.01, speed_px_s=250,
    )]
    generate_multi_object_sequence(
        base_scene_path=[out / 'background.png'], objects=objects, out_dir=out,
        N=len(trace), fps=fps, motion_trace_path=out / 'eye_trace.npy',
        output_crop_size=crop_shape, motion_seed=20260917,
        clear_output=False, run_event_camera=False,
    )
else:
    render_background(image, trace, out / 'frames', crop_shape=crop_shape)
```

Outputs include `frames/` and, for object stimuli, `masks_union/`, per-object masks,
and object metadata. Keep the source images and the settings used for each run.

## 5. Generate events and bin them

```python
from events_camera_simulator.streaming import generate_events_streaming, bin_events_to_disk

generate_events_streaming(
    out, dt=dt, seed=20261012,
    tolerance=0.2, refrac_period=0.0001, sensor_sensitivity=1,
)
binned = bin_events_to_disk(
    out / 'events_file/binaryevents.log', out / 'binned_events.npy',
    frames=len(trace), shape=crop_shape, dt=dt,
)
```

At least three frames are required. The simulator retains the original PNG
quantization, first-transition skip, ±2 ms timestamp jitter, and clock starting at
the first event. `events_file/metadata.json` records the scene-time offset. Bins
use that event clock; masks retain their frame indices. Final empty bins are
zero-padded to the requested sequence length. These are legacy timing conventions
to check against reference results, not a correction of alignment. An event-free
sequence is supported.

## 6. Run the retina

```python
from retina_model.oms_retina import normalize_bipolar_responces, run_oms_retina_

ksize = 39
normalized = normalize_bipolar_responces(
    binned, ksize=ksize, dt=dt, tau_bp=0.012, tau_am=0.140, tau_var=0.060,
    eps=0.001, return_only_normalized=True, output_path=out / 'normalized.npy',
)
spikes = run_oms_retina_(
    normalized, dt=dt, tau_syn=0.005, tau_mem=0.005, threshold=0.3,
    return_spikes_only=True, output_path=out / 'spikes.npy',
)
```

The saved spikes are the raw model response. Event bins and retina arrays are
memory-mapped. To resume work in another session, load a saved array with
`np.load(path, mmap_mode='r')` and restore the relevant settings.

## 7. Measure object selectivity

`p` is the fraction of response counts inside the chosen object region, not a
statistical p-value. Keep counts, areas, and chance fractions alongside it.
Selectivity is undefined for a background-only stimulus; its output is `null`.

```python
from common_utils.utils import compute_object_selectivity

result = dict(scene=image_path.name, subject=subject, condition=condition,
              frames=len(trace), oms_model=None, raw_events=None)
if with_object:
    mask_paths = sorted((out / 'masks_union').glob('*.png'))
    masks = np.stack([np.asarray(Image.open(p)) for p in mask_paths])
    if masks.shape != spikes.shape:
        raise ValueError('Mask and spike dimensions must match')
    for label, response in [('oms_model', spikes), ('raw_events', binned.sum(axis=1))]:
        metrics = compute_object_selectivity(
            response, masks, ksize=ksize, min_visibility=0.5,
            exclude_edge_touch=False, vicinity_method='box', return_more=True,
        )
        result[label] = metrics[1] if isinstance(metrics, tuple) else metrics
save_json(out / 'result.json', result)
```

`vicinity_method='box'` expands masks by the normalization kernel size;
`'exact'` scores only the mask pixels. Set visibility and edge filtering according
to the experiment. Undefined numerical values are saved as JSON `null`.
Mask scoring uses memory proportional to sequence size: check a short sequence
before processing full-resolution runs.

## 8. Compare results across experiments

Load each run's `result.json` with `json.loads(path.read_text())` to build your
results table. Retain scene, subject, condition, speed, and object-size information
when combining runs. Define paired comparisons and independent sampling units
from the actual experimental design; do not treat frames as independent samples.

For an optional paired sign-flip comparison, supply one finite difference in
`p` per independent unit. If a unit contains multiple matched scenes, average its
paired scene differences first:

```python
from common_utils.statistics import paired_sign_flip, holm

# differences = [p_doves - p_circular for each independent unit]
# comparison = paired_sign_flip(differences, seed=42)
# adjusted_p_values = holm([comparison_1['p_value'], comparison_2['p_value']])
```

The helper reports the mean difference, a bootstrap interval over units, and a
two-sided sign-flip p-value (exact up to 16 units, otherwise Monte Carlo). It
requires at least two independent units and sign exchangeability under the null.
`holm` adjusts a specified family of comparisons. These helpers are optional;
the paper's statistical design must still be checked against the original analysis.

## Tests and provenance

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic DOVES-format files and require no dataset download. See
[VALIDATION.md](VALIDATION.md) for what has been checked and
[PROVENANCE.json](PROVENANCE.json) for source commits. The paper's subject–scene
combinations and settings still need to be verified with the actual data. Choose
the code license and final citation with the project owners before publication.
