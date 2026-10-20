# Fixational retina experiments

The experiment code is in separate files in [scripts/](scripts/). Run each step
individually; each reads the previous step's saved files. The underlying modules
contain the stimulus, event-camera, retina, and analysis implementations.

## Download DOVES

Get the dataset from the official [DOVES website](https://live.ece.utexas.edu/research/doves/).
Follow its download and archive-password instructions, and extract it outside this
repository. We use the images in `Images/` and the raw gaze recordings stored in
MATLAB files in `Fixations/`. Keep your moving-object images separately as RGBA
PNGs with meaningful alpha masks.

Follow the site's citation instructions for the DOVES paper and van Hateren's
image database. DOVES and object images are not bundled here.

## Install

From this repository, create and activate a Python environment, then install the
local package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Python 3.10 or later is required. Run the following scripts from the repository
root using that environment.

## Run the steps

Start by editing the paths at the top of each script. Set `OUT` to the same
experiment directory in steps 1–6. Use a new directory for every subject, scene,
and condition. Parameters are visible in each file's `run` function signature;
change those defaults to your experiment's values. No shared notebook state or
experiment manifest is required.

| Step | File and command | What to edit and what it produces |
| --- | --- | --- |
| 1. Load DOVES | [load_doves.py](scripts/load_doves.py) — `python scripts/load_doves.py` | Set image and recording filenames, `SUBJECT`, and `OUT`. Saves the background image, raw gaze samples, and source information. |
| 2. Generate camera motion | [generate_motion.py](scripts/generate_motion.py) — `python scripts/generate_motion.py` | Set `CONDITION`, `N_FRAMES`, and motion/filter parameters. Saves `eye_trace.npy`; DOVES motion also saves the saccade mask and retained trace. |
| 3. Generate stimuli | [generate_stimuli.py](scripts/generate_stimuli.py) — `python scripts/generate_stimuli.py` | Edit `OBJECTS`, crop size, and random seed. Saves frames and object masks. Use an empty object list for background-only stimuli. |
| 4. Simulate events | [generate_events.py](scripts/generate_events.py) — `python scripts/generate_events.py` | Set simulator parameters and seed. Saves the binary event log and `binned_events.npy`. |
| 5. Run the retina | [run_retina.py](scripts/run_retina.py) — `python scripts/run_retina.py` | Set normalization and retina parameters. Saves normalized responses and raw `spikes.npy`. |
| 6. Characterize | [characterize.py](scripts/characterize.py) — `python scripts/characterize.py` | Choose exact masks or expanded vicinity and visibility filtering. Saves OMS and raw-event metrics in `result.json`. |
| 7. Analyse experiments | [analyze.py](scripts/analyze.py) — `python scripts/analyze.py` | Set `RESULTS_ROOT` to the parent of your experiment directories. Creates `summary.csv`. Optionally define paired results and independent unit labels in `PAIRS`. |

Steps save their settings alongside their outputs for inspection. When changing
an upstream step, rerun all dependent steps; there is no automatic stale-output
checking. Loading and frame generation refuse to replace existing experiment or
frame directories. Later stages replace their own outputs when rerun.

## Choose the experimental settings

**DOVES inputs.** Use the actual image and recording filenames in your download.
The image reader expects 768×1024 big-endian uint16 data. `SUBJECT` is a zero-based
index into MATLAB `eye_data`, not necessarily your participant identifier.
Non-finite gaze samples raise an error; apply the study's missing-data policy
explicitly and check recordings for the tracker-loss artifacts described by DOVES.

**Camera motion.** Choose `doves`, `circular`, or `wobbly` in step 2. DOVES motion
removes detected saccades and stitches the increments of retained segments at
200 Hz. Requests longer than the retained trace fail instead of extending it.
Circular and wobbly speeds are in pixels/frame; wobbly cycle counts refer to the
requested sequence duration. Use comparable durations across conditions.

**Moving objects.** Each entry in step 3 specifies an RGBA image, area fraction,
and speed in pixels/second. Add entries for multiple objects. The generator uses
seeded wandering motion with collision avoidance and applies the same camera
translation and crop to frames and masks. Confirm that this matches your intended
object-motion protocol.

**Event timing.** The simulator preserves the original PNG quantization,
first-transition skip, ±2 ms jitter, and clock starting at the first event.
Event metadata records the scene-time offset. Masks retain their original frame
indices and final empty event bins are padded to the requested duration. Verify
these legacy timing conventions against reference results.

**Characterization and statistics.** `p` is the fraction of response counts in
the object region, not a statistical p-value. Counts, areas, and chance fractions
are saved alongside it. Background-only selectivity is undefined. The optional
paired analysis averages differences within each explicitly labelled independent
unit before applying a sign-flip test and bootstrap interval. It requires at least
two independent units and sign exchangeability under the null. Choose the analysis
from the actual study design; frames are not independent experimental units.
For multiple comparisons, the Holm helper is in
[common_utils/statistics.py](common_utils/statistics.py).

The example parameter values still need checking against the paper's actual
runs. Begin with a short sequence: mask scoring uses memory proportional to
sequence size, although event bins and retina outputs are memory-mapped.

## Implementation and checks

- [eye_movements/doves.py](eye_movements/doves.py): DOVES loading and background rendering.
- [eye_movements/utils.py](eye_movements/utils.py): saccade detection, stitching, and motion formulas.
- [moving_objects/](moving_objects/): object trajectories, compositing, and masks.
- [events_camera_simulator/](events_camera_simulator/): event simulation and binning.
- [retina_model/oms_retina.py](retina_model/oms_retina.py): normalization and OMS retina.
- [common_utils/utils.py](common_utils/utils.py): object selectivity metrics.

Run `python -m unittest discover -s tests -v` for synthetic tests requiring no
DOVES download. [VALIDATION.md](VALIDATION.md) describes the checks;
[PROVENANCE.json](PROVENANCE.json) records the source commits. Paper-data validation,
the final citation, and the code license remain to be completed before publication.
