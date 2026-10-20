# 2026-bouanane-npjRobotics-code

Code for fixational-motion stimuli, event-camera simulation, and the OMS retina model.

## Setup

Download [DOVES](https://live.ece.utexas.edu/research/doves/) following the site's
access and citation instructions. Keep its `Images/` and `Fixations/` folders,
and your RGBA object images, outside this repository.

Clone this repository and install the dependencies with Python 3.10 or later:

```bash
git clone <repository-url>
cd 2026-bouanane-npjRobotics-code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Workflow

Edit the paths and parameters in each script, then run it with
`python scripts/<filename>.py`. Use the same `OUT` directory for steps 1–6 and a
new directory for each experiment. `SUBJECT` is a zero-based recording index.

| Step | Script | Purpose |
| --- | --- | --- |
| 1 | [load_doves.py](scripts/load_doves.py) | Load an image and a subject's gaze recording. |
| 2 | [generate_motion.py](scripts/generate_motion.py) | Filter saccades or generate circular/wobbly motion. |
| 3 | [generate_stimuli.py](scripts/generate_stimuli.py) | Render frames and optional moving objects with masks. |
| 4 | [generate_events.py](scripts/generate_events.py) | Simulate and bin camera events. |
| 5 | [run_retina.py](scripts/run_retina.py) | Compute OMS retina spikes. |
| 6 | [characterize.py](scripts/characterize.py) | Measure object selectivity. |
| 7 | [analyze.py](scripts/analyze.py) | Collect results and optionally compare experiments. |

Each step reads saved outputs from the preceding steps. If settings change,
rerun the affected step and all following steps. Choose your subject–scene pairs
and paper parameters before running the full dataset.

## Checks

Run `python -m unittest discover -s tests -v` for synthetic tests.
See [experiment notes](docs/experiment_notes.md) for parameter and timing details,
[validation](VALIDATION.md) for testing status, and [provenance](PROVENANCE.json)
for source commits. Validation against the paper's data is still pending.
