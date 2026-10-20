# Fixational retina reproducibility

Generate DOVES, circular, and wobbly stimuli, optionally add independently moving
objects, simulate event-camera output, run the OMS retina, and measure object
selectivity. This is a standalone extraction of the research code: it does not
need either development repository on `PYTHONPATH`.

**Status:** synthetic pipeline validation is included. The paper's subject–scene
list, final experiment settings, and comparisons still need to be supplied and
checked against reference results. Example settings are not a claim of complete
paper reproduction.

## Install and check without data

Python 3.10 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
retina-reproduce demo results/synthetic
retina-reproduce analyze results/synthetic/demo.json
```

The demo creates synthetic DOVES-format image and recording files, an RGBA object,
and runs all three motion conditions through frames, masks, events, retina, and
metrics. Use a new destination for each demo. `python -m reproduce` is equivalent
to `retina-reproduce`.

## Add your data and experiments

Keep the datasets outside this repository:

```text
/your/data/Doves/
  Images/
    <image filename>
  Fixations/
    <image filename>.mat
/your/data/objects/
  object.png
```

The loader preserves the existing convention: a scene named `image.iml` maps to
`Images/image.iml` and `Fixations/image.iml.mat`. Images are 768×1024 big-endian
uint16 raw arrays; recordings contain MATLAB `eye_data[0, subject]`, with x and y
in rows 0 and 1. **Subject indices are zero-based MATLAB-array indices used by
the Python code**, not automatically the participant numbers in a spreadsheet.
Object PNGs must be RGBA with a meaningful alpha mask.

```bash
cp configs/example.json configs/local-paper.json
# Edit data paths, subject–scene entries, conditions, frames, and parameters.
retina-reproduce validate configs/local-paper.json
retina-reproduce run configs/local-paper.json
retina-reproduce analyze configs/local-paper.json
```

Paths are resolved relative to the configuration file. `validate` checks input
files, image dimensions, subject indexing, gaze values, trace length, and sprites
without producing experiment outputs. Add one entry per experiment with a unique
`id`. All experiments share the explicitly configured event and retina settings.
Use separate manifests for parameter sweeps that change those shared settings.

- `condition: "doves"`: `motion: {}`. Detect saccades and stitch retained
  segments by their increments, as in the source code. Save the detector mask,
  detected events, and stitched-trace indices.
- `condition: "circular"`: `motion: {"radius": 5, "pixels_per_frame": 0.5}`.
- `condition: "wobbly"`:
  `motion: {"v_base": 0.5, "v_amp": 0.3, "velocity_cycles": 1,
  "orientation_cycles": 1, "theta0": 1.5707963267948966}`.
  Speeds are pixels/frame; cycles are over the whole requested sequence.
- `objects: []` generates background-only stimuli and retina responses;
  object selectivity is then undefined and saved as `null`.
- Each object specifies `image`, `area_fraction`, and `speed_px_s`. The retained
  multi-object generator uses seeded wandering trajectories with collision
  avoidance. One object uses that same planner. Earlier straight/arc object
  protocols are not automatically equivalent to this planner.
- DOVES sequences longer than the retained trace fail by default.
  `trace_extension: "pingpong"` explicitly enables forward/reverse extension
  without duplicated turnaround endpoints.
- `contrast_boost` enables the original `clip(image * 1.3 + 0.05, 0, 1)`.
- `analysis.region` selects exact masks or the original box-expanded vicinity.
  `min_visibility` and `exclude_edge_touch` specify frame selection.

Do not infer paper settings from these examples: fill them from the actual runs.
Non-finite gaze samples fail explicitly because their handling must agree with
the original experiment.

## Outputs and reruns

Each experiment writes its resolved manifest and input SHA-256 hashes, environment
versions, gaze trace, frames, object masks, event log, event timing metadata,
binned events, normalized responses, raw spikes, metrics, and a completion marker.
`spikes.npy` is raw model output, without display persistence. Large intermediate
arrays are memory-mapped. Mask scoring still uses memory proportional to sequence
size; start with short runs before processing full-resolution sequences.

```bash
retina-reproduce run configs/local-paper.json --resume
```

Resume skips completed, matching runs and verifies principal output hashes.
Partial runs and changed inputs/configurations fail rather than being overwritten;
choose a new output root for a retry. Inputs and outputs are never deleted by the
release CLI.

The simulator retains historical PNG quantization, the first-transition skip,
±2 ms timestamp jitter, and a clock starting at the first event. Event bins and
mask indices retain that legacy alignment; the event metadata records the scene
time offset. Final bins are zero-padded to the requested frame count, as in the
streaming development pipeline. This is an explicit compatibility convention,
not a correction of the original timing. Event-free sequences are valid and
produce zero event counts. Paper-data equivalence must include timing checks.

## Metrics and statistical comparisons

`summary.csv` contains OMS and raw-event `p`, spike/event counts, selected areas,
chance fractions, enrichment, and the optional selectivity index. No-response
values are `null` in JSON and blank in CSV, not zero. `p` is a response fraction,
not a statistical p-value.

Without explicit comparisons, analysis produces descriptive per-run results only.
To request a paired comparison, add entries to `analysis.comparisons`:

```json
{
  "name": "doves_vs_circular",
  "unit_definition": "participant; average scene differences within participant",
  "source": "oms_model",
  "seed": 42,
  "pairs": [
    {"a": "s0_scene1_doves", "b": "s0_scene1_circular", "unit": "subject0"},
    {"a": "s1_scene1_doves", "b": "s1_scene1_circular", "unit": "subject1"}
  ]
}
```

Pair comparable experiments explicitly. Repeated pairs within an independent unit
are averaged before testing. At least two units are required. The command reports
the mean paired difference, a percentile bootstrap interval over units, a two-sided
paired sign-flip test (exact through 16 units, otherwise Monte Carlo), and Holm
correction across the configured comparisons. The test assumes sign exchangeability
under the null and independent units; a scene–subject table may require a different
design if both sources of variation are modeled. These are available analysis
tools, not a replacement for identifying the paper's original analysis.

## Provenance and release

See `PROVENANCE.json` for source repository commits. Core detector, stitching,
motion formulas, simulator response, normalization, retina recurrence, and
selectivity functions were retained from the development repository. The release
adds portable orchestration, explicit configuration, synthetic fixtures, and tests.
No runtime dependency on the ICONS checkout is required; any missing historical
protocol should be ported and validated explicitly.

Before publishing: populate the experiment manifest, compare a reference run at
each pipeline stage, select the intended statistical comparisons, choose a code
license with the project owners, and add the final citation. No license has been
chosen on the authors' behalf. Obtain DOVES and object assets separately under
their applicable terms; no dataset is bundled.
