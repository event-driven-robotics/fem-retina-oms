# Validation

The standalone modules and individual step scripts were tested using the existing OMS
Python 3.12 environment (NumPy 2.5.2, SciPy 1.18.1, OpenCV 5.0.0.93,
Pillow 12.3.0, bimvee 1.0.23).

Run `python -m unittest discover -s tests -v`.

Five tests cover:

- Calling the individual scripts' `run` functions in sequence with synthetic
  DOVES-format files for DOVES, circular, and wobbly conditions, plus a
  background-only run. Input paths, image dimensions, sequence duration, and
  object speed/size are reduced for the test. Steps communicate through saved
  files; the summary script also reads the resulting per-experiment JSON files.
- Identical encoded events and retina spikes across repeated seeded runs.
- Correct background translation and center crop, missing input errors,
  non-finite trace rejection, and protection against replacing a frame directory.
- Event-free sequences and known exact/vicinity selectivity values.
- Known paired sign-flip probabilities and Holm-adjusted p-values.

The step scripts are run directly from a clone of this repository; the package is
not installed. Dependencies came from the existing OMS environment; a fresh online
dependency installation from `requirements.txt` was not tested.

The synthetic runs use 12 frames at 48×64 output resolution. They verify software
integration, not scientific agreement on DOVES. Full-resolution memory/runtime,
paper subject–scene settings, historical object trajectories, statistical design,
and published-figure equivalence still need reference data.

The retained event simulation code was previously compared against the original
simulator on identical synthetic frames with seed 20261012. Encoded timestamp and
address records matched after sorting equal-time records. The step scripts use
that streaming module without changing its implementation. The detector,
stitching, circular/wobbly formulas, retina recurrence, and selectivity calculation
remain from the source commit listed in `PROVENANCE.json`.
