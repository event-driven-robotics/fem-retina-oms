# Initial validation

Validated locally using the existing OMS Python 3.12 environment:
NumPy 2.5.2, SciPy 1.18.1, OpenCV 5.0.0.93, Pillow 12.3.0, bimvee 1.0.23.
These versions describe the tested environment, not a complete dependency lock.

- Five `unittest` tests pass. They cover all three motion conditions end to end,
  deterministic event/spike reruns, resume checks, changed settings and damaged
  output rejection, missing input detection, trace-extension policy, event-free
  sequences, known selectivity values, and known statistical calculations.
- A wheel was built, installed into a separate temporary directory, and used
  from `/tmp` to run the three-condition demo and analysis. Neither development
  checkout was on its import path. Dependencies came from the existing OMS
  environment; a fresh online dependency installation was not tested.
- AST comparison with the source commit in `PROVENANCE.json` confirmed unchanged
  implementations of `detect_microsaccades_engbert`, `stitch_by_increments`,
  `circular`, `sine_velocity_trajectory`, `normalize_bipolar_responces`,
  `run_oms_retina_`, and `compute_object_selectivity`.
- The original event simulator and release streaming simulator were run on the
  same synthetic frames with seed 20261012. Encoded timestamp/address pairs
  matched exactly after sorting equal-time records: 5,534 DOVES-condition events,
  27,292 circular events, and 25,861 wobbly events.

The demo uses 12 frames per condition at 48×64 output resolution and one moving
object. It checks software integration and deterministic behavior, not scientific
agreement on DOVES. Full-resolution memory/runtime, paper subject–scene settings,
historical object trajectories, statistical design, and published figure
equivalence remain to be checked with the actual data and reference results.
