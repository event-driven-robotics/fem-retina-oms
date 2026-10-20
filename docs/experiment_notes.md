# Experiment notes

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
[common_utils/statistics.py](../common_utils/statistics.py).

