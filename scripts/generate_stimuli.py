"""Step 3: render background motion and optional independent objects and masks."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_utils.io import save_json
from eye_movements.doves import render_background
from moving_objects.add_multi_moving_objects import MovingObjectSpec, generate_multi_object_sequence

OUT = Path('results/example')
OBJECTS = [MovingObjectSpec(Path('/path/to/objects/object.png'), .01, 250)]
# Set OBJECTS = [] for background-only stimuli; add entries for multiple objects.


def run(out=OUT, objects=OBJECTS, crop_shape=(720, 960), motion_seed=20260917):
    out = Path(out)
    if (out / 'frames').exists():
        raise FileExistsError('Frames already exist; use a fresh experiment directory')
    trace = np.load(out / 'eye_trace.npy')
    motion = json.loads((out / 'motion.json').read_text())
    if objects:
        generate_multi_object_sequence([out / 'background.png'], objects, out,
            N=len(trace), fps=motion['fps'], motion_trace_path=out / 'eye_trace.npy',
            output_crop_size=crop_shape, motion_seed=motion_seed,
            clear_output=False, run_event_camera=False)
    else:
        with Image.open(out / 'background.png') as image:
            render_background(np.asarray(image), trace, out / 'frames', crop_shape)
    save_json(out / 'stimulus.json', dict(crop_shape=crop_shape, motion_seed=motion_seed,
              objects=[dict(image=str(o.object_rgba_path), area_fraction=o.area_fraction,
                            speed_px_s=o.speed_px_s) for o in objects]))


if __name__ == '__main__':
    run()
