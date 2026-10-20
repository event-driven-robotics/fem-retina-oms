"""Step 4: simulate and bin events from saved stimulus frames."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_utils.io import save_json
from events_camera_simulator.streaming import generate_events_streaming, bin_events_to_disk

OUT = Path('results/example')


def run(out=OUT, seed=20261012, tolerance=.2, refrac_period=.0001, sensor_sensitivity=1):
    out = Path(out)
    motion = json.loads((out / 'motion.json').read_text())
    stimulus = json.loads((out / 'stimulus.json').read_text())
    dt = 1 / motion['fps']
    generate_events_streaming(out, dt=dt, seed=seed, tolerance=tolerance,
                             refrac_period=refrac_period, sensor_sensitivity=sensor_sensitivity)
    bin_events_to_disk(out / 'events_file/binaryevents.log', out / 'binned_events.npy',
                       frames=motion['frames'], shape=stimulus['crop_shape'], dt=dt)
    save_json(out / 'event_parameters.json', dict(seed=seed, tolerance=tolerance,
              refrac_period=refrac_period, sensor_sensitivity=sensor_sensitivity, dt=dt))


if __name__ == '__main__':
    run()
