"""Step 2: filter DOVES saccades or create circular/wobbly camera motion."""
from pathlib import Path
import numpy as np
from common_utils.io import save_json
from eye_movements.utils import detect_microsaccades_engbert, stitch_by_increments, circular, sine_velocity_trajectory

OUT = Path('results/example')
CONDITION = 'doves'  # doves, circular, wobbly
N_FRAMES = 100


def run(out=OUT, condition=CONDITION, n_frames=N_FRAMES, radius=5, pixels_per_frame=.5,
        v_base=.5, v_amp=.3, velocity_cycles=1, orientation_cycles=1,
        lam=6, min_dur_ms=6, merge_gap_ms=10):
    out = Path(out)
    if n_frames < 3:
        raise ValueError('At least three frames are required')
    fps = 200
    if condition == 'doves':
        eye = np.load(out / 'raw_eye.npy')
        detected = detect_microsaccades_engbert(eye[0], eye[1], fs=fps, lam=lam,
                    min_dur_ms=min_dur_ms, merge_gap_ms=merge_gap_ms)
        x, y, _ = stitch_by_increments(eye[0], eye[1], detected['jitter_mask'], fs=fps,
                                       carry_over=True, detrend_segment=False)
        if len(x) < n_frames:
            raise ValueError('Insufficient retained gaze samples; choose fewer frames')
        np.save(out / 'saccade_mask.npy', detected['saccade_mask'])
        np.save(out / 'retained_trace.npy', np.column_stack((x, y)))
        save_json(out / 'saccades.json', detected['saccades'])
        trace = np.column_stack((x[:n_frames], y[:n_frames]))
    elif condition == 'circular':
        if radius <= 0:
            raise ValueError('radius must be positive')
        trace = np.column_stack(circular(n_frames, radius, pixels_per_frame))
    elif condition == 'wobbly':
        trace = np.column_stack(sine_velocity_trajectory(n_frames, v_base, v_amp,
                velocity_cycles, orientation_cycles=orientation_cycles, theta0=np.pi/2))
    else:
        raise ValueError('Unknown condition')
    np.save(out / 'eye_trace.npy', trace)
    save_json(out / 'motion.json', dict(condition=condition, frames=n_frames, fps=fps,
              radius=radius, pixels_per_frame=pixels_per_frame, v_base=v_base, v_amp=v_amp,
              velocity_cycles=velocity_cycles, orientation_cycles=orientation_cycles,
              lam=lam, min_dur_ms=min_dur_ms, merge_gap_ms=merge_gap_ms))


if __name__ == '__main__':
    run()
