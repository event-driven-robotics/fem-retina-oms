"""Conference trajectories: ping-pong eye motion and a constant-speed semicircle."""
import math
import numpy as np


def extend_eye_trace(trace, frames):
    trace = np.asarray(trace)
    if trace.ndim != 2 or trace.shape[1] != 2 or len(trace) < 2 or frames < 2:
        raise ValueError('Expected an (N, 2) trace with at least two samples and frames >= 2')
    if not np.isfinite(trace).all():
        raise ValueError('Eye trace must be finite')
    period = 2 * (len(trace) - 1)
    phase = np.arange(frames) % period
    index = np.where(phase < len(trace), phase, period - phase)
    return trace[index].copy(), index


def semicircle_centers(scene_rect, sprite_size, frames, dt, speed, gap=1.5):
    """Traverse the upper semicircle left-to-right with constant angular velocity."""
    if frames < 2 or dt <= 0 or speed <= 0:
        raise ValueError('Need at least two frames and positive dt/speed')
    left, top, right, bottom = scene_rect
    width, height = sprite_size
    lower = np.array([left + width / 2 + gap, top + height / 2 + gap])
    upper = np.array([right - width / 2 - gap, bottom - height / 2 - gap])
    duration = (frames - 1) * dt
    radius = speed * duration / math.pi
    if 2 * radius > upper[0] - lower[0] or radius > upper[1] - lower[1]:
        raise ValueError('Constant-speed semicircle does not fit the fully visible region; reduce speed or duration')
    center = (lower + upper) / 2 + [0, radius / 2]
    theta = np.linspace(math.pi, 0, frames)
    positions = center + radius * np.column_stack((np.cos(theta), -np.sin(theta)))
    stats = dict(method='constant_speed_left_to_right_semicircle', radius_px=radius,
                 center_px=center.tolist(), duration_sec=duration,
                 path_length_px=[speed * duration], mean_path_speed_px_s=[speed],
                 max_interval_speed_error_px_s=[float(np.max(np.abs(radius * np.diff(theta) / dt + speed)))],
                 center_span_px=np.ptp(positions, axis=0)[None].tolist(), collision_events=0,
                 speed_reference='arc length in scene coordinates before global camera motion')
    return positions, stats


def circular_eye_trace(frames, radius=5.0, pixels_per_frame=0.5):
    """Continuous (N, 2) camera offsets; speed is arc length per frame."""
    if frames < 2 or not np.isfinite([radius, pixels_per_frame]).all() or min(radius, pixels_per_frame) <= 0:
        raise ValueError('Need at least two frames and positive finite radius/speed')
    theta = np.arange(frames, dtype=float) * (pixels_per_frame / radius)
    return radius * np.column_stack((np.cos(theta), np.sin(theta)))
