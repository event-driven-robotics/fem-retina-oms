"""Fixed-scale single-object placement, using the existing motion integrator.

Only stimulus geometry changes. Eye motion is applied by the original renderer
after scene-coordinate motion; no eye velocity is added to object velocity.
"""
from dataclasses import replace
from functools import lru_cache

import numpy as np

from .add_multi_moving_objects import (
    MovingObjectSpec, _prepare_objects, _collision_body,
    _compute_planning_scene_rect, _center_crop_box,
    _build_joint_trajectories, transform_object,
)


def _bounds(sprite, rect, gap):
    body = _collision_body(sprite)
    margin = .5 + gap / 2
    return (np.asarray(rect[:2]) - body.minimum + margin,
            np.asarray(rect[2:]) - body.maximum - margin)


def _vertical_areas(sprite, height, threshold):
    rows = (np.asarray(sprite.getchannel('A')) > threshold).sum(axis=1)
    cumulative = np.r_[0, np.cumsum(np.pad(rows, (height, height)))]
    return cumulative[height:] - cumulative[:-height]


@lru_cache(maxsize=64)
def _geometry(asset, fraction, base_size, output_size, rect, eye_y_min,
              eye_y_max, gap, threshold, tolerance, variation):
    # Preparing in permissive mode lets us inspect the original scale even when
    # its sprite exceeds the background. Full visibility is always tried first.
    plan = _prepare_objects(base_size, [MovingObjectSpec(asset, fraction, 1)],
                            8, (output_size[1], output_size[0]), True)[0]
    height, width = output_size
    pixels = height * width
    target = fraction * pixels
    original_scale = plan.scale

    def render(scale):
        return transform_object(plan.image, scale=scale)

    sprite = render(original_scale)
    lower, upper = _bounds(sprite, rect, gap)
    mode = 'fully_visible'
    scale = original_scale
    if np.any(upper <= lower):
        # A small raster-scale adjustment may fit (notably the 45% condition).
        body = _collision_body(sprite)
        available = np.asarray(rect[2:]) - rect[:2] - (1 + gap) - 2
        fit_scale = original_scale * min(1., np.min(available / (body.maximum - body.minimum)))
        fitted = render(fit_scale)
        fitted_area = np.count_nonzero(np.asarray(fitted.getchannel('A')) > threshold)
        fit_lower, fit_upper = _bounds(fitted, rect, gap)
        if np.all(fit_upper > fit_lower) and abs(fitted_area - target) <= tolerance * pixels:
            sprite, scale, lower, upper = fitted, fit_scale, fit_lower, fit_upper
        else:
            mode = 'bounded_partial_visibility'
            # Calibrate one fixed scale to the maximum visible vertical window.
            # Integer resizing makes this a discrete search; keep the closest.
            low, high = original_scale * .9, original_scale * 1.2
            best = None
            for _ in range(16):
                candidate_scale = (low + high) / 2
                candidate = render(candidate_scale)
                areas = _vertical_areas(candidate, height, threshold)
                maximum = int(areas.max())
                error = abs(maximum - target)
                if best is None or error < best[0]:
                    best = (error, candidate_scale, candidate, areas)
                if maximum < target:
                    low = candidate_scale
                else:
                    high = candidate_scale
            _, scale, sprite, areas = best
            lower, upper = _bounds(sprite, rect, gap)
            if upper[0] <= lower[0]:
                raise ValueError('Object is too wide for the vertical-window placement strategy.')
            maximum_index = int(np.argmax(areas))
            allowed = ((np.abs(areas - target) <= tolerance * pixels)
                       & (areas >= areas.max() - variation * pixels))
            if not allowed[maximum_index]:
                raise ValueError('Cannot calibrate visible object area within tolerance.')
            start = end = maximum_index
            while start > 0 and allowed[start - 1]:
                start -= 1
            while end + 1 < len(areas) and allowed[end + 1]:
                end += 1
            # Window index minus height is the first visible sprite row.
            # Leave one pixel at both bounds for placement/warp rounding.
            crop_top = _center_crop_box(base_size, output_size)[1]
            lower[1] = crop_top - eye_y_min + sprite.height / 2 - (end - height) + 1
            upper[1] = crop_top - eye_y_max + sprite.height / 2 - (start - height) - 1
            if np.any(upper <= lower):
                raise ValueError('No motion corridor meets the visible-area variation tolerance.')

    area = np.count_nonzero(np.asarray(sprite.getchannel('A')) > threshold)
    if mode == 'fully_visible' and abs(area - target) > tolerance * pixels:
        raise ValueError('Original fully visible sprite exceeds the requested area tolerance.')
    updated = replace(plan, scale=scale, scaled_size=sprite.size)
    config = dict(mode=mode, fixed_scale=float(scale), original_scale=float(original_scale),
                  target_fraction=float(fraction), target_tolerance=float(tolerance),
                  temporal_variation_tolerance=float(variation),
                  center_lower=lower.tolist(), center_upper=upper.tolist())
    return updated, sprite, lower, upper, config


def plan_single_visible_area(base_size, output_size, motion_trace, spec, n_frames,
                             dt=.005, seed=20260917, gap=1.5, wander=.6,
                             threshold=1, tolerance=.003, variation=.002):
    """Plan constant scene-speed motion in a fixed, area-validated corridor."""
    if output_size is None:
        output_size = (base_size[1], base_size[0])
    rect = _compute_planning_scene_rect(base_size, output_size, motion_trace)
    offsets = np.zeros((1, 2)) if motion_trace is None else np.asarray(motion_trace)
    # The mask renderer uses nearest-neighbour shifts. Floor/ceil conservatively
    # cover every resulting integer row displacement, including half pixels.
    eye_min, eye_max = np.floor(offsets[:, 1].min()), np.ceil(offsets[:, 1].max())
    plan, sprite, lower, upper, config = _geometry(
        spec.object_rgba_path, spec.area_fraction, tuple(base_size), tuple(output_size),
        tuple(rect), float(eye_min), float(eye_max), gap, threshold, tolerance, variation,
    )
    plan = replace(plan, spec=spec)
    if config['mode'] == 'fully_visible':
        motion_rect, initial = rect, None
    else:
        # Express the desired center corridor as walls for the existing
        # collision integrator. The virtual walls may lie outside the viewport.
        body = _collision_body(sprite)
        margin = .5 + gap / 2
        motion_rect = (* (lower + body.minimum - margin),
                       * (upper + body.maximum + margin))
        initial = np.array([(lower + upper) / 2])
    trajectories, sprites, stats = _build_joint_trajectories(
        [plan], motion_rect, n_frames, dt, seed, gap, wander, 80, 15,
        'alpha', initial, False, [False],
    )
    stats['visible_area_constraint'] = dict(config)
    return [plan], trajectories, sprites, stats
