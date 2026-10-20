"""Multi-object rendering with jointly planned, collision-safe wandering motion.

Dependencies: NumPy, Pillow and SciPy for the default alpha-mask collision model.
OpenCV is needed only for a camera motion trace. Original project rendering
helpers are used when available, with local fallbacks for standalone use.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

try:
    import cv2
except ImportError:  # Only required when applying a camera motion trace.
    cv2 = None
import numpy as np
from PIL import Image, ImageOps

create_gif = None

try:
    from .add_moving_objects import (
        combine_masks,
        ensure_grayscale_L,
        ensure_rgba,
        make_mask,
        transform_object,
        alpha_area,
    )
except ImportError:
    try:
        from moving_objects.add_moving_objects import (
            combine_masks,
            ensure_grayscale_L,
            ensure_rgba,
            make_mask,
            transform_object,
            alpha_area,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"moving_objects", "moving_objects.add_moving_objects"}:
            raise

        # Standalone equivalents when the original project's helper is absent.
        def ensure_grayscale_L(image):
            return image.convert("L")

        def ensure_rgba(image):
            return image.convert("RGBA")

        def alpha_area(image):
            return float(np.asarray(image.getchannel("A"), dtype=np.float64).sum() / 255)

        def transform_object(image, angle_deg=0.0, scale=1.0):
            size = tuple(max(1, int(round(d * scale))) for d in image.size)
            result = image.resize(size, Image.Resampling.LANCZOS)
            if angle_deg:
                result = result.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
            return result

        def make_mask(base_size, image, xy, mode="binary", thresh=1):
            alpha = image.getchannel("A")
            if mode == "binary":
                alpha = alpha.point(lambda value: 255 if value > thresh else 0)
            elif mode not in {"alpha", "soft"}:
                raise ValueError("mask_mode must be 'binary', 'alpha' or 'soft'")
            result = Image.new("L", base_size)
            result.paste(alpha, xy)
            return result

        def combine_masks(a, b):
            return Image.fromarray(np.maximum(np.asarray(a), np.asarray(b)))


TrajectoryFn = Callable[[float, int, int, Tuple[int, int], Tuple[int, int]], Tuple[float, float, float, float]]


@dataclass(frozen=True)
class MovingObjectSpec:
    object_rgba_path: Path
    area_fraction: float
    speed_px_s: float
    direction: Optional[str] = None
    phase_px: float = 0.0
    allow_partial_visibility: bool = False


@dataclass(frozen=True)
class PlannedMovingObject:
    spec: MovingObjectSpec
    image: Image.Image
    scale: float
    scaled_size: Tuple[int, int]
    lane_y: float
    trajectory: TrajectoryFn


_DEF_MIN_SCALE = 0.00001


def scale_for_area_fraction(
    base_size: Tuple[int, int],
    obj_area_px: float,
    frac: float,
    min_scale: float = _DEF_MIN_SCALE,
    max_scale: float = 1e6,
) -> float:
    """Scale an object so its soft alpha area matches a fraction of the scene area."""
    w, h = base_size
    frac = max(0.0, min(1.0, float(frac)))
    target_area = max(1.0, frac * float(w * h))
    if obj_area_px <= 0:
        return 1.0
    scale = math.sqrt(target_area / obj_area_px)
    return max(min_scale, min(scale, max_scale))


def load_object_specs(objects_json: str | Path) -> list[MovingObjectSpec]:
    """Load object specs from a JSON list.

    The JSON schema is intentionally minimal: each object needs
    object_rgba_path, area_fraction, and speed_px_s. Optional fields:
    - allow_partial_visibility (bool): if True, this object is allowed to extend outside frame bounds.
    - direction, phase_px (legacy, ignored)
    """
    path = Path(objects_json)
    raw_objects = json.loads(path.read_text())
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("objects-json must contain a non-empty JSON list")

    specs: list[MovingObjectSpec] = []
    for index, item in enumerate(raw_objects):
        if not isinstance(item, dict):
            raise ValueError(f"Object entry {index} must be a JSON object")

        spec = MovingObjectSpec(
            object_rgba_path=Path(item["object_rgba_path"]),
            area_fraction=float(item["area_fraction"]),
            speed_px_s=float(item["speed_px_s"]),
            direction=str(item["direction"]) if item.get("direction") is not None else None,
            phase_px=float(item["phase_px"]) if item.get("phase_px") is not None else 0.0,
            allow_partial_visibility=bool(item["allow_partial_visibility"]) if item.get("allow_partial_visibility") is not None else False,
        )
        if not spec.object_rgba_path.is_absolute() and not spec.object_rgba_path.exists():
            spec = MovingObjectSpec(
                path.parent / spec.object_rgba_path,
                spec.area_fraction,
                spec.speed_px_s,
                spec.direction,
                spec.phase_px,
            )
        specs.append(spec)
    return specs


def _ensure_positive_specs(specs: Sequence[MovingObjectSpec]) -> None:
    if not specs:
        raise ValueError("At least one moving object spec is required.")
    for index, spec in enumerate(specs):
        if not math.isfinite(spec.area_fraction) or not 0 < spec.area_fraction <= 1:
            raise ValueError(f"Spec {index} has invalid area_fraction={spec.area_fraction!r}.")
        if not math.isfinite(spec.speed_px_s) or spec.speed_px_s < 0:
            raise ValueError(f"Spec {index} has invalid speed_px_s={spec.speed_px_s!r}.")
        if spec.direction is not None and spec.direction.lower() not in {"lr", "rl"}:
            raise ValueError(
                f"Spec {index} must use direction='lr' or 'rl', got {spec.direction!r}."
            )


def _build_random_continuous_trajectory(
    scene_size: Tuple[int, int],
    obj_size: Tuple[int, int],
    speed_px_s: float,
    path_seed: int,
    margin_px: int = 24,
    scene_rect: Optional[tuple[float, float, float, float]] = None,
) -> TrajectoryFn:
    """Compatibility helper used by tests; independent from collision planner."""
    scene_w, scene_h = scene_size
    obj_w, obj_h = obj_size
    half_w = max(1.0, obj_w / 2.0)
    half_h = max(1.0, obj_h / 2.0)
    if scene_rect is None:
        rect_left = 0.0
        rect_top = 0.0
        rect_right = float(scene_w)
        rect_bottom = float(scene_h)
    else:
        rect_left, rect_top, rect_right, rect_bottom = [float(value) for value in scene_rect]
    x_min = rect_left + margin_px + half_w
    x_max = max(x_min + 1.0, rect_right - margin_px - half_w)
    y_min = rect_top + margin_px + half_h
    y_max = max(y_min + 1.0, rect_bottom - margin_px - half_h)

    rng = np.random.default_rng(path_seed)
    points: list[Tuple[float, float]] = []
    current = (float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)))
    points.append(current)

    for _ in range(7):
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        step = float(rng.uniform(40.0, min(scene_w, scene_h) * 0.18))
        nx = min(max(current[0] + math.cos(angle) * step, x_min), x_max)
        ny = min(max(current[1] + math.sin(angle) * step, y_min), y_max)
        current = (nx, ny)
        points.append(current)

    if len(points) >= 2:
        points.append(points[0])

    cumulative = [0.0]
    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        segment_len = math.hypot(x1 - x0, y1 - y0)
        cumulative.append(cumulative[-1] + segment_len)

    total_len = cumulative[-1] if cumulative else 0.0

    def fn(
        t: float,
        i: int,
        N: int,
        base_size: Tuple[int, int],
        obj_size_inner: Tuple[int, int],
    ):
        if speed_px_s <= 0.0 or total_len <= 0.0 or len(points) < 2:
            x, y = points[0]
            return (x, y, 0.0, 1.0)

        wrapped = (speed_px_s * t) % total_len
        if wrapped < 0:
            wrapped += total_len

        for idx in range(1, len(cumulative)):
            if wrapped <= cumulative[idx]:
                x0, y0 = points[idx - 1]
                x1, y1 = points[idx]
                seg_len = cumulative[idx] - cumulative[idx - 1]
                if seg_len <= 0.0:
                    return (x0, y0, 0.0, 1.0)
                frac = (wrapped - cumulative[idx - 1]) / seg_len
                x = x0 + (x1 - x0) * frac
                y = y0 + (y1 - y0) * frac
                return (x, y, 0.0, 1.0)

        x1, y1 = points[-1]
        return (x1, y1, 0.0, 1.0)

    return fn


def _combine_many_masks(masks: Sequence[Image.Image]) -> Optional[Image.Image]:
    combined: Optional[Image.Image] = None
    for mask in masks:
        combined = mask if combined is None else combine_masks(combined, mask)
    return combined


def _prepare_output_dirs(
    out_dir: Path,
    object_count: int,
    clear_output: bool,
) -> tuple[Path, list[Path], Path]:
    frames_dir = out_dir / "frames"
    masks_root_dir = out_dir / "masks"
    union_dir = out_dir / "masks_union"

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_root_dir.mkdir(parents=True, exist_ok=True)
    union_dir.mkdir(parents=True, exist_ok=True)

    object_mask_dirs = []
    for index in range(object_count):
        object_dir = masks_root_dir / f"object_{index + 1:02d}"
        object_dir.mkdir(parents=True, exist_ok=True)
        object_mask_dirs.append(object_dir)

    if clear_output:
        for directory in [frames_dir, union_dir, *object_mask_dirs]:
            for path in directory.glob("*"):
                try:
                    path.unlink()
                except Exception:
                    pass

    return frames_dir, object_mask_dirs, union_dir


def _load_motion_trace(motion_trace_path: Path) -> np.ndarray:
    trace = np.asarray(np.load(motion_trace_path), dtype=np.float32)
    if trace.ndim != 2:
        raise ValueError(f"Motion trace must be a 2D array, got shape {trace.shape}.")
    if trace.shape[0] == 2:
        trace = trace.T
    elif trace.shape[1] != 2:
        raise ValueError(f"Motion trace must have shape (2, N) or (N, 2), got {trace.shape}.")
    return trace


def _crop_center_to_size(
    image: Image.Image | np.ndarray,
    target_size: tuple[int, int] | None = (720, 960),
) -> Image.Image | np.ndarray:
    """Center-crop a larger image/array to the requested final output size."""
    if target_size is None:
        return image
    target_h, target_w = int(target_size[0]), int(target_size[1])
    if isinstance(image, Image.Image):
        src_w, src_h = image.size
        if src_w < target_w or src_h < target_h:
            raise ValueError(
                f"Image size {image.size} is smaller than the requested crop size {(target_w, target_h)}. "
                "Use a larger base frame or set a smaller crop target."
            )
        left = (src_w - target_w) // 2
        top = (src_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        return image.crop((left, top, right, bottom))

    arr = np.asarray(image)
    h, w = arr.shape[:2]
    if w < target_w or h < target_h:
        raise ValueError(
            f"Array size {(w, h)} is smaller than the requested crop size {(target_w, target_h)}. "
            "Use a larger base frame or set a smaller crop target."
        )
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    return arr[top:bottom, left:right]


def _center_crop_box(
    base_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    base_w, base_h = int(base_size[0]), int(base_size[1])
    target_h, target_w = int(target_size[0]), int(target_size[1])
    if base_w < target_w or base_h < target_h:
        raise ValueError(
            f"Base frame size {(base_w, base_h)} is smaller than the requested crop size {(target_w, target_h)}."
        )
    left = float((base_w - target_w) // 2)
    top = float((base_h - target_h) // 2)
    return (left, top, left + float(target_w), top + float(target_h))


def _compute_planning_scene_rect(
    base_size: tuple[int, int],
    target_size: tuple[int, int] | None,
    motion_trace: Optional[np.ndarray],
) -> tuple[float, float, float, float]:
    """Return the pre-shift region guaranteed to remain visible in final output."""
    if target_size is None:
        crop_left = 0.0
        crop_top = 0.0
        crop_right = float(base_size[0])
        crop_bottom = float(base_size[1])
    else:
        crop_left, crop_top, crop_right, crop_bottom = _center_crop_box(base_size, target_size)

    if motion_trace is None or len(motion_trace) == 0:
        return (crop_left, crop_top, crop_right, crop_bottom)

    x_offsets = np.asarray(motion_trace[:, 0], dtype=np.float64)
    y_offsets = np.asarray(motion_trace[:, 1], dtype=np.float64)
    left = max(0.0, crop_left - float(np.min(x_offsets)))
    top = max(0.0, crop_top - float(np.min(y_offsets)))
    right = min(float(base_size[0]), crop_right - float(np.max(x_offsets)))
    bottom = min(float(base_size[1]), crop_bottom - float(np.max(y_offsets)))
    if left >= right or top >= bottom:
        raise ValueError(
            "Motion trace leaves no stable visible region after applying the requested crop. "
            "Use a larger base frame, smaller crop, or a smaller motion trace."
        )
    return (left, top, right, bottom)


def _shift_frame(frame: np.ndarray, x_offset: float, y_offset: float, interpolation: int) -> np.ndarray:
    height, width = frame.shape[:2]
    translation = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    return cv2.warpAffine(
        frame,
        translation,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _build_sampled_trajectory(samples: Sequence[tuple[float, float]]) -> TrajectoryFn:
    """Replay a precomputed center trajectory by frame index."""
    if len(samples) == 0:
        raise ValueError("Sampled trajectory requires at least one sample.")
    frozen = [(float(x), float(y)) for x, y in samples]

    def fn(
        t: float,
        i: int,
        N: int,
        base_size: Tuple[int, int],
        obj_size_inner: Tuple[int, int],
    ):
        frame_index = min(max(int(i), 0), len(frozen) - 1)
        x, y = frozen[frame_index]
        return (x, y, 0.0, 1.0)

    return fn


def _partition_scene_grid(
    scene_size: tuple[int, int],
    obj_sizes: Sequence[tuple[float, float]],
) -> Optional[list[tuple[float, float, float, float]]]:
    """Partition the scene into equal cells when all objects fit."""
    n_obj = len(obj_sizes)
    if n_obj <= 0:
        return []

    scene_w, scene_h = float(scene_size[0]), float(scene_size[1])
    best_layout: Optional[tuple[int, int, float, float]] = None
    best_score: Optional[tuple[int, float]] = None

    for rows in range(1, n_obj + 1):
        cols = int(math.ceil(n_obj / rows))
        cell_w = scene_w / float(cols)
        cell_h = scene_h / float(rows)
        if any(obj_w > cell_w + 1e-6 or obj_h > cell_h + 1e-6 for obj_w, obj_h in obj_sizes):
            continue
        unused = rows * cols - n_obj
        cell_area = cell_w * cell_h
        score = (unused, -cell_area)
        if best_score is None or score < best_score:
            best_score = score
            best_layout = (rows, cols, cell_w, cell_h)

    if best_layout is None:
        return None

    rows, cols, cell_w, cell_h = best_layout
    cells: list[tuple[float, float, float, float]] = []
    for row in range(rows):
        used_in_row = cols
        if row == rows - 1 and rows * cols > n_obj:
            used_in_row = n_obj - row * cols
        row_width = used_in_row * cell_w
        x_offset = 0.5 * max(0.0, scene_w - row_width)
        for col in range(used_in_row):
            left = x_offset + col * cell_w
            top = row * cell_h
            cells.append((left, top, cell_w, cell_h))
            if len(cells) == n_obj:
                return cells
    return cells


def _pack_shelf_cells(
    scene_size: tuple[int, int],
    cell_sizes: Sequence[tuple[float, float]],
) -> Optional[list[tuple[float, float, float, float]]]:
    """Pack motion cells with a deterministic shelf layout."""
    scene_w, scene_h = float(scene_size[0]), float(scene_size[1])
    heuristics = [
        lambda item: item[1][1],
        lambda item: item[1][0],
        lambda item: item[1][0] * item[1][1],
        lambda item: max(item[1][0], item[1][1]),
    ]

    for key_fn in heuristics:
        order = [index for index, _ in sorted(enumerate(cell_sizes), key=key_fn, reverse=True)]
        placements: list[Optional[tuple[float, float, float, float]]] = [None] * len(cell_sizes)
        row: list[tuple[int, float, float, float, float]] = []
        x_cursor = 0.0
        y_cursor = 0.0
        row_height = 0.0
        success = True

        def flush_row() -> bool:
            nonlocal row, y_cursor, row_height
            if not row:
                return True
            used_width = max(item[3] for item in row)
            x_offset = max(0.0, 0.5 * (scene_w - used_width))
            for idx, left, top, right, height in row:
                placements[idx] = (left + x_offset, top, right - left, height)
            y_cursor += row_height
            row = []
            row_height = 0.0
            return y_cursor <= scene_h + 1e-6

        for idx in order:
            cell_w, cell_h = float(cell_sizes[idx][0]), float(cell_sizes[idx][1])
            if cell_w > scene_w + 1e-6 or cell_h > scene_h + 1e-6:
                success = False
                break

            if row and x_cursor + cell_w > scene_w + 1e-6:
                if not flush_row():
                    success = False
                    break
                x_cursor = 0.0

            if y_cursor + max(row_height, cell_h) > scene_h + 1e-6:
                success = False
                break

            left = x_cursor
            right = x_cursor + cell_w
            row.append((idx, left, y_cursor, right, cell_h))
            x_cursor = right
            row_height = max(row_height, cell_h)

        if success and flush_row() and all(item is not None for item in placements):
            return [item for item in placements if item is not None]

    return None


@dataclass(frozen=True)
class _CollisionBody:
    # Pixel-edge hull in renderer coordinates, relative to the sprite center.
    hull: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    area: float


@dataclass(frozen=True)
class _PairGeometry:
    i: int
    j: int
    axes: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class _AlphaPairGeometry(_PairGeometry):
    field: np.ndarray
    origin: np.ndarray
    clearance: float


def _alpha_pair_geometry(bodies, sprites, gap):
    """Cache each pair's forbidden translations from its real alpha pixels."""
    try:
        from scipy.signal import fftconvolve
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise ImportError(
            "Alpha collision geometry requires SciPy. Install scipy, "
            "or select collision_geometry='convex'."
        ) from exc
    hull_pairs = _pair_geometry(bodies, gap)
    masks = [np.asarray(sprite.getchannel("A")) > 0 for sprite in sprites]
    clearance = 1.5 * math.sqrt(2.0) + 0.05 + gap
    padding = int(math.ceil(clearance)) + 3
    result = []
    for pair in hull_pairs:
        a, b = masks[pair.i], masks[pair.j]
        correlation = fftconvolve(a.astype(np.float64), b[::-1, ::-1].astype(np.float64))
        blocked = np.pad(correlation > 0.5, padding)
        field = (distance_transform_edt(~blocked) - distance_transform_edt(blocked)).astype(np.float32)
        ah, aw = a.shape
        bh, bw = b.shape
        origin = np.array([
            (aw - bw) / 2 + bw - 1 + padding,
            (ah - bh) / 2 + bh - 1 + padding,
        ])
        result.append(
            _AlphaPairGeometry(
                pair.i,
                pair.j,
                pair.axes,
                pair.lower,
                pair.upper,
                field,
                origin,
                clearance,
            )
        )
    return result


def _field_value_gradient(pair, relative):
    x, y = relative + pair.origin
    h, w = pair.field.shape
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return float("inf"), np.zeros(2)
    ix, iy = int(math.floor(x)), int(math.floor(y))
    u, v = x - ix, y - iy
    z00, z10 = float(pair.field[iy, ix]), float(pair.field[iy, ix + 1])
    z01, z11 = float(pair.field[iy + 1, ix]), float(pair.field[iy + 1, ix + 1])
    a, b, c = z10 - z00, z01 - z00, z11 - z10 - z01 + z00
    return z00 + a * u + b * v + c * u * v, np.array([a + c * v, b + c * u])


def _alpha_penetration(pair, centers):
    relative = centers[pair.j] - centers[pair.i]
    value, gradient = _field_value_gradient(pair, relative)
    depth = pair.clearance - value
    if depth <= 1e-8:
        return None
    norm = float(np.linalg.norm(gradient))
    if norm < 1e-8:
        candidates = [np.array([math.cos(a), math.sin(a)]) for a in np.arange(8) * math.pi / 4]
        gradient = max(candidates, key=lambda n: _field_value_gradient(pair, relative + n * 2)[0])
        norm = 1.0
    return float(depth), gradient / norm


def _first_contour_root(a, b, c, horizon):
    """First inward crossing of a + b*t + c*t*t = 0 within one grid cell."""
    tolerance = 1e-8
    if a <= tolerance and b < -1e-10:
        return 0.0
    if abs(c) < 1e-14:
        if b >= -1e-12:
            return None
        root = -a / b
        return max(0.0, root) if -1e-10 <= root <= horizon + 1e-12 else None
    discriminant = b * b - 4 * c * a
    if discriminant < 0:
        return None
    discriminant = math.sqrt(discriminant)
    q = -0.5 * (b + math.copysign(discriminant, b))
    roots = [q / c]
    if abs(q) > 1e-30:
        roots.append(a / q)
    for root in sorted(roots):
        if -1e-10 <= root <= horizon + 1e-12 and b + 2 * c * root < -1e-10:
            return max(0.0, root)
    return None


def _alpha_hit(pair, centers, velocities, horizon):
    """Exact ray/contour intersections of the cached bilinear distance field."""
    point = centers[pair.j] - centers[pair.i] + pair.origin
    velocity = velocities[pair.j] - velocities[pair.i]
    if np.max(np.abs(velocity)) < 1e-12:
        return None
    h, w = pair.field.shape
    start, finish = 0.0, float(horizon)
    for axis, limit in enumerate((w - 1, h - 1)):
        if abs(velocity[axis]) < 1e-12:
            if point[axis] < 0 or point[axis] > limit:
                return None
        else:
            times = (-point[axis] / velocity[axis], (limit - point[axis]) / velocity[axis])
            start, finish = max(start, min(times)), min(finish, max(times))
    if start > finish or finish < 0:
        return None
    now = max(0.0, start)
    while now <= finish + 1e-12:
        x, y = point + velocity * now
        ix = int(math.floor(x)) if velocity[0] >= 0 else int(math.ceil(x)) - 1
        iy = int(math.floor(y)) if velocity[1] >= 0 else int(math.ceil(y)) - 1
        ix, iy = min(max(ix, 0), w - 2), min(max(iy, 0), h - 2)
        u, v = x - ix, y - iy
        z00, z10 = float(pair.field[iy, ix]), float(pair.field[iy, ix + 1])
        z01, z11 = float(pair.field[iy + 1, ix]), float(pair.field[iy + 1, ix + 1])
        dx, dy, mixed = z10 - z00, z01 - z00, z11 - z10 - z01 + z00
        vx, vy = velocity
        a = z00 + dx * u + dy * v + mixed * u * v - pair.clearance
        b = (dx + mixed * v) * vx + (dy + mixed * u) * vy
        c = mixed * vx * vy
        crossings = []
        for coordinate, cell, speed in ((x, ix, vx), (y, iy, vy)):
            if abs(speed) > 1e-12:
                boundary = cell + 1 if speed > 0 else cell
                crossings.append(max(0.0, (boundary - coordinate) / speed))
        cell_duration = min([finish - now, *crossings])
        if a < -1e-5:
            raise RuntimeError("Internal alpha collision error: penetrating initial state.")
        hit = _first_contour_root(a, b, c, cell_duration)
        if hit is not None:
            normal = np.array([dx + mixed * (v + vy * hit), dy + mixed * (u + vx * hit)])
            norm = float(np.linalg.norm(normal))
            if norm > 1e-12:
                return now + hit, normal / norm
        if now + cell_duration >= finish - 1e-12:
            return None
        next_time = now + cell_duration
        now = max(float(np.nextafter(next_time, math.inf)), now + 1e-12)
    return None


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain hull; no additional geometry dependency required."""
    ordered = sorted(set(map(tuple, points)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _collision_body(sprite: Image.Image) -> _CollisionBody:
    alpha = np.asarray(sprite.getchannel("A")) > 0
    rows = np.flatnonzero(alpha.any(axis=1))
    if not len(rows):
        raise ValueError("An object has no visible pixels after scaling.")
    left = alpha[rows].argmax(axis=1)
    right = alpha.shape[1] - alpha[rows, ::-1].argmax(axis=1)
    corners = np.concatenate(
        [
            np.column_stack((left, rows)),
            np.column_stack((right, rows)),
            np.column_stack((left, rows + 1)),
            np.column_stack((right, rows + 1)),
        ]
    )
    hull = _convex_hull(corners) - np.asarray(sprite.size, dtype=float) / 2
    area = abs(float(np.sum(hull[:, 0] * np.roll(hull[:, 1], -1) - hull[:, 1] * np.roll(hull[:, 0], -1)))) / 2
    return _CollisionBody(hull, hull.min(axis=0), hull.max(axis=0), area)


def _pair_geometry(bodies: Sequence[_CollisionBody], gap: float) -> list[_PairGeometry]:
    pairs = []
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            axes = []
            for body in (bodies[i], bodies[j]):
                edges = np.roll(body.hull, -1, axis=0) - body.hull
                normals = np.column_stack((-edges[:, 1], edges[:, 0]))
                normals /= np.linalg.norm(normals, axis=1)[:, None]
                for normal in normals:
                    if normal[0] < -1e-12 or (abs(normal[0]) <= 1e-12 and normal[1] < 0):
                        normal = -normal
                    if not any(abs(float(normal @ old)) > 1 - 1e-12 for old in axes):
                        axes.append(normal)
            axes = np.asarray(axes)
            a, b = bodies[i].hull @ axes.T, bodies[j].hull @ axes.T
            pad = np.abs(axes).sum(axis=1) + gap
            pairs.append(
                _PairGeometry(
                    i,
                    j,
                    axes,
                    a.min(axis=0) - b.max(axis=0) - pad,
                    a.max(axis=0) - b.min(axis=0) + pad,
                )
            )
    return pairs


def _pair_penetration(pair: _PairGeometry, centers: np.ndarray):
    if isinstance(pair, _AlphaPairGeometry):
        return _alpha_penetration(pair, centers)
    relative = (centers[pair.j] - centers[pair.i]) @ pair.axes.T
    low, high = relative - pair.lower, pair.upper - relative
    if np.any(low <= 0) or np.any(high <= 0):
        return None
    depths = np.minimum(low, high)
    axis = int(np.argmin(depths))
    normal = pair.axes[axis] * (1 if high[axis] <= low[axis] else -1)
    return float(depths[axis]), normal


def _mixed_integer_placement(pairs, lower, upper, time_limit):
    """Solve the separation disjunctions jointly if SciPy is installed."""
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError:
        return None
    n_coordinates = lower.size
    n_indicators = sum(2 * len(pair.axes) for pair in pairs)
    if n_indicators == 0:
        return (lower + upper) / 2
    matrix = lil_matrix((n_indicators + len(pairs), n_coordinates + n_indicators))
    bounds = np.zeros(matrix.shape[0])
    row, variable = 0, n_coordinates
    clearance = 0.05
    for pair in pairs:
        indicators = []
        min_delta = lower[pair.j] - upper[pair.i]
        max_delta = upper[pair.j] - lower[pair.i]
        for normal, low, high in zip(pair.axes, pair.lower, pair.upper):
            for sign, limit in ((1, low - clearance), (-1, -high - clearance)):
                coefficients = normal * sign
                max_projection = np.sum(
                    np.where(coefficients >= 0, coefficients * max_delta, coefficients * min_delta)
                )
                big_m = max(0.0, max_projection - limit) + 1e-6
                matrix[row, 2 * pair.j : 2 * pair.j + 2] = coefficients
                matrix[row, 2 * pair.i : 2 * pair.i + 2] = -coefficients
                matrix[row, variable] = big_m
                bounds[row] = limit + big_m
                indicators.append(variable)
                row += 1
                variable += 1
        matrix[row, indicators] = -1
        bounds[row] = -1
        row += 1
    result = milp(
        np.zeros(matrix.shape[1]),
        integrality=np.r_[np.zeros(n_coordinates), np.ones(n_indicators)],
        bounds=Bounds(np.r_[lower.ravel(), np.zeros(n_indicators)], np.r_[upper.ravel(), np.ones(n_indicators)]),
        constraints=LinearConstraint(matrix.tocsr(), -np.inf, bounds),
        options={"time_limit": float(time_limit)},
    )
    if result.x is None:
        return None
    centers = np.clip(result.x[:n_coordinates].reshape(lower.shape), lower, upper)
    if all(_pair_penetration(pair, centers) is None for pair in pairs) or isinstance(pairs[0], _AlphaPairGeometry):
        return centers
    return None


def _place_bodies(
    bodies,
    pairs,
    scene_rect,
    gap,
    rng,
    attempts=80,
    packing_time_limit=15.0,
    initial_centers=None,
    allow_partial_visibility=False,
    per_object_visibility=None,
):
    """Joint placement using the selected pair geometry and bounded retries.

    Args:
        allow_partial_visibility: Legacy flag; if True, all objects allow partial visibility.
        per_object_visibility: Optional list of bools; if provided, overrides allow_partial_visibility per object.
    """
    left, top, right, bottom = map(float, scene_rect)
    margin = 0.5 + gap / 2

    # Determine visibility mode per object
    if per_object_visibility is not None:
        visibility_flags = per_object_visibility
    else:
        visibility_flags = [allow_partial_visibility] * len(bodies)

    # Apply bounds based on per-object visibility
    lower_list = []
    upper_list = []
    for body, allow_partial in zip(bodies, visibility_flags):
        if allow_partial:
            # Relax bounds so body may extend outside frame
            lower_list.append([left, top] - body.maximum + margin)
            upper_list.append([right, bottom] - body.minimum - margin)
        else:
            # Strict bounds: keep body fully inside frame
            lower_list.append([left, top] - body.minimum + margin)
            upper_list.append([right, bottom] - body.maximum - margin)

    lower = np.array(lower_list)
    upper = np.array(upper_list)

    if np.any(upper <= lower):
        raise RuntimeError(
            "A scaled object leaves no motion clearance for the selected visibility mode. "
            "Check its aspect ratio, area_fraction, crop and motion trace."
        )
    if initial_centers is not None:
        centers = np.asarray(initial_centers, dtype=np.float64)
        if centers.shape != lower.shape or not np.isfinite(centers).all():
            raise ValueError("Initial centers must be a finite (object_count, 2) array.")
        if np.any(centers < lower) or np.any(centers > upper):
            raise ValueError("Initial centers place an object outside the allowed scene.")
        if any(_pair_penetration(pair, centers) is not None for pair in pairs):
            raise ValueError("Initial centers violate object collision clearance.")
        return centers.copy(), lower, upper
    scene_area = (right - left) * (bottom - top)
    hull_area = sum(body.area for body in bodies)
    alpha_geometry = bool(pairs) and isinstance(pairs[0], _AlphaPairGeometry)
    if not alpha_geometry and hull_area > scene_area:
        raise RuntimeError(
            f"Convex collision outlines occupy {hull_area / scene_area:.1%} "
            "of the available scene. Concave/transparent assets can need "
            "more collision space than their alpha area suggests."
        )
    sizes = [tuple(body.maximum - body.minimum + 2 * margin + 0.02) for body in bodies]
    cells = _partition_scene_grid((right - left, bottom - top), sizes)
    if cells is None:
        cells = _pack_shelf_cells((right - left, bottom - top), sizes)
    grid_seed = None
    if cells is not None:
        grid_seed = np.array(
            [
                [left + x + w / 2, top + y + h / 2] - (body.minimum + body.maximum) / 2
                for body, (x, y, w, h) in zip(bodies, cells)
            ]
        )
    if grid_seed is not None and all(_pair_penetration(pair, grid_seed) is None for pair in pairs):
        return grid_seed, lower, upper
    solved = None if alpha_geometry else _mixed_integer_placement(pairs, lower, upper, packing_time_limit)
    if solved is not None:
        return solved, lower, upper
    best_depth = float("inf")
    for attempt in range(attempts):
        if alpha_geometry and attempt == min(8, attempts - 1):
            solved = _mixed_integer_placement(pairs, lower, upper, packing_time_limit)
            if solved is not None and all(_pair_penetration(pair, solved) is None for pair in pairs):
                return solved, lower, upper
        if alpha_geometry and attempt == min(8, attempts - 1) and solved is not None:
            centers = solved.copy()
        else:
            centers = grid_seed.copy() if attempt == 0 and grid_seed is not None else rng.uniform(lower, upper)
        for iteration in range(600):
            total_depth = 0.0
            for index in rng.permutation(len(pairs)):
                pair = pairs[index]
                contact = _pair_penetration(pair, centers)
                if contact is None:
                    continue
                depth, normal = contact
                total_depth += depth
                correction = normal * (depth + 0.02) / 2
                centers[pair.i] -= correction
                centers[pair.j] += correction
                centers[pair.i] = np.clip(centers[pair.i], lower[pair.i], upper[pair.i])
                centers[pair.j] = np.clip(centers[pair.j], lower[pair.j], upper[pair.j])
            if total_depth == 0:
                return centers, lower, upper
            best_depth = min(best_depth, total_depth)
            if iteration and iteration % 100 == 0:
                centers = np.clip(centers + rng.normal(0, 2.0, centers.shape), lower, upper)
    raise RuntimeError(
        f"Could not pack {len(bodies)} collision outlines after {attempts} restarts "
        f"(hull occupancy {hull_area / scene_area:.1%}; best summed penetration "
        f"{best_depth:.3f}px). This does not prove the alpha masks cannot fit. "
        "Try another motion_seed, a larger placement_attempts/packing_time_limit_s "
        "budget, or reduce the object area/gap."
    )


def _pair_hit(pair: _PairGeometry, centers: np.ndarray, velocities: np.ndarray, horizon: float):
    """Continuous collision test for the configured pair geometry."""
    if isinstance(pair, _AlphaPairGeometry):
        return _alpha_hit(pair, centers, velocities, horizon)
    offset = (centers[pair.j] - centers[pair.i]) @ pair.axes.T
    speed = (velocities[pair.j] - velocities[pair.i]) @ pair.axes.T
    moving = np.abs(speed) > 1e-12
    if np.any(~moving & ((offset <= pair.lower + 1e-9) | (offset >= pair.upper - 1e-9))):
        return None
    entry = np.full(len(speed), -np.inf)
    leave = np.full(len(speed), np.inf)
    t0 = (pair.lower[moving] - offset[moving]) / speed[moving]
    t1 = (pair.upper[moving] - offset[moving]) / speed[moving]
    entry[moving], leave[moving] = np.minimum(t0, t1), np.maximum(t0, t1)
    axis = int(np.argmax(entry))
    enter, end = float(entry[axis]), float(np.min(leave))
    if not math.isfinite(enter) or end <= 1e-12 or enter > end - 1e-12 or enter > horizon:
        return None
    if enter < -1e-7:
        raise RuntimeError("Internal collision error: penetrating initial state.")
    normal = pair.axes[axis] * (-1 if speed[axis] > 0 else 1)
    return max(0.0, enter), normal


def _advance_motion(centers, velocities, lower, upper, pairs, duration, segments=None, rng=None):
    """Advance exactly to impact, reflect, then use the remaining time."""
    remaining = float(duration)
    distances = np.zeros(len(centers))
    collisions = 0
    zero_events = 0
    while remaining > 1e-14:
        earliest = remaining
        events = []
        for i, velocity in enumerate(velocities):
            for axis in (0, 1):
                component = float(velocity[axis])
                if abs(component) <= 1e-12:
                    continue
                boundary = upper[i, axis] if component > 0 else lower[i, axis]
                hit = (boundary - centers[i, axis]) / component
                if hit < -1e-7:
                    raise RuntimeError("Internal collision error: object outside scene.")
                hit = max(0.0, hit)
                if hit <= earliest:
                    if hit < earliest - 1e-12:
                        events = []
                    earliest = hit
                    normal = np.zeros(2)
                    normal[axis] = -1 if component > 0 else 1
                    events.append((i, None, normal))
        for pair in pairs:
            hit = _pair_hit(pair, centers, velocities, earliest)
            if hit is not None:
                when, normal = hit
                if when < earliest - 1e-12:
                    events = []
                earliest = when
                events.append((pair.i, pair.j, normal))
        if earliest > 0:
            if segments is not None:
                segments.append((earliest, centers.copy(), velocities.copy()))
            centers += velocities * earliest
            distances += np.linalg.norm(velocities, axis=1) * earliest
            remaining = max(0.0, remaining - earliest)
            zero_events = 0
        else:
            zero_events += 1
        if not events:
            break
        for i, j, normal in events:
            if j is None:
                component = float(velocities[i] @ normal)
                if component < 0:
                    velocities[i] -= 2 * component * normal
            else:
                component_i = float(velocities[i] @ normal)
                component_j = float(velocities[j] @ normal)
                if component_i > 0:
                    velocities[i] -= 2 * component_i * normal
                if component_j < 0:
                    velocities[j] -= 2 * component_j * normal
                if rng is not None:
                    for index, outward in ((i, -normal), (j, normal)):
                        angle = rng.uniform(-0.35, 0.35)
                        cosine, sine = math.cos(angle), math.sin(angle)
                        vx, vy = velocities[index]
                        candidate = np.array([cosine * vx - sine * vy, sine * vx + cosine * vy])
                        if float(candidate @ outward) > 0:
                            velocities[index] = candidate
            collisions += 1
        if zero_events > 64 or collisions > 100000:
            raise RuntimeError(
                "Motion jam at simultaneous contacts. Try another motion_seed "
                "or give the objects more clearance; speeds were not reduced."
            )
    return distances, collisions


def _simulate_joint_motion(
    bodies,
    speeds,
    scene_rect,
    n_frames,
    dt,
    seed=20260917,
    gap=1.5,
    wander_turn_rate=0.6,
    placement_attempts=80,
    packing_time_limit=15.0,
    sprites=None,
    initial_centers=None,
    allow_partial_visibility=False,
    per_object_visibility=None,
):
    rng = np.random.default_rng(seed)
    pairs = _alpha_pair_geometry(bodies, sprites, gap) if sprites is not None else _pair_geometry(bodies, gap)
    centers, lower, upper = _place_bodies(
        bodies,
        pairs,
        scene_rect,
        gap,
        rng,
        placement_attempts,
        packing_time_limit,
        initial_centers,
        allow_partial_visibility,
        per_object_visibility,
    )
    angles = rng.uniform(-math.pi, math.pi, len(bodies))
    speeds = np.asarray(speeds, dtype=np.float64)
    velocities = np.column_stack((np.cos(angles), np.sin(angles))) * speeds[:, None]
    positions = np.empty((n_frames, len(bodies), 2), dtype=np.float64)
    distances = np.zeros(len(bodies))
    max_interval_speed_error = np.zeros(len(bodies))
    collisions = 0
    steering_dt = 1 / 30
    next_steering = steering_dt
    angular_rates = rng.normal(0, wander_turn_rate, len(bodies))
    now = 0.0
    positions[0] = centers
    for frame in range(1, n_frames):
        target = frame * dt
        interval_distances = np.zeros(len(bodies))
        while now < target - 1e-12:
            end = min(target, next_steering)
            travelled, hits = _advance_motion(centers, velocities, lower, upper, pairs, end - now, rng=rng)
            distances += travelled
            interval_distances += travelled
            collisions += hits
            now = end
            if now >= next_steering - 1e-12:
                decay = math.exp(-steering_dt / 0.8)
                angular_rates = (
                    decay * angular_rates
                    + math.sqrt(1 - decay * decay) * rng.normal(0, wander_turn_rate, len(bodies))
                )
                angles = np.arctan2(velocities[:, 1], velocities[:, 0]) + angular_rates * steering_dt
                velocities = np.column_stack((np.cos(angles), np.sin(angles))) * speeds[:, None]
                next_steering += steering_dt
        positions[frame] = centers
        interval_speeds = interval_distances / dt
        max_interval_speed_error = np.maximum(max_interval_speed_error, np.abs(interval_speeds - speeds))
        if not np.allclose(interval_speeds, speeds, rtol=1e-8, atol=1e-7):
            raise RuntimeError(f"Object path speed differs from its requested speed at frame {frame}.")
    duration = (n_frames - 1) * dt
    expected = speeds * duration
    if not np.allclose(distances, expected, rtol=1e-8, atol=1e-7):
        raise RuntimeError("Internal collision error: path length did not match the requested speed.")
    stats = {
        "method": "continuous_alpha_collision" if sprites is not None else "continuous_convex_collision",
        "seed": int(seed),
        "collision_gap_px": float(gap),
        "collision_events": collisions,
        "duration_sec": duration,
        "path_length_px": distances.tolist(),
        "mean_path_speed_px_s": (distances / duration).tolist() if duration > 0 else [None] * len(speeds),
        "max_interval_speed_error_px_s": max_interval_speed_error.tolist() if duration > 0 else [None] * len(speeds),
        "convex_hull_area_px": [body.area for body in bodies],
        "center_span_px": np.ptp(positions, axis=0).tolist(),
        "alpha_support_area_px": [
            int(np.count_nonzero(np.asarray(sprite.getchannel("A")))) for sprite in sprites
        ] if sprites is not None else None,
        "speed_reference": "scene coordinates before global camera motion; distance along the path",
    }
    return positions, stats


def _prepare_objects(
    base_size: Tuple[int, int],
    specs: Sequence[MovingObjectSpec],
    lane_gap_px: int,
    area_reference_size: Tuple[int, int] | None = None,
    allow_partial_visibility_single: bool = False,
) -> list[PlannedMovingObject]:
    _ensure_positive_specs(specs)

    scene_area_fraction = sum(spec.area_fraction for spec in specs)
    if scene_area_fraction > 1.0:
        raise ValueError(
            "The requested total occupied area exceeds the scene area. "
            f"Sum of area fractions={scene_area_fraction:.6f}, limit=1.0."
        )

    area_size = area_reference_size if area_reference_size is not None else base_size
    base_w, base_h = base_size
    source_images: list[Image.Image] = []
    scales: list[float] = []
    scaled_sizes: list[Tuple[int, int]] = []

    strict_full_visibility = not (allow_partial_visibility_single and len(specs) == 1)

    for spec in specs:
        with Image.open(spec.object_rgba_path) as source:
            image = ensure_rgba(source).copy()
        bbox = image.getchannel("A").getbbox()
        if bbox is None:
            raise ValueError(f"Object {spec.object_rgba_path} is fully transparent.")
        image = image.crop(bbox)
        source_images.append(image)
        scale = scale_for_area_fraction(area_size, alpha_area(image), spec.area_fraction)
        scaled_size = (
            max(1, int(round(image.size[0] * scale))),
            max(1, int(round(image.size[1] * scale))),
        )
        if strict_full_visibility and scaled_size[0] > base_w:
            raise ValueError(
                f"Object {spec.object_rgba_path} is too wide to stay fully visible after scaling: "
                f"scaled width={scaled_size[0]}, scene width={base_w}. Reduce area_fraction or use a narrower asset."
            )
        if strict_full_visibility and scaled_size[1] > base_h:
            raise ValueError(
                f"Object {spec.object_rgba_path} is too tall to stay fully visible after scaling: "
                f"scaled height={scaled_size[1]}, scene height={base_h}. Reduce area_fraction or use a shorter asset."
            )
        scales.append(scale)
        scaled_sizes.append(scaled_size)

    planned: list[PlannedMovingObject] = []
    for index, spec in enumerate(specs):
        image = source_images[index]
        scale = scales[index]
        scaled_size = scaled_sizes[index]
        trajectory = _build_sampled_trajectory([(base_w / 2, base_h / 2)])
        planned.append(
            PlannedMovingObject(
                spec=spec,
                image=image,
                scale=scale,
                scaled_size=scaled_size,
                lane_y=base_h / 2.0,
                trajectory=trajectory,
            )
        )

    return planned


def _build_joint_trajectories(
    planned_objects,
    scene_rect,
    n_frames,
    dt,
    seed,
    gap,
    wander_turn_rate,
    placement_attempts,
    packing_time_limit,
    collision_geometry="alpha",
    initial_centers=None,
    allow_partial_visibility=False,
    per_object_visibility=None,
):
    sprites = [transform_object(p.image, angle_deg=0.0, scale=p.scale) for p in planned_objects]
    bodies = [_collision_body(sprite) for sprite in sprites]
    positions, stats = _simulate_joint_motion(
        bodies,
        [p.spec.speed_px_s for p in planned_objects],
        scene_rect,
        n_frames,
        dt,
        seed,
        gap,
        wander_turn_rate,
        placement_attempts,
        packing_time_limit,
        sprites if collision_geometry == "alpha" else None,
        initial_centers,
        allow_partial_visibility,
        per_object_visibility,
    )
    trajectories = [_build_sampled_trajectory(positions[:, i]) for i in range(len(bodies))]
    return trajectories, sprites, stats


def generate_multi_object_sequence(
    base_scene_path: Path | Sequence[Path],
    objects: Sequence[MovingObjectSpec],
    out_dir: Path,
    N: int = 1000,
    fps: int = 200,
    mask_mode: str = "binary",
    alpha_threshold: int = 1,
    save_color_frames: bool = False,
    save_npz: bool = False,
    meta_csv: bool = True,
    clear_output: bool = True,
    lane_gap_px: int = 8,
    save_union_mask: bool = True,
    motion_trace_path: Path | None = None,
    max_frames: int | None = None,
    output_crop_size: tuple[int, int] | None = (720, 960),
    run_event_camera: bool = False,
    event_tolerance: float = 0.2,
    event_refrac_period: float = 0.0001,
    event_sensor_sensitivity: float = 1,
    event_cropping_limits: list | None = None,
    planning_allowed_overlap: float = 0.0,
    allow_object_replacement_on_failure: bool = False,
    replacement_objects_dir: Path | None = None,
    max_pair_overlap_px: float | None = None,
    motion_seed: int = 20260917,
    collision_gap_px: float = 1.5,
    wander_turn_rate_rad_s: float = 0.6,
    placement_attempts: int = 80,
    packing_time_limit_s: float = 15.0,
    collision_geometry: str = "alpha",
    initial_centers_path: Path | None = None,
    allow_partial_visibility_single: bool = False,
    allow_partial_visibility: bool = False,
    single_object_area_tolerance: float | None = None,
    single_object_area_variation: float = 0.002,
    save_gifs: bool = True,
    single_object_arc: bool = False,
) -> None:
    """Generate independent-speed wandering objects with continuous collisions.

    Legacy overlap/replacement args remain accepted for CLI compatibility.
    The planner always enforces zero overlap in motion geometry.

    Args:
        allow_partial_visibility_single: Allow single objects to extend beyond frame bounds.
        allow_partial_visibility: Allow any number of objects to extend beyond frame bounds.
        single_object_area_tolerance: Optional absolute visible fraction tolerance;
            tries full visibility first, then fixed-scale bounded partial visibility.
        single_object_area_variation: Maximum temporal range of visible fraction
            when the single-object visible-area constraint is enabled.
        save_gifs: Save optional scene/mask GIFs in addition to individual frames.
        single_object_arc: Traverse a fully visible, constant-speed upper semicircle left-to-right.
    """
    _ = planning_allowed_overlap
    _ = allow_object_replacement_on_failure
    _ = replacement_objects_dir

    if collision_geometry not in {"alpha", "convex"}:
        raise ValueError("collision_geometry must be alpha or convex.")
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError("N must be a positive integer.")
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("fps must be positive and finite.")
    if not 0 <= alpha_threshold < 255:
        raise ValueError("alpha_threshold must be in [0, 255).")
    if not math.isfinite(collision_gap_px) or collision_gap_px < 0:
        raise ValueError("collision_gap_px must be finite and nonnegative.")
    if not math.isfinite(wander_turn_rate_rad_s) or wander_turn_rate_rad_s < 0:
        raise ValueError("wander_turn_rate_rad_s must be finite and nonnegative.")
    if not math.isfinite(packing_time_limit_s) or packing_time_limit_s <= 0:
        raise ValueError("packing_time_limit_s must be positive and finite.")
    if not isinstance(placement_attempts, int) or placement_attempts <= 0:
        raise ValueError("placement_attempts must be a positive integer.")
    if not isinstance(motion_seed, (int, np.integer)) or motion_seed < 0:
        raise ValueError("motion_seed must be a nonnegative integer.")
    if max_pair_overlap_px is not None and (
        not math.isfinite(float(max_pair_overlap_px)) or float(max_pair_overlap_px) < 0.0
    ):
        raise ValueError(
            f"max_pair_overlap_px must be >= 0 when provided. Got {max_pair_overlap_px!r}."
        )
    max_pair_overlap_px_value = None if max_pair_overlap_px is None else float(max_pair_overlap_px)

    if isinstance(base_scene_path, (str, Path)):
        frame_paths = [Path(base_scene_path)]
    else:
        frame_paths = list(base_scene_path)

    if not frame_paths:
        raise ValueError("No input frames were provided for the scene sequence.")

    motion_trace = None
    if motion_trace_path is not None:
        if fps != 200:
            raise ValueError("Eye-motion traces are sampled at 200 Hz; fps must be 200 (5 ms per sample).")
        if cv2 is None:
            raise ImportError("OpenCV is required for motion_trace_path (install opencv-python).")
        motion_trace = _load_motion_trace(Path(motion_trace_path))
        if len(frame_paths) != 1:
            raise ValueError(
                "motion_trace_path requires exactly one static background frame; "
                "the trace controls the generated sequence length."
            )
        N = len(motion_trace)

    if max_frames is not None:
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive when provided, got {max_frames}.")
        N = min(N, max_frames)

    if len(frame_paths) > 1:
        N = min(len(frame_paths), int(max_frames)) if max_frames is not None else len(frame_paths)
    if motion_trace is not None:
        motion_trace = motion_trace[:N]
        if len(motion_trace) == 0 or not np.isfinite(motion_trace).all():
            raise ValueError("Motion trace must be nonempty and finite.")

    target_scene_size = output_crop_size if output_crop_size is not None else None
    if target_scene_size is not None:
        if len(target_scene_size) != 2 or any(
            not isinstance(v, (int, np.integer)) or v <= 0 for v in target_scene_size
        ):
            raise ValueError("output_crop_size must contain positive integer (height, width).")
    with Image.open(frame_paths[0]) as base_img:
        base_L = ensure_grayscale_L(base_img.copy())
    scene_w, scene_h = base_L.size
    base_size = (scene_w, scene_h)

    for frame_path in frame_paths[1:]:
        with Image.open(frame_path) as current_img:
            current = ensure_grayscale_L(current_img.copy())
        if current.size != base_size:
            raise ValueError(
                f"Frame {frame_path} has size {current.size}, but the first frame has size {base_size}. "
                "All frames must share the same resolution."
            )

    active_specs = list(objects)
    area_reference_size = (target_scene_size[1], target_scene_size[0]) if target_scene_size else base_size
    planning_scene_rect = _compute_planning_scene_rect(base_size, target_scene_size, motion_trace)
    # Partial visibility: either allow_partial_visibility_single (for single objects) or allow_partial_visibility (for any)
    use_partial_visibility = allow_partial_visibility or (allow_partial_visibility_single and len(active_specs) == 1)
    # Extract per-object visibility flags from specs
    per_object_visibility_flags = [spec.allow_partial_visibility for spec in active_specs]
    if single_object_arc and (len(active_specs) != 1 or single_object_area_tolerance is not None):
        raise ValueError("Arc motion requires one object and no bounded-area placement mode")
    if single_object_area_tolerance is not None:
        if len(active_specs) != 1 or mask_mode != "binary":
            raise ValueError("Visible-area constraints require one object and binary masks.")
        if not (0 < single_object_area_tolerance < 1 and 0 <= single_object_area_variation < 1):
            raise ValueError("Invalid visible-area tolerances.")
    planned_objects = _prepare_objects(
        base_size,
        active_specs,
        lane_gap_px=lane_gap_px,
        area_reference_size=area_reference_size,
        allow_partial_visibility_single=use_partial_visibility or single_object_area_tolerance is not None,
    )

    dt = 1.0 / float(fps)
    if single_object_arc:
        from moving_objects.conference_motion import semicircle_centers
        rendered_sprites = [transform_object(p.image, angle_deg=0.0, scale=p.scale) for p in planned_objects]
        positions, motion_stats = semicircle_centers(
            planning_scene_rect, rendered_sprites[0].size, N, dt,
            active_specs[0].speed_px_s, collision_gap_px,
        )
        planned_trajectories = [_build_sampled_trajectory(positions.tolist())]
    elif single_object_area_tolerance is not None:
        from moving_objects.constant_visible_area import plan_single_visible_area
        planned_objects, planned_trajectories, rendered_sprites, motion_stats = plan_single_visible_area(
            base_size, target_scene_size, motion_trace, active_specs[0], N,
            dt=dt, seed=motion_seed, gap=collision_gap_px,
            wander=wander_turn_rate_rad_s, threshold=alpha_threshold,
            tolerance=single_object_area_tolerance, variation=single_object_area_variation,
        )
    else:
        planned_trajectories, rendered_sprites, motion_stats = _build_joint_trajectories(
            planned_objects,
            planning_scene_rect,
            N,
            dt,
            motion_seed,
            collision_gap_px,
            wander_turn_rate_rad_s,
            placement_attempts,
            packing_time_limit_s,
            collision_geometry,
            np.load(initial_centers_path, allow_pickle=False) if initial_centers_path is not None else None,
            use_partial_visibility,
            per_object_visibility_flags,
        )
    planned_objects = [
        PlannedMovingObject(
            p.spec,
            p.image,
            p.scale,
            rendered_sprites[i].size,
            p.lane_y,
            planned_trajectories[i],
        )
        for i, p in enumerate(planned_objects)
    ]

    frames_dir, object_mask_dirs, union_dir = _prepare_output_dirs(
        Path(out_dir), len(planned_objects), clear_output=clear_output
    )
    out_dir = Path(out_dir)
    output_w, output_h = (target_scene_size[1], target_scene_size[0]) if target_scene_size else base_size
    output_area = float(output_w * output_h)
    generated_frames = [] if run_event_camera else None

    frames_stack = None
    masks_stack = None
    union_stack = None
    if save_npz:
        frames_stack = np.zeros(
            (N, output_h, output_w, 3) if save_color_frames else (N, output_h, output_w),
            dtype=np.uint8,
        )
        masks_stack = np.zeros((N, len(planned_objects), output_h, output_w), dtype=np.uint8)
        if save_union_mask:
            union_stack = np.zeros((N, output_h, output_w), dtype=np.uint8)

    frame_summary: list[dict[str, float | int]] = []
    csv_path = out_dir / "meta.csv"
    if meta_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["frame", "t_sec"]
            for index in range(len(planned_objects)):
                header += [
                    f"x{index + 1}",
                    f"y{index + 1}",
                    f"scale{index + 1}",
                    f"speed{index + 1}_px_s",
                    f"area{index + 1}_px",
                ]
            header += ["total_area_px", "total_area_fraction"]
            writer.writerow(header)

    for frame_index in range(N):
        background_path = frame_paths[frame_index % len(frame_paths)]
        with Image.open(background_path) as background_img:
            background = ensure_grayscale_L(background_img.copy())
        if background.size != base_size:
            raise ValueError(
                f"Frame {background_path} has size {background.size}, but the scene size is {base_size}."
            )

        t = frame_index * dt
        object_masks: list[Image.Image] = []
        object_frames: list[tuple[Image.Image, Tuple[int, int]]] = []
        metadata_row: list[object] = [frame_index, f"{t:.6f}"]
        visible_area_total = 0.0

        for object_index, planned in enumerate(planned_objects):
            x, y, angle_deg, traj_scale = planned.trajectory(t, frame_index, N, base_size, planned.scaled_size)
            applied_scale = float(planned.scale * traj_scale)
            obj_t = rendered_sprites[object_index]
            obj_w, obj_h = obj_t.size
            x_left = int(round(x - obj_w / 2.0))
            y_top = int(round(y - obj_h / 2.0))
            xy = (x_left, y_top)

            mask = make_mask(base_size, obj_t, xy, mode=mask_mode, thresh=alpha_threshold)
            visible_area_px = float(np.asarray(mask, dtype=np.uint8).sum() / 255.0)
            visible_area_total += visible_area_px

            object_masks.append(mask)
            object_frames.append((obj_t, xy))
            metadata_row += [
                f"{x:.3f}",
                f"{y:.3f}",
                f"{applied_scale:.5f}",
                f"{planned.spec.speed_px_s:.3f}",
                f"{visible_area_px:.3f}",
            ]

        union_mask = _combine_many_masks(object_masks)
        total_area_fraction = visible_area_total / float(scene_w * scene_h)
        metadata_row += [f"{visible_area_total:.3f}", f"{total_area_fraction:.6f}"]

        if save_color_frames:
            base_rgba = background.convert("RGBA")
            for obj_t, xy in object_frames:
                base_rgba.alpha_composite(obj_t, dest=xy)
            frame_out = base_rgba.convert("RGB")
        else:
            frame_rgba = background.convert("RGBA")
            for obj_t, xy in object_frames:
                frame_rgba.alpha_composite(obj_t, dest=xy)
            frame_out = ImageOps.grayscale(frame_rgba)

        if motion_trace is not None:
            x_offset, y_offset = motion_trace[frame_index]
            frame_out = Image.fromarray(_shift_frame(np.asarray(frame_out), x_offset, y_offset, cv2.INTER_LINEAR))
            object_masks = [
                Image.fromarray(_shift_frame(np.asarray(mask), x_offset, y_offset, cv2.INTER_NEAREST))
                for mask in object_masks
            ]
            if union_mask is not None:
                union_mask = Image.fromarray(
                    _shift_frame(np.asarray(union_mask), x_offset, y_offset, cv2.INTER_NEAREST)
                )
            visible_area_total = sum(
                float(np.asarray(mask, dtype=np.uint8).sum() / 255.0) for mask in object_masks
            )
            total_area_fraction = visible_area_total / float(scene_w * scene_h)
            metadata_row[-2:] = [f"{visible_area_total:.3f}", f"{total_area_fraction:.6f}"]

        if target_scene_size is not None:
            frame_out = Image.fromarray(_crop_center_to_size(np.asarray(frame_out), target_scene_size))
            object_masks = [
                Image.fromarray(_crop_center_to_size(np.asarray(mask), target_scene_size))
                for mask in object_masks
            ]
            if union_mask is not None:
                union_mask = Image.fromarray(_crop_center_to_size(np.asarray(union_mask), target_scene_size))
            visible_area_total = sum(
                float(np.asarray(mask, dtype=np.uint8).sum() / 255.0) for mask in object_masks
            )
            total_area_fraction = visible_area_total / float(target_scene_size[1] * target_scene_size[0])
            metadata_row[-2:] = [f"{visible_area_total:.3f}", f"{total_area_fraction:.6f}"]

        final_areas = [float(np.asarray(mask, dtype=np.uint8).sum() / 255.0) for mask in object_masks]
        visible_area_total = sum(final_areas)
        total_area_fraction = visible_area_total / output_area
        for object_index, area in enumerate(final_areas):
            metadata_row[2 + 5 * object_index + 4] = f"{area:.3f}"
        metadata_row[-2:] = [f"{visible_area_total:.3f}", f"{total_area_fraction:.6f}"]

        frame_path = frames_dir / f"frame_{frame_index:04d}.png"
        frame_out.save(frame_path)

        if generated_frames is not None:
            generated_frames.append(np.asarray(frame_out.convert("RGB"), dtype=np.uint8))

        for object_index, mask in enumerate(object_masks):
            mask_path = object_mask_dirs[object_index] / f"mask_{frame_index:04d}.png"
            mask.save(mask_path)

        if max_pair_overlap_px_value is not None and len(object_masks) >= 2:
            mask_arrays = [np.asarray(mask, dtype=np.uint8) > 0 for mask in object_masks]
            for i in range(len(mask_arrays)):
                for j in range(i + 1, len(mask_arrays)):
                    overlap_px = float(np.count_nonzero(mask_arrays[i] & mask_arrays[j]))
                    if overlap_px > max_pair_overlap_px_value + 1e-6:
                        raise RuntimeError(
                            f"Pair overlap limit violated at frame {frame_index}: "
                            f"objects ({i + 1},{j + 1}) overlap by {overlap_px:.3f}px "
                            f"(limit={max_pair_overlap_px_value:.3f}px)."
                        )

        if save_union_mask and union_mask is not None:
            union_path = union_dir / f"mask_union_{frame_index:04d}.png"
            union_mask.save(union_path)

        union_area_px = 0.0
        if union_mask is not None:
            union_area_px = float(np.asarray(union_mask, dtype=np.uint8).sum() / 255.0)
        overlap_area_px = max(0.0, visible_area_total - union_area_px)

        if overlap_area_px > 1e-6:
            raise RuntimeError(
                f"Strict no-overlap violation at frame {frame_index}: "
                f"overlap_area_px={overlap_area_px:.3f}."
            )

        frame_summary.append(
            {
                "frame": frame_index,
                "t_sec": float(t),
                "total_area_px": float(visible_area_total),
                "total_area_fraction": float(total_area_fraction),
                "union_area_px": float(union_area_px),
                "union_area_fraction": float(union_area_px / output_area),
                "overlap_area_px": float(overlap_area_px),
                "overlap_area_fraction": float(overlap_area_px / output_area),
            }
        )

        if save_npz and frames_stack is not None and masks_stack is not None:
            frames_stack[frame_index] = np.asarray(frame_out, dtype=np.uint8)
            for object_index, mask in enumerate(object_masks):
                masks_stack[frame_index, object_index] = np.asarray(mask, dtype=np.uint8)
            if save_union_mask and union_stack is not None and union_mask is not None:
                union_stack[frame_index] = np.asarray(union_mask, dtype=np.uint8)

        if meta_csv:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(metadata_row)

    if single_object_area_tolerance is not None:
        visible = np.array([row["union_area_fraction"] for row in frame_summary])
        target = active_specs[0].area_fraction
        if np.max(np.abs(visible - target)) > single_object_area_tolerance + 1e-12:
            raise RuntimeError("Rendered masks exceed the requested visible-area tolerance.")
        if np.ptp(visible) > single_object_area_variation + 1e-12:
            raise RuntimeError("Rendered masks exceed the visible-area temporal variation limit.")
        motion_stats["visible_area_constraint"].update(
            measured_min_fraction=float(visible.min()),
            measured_max_fraction=float(visible.max()),
        )
    summary_path = out_dir / "sequence_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "fps": fps,
                "n_frames": N,
                "object_count": len(planned_objects),
                "planning_allowed_overlap": float(planning_allowed_overlap),
                "effective_allowed_overlap": 0.0,
                "collision_geometry": collision_geometry,
                "initial_centers_path": str(initial_centers_path) if initial_centers_path is not None else None,
                "motion": motion_stats,
                "allow_object_replacement_on_failure": bool(allow_object_replacement_on_failure),
                "max_pair_overlap_px": max_pair_overlap_px_value,
                "allow_partial_visibility_single": bool(allow_partial_visibility_single),
                "allow_partial_visibility": bool(allow_partial_visibility),
                "objects": [
                    {
                        "object_rgba_path": str(spec.object_rgba_path),
                        "area_fraction": float(spec.area_fraction),
                        "speed_px_s": float(spec.speed_px_s),
                    }
                    for spec in active_specs
                ],
                "frames": frame_summary,
            },
            indent=2,
        )
    )

    if save_npz and frames_stack is not None and masks_stack is not None:
        npz_path = out_dir / "sequence.npz"
        save_dict = {
            "frames": frames_stack,
            "masks": masks_stack,
            "fps": fps,
        }
        if save_union_mask and union_stack is not None:
            save_dict["mask_union"] = union_stack
        np.savez_compressed(npz_path, **save_dict)

    try:
        if save_gifs and create_gif is not None:
            duration_ms = max(10, int(1000.0 / float(fps)))
            gif_path = out_dir / "frames_animation.gif"
            create_gif(str(frames_dir), str(gif_path), duration=duration_ms)
            if save_union_mask:
                union_gif_path = out_dir / "masks_union_animation.gif"
                create_gif(str(union_dir), str(union_gif_path), duration=duration_ms)
    except Exception:
        pass

    if run_event_camera:
        from events_camera_simulator.conference_stream import generate_events_streaming
        if event_cropping_limits is not None and any(v is not None for v in event_cropping_limits):
            raise ValueError("Crop frames and masks together before event generation")
        generate_events_streaming(out_dir, dt=dt, tolerance=event_tolerance,
                                  refrac_period=event_refrac_period,
                                  sensor_sensitivity=event_sensor_sensitivity)
