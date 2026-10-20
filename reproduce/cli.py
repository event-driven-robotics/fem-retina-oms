"""Manifest-driven experiments; paths are relative to the manifest, not the shell."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat

from eye_movements.utils import (
    detect_microsaccades_engbert, stitch_by_increments, circular,
    sine_velocity_trajectory,
)
from events_camera_simulator.conference_stream import generate_events_streaming, bin_events_to_disk
from retina_model.oms_retina import normalize_bipolar_responces, run_oms_retina_
from moving_objects.add_multi_moving_objects import MovingObjectSpec, generate_multi_object_sequence
from common_utils.utils import compute_object_selectivity


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_signature():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): digest(p)
            for package in ["reproduce", "eye_movements", "events_camera_simulator",
                            "retina_model", "moving_objects", "common_utils"]
            for p in sorted((root / package).glob("*.py"))}


def resolve(base, value):
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path):
    path = Path(path).resolve()
    c = json.loads(path.read_text())
    if c.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    c["doves_root"] = str(resolve(path.parent, c["doves_root"]))
    c["output_root"] = str(resolve(path.parent, c["output_root"]))
    if c.get("fps") != 200:
        raise ValueError("The preserved DOVES pipeline requires fps=200")
    ids = set()
    for e in c["experiments"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", e["id"]) or e["id"] in ids:
            raise ValueError("Experiment ids must be unique, safe directory names")
        ids.add(e["id"])
        if Path(e["scene"]).name != e["scene"]:
            raise ValueError("scene must be a filename, not a path")
        if not isinstance(e["subject"], int) or e["subject"] < 0:
            raise ValueError("subject must be a zero-based nonnegative integer")
        if e["condition"] not in {"doves", "circular", "wobbly"}:
            raise ValueError("Unknown eye-motion condition")
        if not isinstance(e["frames"], int) or e["frames"] < 3:
            raise ValueError("frames must be an integer >= 3")
        if e.get("trace_extension", "error") not in {"error", "pingpong"}:
            raise ValueError("trace_extension must be error or pingpong")
        for obj in e.get("objects", []):
            obj["image"] = str(resolve(path.parent, obj["image"]))
            if not 0 < obj["area_fraction"] < 1 or obj["speed_px_s"] < 0:
                raise ValueError("Invalid object size or speed")
    if c["analysis"]["region"] not in {"exact", "vicinity"}:
        raise ValueError("analysis.region must be exact or vicinity")
    if not 0 <= c["analysis"]["min_visibility"] <= 1:
        raise ValueError("min_visibility must be in [0, 1]")
    for name in ["tau_bp", "tau_am", "tau_var", "eps", "tau_syn", "tau_mem"]:
        if c["retina"][name] <= 0:
            raise ValueError(f"retina.{name} must be positive")
    if c["retina"]["ksize"] < 1:
        raise ValueError("retina.ksize must be positive")
    for name in ["tolerance", "refrac_period", "sensor_sensitivity"]:
        if c["events"][name] <= 0:
            raise ValueError(f"events.{name} must be positive")
    return c


def input_paths(c, e):
    root = Path(c["doves_root"])
    return [root / "Images" / e["scene"],
            root / "Fixations" / (e["scene"] + ".mat"),
            *[Path(o["image"]) for o in e.get("objects", [])]]


def prepare(c, e):
    """Read raw DOVES files and retain the original detector/increment stitching."""
    image_path, fixation_path, *_ = input_paths(c, e)
    h, w = c["image_shape"]
    pixels = np.fromfile(image_path, dtype=">u2")
    if pixels.size != h * w:
        raise ValueError(f"{image_path}: expected {h*w} big-endian uint16 pixels")
    image = pixels.reshape(h, w).astype(np.float32)
    if image.max() <= 0:
        raise ValueError("DOVES image is empty")
    image /= image.max()
    if c["contrast_boost"]:
        image = np.clip(image * 1.3 + .05, 0, 1)
    image = (image * 255).astype(np.uint8)
    data = loadmat(fixation_path)["eye_data"]
    if e["subject"] >= data.shape[1]:
        raise ValueError(f"Subject {e['subject']} is outside eye_data (zero-based indexing)")
    eye = np.asarray(data[0, e["subject"]])
    if eye.ndim != 2 or eye.shape[0] < 2 or eye.shape[1] < 3:
        raise ValueError("eye_data entry must have at least two rows and three samples")
    if not np.isfinite(eye[:2]).all():
        raise ValueError("Non-finite gaze samples; specify the paper's missing-data policy before running")
    n = e["frames"]
    motion = e["motion"]
    info = {}
    if e["condition"] == "doves":
        detected = detect_microsaccades_engbert(eye[0], eye[1], fs=c["fps"], **c["saccades"])
        x, y, _ = stitch_by_increments(eye[0], eye[1], detected["jitter_mask"], c["fps"])
        trace = np.column_stack((x, y))
        info = {"saccades": detected["saccades"], "saccade_mask": detected["saccade_mask"],
                "retained_trace_samples": len(trace)}
        if len(trace) < 2:
            raise ValueError("Fewer than two samples remain after saccade filtering")
        indices = np.arange(n)
        if n > len(trace):
            if e.get("trace_extension", "error") != "pingpong":
                raise ValueError(f"Requested {n} frames, only {len(trace)} retained samples; shorten or explicitly choose pingpong")
            indices %= 2 * (len(trace) - 1)
            indices = np.where(indices < len(trace), indices, 2*(len(trace)-1)-indices)
        info["stitched_trace_indices"] = indices
        trace = trace[indices]
    elif e["condition"] == "circular":
        if motion["radius"] <= 0:
            raise ValueError("Circular radius must be positive")
        trace = np.column_stack(circular(n, **motion))
    else:
        trace = np.column_stack(sine_velocity_trajectory(n, **motion))
    return image, trace, info


def validate(c):
    if not c["experiments"]:
        raise ValueError("Add subject-scene experiments to the manifest first")
    for e in c["experiments"]:
        for p in input_paths(c, e):
            if not p.is_file():
                raise FileNotFoundError(p)
        image, _, _ = prepare(c, e)
        h, w = c["crop_shape"]
        if h <= 0 or w <= 0 or h > image.shape[0] or w > image.shape[1] or h > 1024 or w > 2048:
            raise ValueError("crop_shape must fit the image and the event address format")
        for obj in e.get("objects", []):
            with Image.open(obj["image"]) as sprite:
                if sprite.mode != "RGBA" or not np.any(np.asarray(sprite.getchannel("A"))):
                    raise ValueError("Object images must be RGBA with a nonempty alpha mask")
        print(f"Validated {e['id']}")


def render(c, e, out):
    image, trace, filtering = prepare(c, e)
    Image.fromarray(image).save(out / "background.png")
    np.save(out / "eye_trace.npy", trace)
    save_json(out / "eye_filtering.json", filtering)
    if e.get("objects"):
        specs = [MovingObjectSpec(Path(o["image"]), o["area_fraction"], o["speed_px_s"])
                 for o in e["objects"]]
        generate_multi_object_sequence(
            [out / "background.png"], specs, out, N=e["frames"], fps=c["fps"],
            clear_output=False, motion_trace_path=out / "eye_trace.npy",
            output_crop_size=tuple(c["crop_shape"]), run_event_camera=False,
            save_gifs=False, motion_seed=e["motion_seed"],
        )
    else:
        (out / "frames").mkdir()
        h, w = c["crop_shape"]
        y0, x0 = (image.shape[0]-h)//2, (image.shape[1]-w)//2
        for i, (x, y) in enumerate(trace):
            frame = cv2.warpAffine(image, np.float32([[1, 0, x], [0, 1, y]]), image.shape[::-1])
            if not cv2.imwrite(str(out / "frames" / f"frame_{i:06d}.png"), frame[y0:y0+h, x0:x0+w]):
                raise OSError("Failed to write frame")


def run_one(c, e, resume=False):
    out = Path(c["output_root"]) / e["id"]
    inputs = {str(p): digest(p) for p in input_paths(c, e)}
    settings = {k: v for k, v in c.items() if k not in {"experiments", "output_root"}}
    signature = {"settings": settings, "experiment": e, "inputs_sha256": inputs,
                 "source_sha256": source_signature()}
    if out.exists():
        if resume and (out / "complete.json").is_file():
            if json.loads((out / "manifest.json").read_text()) != signature:
                raise ValueError(f"{out}: configuration or inputs changed; choose a new output directory")
            for name, expected in json.loads((out / "complete.json").read_text())["artifacts"].items():
                if not (out / name).is_file() or digest(out / name) != expected:
                    raise ValueError(f"{out}: missing or changed artifact {name}")
            print(f"Already complete: {e['id']}")
            return
        raise FileExistsError(f"{out} exists; use --resume for completed runs or choose a new output_root")
    out.mkdir(parents=True)
    save_json(out / "manifest.json", signature)
    versions = {}
    for package in ["numpy", "scipy", "opencv-python", "Pillow", "bimvee"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    save_json(out / "environment.json", versions)
    render(c, e, out)
    dt = 1 / c["fps"]
    generate_events_streaming(out, dt=dt, seed=e["event_seed"], **c["events"])
    binned = bin_events_to_disk(out / "events_file/binaryevents.log", out / "binned_events.npy",
                               e["frames"], shape=tuple(c["crop_shape"]), dt=dt)
    r = c["retina"]
    normalized = normalize_bipolar_responces(
        binned, dt=dt, **{k: r[k] for k in ["ksize", "tau_bp", "tau_am", "tau_var", "eps"]},
        return_only_normalized=True, output_path=out / "normalized.npy")
    spikes = run_oms_retina_(
        normalized, dt=dt, **{k: r[k] for k in ["tau_syn", "tau_mem", "threshold"]},
        return_spikes_only=True, output_path=out / "spikes.npy")
    result = {"experiment": e, "num_frames": e["frames"],
              "total_spikes": int(spikes.sum()), "oms_model": None, "raw_events": None}
    if e.get("objects"):
        paths = sorted((out / "masks_union").glob("*.png"))
        if len(paths) != len(spikes):
            raise ValueError("Mask and spike frame counts differ")
        masks = np.stack([np.asarray(Image.open(p)) for p in paths])
        kwargs = dict(ksize=r["ksize"], min_visibility=c["analysis"]["min_visibility"],
                      exclude_edge_touch=c["analysis"]["exclude_edge_touch"], return_more=True,
                      vicinity_method="exact" if c["analysis"]["region"] == "exact" else "box")
        for label, response in [("oms_model", spikes), ("raw_events", binned.sum(axis=1))]:
            value = compute_object_selectivity(response, masks, **kwargs)
            result[label] = value[1] if isinstance(value, tuple) else value
    save_json(out / "result.json", result)
    artifacts = ["result.json", "spikes.npy", "eye_trace.npy", "events_file/binaryevents.log",
                 "events_file/metadata.json", "binned_events.npy", "normalized.npy"]
    save_json(out / "complete.json", {"artifacts": {p: digest(out / p) for p in artifacts}})
    print(f"Completed {e['id']}: {out}")


def make_demo(target):
    """Synthetic inputs follow DOVES file structure; no downloaded data required."""
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError(f"Choose a new directory: {target}")
    (target / "Doves/Images").mkdir(parents=True)
    (target / "Doves/Fixations").mkdir()
    rng = np.random.default_rng(42)
    (rng.uniform(500, 60000, (64, 80)).astype(">u2")).tofile(target / "Doves/Images/synthetic.iml")
    eye = np.empty((1, 1), dtype=object)
    eye[0, 0] = np.cumsum(rng.normal(0, .15, (2, 60)), axis=1)
    savemat(target / "Doves/Fixations/synthetic.iml.mat", {"eye_data": eye})
    sprite = np.zeros((10, 10, 4), dtype=np.uint8)
    sprite[1:9, 1:9] = [255, 255, 255, 255]
    Image.fromarray(sprite).save(target / "object.png")
    template = Path(__file__).with_name("example.json")
    c = json.loads(template.read_text())
    c.update(doves_root="Doves", output_root="outputs", image_shape=[64, 80], crop_shape=[48, 64])
    c["retina"]["ksize"] = 5
    experiments = []
    for condition in ["doves", "circular", "wobbly"]:
        e = json.loads(json.dumps(c["experiments"][0]))
        e.update(id=condition, scene="synthetic.iml", subject=0, condition=condition, frames=12)
        e["objects"][0].update(image="object.png", speed_px_s=25, area_fraction=.04)
        e["motion"] = ({} if condition == "doves" else
                       {"radius": 2, "pixels_per_frame": .5} if condition == "circular" else
                       {"v_base": .5, "v_amp": .3, "velocity_cycles": 1, "orientation_cycles": 1, "theta0": math.pi/2})
        experiments.append(e)
    c["experiments"] = experiments
    save_json(target / "demo.json", c)
    return target / "demo.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["validate", "run"]:
        p = sub.add_parser(name)
        p.add_argument("config", type=Path)
        if name == "run":
            p.add_argument("--resume", action="store_true")
    p = sub.add_parser("demo")
    p.add_argument("directory", type=Path)
    p = sub.add_parser("analyze")
    p.add_argument("config", type=Path)
    args = parser.parse_args()
    if args.command == "demo":
        path = make_demo(args.directory)
        c = load_config(path)
        validate(c)
        for e in c["experiments"]:
            run_one(c, e)
        print(f"Demo configuration: {path}")
    else:
        c = load_config(args.config)
        if args.command == "analyze":
            from .analysis import summarize
            summarize(c)
        else:
            validate(c)
            if args.command == "run":
                for e in c["experiments"]:
                    run_one(c, e, resume=args.resume)


if __name__ == "__main__":
    main()
