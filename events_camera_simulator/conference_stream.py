"""Bounded-memory version of the existing PNG-quantized event simulator/export."""
import json
from pathlib import Path
import cv2
import numpy as np
from bimvee.exportIitYarp import encodeEvents24Bit
from common_utils.memory import release_pages
from events_camera_simulator.event_camera import EventCameraSimulator


def generate_events_streaming(dataset, dt=0.005, seed=20261012, tolerance=0.2, refrac_period=0.0001, sensor_sensitivity=1):
    dataset = Path(dataset)
    camera = EventCameraSimulator(str(dataset / 'frames'), tolerance, refrac_period, sensor_sensitivity, [None]*4)
    paths = sorted((dataset / 'frames').glob('*.png'))
    event_dir = dataset / 'events_frames'
    event_dir.mkdir(exist_ok=True)
    (dataset / 'events_file').mkdir(exist_ok=True)
    if len(paths) < 3:
        raise ValueError('At least three frames are required')
    previous = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    last = np.zeros(previous.shape)
    rng = np.random.RandomState(seed)
    pending = np.empty((0, 4), dtype=np.float64)  # t, x, y, polarity
    origin = None
    event_count = 0
    with (dataset / 'events_file/binaryevents.log').open('wb') as stream:
        for i, path in enumerate(paths[1:], 1):
            current = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            image, last = camera.process_image(current, previous, i*dt, last)
            previous = current
            quantized = (image * 255).astype(np.uint8)
            cv2.imwrite(str(event_dir / f'{i-1:06d}_ec.png'), quantized)
            (event_dir / f'{i-1:06d}_ec.json').write_text(json.dumps({'timestamp': i*dt}))
            # Preserve frame2events' first-transition skip and PNG quantization.
            if i == 1:
                continue
            counts = np.round(quantized.astype(np.float32) / 255 / 0.05 - 10).astype(int)
            y, x = np.nonzero(counts)
            n = np.abs(counts[y, x])
            starts = np.repeat(np.cumsum(n) - n, n)
            ordinal = np.arange(int(n.sum())) - starts
            t = (i-1)*dt + ordinal * (dt / np.repeat(n, n)) + (rng.random(len(ordinal))-.5)/250
            batch = np.column_stack((t, np.repeat(x, n), np.repeat(y, n), np.repeat(counts[y, x] > 0, n)))
            pending = np.concatenate((pending, batch))
            pending = pending[np.argsort(pending[:, 0], kind='stable')]
            # Future intervals cannot have events earlier than current time - jitter.
            cut = len(pending) if i == len(paths)-1 else np.searchsorted(pending[:, 0], i*dt-0.002)
            ready, pending = pending[:cut], pending[cut:]
            if len(ready):
                if origin is None:
                    origin = float(ready[0, 0])
                encoded = encodeEvents24Bit(ready[:, 0]-origin, ready[:, 1], ready[:, 2], ready[:, 3].astype(bool))
                encoded.astype(np.uint32).tofile(stream)
                event_count += len(ready)
            if i % 100 == 0:
                print(f'Events: {i}/{len(paths)-1}', flush=True)
    (dataset / 'events_file/metadata.json').write_text(json.dumps(dict(
        event_seed=seed, event_count=event_count, time_origin_scene_s=origin,
        note='Legacy PNG quantization, first-transition skip, uniform +/-2 ms jitter, first-event time origin.'
    ), indent=2)+'\n')


def bin_events_to_disk(events_file, out_path, frames, shape=(720, 960), dt=0.005):
    """Decode this simulator's 24-bit address stream without a full event table."""
    h, w = shape
    output = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.uint32, shape=(frames, 2, h, w))
    # w+ creates a new zero-filled file; avoid touching the whole array in RAM.
    bin_us = round(dt*1e6)
    if abs(bin_us/1e6-dt) > 1e-12:
        raise ValueError('dt must be an integer number of microseconds')
    with open(events_file, 'rb') as stream:
        while True:
            words = np.fromfile(stream, dtype=np.uint32, count=2*65536)
            if not len(words):
                break
            if len(words) % 2:
                raise ValueError('Incomplete binary event')
            words = words.reshape(-1, 2)
            ts, address = words[:, 0], words[:, 1]
            idx = (ts // bin_us).astype(np.int64)
            x = ((address >> 1) & 2047).astype(np.int64)
            y = ((address >> 12) & 1023).astype(np.int64)
            pol = (1-(address & 1)).astype(np.int64)
            if np.any(idx >= frames) or np.any(x >= w) or np.any(y >= h):
                raise ValueError('Event outside requested sequence or sensor')
            np.add.at(output, (idx, pol, y, x), 1)
            release_pages(output)
    return output
