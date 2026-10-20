import copy
import json
from pathlib import Path

import cv2
import numpy as np
import unittest
import tempfile

from reproduce.cli import make_demo, load_config, validate, run_one, digest, prepare
from reproduce.analysis import summarize, paired_sign_flip, holm
from common_utils.utils import compute_object_selectivity
from events_camera_simulator.conference_stream import generate_events_streaming, bin_events_to_disk


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tmp_path = Path(folder.name)

    def test_three_conditions_reproduce_and_resume(self):
        tmp_path = self.tmp_path
        config = load_config(make_demo(tmp_path / "demo"))
        validate(config)
        for e in config["experiments"]:
            run_one(config, e)
            run_one(config, e, resume=True)
            out = Path(config["output_root"]) / e["id"]
            spikes = np.load(out / "spikes.npy")
            assert spikes.shape == (12, 48, 64)
            assert np.isin(spikes, [0, 1]).all()
            result = json.loads((out / "result.json").read_text())
            assert result["raw_events"]["N_tot"] > 0
            assert result["oms_model"]["N_obj"] + result["oms_model"]["N_bg"] == result["oms_model"]["N_tot"]
        summarize(config)
        other = copy.deepcopy(config)
        other["output_root"] = str(tmp_path / "repeat")
        for e in other["experiments"]:
            run_one(other, e)
            for name in ["eye_trace.npy", "spikes.npy", "events_file/binaryevents.log"]:
                assert digest(Path(config["output_root"]) / e["id"] / name) == digest(Path(other["output_root"]) / e["id"] / name)
        changed = copy.deepcopy(config)
        changed["retina"]["threshold"] = .4
        with self.assertRaisesRegex(ValueError, "changed"):
            run_one(changed, changed["experiments"][0], resume=True)
        out = Path(config["output_root"]) / config["experiments"][0]["id"]
        (out / "spikes.npy").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "artifact"):
            run_one(config, config["experiments"][0], resume=True)


    def test_missing_data_and_trace_policy(self):
        tmp_path = self.tmp_path
        config = load_config(make_demo(tmp_path / "demo"))
        e = config["experiments"][0]
        e["frames"] = 1000
        with self.assertRaisesRegex(ValueError, "retained"):
            prepare(config, e)
        e["trace_extension"] = "pingpong"
        _, trace, info = prepare(config, e)
        assert trace.shape == (1000, 2)
        assert np.all(np.abs(np.diff(info["stitched_trace_indices"])) == 1)
        (Path(config["doves_root"]) / "Images/synthetic.iml").unlink()
        with self.assertRaises(FileNotFoundError):
            validate(config)


    def test_zero_event_sequence(self):
        tmp_path = self.tmp_path
        (tmp_path / "frames").mkdir()
        for i in range(4):
            cv2.imwrite(str(tmp_path / "frames" / f"{i}.png"), np.full((8, 10), 100, np.uint8))
        generate_events_streaming(tmp_path)
        counts = bin_events_to_disk(tmp_path / "events_file/binaryevents.log", tmp_path / "counts.npy", 4, shape=(8, 10))
        assert counts.shape == (4, 2, 8, 10)
        assert counts.sum() == 0
        assert json.loads((tmp_path / "events_file/metadata.json").read_text())["event_count"] == 0


    def test_known_selectivity_and_undefined_response(self):
        mask = np.zeros((2, 9, 9), np.uint8)
        mask[:, 4, 4] = 255
        spikes = np.zeros_like(mask, dtype=float)
        spikes[:, 4, 4] = 2
        spikes[:, 4, 5] = 3
        spikes[:, 0, 0] = 5
        exact = compute_object_selectivity(spikes, mask, ksize=3, vicinity_method="exact", return_more=True)
        vicinity = compute_object_selectivity(spikes, mask, ksize=3, return_more=True)
        assert exact["p"] == .2
        assert vicinity["p"] == .5
        assert exact["N_tot"] == vicinity["N_tot"] == 20
        assert np.isnan(compute_object_selectivity(np.zeros_like(spikes), mask))


    def test_statistical_calculation(self):
        result = paired_sign_flip([1, 1, 1, 1])
        assert result["p_value"] == 2 / 16
        assert result["mean_difference_a_minus_b"] == 1
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
        with self.assertRaisesRegex(ValueError, "two"):
            paired_sign_flip([1])
