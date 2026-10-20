import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.io import savemat

from common_utils.utils import compute_object_selectivity
from common_utils.statistics import paired_sign_flip, holm
from eye_movements.doves import load_doves, render_background
from events_camera_simulator.streaming import generate_events_streaming, bin_events_to_disk
from scripts import load_doves as load_step, generate_motion, generate_stimuli, generate_events, run_retina, characterize, analyze
from moving_objects.add_multi_moving_objects import MovingObjectSpec


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tmp_path = Path(folder.name)

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

    def test_independent_steps_for_all_conditions(self):
        root = self.tmp_path
        rng = np.random.default_rng(42)
        image_path = root / 'scene.iml'
        rng.uniform(500, 60000, (64, 80)).astype('>u2').tofile(image_path)
        eye = np.empty((1, 1), dtype=object)
        eye[0, 0] = np.cumsum(rng.normal(0, .15, (2, 60)), axis=1)
        recording = root / 'scene.mat'
        savemat(recording, {'eye_data': eye})
        sprite = np.zeros((10, 10, 4), np.uint8)
        sprite[1:9, 1:9] = [255, 255, 255, 255]
        object_path = root / 'object.png'
        Image.fromarray(sprite).save(object_path)
        for condition in ['doves', 'circular', 'wobbly']:
            previous = None
            for repeat in range(2):
                out = root / f'{condition}_{repeat}'
                load_step.run(image_path, recording, 0, out, image_shape=(64, 80))
                generate_motion.run(out, condition, 12)
                generate_stimuli.run(out, [MovingObjectSpec(object_path, .04, 25)], (48, 64))
                generate_events.run(out)
                run_retina.run(out, ksize=5)
                characterize.run(out)
                spikes = np.load(out / 'spikes.npy')
                self.assertEqual(spikes.shape, (12, 48, 64))
                self.assertTrue(np.isin(spikes, [0, 1]).all())
                result = json.loads((out / 'result.json').read_text())
                self.assertGreater(result['raw_events']['N_tot'], 0)
                self.assertEqual(result['oms_model']['N_tot'],
                                 result['oms_model']['N_obj'] + result['oms_model']['N_bg'])
                events = (out / 'events_file/binaryevents.log').read_bytes()
                if previous is not None:
                    self.assertEqual(previous[0], events)
                    np.testing.assert_array_equal(previous[1], spikes)
                previous = (events, spikes)
                if condition == 'doves' and repeat == 0:
                    background = root / 'background'
                    load_step.run(image_path, recording, 0, background, image_shape=(64, 80))
                    generate_motion.run(background, 'doves', 12)
                    generate_stimuli.run(background, [], (48, 64))
                    generate_events.run(background)
                    run_retina.run(background, ksize=5)
                    characterize.run(background)
                    result = json.loads((root / 'background/result.json').read_text())
                    self.assertIsNone(result['oms_model'])

        analyze.run(root)
        self.assertTrue((root / 'summary.csv').is_file())

    def test_background_translation_and_input_checks(self):
        image = np.arange(80, dtype=np.uint8).reshape(8, 10)
        render_background(image, np.array([[0, 0], [1, 0]]), self.tmp_path / 'frames', (6, 8))
        first = cv2.imread(str(self.tmp_path / 'frames/frame_000000.png'), 0)
        second = cv2.imread(str(self.tmp_path / 'frames/frame_000001.png'), 0)
        np.testing.assert_array_equal(first, image[1:7, 1:9])
        np.testing.assert_array_equal(second, image[1:7, :8])
        with self.assertRaises(FileExistsError):
            render_background(image, np.zeros((3, 2)), self.tmp_path / 'frames', (6, 8))
        with self.assertRaises(ValueError):
            render_background(image, np.array([[np.nan, 0]]), self.tmp_path / 'other', (6, 8))
        with self.assertRaises(FileNotFoundError):
            load_doves(self.tmp_path / 'missing', self.tmp_path / 'missing.mat', 0)
