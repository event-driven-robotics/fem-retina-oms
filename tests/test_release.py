import json
import re
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

    def test_readme_stages_for_all_conditions(self):
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
        readme = (Path(__file__).resolve().parents[1] / 'README.md').read_text()
        blocks = re.findall(r'```python\n(.*?)```', readme, re.S)
        self.assertEqual(len(blocks), 7)
        original = '\n'.join(blocks[:6])
        for condition in ['doves', 'circular', 'wobbly']:
            previous = None
            for repeat in range(2):
                out = root / f'{condition}_{repeat}'
                code = original
                replacements = {
                    "'/path/to/Doves/Images/imk01366.iml'": repr(str(image_path)),
                    "'/path/to/Doves/Fixations/imk01366.iml.mat'": repr(str(recording)),
                    "'/path/to/objects/object.png'": repr(str(object_path)),
                    "'results/example_doves_subject0'": repr(str(out)),
                    "condition = 'doves'": f'condition = {condition!r}',
                    'contrast_boost=False)': 'contrast_boost=False, image_shape=(64, 80))',
                    'crop_shape = (720, 960)': 'crop_shape = (48, 64)',
                    'n_frames = 100': 'n_frames = 12',
                    'ksize = 39': 'ksize = 5',
                    'area_fraction=0.01, speed_px_s=250': 'area_fraction=0.04, speed_px_s=25',
                }
                for source, replacement in replacements.items():
                    self.assertIn(source, code)
                    code = code.replace(source, replacement)
                namespace = {}
                exec(compile(code, 'README.md', 'exec'), namespace)
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
                    background_code = code.replace('with_object = True', 'with_object = False')
                    background_code = background_code.replace(repr(str(out)), repr(str(root / 'background')))
                    exec(compile(background_code, 'README.md', 'exec'), {})
                    result = json.loads((root / 'background/result.json').read_text())
                    self.assertIsNone(result['oms_model'])

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
