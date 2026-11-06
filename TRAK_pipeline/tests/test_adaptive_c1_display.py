"""Meaningful display-only limits contracts; no scientific workers."""
from pathlib import Path
import importlib.util
import json
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('c1_display_under_test', ROOT/'src/trak_pipeline/display.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
resolve = _module.resolve_c1_display
AUTO = {'C1_display_mode': 'roi_percentile', 'C1_display_percentiles': [1, 99.9],
        'C1_display_ADU': [400, 2300]}


class AdaptiveC1DisplayTests(unittest.TestCase):
    def test_fixed_mode_never_reads(self):
        def forbidden(frame):
            self.fail('Fixed display must not read any original frame')
        answer = resolve({'C1_display_ADU': [400, 2300]}, range(10), forbidden)
        self.assertEqual(answer['limits_ADU'], [400.0, 2300.0])
        self.assertEqual(answer['sample_frames'], [])
        self.assertTrue(answer['display_only'])
        json.dumps(answer, allow_nan=False)

    def test_roi_zero_and_nonfinite_exclusion_linear_quantiles(self):
        roi = np.array([[1, 1, 1], [1, 1, 0]], dtype=bool)
        frame = np.array([[0, 10, 20], [30, np.nan, 65535]], dtype=float)
        answer = resolve(dict(AUTO, C1_display_percentiles=[0, 100]), [7], lambda f: frame, roi)
        self.assertEqual(answer['limits_ADU'], [10.0, 30.0])
        self.assertEqual(answer['nonzero_ROI_pixels'], 3)
        self.assertEqual(answer['excluded_nonfinite_ROI_pixels'], 1)
        self.assertEqual(answer['sample_frames'], [7])
        json.dumps(answer, allow_nan=False)

    def test_exact_ten_frames_one_pooled_window_and_no_mutation(self):
        ids = [0, 49, 200, 625, 1200, 1250, 1875, 2000, 2450, 2499]
        images = {f: np.array([[100+i, 500+i], [900+i, 60000]], dtype=np.uint16)
                  for i, f in enumerate(ids)}
        before = {f: image.copy() for f, image in images.items()}
        roi = np.array([[True, True], [True, False]])
        original_roi = roi.copy()
        calls = []
        def read(frame):
            calls.append(frame)
            return images[frame]
        answer = resolve(AUTO, ids, read, roi)
        expected = np.percentile(np.concatenate([images[f][roi] for f in ids]), [1, 99.9], method='linear')
        np.testing.assert_array_equal(answer['limits_ADU'], expected)
        self.assertEqual(calls, ids)
        self.assertEqual(answer['sample_frames'], ids)
        self.assertEqual(answer['nonzero_ROI_pixels'], 30)
        for frame in ids:
            np.testing.assert_array_equal(images[frame], before[frame])
        np.testing.assert_array_equal(roi, original_roi)

    def test_missing_frames_and_conflicting_shapes_are_errors(self):
        roi = np.ones((2, 2), dtype=bool)
        def missing(frame):
            raise IndexError('Original frame missing')
        with self.assertRaisesRegex(IndexError, 'missing'):
            resolve(AUTO, [2], missing, roi)
        with self.assertRaisesRegex(ValueError, 'shape'):
            resolve(AUTO, [2], lambda f: np.ones((3, 3)), roi)

    def test_zero_signal_and_uniform_signal_have_explicit_fallbacks(self):
        roi = np.ones((2, 2), dtype=bool)
        answer = resolve(AUTO, [0], lambda f: np.zeros((2, 2), dtype=np.uint16), roi)
        self.assertEqual(answer['limits_ADU'], [400., 2300.])
        self.assertIn('no_finite_nonzero', answer['fallback_reason'])
        for value, limits in [(100, [99., 101.]), (65535, [65534., 65535.])]:
            answer = resolve(AUTO, [0], lambda f: np.full((2, 2), value, dtype=np.uint16), roi)
            self.assertEqual(answer['limits_ADU'], limits)
            self.assertIn('degenerate', answer['fallback_reason'])

    def test_invalid_configuration_and_empty_roi_fail(self):
        roi = np.ones((2, 2), dtype=bool)
        reader = lambda f: np.ones((2, 2), dtype=np.uint16)
        for ids in ([], [0, 0], [-1], [1.5], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                resolve(AUTO, ids, reader, roi)
        for percentiles in ([99, 1], [-1, 99], [1, 101], [1, 1]):
            with self.subTest(percentiles=percentiles), self.assertRaises(ValueError):
                resolve(dict(AUTO, C1_display_percentiles=percentiles), [0], reader, roi)
        with self.assertRaisesRegex(ValueError, 'no pixels'):
            resolve(AUTO, [0], reader, np.zeros((2, 2), dtype=bool))
        with self.assertRaises(ValueError):
            resolve({'C1_display_mode': 'unknown'})


if __name__ == '__main__':
    unittest.main()
