"""Mask-parameter overrides (2026-11-13): Nellie Hessian gate division and dataset --method-set.

No Nellie, detection, fitting or TARDIS runs: the native worker is replaced by a stub
module and the native process by an in-process fake.
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from trak_pipeline.identity import scientific_key
from trak_pipeline.stages import mask
from trak_pipeline.types import Artifact, Sample, StageContext

ROOT = Path(__file__).resolve().parents[1]
FRAMES = 4; SHAPE = (6, 5)


def load_tool(name):
    spec = importlib.util.spec_from_file_location('test_override_' + name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


class FakePopen:
    requests = []
    def __init__(self, command, cwd=None, env=None, stdout=None, stderr=None):
        self.args = command; self.returncode = None
    def wait(self, timeout=None):
        request = json.loads(Path(self.args[-1]).read_text(encoding='utf-8')); FakePopen.requests.append(request)
        raw = tifffile.imread(request['input']); np.save(request['output'], raw % 2 == 0)
        for name in ('thresholds.json', 'actual_parameters.json'):
            (Path(request['work']) / name).write_text('{}', encoding='utf-8')
        self.returncode = 0; return 0
    def poll(self): return self.returncode
    def terminate(self): pass
    def kill(self): pass


class BroadGateTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'validation').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'validation', prefix='gate_'); base = Path(self.tmp.name)
        self.data = base / 'data'; self.common = base / 'common'; self.data.mkdir(); self.common.mkdir()
        self.c1 = self.data / 'C1.tif'
        tifffile.imwrite(self.c1, (np.arange(FRAMES * SHAPE[0] * SHAPE[1]) % 251).astype(np.uint16).reshape(FRAMES, *SHAPE),
                         imagej=True, metadata=dict(axes='TYX'))
        self.env = patch.dict(os.environ, {'TRAK_COMMON_ROOT': str(self.common)}); self.env.start()

    def tearDown(self):
        self.env.stop(); self.tmp.cleanup()

    def run_broad(self, extra, name):
        config = dict(data_root=str(self.data), acquisition=dict(shape=list(SHAPE), frames=FRAMES, pixel_size_nm=117.0, frame_interval_s=0.02),
                      sample_frames=[0], execution=dict(chunk_frames=2, nellie_chunk_workers=1), runtime=dict(nellie_python='nellie-python'))
        params = dict(frames=FRAMES, shape=list(SHAPE), pixel_size_nm=117.0, frame_interval_s=0.02, filter_radii_um=[0.25, 1.0], nellie_version='1.0.4', **extra)
        ctx = StageContext(config, Sample('WT_TEST_BR1_T1', self.data, {}, self.c1, self.c1, self.c1), params,
                           dict(C1=Artifact('c1', 'C1', dict(image=str(self.c1)))), self.data / name, self.common / 'releases/x', config['runtime'])
        FakePopen.requests = []
        with patch.object(mask.subprocess, 'Popen', FakePopen), patch('builtins.print'):
            mask.execute_broad(ctx)
        return FakePopen.requests

    def test_default_gate_is_two_and_declared_gate_reaches_native_request(self):
        self.assertEqual({r['frob_thresh_division'] for r in self.run_broad({}, 'default')}, {2.0})
        self.assertEqual({r['frob_thresh_division'] for r in self.run_broad(dict(frob_thresh_division=4), 'gate4')}, {4.0})

    def test_invalid_gate_is_rejected_before_any_native_call(self):
        for bad in (0, -1, float('inf')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.run_broad(dict(frob_thresh_division=bad), 'bad%s' % bad)

    def test_native_worker_passes_gate_to_nellie_filter(self):
        seen = {}
        class FakeFilter:
            def __init__(self, im, **kw): seen.update(kw); self.sigmas = [1.0]
            def run(self): pass
        class FakeLabel:
            def __init__(self, im, **kw): self.min_area_pixels = 1
            def _compute_frame_thresholds(self, a, b): return None, 1.0
            def run(self):
                for _ in range(2): self._compute_frame_thresholds(None, None)
        class FakeInfo:
            axes = 'TYX'
            def __init__(self, *a, **k): pass
            def find_metadata(self): pass
            def load_metadata(self): pass
            def change_dim_res(self, *a): pass
        work = self.data / 'native'; work.mkdir()
        labels = work / 'labels.tif'; tifffile.imwrite(labels, np.ones((2, *SHAPE), np.uint16))
        im = types.SimpleNamespace(shape=(2, *SHAPE), no_z=True, pipeline_paths=dict(im_instance_label=str(labels)))
        modules = {'nellie': types.ModuleType('nellie'), 'nellie.im_info': types.ModuleType('nellie.im_info'),
                   'nellie.im_info.verifier': types.SimpleNamespace(FileInfo=FakeInfo, ImInfo=lambda fi: im),
                   'nellie.segmentation': types.ModuleType('nellie.segmentation'),
                   'nellie.segmentation.filtering': types.SimpleNamespace(Filter=FakeFilter),
                   'nellie.segmentation.labelling': types.SimpleNamespace(Label=FakeLabel)}
        for gate, expected in ((None, 2.0), (4, 4.0)):
            request = dict(work=str(work), input=str(self.c1), output=str(work / 'out.npy'), frame_ids=[0, 1], shape_yx=list(SHAPE),
                           pixel_size_nm=117.0, frame_interval_s=0.02, filter_radii_um=[0.25, 1.0], nellie_version='1.0.4')
            if gate is not None: request['frob_thresh_division'] = gate
            path = work / 'request.json'; path.write_text(json.dumps(request), encoding='utf-8'); seen.clear()
            with patch.dict(sys.modules, modules), patch('importlib.metadata.version', return_value='1.0.4'):
                mask._native_worker(path)
            with self.subTest(gate=gate):
                self.assertEqual(seen['frob_thresh_division'], expected)
                self.assertEqual(json.loads((work / 'actual_parameters.json').read_text())['frob_thresh_division'], expected)

    def test_absent_gate_keeps_existing_scientific_identity(self):
        base = dict(filter_radii_um=[0.25, 1.0], nellie_version='1.0.4')
        self.assertNotEqual(scientific_key('nellie_broad', '1', base, {'C1': 'x'}),
                            scientific_key('nellie_broad', '1', dict(base, frob_thresh_division=4), {'C1': 'x'}))
        self.assertEqual(scientific_key('nellie_broad', '1', base, {'C1': 'x'}),
                         scientific_key('nellie_broad', '1', dict(base), {'C1': 'x'}))


class MethodSetTests(unittest.TestCase):
    prepare = load_tool('prepare_dataset_batch')

    def test_mask_parameters_are_parsed_and_applied_with_a_record(self):
        sets = self.prepare.parse_method_sets(['broad.frob_thresh_division=4', 'mask_threshold.minimum_support=20',
                                               'mask_cleanup.min_foreground_pixels=60'])
        methods = dict(broad=dict(parameters=dict(nellie_version='1.0.4')), mask_threshold=dict(parameters=dict(minimum_support=10)),
                       mask_cleanup=dict(parameters=dict(min_foreground_pixels=32)), fit=dict(parameters=dict(a=1)))
        new, record = self.prepare.apply_method_sets(methods, sets)
        self.assertEqual(new['broad']['parameters']['frob_thresh_division'], 4)
        self.assertEqual(new['mask_threshold']['parameters']['minimum_support'], 20)
        self.assertEqual(new['mask_cleanup']['parameters']['min_foreground_pixels'], 60)
        self.assertEqual(new['fit'], methods['fit']); self.assertEqual(methods['mask_threshold']['parameters']['minimum_support'], 10)
        self.assertEqual([(r['stage'], r['parameter'], r['template_value'], r['value']) for r in record],
                         [('broad', 'frob_thresh_division', None, 4), ('mask_threshold', 'minimum_support', 10, 20),
                          ('mask_cleanup', 'min_foreground_pixels', 32, 60)])

    def test_non_mask_duplicate_or_invalid_settings_are_rejected(self):
        for bad in (['fit.workers=4'], ['assignment.k=2'], ['mask_threshold.minimum_support=0'], ['mask_threshold.minimum_support=51'],
                    ['mask_threshold.minimum_support=20.5'], ['broad.frob_thresh_division=0'], ['broad.frob_thresh_division="4"'],
                    ['mask_cleanup.min_foreground_pixels=true'], ['broad.frob_thresh_division'], ['broad.frob_thresh_division=x'],
                    ['mask_threshold.minimum_support=20', 'mask_threshold.minimum_support=30']):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.prepare.parse_method_sets(bad)


if __name__ == '__main__':
    unittest.main()
