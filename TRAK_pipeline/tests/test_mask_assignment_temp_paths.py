"""Short atomic siblings: Windows path budget and small I/O round trips only."""
from __future__ import annotations
import csv
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path,PureWindowsPath
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'src'))
from trak_pipeline.stages import mask,assignment


class MaskAssignmentTempPathsTest(unittest.TestCase):
    def setUp(self):
        parent=PACKAGE/'validation/mask_assignment_temp_paths_20261104';parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='io_',dir=parent);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def test_json_array_packed_and_csv_roundtrip(self):
        values=np.arange(27,dtype=np.uint16).reshape(3,3,3);binary=values%2==0
        mask._json(self.root/'record.json',dict(a=1,b=[2,3]));self.assertEqual(json.loads((self.root/'record.json').read_text()),dict(a=1,b=[2,3]))
        mask._array(self.root/'array.npy',values);np.testing.assert_array_equal(np.load(self.root/'array.npy'),values)
        mask.save_packed(self.root/'chunk_000000.npz',binary,[0,1,2]);actual,ids=mask.load_packed(self.root/'chunk_000000.npz')
        np.testing.assert_array_equal(actual,binary);self.assertEqual(ids,[0,1,2])
        assignment._write_csv(self.root/'counts.csv',[dict(frame=0,On=4),dict(frame=1,On=3)])
        with (self.root/'counts.csv').open() as stream:self.assertEqual(list(csv.DictReader(stream)),[dict(frame='0',On='4'),dict(frame='1',On='3')])
        self.assertEqual(list(self.root.glob('.*.tmp')),[])
    def test_actual_halo_windows_path_candidates_remain_below_260(self):
        cell=PureWindowsPath('D:/Xinran_TRAK1_TRAK2_negativeWT/NegativeControl/WT/20260930_WT_TagOnly_BR1/093026_WT_TagOnly_T1f/analysis_results/modular_cache')
        candidates=[cell/'broad'/('a'*64)/'packed/chunk_000000.npz',cell/'broad'/('a'*64)/'packed/chunk_000000.json',
            cell/'assignment'/('a'*64)/'parts/labels_000000.csv.gz',cell/'assignment'/('a'*64)/'per_frame_counts.csv',
            cell/'assignment'/('a'*64)/'classification_summary.json']
        short=mask._temporary_path(self.root/'destination.json')
        try:
            self.assertEqual(short.parent,self.root);self.assertEqual(len(short.name),13)
            for destination in candidates:
                old=str(destination)+'.'+'f'*32+'.tmp';new=str(destination.with_name(short.name))
                self.assertGreaterEqual(len(old),260,str(destination));self.assertLess(len(new),260,str(destination))
                self.assertLess(len(str(destination)),260)
        finally:short.unlink()
    def test_same_destination_concurrent_writers_use_unique_atomic_siblings(self):
        original=mask._temporary_path;paths=[];lock=threading.Lock();target=self.root/'shared.json'
        def reserve(destination):
            result=original(destination)
            with lock:paths.append(result)
            return result
        with patch.object(mask,'_temporary_path',side_effect=reserve),ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i:mask._json(target,dict(writer=i,payload=[i]*100)),range(48)))
        self.assertEqual(len(paths),48);self.assertEqual(len(set(paths)),48)
        self.assertTrue(all(p.parent==target.parent and len(p.name)==13 for p in paths))
        final=json.loads(target.read_text());self.assertEqual(final['payload'],[final['writer']]*100)
        self.assertTrue(all(not p.exists() for p in paths))
    def test_assignment_parts_and_final_gzip_use_short_siblings(self):
        # The existing four-frame fixture exercises both gzip checkpoint parts,
        # final gzip merge, JSON and CSV publication, including resume behavior.
        spec=importlib.util.spec_from_file_location('_small_assignment_fixture',PACKAGE/'tests/test_mask_assignment.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        case=fixture.MaskAssignmentTest('test_full_assignment_readonly_master_and_empty_frames')
        original=mask._temporary_path;destinations=[]
        def reserve(destination):
            path=original(destination);destinations.append((Path(destination),path));return path
        case.setUp()
        try:
            with patch.object(mask,'_temporary_path',side_effect=reserve),patch.object(assignment,'_temporary_path',side_effect=reserve):
                case.test_full_assignment_readonly_master_and_empty_frames()
            names=[d.name for d,p in destinations]
            self.assertIn('labels_000000.csv.gz',names);self.assertIn('labels.csv.gz',names)
            self.assertIn('classification_summary.json',names);self.assertIn('per_frame_counts.csv',names)
            self.assertTrue(all(p.parent==d.parent and len(p.name)==13 for d,p in destinations))
            self.assertTrue(all(not p.exists() for d,p in destinations))
        finally:case.tearDown()
    def test_failed_write_preserves_existing_destination_and_removes_temporary(self):
        destination=self.root/'record.json';mask._json(destination,dict(saved=True));before=destination.read_bytes()
        with self.assertRaises(ValueError):mask._json(destination,dict(invalid=float('nan')))
        self.assertEqual(destination.read_bytes(),before);self.assertEqual(list(self.root.glob('.*.tmp')),[])


if __name__=='__main__':unittest.main(verbosity=2)
