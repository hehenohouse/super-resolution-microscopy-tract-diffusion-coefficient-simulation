"""Bounded BR4/BR5 inventory/config tests; no science or TIFF pixels are read."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_br45_batch', ROOT/'tools/prepare_br45_batch.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def header_file(path, endian='>', frames=2500, images=2500, width=428, height=684, bits=16, orientation=1):
    description = ('ImageJ=1.54p\nimages='+str(images)+'\nframes='+str(frames)+'\nunit=micron\nfinterval=0.02\n').encode()+b'\0'
    values = [(256,4,1,struct.pack(endian+'I',width)),(257,4,1,struct.pack(endian+'I',height)),
        (258,3,1,struct.pack(endian+'H',bits)),(259,3,1,struct.pack(endian+'H',5)),
        (270,2,len(description),description),(274,3,1,struct.pack(endian+'H',orientation)),
        (277,3,1,struct.pack(endian+'H',1)),(282,5,1,struct.pack(endian+'II',8547008,1000000)),
        (283,5,1,struct.pack(endian+'II',8547008,1000000)),(296,3,1,struct.pack(endian+'H',1))]
    offset=8+2+12*len(values)+4;extra=bytearray();entries=bytearray()
    for tag,kind,count,data in values:
        if len(data)<=4:payload=data.ljust(4,b'\0')
        else:payload=struct.pack(endian+'I',offset+len(extra));extra.extend(data)
        entries.extend(struct.pack(endian+'HHI',tag,kind,count)+payload)
    Path(path).write_bytes((b'MM' if endian=='>' else b'II')+struct.pack(endian+'HI',42,8)+struct.pack(endian+'H',len(values))+entries+struct.pack(endian+'I',0)+extra)


class NewReplicaInventoryTests(unittest.TestCase):
    def test_first_ifd_metadata_accepts_both_byte_orders_without_pixel_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            for order in ('<','>'):
                p=Path(folder)/('sample'+order.replace('<','little').replace('>','big')+'.tiff');header_file(p,endian=order)
                self.assertLess(p.stat().st_size,1024)
                result=module.tiff_metadata(p)
                self.assertEqual(result['shape'],[2500,684,428]);self.assertEqual(result['dtype'],'uint16')
                self.assertAlmostEqual(result['frame_interval_s'],.02)
                self.assertAlmostEqual(result['pixel_size_nm_xy'][0],117.,places=4)
                self.assertIn('no_pixels',result['metadata_scope'])

    def test_header_mismatch_is_reported_without_a_scientific_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'sample.tiff'
            for options in [dict(frames=2000),dict(width=427),dict(bits=8),dict(orientation=2)]:
                header_file(p,**options)
                with self.assertRaises(ValueError):module.tiff_metadata(p)

    def test_confirmed_mdk_mapping_and_parent_br_identity(self):
        self.assertEqual(module.replica_name('TRAK2','MDK','20260820_MDK_trak2_prcs'),'BR4')
        self.assertEqual(module.replica_name('TRAK2','MDK','20260910_MDK_TRAK2_prcs'),'BR5')
        with self.assertRaises(ValueError):module.replica_name('TRAK2','MDK','20260911_MDK_TRAK2_prcs')
        self.assertEqual(module.replica_name('TRAK1','WT','20260902_WT_TRAK1_BR4'),'BR4')
        self.assertEqual(module.cell_name('090226_WTbr02_TRAK1_T21f'),'T21')
        self.assertEqual(module.cell_name('090226_WTbr1_TRAK2_T61f'),'T61')

    def test_unique_inputs_ignore_hidden_files_and_reject_ambiguity(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            for name in ['C1-test.tiff','C2-test.tiff','cell.roi','._C1-test.tiff','notes.txt']:(p/name).write_bytes(b'')
            self.assertEqual([v.name for v in module.unique_inputs(p)],['C1-test.tiff','C2-test.tiff','cell.roi'])
            (p/'C1-copy.tiff').write_bytes(b'')
            with self.assertRaises(ValueError):module.unique_inputs(p)
            (p/'C1-copy.tiff').unlink();(p/'cell.roi').unlink()
            with self.assertRaises(ValueError):module.unique_inputs(p)

    def test_batch_output_name_cannot_escape_prepared_root(self):
        with self.assertRaisesRegex(ValueError, 'simple name'):
            module.build('/unused', '../outside')

    def test_portable_config_rebases_paths_without_changing_science(self):
        base=dict(data_root='../../Xinran_TRAK1',legacy_releases={'mac':'../../Xinran_TRAK1/_batch_pipeline/releases/old'},
            runtime_profiles={'mac':{'reference_file':'../../Xinran_TRAK1/_batch_pipeline/runtime_paths.json','reference_base':'../../Xinran_TRAK1/_batch_pipeline'}},
            methods={'broad':{'method_id':'nellie_broad','method_version':'1','parameters':{'radius':1.}}},
            acquisition={'shape':[2500,684,428]})
        original=deepcopy(base)
        old=Path('/external/_TRAK_pipeline/configs');new=Path('/external/_TRAK_pipeline/batches/id/configs')
        result=module.rebase_config_paths(base,old,new)
        self.assertEqual((new/result['data_root']).resolve(),Path('/external/Xinran_TRAK1'))
        self.assertEqual(result['methods'],base['methods']);self.assertEqual(result['acquisition'],base['acquisition']);self.assertEqual(base,original)

    def test_new_inventory_skips_generated_outputs_and_rejects_duplicate_br(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for condition in module.CONDITIONS:
                for br in ('BR4','BR5'):
                    b=root/'TRAK1'/condition/('20260902_'+condition+'_TRAK1_'+br);cell=b/('090226_'+condition+'_TRAK1_T1f');cell.mkdir(parents=True)
                    for name in ('C1-test.tiff','C2-test.tiff','cell.roi'):(cell/name).write_bytes(b'')
                    (b/'analysis_results').mkdir();(b/'Analysis_Figures_Intra').mkdir()
            fake=lambda p:dict(shape=[2500,684,428],axes='TYX',dtype='uint16')
            with patch.dict(module.EXPECTED_NEW,TRAK1=6):
                rows,counts,audit,_=module.collect_new(root,'TRAK1',fake)
                self.assertEqual(len(rows),6);self.assertEqual(len({r['sample_id'] for r in rows}),6)
                self.assertEqual(sum(counts.values()),6)
                duplicate=root/'TRAK1'/'WT'/'20260903_WT_TRAK1_BR4';duplicate.mkdir()
                with self.assertRaisesRegex(ValueError,'same condition/BR'):module.collect_new(root,'TRAK1',fake)


if __name__=='__main__':unittest.main()
