"""Bounded tests for new-dataset queues: no detection, fitting, segmentation or TARDIS calls."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location('test_dataset_' + name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


dataset = load('run_dataset_batch'); outer = load('run_combined_analysis_batch'); prepare = load('prepare_dataset_batch')
IDS = ['WT_Miro1_BR1_T1', 'WT_Miro1_BR2_T1', 'TDK_Miro1_BR1_T1', 'TDK_Miro1_BR2_T1']


class Fixture(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'validation/dataset_batch_20261111'; parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='unit_', dir=parent); self.addCleanup(self.temp.cleanup)
        self.common = Path(self.temp.name); self.batch = self.common / 'batches/Miro1'
        config = self.batch / 'configs/Miro1.json'; config.parent.mkdir(parents=True)
        config.write_text(json.dumps(dict(molecule='Miro1')), encoding='utf-8')
        self.manifest = dict(schema_version=1, batch_id='Miro1', common_root_relative='../..', path_basis='common_root',
            release_relative='releases/r', ready_for_execution=True, experiment_type='target_dataset',
            merge_with_target_datasets=False, conditions=['WT', 'TDK'], replicas=['BR1', 'BR2'],
            new_cells=4, existing_cells=0, summary_cells=4, require_first_complete_cell=True,
            datasets=[dict(molecule='Miro1', config_relative='batches/Miro1/configs/Miro1.json',
                           new_sample_ids=list(IDS), new_cells=4, total_cells=4)])

    def write(self, manifest, name='batches/Miro1/batch_manifest.json'):
        path = self.common / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding='utf-8'); return path


class ManifestTests(Fixture):
    def test_valid_manifest_is_accepted(self):
        m, common, batch = dataset.manifest_paths(self.write(self.manifest))
        self.assertEqual(m['datasets'][0]['molecule'], 'Miro1'); self.assertEqual(batch, self.batch.resolve())

    def test_unsafe_or_inconsistent_manifests_are_rejected(self):
        cases = {
            'reserved TRAK1': lambda m: m['datasets'][0].update(molecule='TRAK1'),
            'reserved HaloTag': lambda m: m['datasets'][0].update(molecule='HaloTag_only'),
            'pooled with targets': lambda m: m.update(merge_with_target_datasets=True),
            'wrong experiment type': lambda m: m.update(experiment_type='negative_control'),
            'existing cells': lambda m: m.update(existing_cells=3),
            'partial queue': lambda m: m['datasets'][0].update(new_sample_ids=IDS[:3], new_cells=3),
            'duplicate ID': lambda m: m['datasets'][0].update(new_sample_ids=IDS[:3] + IDS[:1]),
            'summary count': lambda m: m.update(summary_cells=5),
            'two datasets': lambda m: m['datasets'].append(deepcopy(m['datasets'][0])),
            'missing replicas': lambda m: m.update(replicas=[]),
            'duplicate condition': lambda m: m.update(conditions=['WT', 'WT']),
            'not frozen': lambda m: m.update(ready_for_execution=False),
        }
        for label, change in cases.items():
            with self.subTest(label=label):
                m = deepcopy(self.manifest); change(m)
                with self.assertRaises(ValueError): dataset.manifest_paths(self.write(m))

    def test_plan_and_stop_do_not_require_a_ready_batch(self):
        m = deepcopy(self.manifest); m.update(ready_for_execution=False)
        dataset.manifest_paths(self.write(m), require_ready=False)


def sample(sid, molecule='Miro1'):
    condition, _, br, _ = sid.split('_')
    return types.SimpleNamespace(sample_id=sid, metadata=dict(molecule=molecule, genotype=condition, BR=br))


class BackendTests(Fixture):
    def backend(self, samples, config_molecule='Miro1'):
        b = object.__new__(dataset.DatasetBackend)
        b.api = types.SimpleNamespace(load_config=lambda path: dict(molecule=config_molecule), discover_samples=lambda c: samples)
        b.common = self.common; b.manifest = self.manifest; return b

    def test_every_configured_cell_is_queued(self):
        values = self.backend([sample(s) for s in IDS]).load_datasets()
        self.assertEqual([s.sample_id for s in values[0]['new_samples']], IDS)
        self.assertEqual(len(values[0]['all_samples']), 4)

    def test_inventory_mismatches_stop_before_any_cell(self):
        cases = {
            'extra replica': [sample(s) for s in IDS[:3]] + [sample('TDK_Miro1_BR3_T1')],
            'missing condition': [sample(s) for s in IDS[:2]] + [sample('WT_Miro1_BR1_T2'), sample('WT_Miro1_BR2_T2')],
            'foreign molecule': [sample(s) for s in IDS[:3]] + [sample(IDS[3], 'TRAK1')],
            'size': [sample(s) for s in IDS[:3]],
        }
        for label, samples in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError): self.backend(samples).load_datasets()
        with self.assertRaises(ValueError): self.backend([sample(s) for s in IDS], config_molecule='TRAK2').load_datasets()


class CombinedTests(Fixture):
    def test_dataset_jobs_launch_the_dataset_queue_from_the_pinned_release(self):
        release = Path('/release'); job = dict(job_id='Miro1', kind='dataset', manifest_relative='batches/Miro1/batch_manifest.json')
        command = outer.child_command(release, job, self.common, 'resume')
        self.assertEqual(command[-5:], [str(release / 'tools/run_dataset_batch.py'), '--manifest',
                                       str(self.common / job['manifest_relative']), '--action', 'resume'])

    def test_combined_manifest_accepts_dataset_kind_and_rejects_unknown_kinds(self):
        self.write(self.manifest)
        combined = dict(schema_version=1, batch_id='all', common_root_relative='../..', path_basis='common_root',
                        release_relative='releases/r', ready_for_execution=True,
                        jobs=[dict(job_id='Miro1', kind='dataset', manifest_relative='batches/Miro1/batch_manifest.json')])
        path = self.write(combined, 'batches/all/batch_manifest.json')
        outer.manifest_paths(path)
        combined['jobs'][0]['kind'] = 'pilot'; self.write(combined, 'batches/all/batch_manifest.json')
        with self.assertRaises(ValueError): outer.manifest_paths(path)
        with self.assertRaises(ValueError): outer.child_command(Path('/r'), dict(kind='pilot', manifest_relative='x'), self.common, 'start')

    def test_dataset_child_must_share_the_combined_release(self):
        m = deepcopy(self.manifest); m['release_relative'] = 'releases/other'; self.write(m)
        combined = dict(schema_version=1, batch_id='all', common_root_relative='../..', path_basis='common_root',
                        release_relative='releases/r', ready_for_execution=True,
                        jobs=[dict(job_id='Miro1', kind='dataset', manifest_relative='batches/Miro1/batch_manifest.json')])
        with self.assertRaises(ValueError): outer.manifest_paths(self.write(combined, 'batches/all/batch_manifest.json'))


def header():
    return dict(shape=[2500, 684, 428], axes='TYX')


class PrepareTests(Fixture):
    def make_cell(self, root, condition, br_folder, cell_folder):
        d = root / condition / br_folder / cell_folder; d.mkdir(parents=True)
        stem = cell_folder[:-1]
        for name in ('C1-' + stem + '.tiff', 'C2-' + stem + '.tiff', '0001-0002-0003.roi'): (d / name).write_bytes(b'x')
        return d

    def test_collect_uses_parent_BR_suffix_and_cell_date(self):
        root = self.common / 'Xinran_Miro1'
        self.make_cell(root, 'WT', '20251016_WT_Miro1_BR1', '101625_WT_Miro1_T1f')
        self.make_cell(root, 'WT', '110625_WT_Miro1_BR2', '110625_WT_Miro1_T2f')
        rows, counts, audit, paths = prepare.collect(root, 'Miro1', ['WT'], ['BR1', 'BR2'], lambda p: header())
        self.assertEqual([r['sample_id'] for r in rows], ['WT_Miro1_BR1_T1', 'WT_Miro1_BR2_T2'])
        self.assertEqual([r['acquisition_date'] for r in rows], ['2025-10-16', '2025-11-06'])
        self.assertEqual(paths['WT', 'BR2'], 'WT/110625_WT_Miro1_BR2')

    def test_collect_rejects_missing_or_unexpected_replicas_and_ambiguous_inputs(self):
        root = self.common / 'Xinran_Miro1'
        self.make_cell(root, 'WT', '20251016_WT_Miro1_BR1', '101625_WT_Miro1_T1f')
        with self.assertRaises(ValueError): prepare.collect(root, 'Miro1', ['WT'], ['BR1', 'BR2'], lambda p: header())
        self.make_cell(root, 'WT', '20251023_WT_Miro1_BR7', '102325_WT_Miro1_T1f')
        with self.assertRaises(ValueError): prepare.collect(root, 'Miro1', ['WT'], ['BR1'], lambda p: header())
        extra = self.make_cell(self.common / 'Other', 'WT', 'x_BR1', '101625_WT_Miro1_T1f'); (extra / 'second.roi').write_bytes(b'x')
        with self.assertRaises(ValueError): prepare.collect(self.common / 'Other', 'Miro1', ['WT'], ['BR1'], lambda p: header())

    def make_flat_cell(self, root, br_folder, cell_folder):
        d = root / br_folder / cell_folder; d.mkdir(parents=True)
        for name in ('C1-x.tiff', 'C2-x.tiff', 't1.roi'): (d / name).write_bytes(b'x')
        return d

    def test_flat_layout_YYYYMMDD_dates_and_recorded_exclusions(self):
        root = self.common / 'Xinran_New'
        self.make_flat_cell(root, '20260423_TDK_TRAK1OE_TRAK2_BR1', '20260423_TDK_650TK2_TK1fgOE_T1f')
        self.make_flat_cell(root, '20260423_TDK_TRAK1OE_TRAK2_BR1', '20260423_TDK_650TK2_TK1fgOE_T4f')
        self.make_flat_cell(root, '20260528_TDK_TRAK1OE_TRAK2_BR2', '20260528_TDK_TK1OE_T1f')
        read = []
        def reader(p):
            read.append(p.parent.name)
            if p.parent.name.endswith('_T4f'): raise ValueError('New TIFF acquisition differs from 2500x684x428')
            return header()
        rows, counts, audit, paths = prepare.collect(root, 'P-OE', ['TDK'], ['BR1', 'BR2'], reader, flat=True,
                                                     exclusions={'TDK_P-OE_BR1_T4': '1000 frames, not 2500'})
        self.assertEqual([r['sample_id'] for r in rows], ['TDK_P-OE_BR1_T1', 'TDK_P-OE_BR2_T1'])
        self.assertEqual([r['acquisition_date'] for r in rows], ['2026-04-23', '2026-05-28'])
        self.assertEqual(paths['TDK', 'BR2'], '20260528_TDK_TRAK1OE_TRAK2_BR2')
        self.assertNotIn('20260423_TDK_650TK2_TK1fgOE_T4f', read)
        excluded = [a for a in audit if 'excluded_reason' in a]
        self.assertEqual([(a['sample_id'], a['excluded_reason']) for a in excluded], [('TDK_P-OE_BR1_T4', '1000 frames, not 2500')])
        self.assertEqual(counts['TDK', 'BR1'], 1)
        with self.assertRaisesRegex(ValueError, 'differs'):
            prepare.collect(root, 'P-OE', ['TDK'], ['BR1', 'BR2'], reader, flat=True)
        with self.assertRaisesRegex(ValueError, 'not found'):
            prepare.collect(root, 'P-OE', ['TDK'], ['BR1', 'BR2'], lambda p: header(), flat=True, exclusions={'TDK_P-OE_BR1_T9': 'x'})
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            prepare.collect(root, 'P-OE', ['WT', 'TDK'], ['BR1', 'BR2'], lambda p: header(), flat=True)
        with self.assertRaises(ValueError): prepare.parse_exclusions(['TDK_P-OE_BR1_T4'])

    def test_acquisition_date_prefixes_are_explicit(self):
        self.assertEqual(prepare.acquisition_date('101625_WT_Miro1_T1f'), '2025-10-16')
        self.assertEqual(prepare.acquisition_date('20260611_TDK_TK1OE_T1f'), '2026-06-11')
        for bad in ('T1f', '2026061_x', '20261399_x'):
            with self.subTest(bad=bad), self.assertRaises(ValueError): prepare.acquisition_date(bad)

    def test_condition_read_from_flat_replica_folder_names(self):
        root = self.common / 'Xinran_M2'
        self.make_flat_cell(root, '20251016_WT_M2_BR1', '101625_WT_M2_T1f')
        self.make_flat_cell(root, '20251016_TDK_M2_BR1', '101625_TDKd3_M2_T1f')
        self.make_flat_cell(root, '20251023_TDK_M2_BR2', '102325_TDKd3_M2_T2f')
        self.make_flat_cell(root, '110625_WT_M2_BR2', '110625_WT_M2_T3f')
        rows, counts, audit, paths = prepare.collect(root, 'M2', ['WT', 'TDK'], ['BR1', 'BR2'], lambda p: header(), condition_from_folder=True)
        self.assertEqual([r['sample_id'] for r in rows], ['WT_M2_BR1_T1', 'WT_M2_BR2_T3', 'TDK_M2_BR1_T1', 'TDK_M2_BR2_T2'])
        self.assertEqual([r['genotype'] for r in rows], ['WT', 'WT', 'TDK', 'TDK'])
        self.assertEqual(paths['TDK', 'BR2'], '20251023_TDK_M2_BR2')
        self.assertEqual(rows[1]['acquisition_date'], '2025-11-06')
        with self.assertRaisesRegex(ValueError, 'Replicas for MDK'):   # declared condition without folders
            prepare.collect(root, 'M2', ['WT', 'TDK', 'MDK'], ['BR1', 'BR2'], lambda p: header(), condition_from_folder=True)
        with self.assertRaisesRegex(ValueError, 'exactly one of WT'):   # folder naming no declared condition
            prepare.collect(root, 'M2', ['WT'], ['BR1', 'BR2'], lambda p: header(), condition_from_folder=True)
        self.make_flat_cell(root, '20251030_WT_TDK_M2_BR3', '103025_x_T1f')
        with self.assertRaisesRegex(ValueError, 'exactly one of'):      # ambiguous folder
            prepare.collect(root, 'M2', ['WT', 'TDK'], ['BR1', 'BR2', 'BR3'], lambda p: header(), condition_from_folder=True)
        with self.assertRaises(ValueError):
            prepare.collect(root, 'M2', ['WT'], ['BR1'], lambda p: header(), flat=True, condition_from_folder=True)

    def test_build_refuses_existing_queue_names(self):
        for molecule in ('TRAK1', 'TRAK2', 'HaloTag_only'):
            with self.subTest(molecule=molecule):
                with self.assertRaises(ValueError):
                    prepare.build(self.common, molecule, 'Xinran_Miro1', ['WT'], ['BR1'], 'b', metadata_reader=lambda p: header())


if __name__ == '__main__':
    unittest.main()
