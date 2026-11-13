#!/usr/bin/env python3
"""Pinned, resumable queue for one new, independent target-protein dataset.

Every configured cell of the dataset is queued. Conditions and replicas are
declared explicitly in the manifest and checked against the frozen inventory.
Cells run through the same frozen production runner as the TRAK1/TRAK2 queues
(SharedBackend); this controller implements no detection, fitting, segmentation
or TARDIS, and its summaries never pool with any other dataset.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import traceback

RESERVED_MOLECULES = {'TRAK1', 'TRAK2', 'HaloTag_only'}


def load_extension():
    path = Path(__file__).with_name('run_replica_extension_batch.py')
    spec = importlib.util.spec_from_file_location('_dataset_extension', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


extension = load_extension()


def _unique_list(value):
    return isinstance(value, list) and value and len(value) == len(set(value)) and all(isinstance(v, str) and v for v in value)


def manifest_paths(path, require_ready=True):
    path = Path(path).resolve(); m = extension.read(path); batch = path.parent
    common = (batch / m.get('common_root_relative', '../..')).resolve()
    if m.get('schema_version') != 1 or m.get('path_basis') != 'common_root':
        raise ValueError('Unsupported dataset manifest')
    if not batch.is_relative_to(common / 'batches'):
        raise ValueError('Dataset manifest escapes the shared batches directory')
    if require_ready and (not m.get('ready_for_execution') or not m.get('release_relative')):
        raise ValueError('Dataset batch is not frozen and ready; no cell was started')
    if m.get('experiment_type') != 'target_dataset' or m.get('merge_with_target_datasets') is not False:
        raise ValueError('A new dataset queue must be explicitly independent of all other datasets')
    if not _unique_list(m.get('conditions')) or not _unique_list(m.get('replicas')):
        raise ValueError('Conditions and replicas must be explicit, non-empty and unique')
    records = m.get('datasets', [])
    if len(records) != 1:
        raise ValueError('A dataset queue declares exactly one dataset')
    d = records[0]; molecule = d.get('molecule')
    if not isinstance(molecule, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,40}', molecule) or molecule in RESERVED_MOLECULES:
        raise ValueError('Dataset queues are for new target proteins; TRAK1, TRAK2 and HaloTag-only keep their own queues')
    ids = d.get('new_sample_ids', [])
    if (len(ids) != len(set(ids)) or not ids or len(ids) != d.get('new_cells') or d.get('new_cells') != d.get('total_cells')
            or len(ids) != m.get('new_cells') or m.get('existing_cells') != 0 or m.get('summary_cells') != len(ids)):
        raise ValueError('A new dataset queue must run every configured cell exactly once')
    config = (common / d['config_relative']).resolve()
    if not config.is_relative_to(common) or not config.is_file():
        raise ValueError('Missing or out-of-package dataset configuration')
    return m, common, batch


class DatasetBackend(extension.SharedBackend):
    def load_datasets(self):
        values = []
        for record in self.manifest['datasets']:
            config = self.api.load_config(self.common / record['config_relative'])
            if config.get('molecule') != record['molecule']:
                raise ValueError('Configuration molecule differs from the manifest')
            samples = self.api.discover_samples(config); byid = {s.sample_id: s for s in samples}
            if len(samples) != record['total_cells'] or len(byid) != len(samples):
                raise ValueError('Dataset inventory size/identity differs for ' + record['molecule'])
            if any(s.metadata.get('molecule') != record['molecule'] for s in samples):
                raise ValueError('Mixed dataset membership')
            if set(self.manifest['conditions']) != {s.metadata.get('genotype') for s in samples}:
                raise ValueError('Dataset conditions differ from the explicit manifest')
            if set(self.manifest['replicas']) != {s.metadata.get('BR') for s in samples}:
                raise ValueError('Dataset replicas differ from the explicit manifest')
            if set(record['new_sample_ids']) != set(byid):
                raise ValueError('Every configured cell must be queued exactly once')
            new = [byid[sid] for sid in record['new_sample_ids']]
            values.append(dict(record=record, config=config, all_samples=samples, new_samples=new, molecule=record['molecule']))
        self.datasets = values
        return values


def completed_ids(datasets, backend, state):
    """Read-only: queued cells whose recorded completion and outputs are still valid."""
    done = []
    for d in datasets:
        family = backend.family(d)
        for sample in d['new_samples']:
            row = state.get('cells', {}).get(sample.sample_id, {})
            if row.get('status') == 'complete' and row.get('family') == family \
                    and row.get('source_stamp') == backend.source_stamp(d, sample) \
                    and backend.completion_outputs_valid(d, sample, row):
                done.append(sample.sample_id)
    return done


def install_external_stop_guard(module=extension):
    external = os.environ.get('TRAK_COMBINED_STOP_FILE')
    if not external:
        return
    old = module.clear_stop
    def guarded_clear(batch, datasets):
        if Path(external).exists(): module.request_stop(batch, datasets)
        else: old(batch, datasets)
    module.clear_stop = guarded_clear


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--action', choices=['start', 'resume', 'stop', 'plan', 'summarize', 'refresh'], default='start')
    args = parser.parse_args(argv)
    manifest, common, batch = manifest_paths(args.manifest, require_ready=args.action not in ('stop', 'plan'))
    if not common.is_dir() or not os.path.ismount(str(common.parent)):
        raise OSError('The external data drive must be mounted; no output was written')
    extension.early_environment(batch); release = extension.activate_pinned(common, manifest)
    backend = DatasetBackend(common, manifest); backend.platform.install_stop_handlers(); datasets = backend.load_datasets()
    install_external_stop_guard()
    if args.action == 'stop':
        extension.request_stop(batch, datasets); return 0
    if args.action == 'plan':
        path = batch / 'state/batch_state.json'
        state = extension.read(path) if path.is_file() else extension.new_state(manifest, datasets, backend)
        done = completed_ids(datasets, backend, state)
        print(json.dumps(dict(mode='read_only_queue_plan', batch_id=manifest['batch_id'], pinned_release=str(release),
            molecule=datasets[0]['molecule'], conditions=manifest['conditions'], replicas=manifest['replicas'],
            queued_cells=manifest['new_cells'], completed_cells=len(done), pending_cells=manifest['new_cells'] - len(done),
            summary_cells=manifest['summary_cells'], scientific_stage_calls=0), indent=2)); return 0
    with extension.BatchLock(batch, backend.platform), backend.lock_contexts(datasets):
        if args.action in ('summarize', 'refresh'):
            complete = True
            for d in datasets:
                try:
                    result = backend.summarize(d) if args.action == 'summarize' else backend.refresh(d)
                    if result.get('missing') or result.get('complete') is False: complete = False
                    print(json.dumps(dict(molecule=d['molecule'], result=result), indent=2), flush=True)
                except Exception as error:
                    if backend.infrastructure(error): raise
                    complete = False
                    print(json.dumps(dict(molecule=d['molecule'], error=str(error), scientific_upstream_calls=0)), flush=True)
            return 0 if complete else 1
        d = datasets[0]
        print(d['molecule'] + ': ' + str(manifest['new_cells']) + ' cells (' + '/'.join(manifest['conditions']) + ' x '
              + '/'.join(manifest['replicas']) + '); independent summaries, never pooled with other datasets.', flush=True)
        state = extension.execute_queue(manifest, datasets, backend, batch, args.action)
        print(json.dumps({k: state.get(k) for k in ('batch_id', 'status', 'completed_new_cells', 'failed_new_cells', 'remaining_new_cells')}, indent=2), flush=True)
        return 0 if state['status'] == 'COMPLETE' or state.get('stopped_after_cell') else 1


if __name__ == '__main__':
    try: raise SystemExit(main())
    except KeyboardInterrupt: print('Interrupted; committed stage checkpoints are retained.', file=sys.stderr); raise SystemExit(130)
    except BlockingIOError as error: print('BATCH BUSY: ' + str(error), file=sys.stderr); raise SystemExit(75)
    except Exception: traceback.print_exc(); raise SystemExit(1)
