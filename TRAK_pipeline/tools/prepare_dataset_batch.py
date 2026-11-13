#!/usr/bin/env python3
"""Prepare a pinned queue for one new, independent target-protein dataset.

Every configured cell is new. Only directory entries, first TIFF IFDs and small
JSON files are read, with the same header checks as prepare_br45_batch. This tool
never reads pixels, renames or copies inputs, imports a scientific stage, or
starts a cell. Scientific methods, acquisition, summary, execution, style and
runtime settings are copied unchanged from the production TRAK1 BR4/BR5
configuration, so the new dataset is analysed exactly like TRAK1 and TRAK2,
unless explicit --method-set overrides are given. Only the mitochondrial-mask
parameters in ALLOWED_METHOD_SETS can be overridden; every override is recorded
in the batch manifest and inventory, and all other settings stay unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

SOURCE_ROOT = Path(__file__).resolve().parents[1]
COMMON_ROOT = SOURCE_ROOT.parents[1] if SOURCE_ROOT.parent.name == 'releases' else SOURCE_ROOT
TEMPLATE_RELATIVE = 'batches/20261102_BR4_BR5/configs/TRAK1_BR4_BR5.json'
RESERVED_MOLECULES = {'TRAK1', 'TRAK2', 'HaloTag_only'}
NAME = r'[A-Za-z0-9][A-Za-z0-9_-]{0,40}'
UNCHANGED_SECTIONS = ('methods', 'acquisition', 'summary', 'execution', 'style', 'sample_frames', 'runtime_profiles')
# Mask-only scientific parameters that a new dataset may set explicitly (stage, parameter) -> validator.
ALLOWED_METHOD_SETS = {
    ('broad', 'frob_thresh_division'): lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < float('inf'),
    ('mask_threshold', 'minimum_support'): lambda v: isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 50,
    ('mask_cleanup', 'min_foreground_pixels'): lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 1,
}


def parse_method_sets(items):
    """'stage.parameter=JSON' strings -> ordered list of (stage, parameter, value); mask parameters only."""
    result = []; seen = set()
    for item in items or ():
        left, sep, raw = item.partition('=')
        stage, dot, name = left.partition('.')
        if not sep or not dot or not name:
            raise ValueError('--method-set requires STAGE.PARAMETER=JSON: ' + item)
        try: value = json.loads(raw)
        except json.JSONDecodeError as error: raise ValueError('--method-set value must be JSON: ' + item) from error
        key = (stage, name)
        if key not in ALLOWED_METHOD_SETS:
            raise ValueError('Only mitochondrial-mask parameters may be set: ' + ', '.join(s + '.' + n for s, n in sorted(ALLOWED_METHOD_SETS)))
        if key in seen: raise ValueError('Duplicate --method-set for ' + left)
        if not ALLOWED_METHOD_SETS[key](value): raise ValueError('Invalid value for ' + left + ': ' + raw)
        seen.add(key); result.append((stage, name, value))
    return result


def apply_method_sets(methods, method_sets):
    methods = deepcopy(methods); record = []
    for stage, name, value in method_sets:
        parameters = methods[stage].setdefault('parameters', {})
        record.append(dict(stage=stage, parameter=name, template_value=parameters.get(name), value=value))
        parameters[name] = value
    return methods, record


def _load_br45():
    path = Path(__file__).with_name('prepare_br45_batch.py')
    spec = importlib.util.spec_from_file_location('_dataset_prepare_br45', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


br45 = _load_br45()


def replica_of(directory, replicas):
    match = re.search(r'_(BR\d+)$', Path(directory).name)
    if not match or match.group(1) not in replicas:
        raise ValueError('Replica folder must end in one of ' + '/'.join(replicas) + ': ' + str(directory))
    return match.group(1)


def collect(root, molecule, conditions, replicas, metadata_reader=None):
    metadata_reader = metadata_reader or br45.tiff_metadata
    rows = []; counts = Counter(); audits = []; replica_paths = {}
    for condition in conditions:
        folder = root / condition
        if not folder.is_dir():
            raise FileNotFoundError('Condition directory is missing: ' + str(folder))
        seen = set()
        for br_dir in br45.visible_entries(folder, directories=True):
            br = replica_of(br_dir, replicas)
            if br in seen:
                raise ValueError('Multiple physical folders map to the same condition/BR: ' + str(br_dir))
            seen.add(br); replica_paths[(condition, br)] = br_dir.relative_to(root).as_posix()
            for directory in br45.visible_entries(br_dir, directories=True):
                if directory.name in ('analysis_results', 'test') or directory.name.startswith('Analysis_Figures'):
                    continue
                cell = br45.cell_name(directory); c1, c2, roi = br45.unique_inputs(directory)
                headers = {key: metadata_reader(path) for key, path in [('C1', c1), ('C2', c2)]}
                if headers['C1']['shape'] != headers['C2']['shape']:
                    raise ValueError('C1/C2 declared grids differ: ' + str(directory))
                acquired = datetime.strptime(directory.name[:6], '%m%d%y').date().isoformat()
                sid = '_'.join([condition, molecule, br, cell]); rel = directory.relative_to(root).as_posix()
                rows.append(dict(sample_id=sid, sample_name=' '.join([condition, molecule, br, 'cell', cell]),
                    genotype=condition, BR=br, cell=cell, molecule=molecule, acquisition_date=acquired,
                    source_directory=rel, source_directory_relative=rel,
                    C1_file=c1.relative_to(root).as_posix(), C2_file=c2.relative_to(root).as_posix(),
                    ROI_file=roi.relative_to(root).as_posix(), status='pending', batch_role='new_dataset', headers=headers))
                counts[condition, br] += 1
                audits.append(dict(sample_id=sid, source_directory_relative=rel, original_cell_directory_name=directory.name,
                    parent_replica_directory=br_dir.name,
                    naming_policy='Parent BR folder suffix is authoritative; original filenames retained; no cross-dataset pairing inferred',
                    files=[dict(relative_path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                           for p in (c1, c2, roi)],
                    TIFF_metadata=headers))
        if seen != set(replicas):
            raise ValueError('Replicas for ' + condition + ' are ' + str(sorted(seen)) + ', expected ' + str(list(replicas)))
    rows.sort(key=lambda r: (list(conditions).index(r['genotype']), br45.natural(r['BR']), br45.natural(r['cell'])))
    ids = [r['sample_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample IDs; no directory/coordinate based deduplication was attempted')
    return rows, counts, audits, replica_paths


def build(common_root, molecule, dataset_dir, conditions, replicas, batch_id, release_relative=None,
          expected_cells=None, metadata_reader=None, method_sets=()):
    for label, value in (('molecule', molecule), ('dataset directory', dataset_dir), ('batch ID', batch_id)):
        if not re.fullmatch(NAME if label != 'batch ID' else r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}', value):
            raise ValueError('Invalid ' + label + ': ' + value)
    if molecule in RESERVED_MOLECULES:
        raise ValueError(molecule + ' already has its own queue type; use a new dataset name')
    if not conditions or len(set(conditions)) != len(conditions) or not replicas or len(set(replicas)) != len(replicas):
        raise ValueError('Conditions and replicas must be explicit, non-empty and unique')
    if any(not re.fullmatch(r'BR\d+', r) for r in replicas):
        raise ValueError('Replicas must be named BR<number>')
    common = Path(common_root).resolve(); batch = (common / 'batches' / batch_id).resolve()
    if not batch.is_relative_to(common / 'batches'):
        raise ValueError('New batch output escapes the common batch root')
    root = (common.parent / dataset_dir).resolve()
    if not root.is_dir() or root.parent != common.parent:
        raise ValueError('Dataset directory must sit next to _TRAK_pipeline on the same drive: ' + str(root))
    template_path = common / TEMPLATE_RELATIVE; template = br45.read(template_path)
    rows, counts, audit, replica_paths = collect(root, molecule, conditions, replicas, metadata_reader)
    if expected_cells is not None and len(rows) != expected_cells:
        raise ValueError('Found ' + str(len(rows)) + ' cells, expected ' + str(expected_cells))
    config_dir = batch / 'configs'
    config = br45.rebase_config_paths(template, template_path.parent, config_dir)
    for key in ('legacy_control_relative', 'inventory_relative', 'legacy_releases'):
        config.pop(key, None)
    config.update(dataset_id=molecule, molecule=molecule,
        data_root=os.path.relpath(root, config_dir).replace(os.sep, '/'),
        conditions=list(conditions), replicas=list(replicas), samples=rows, legacy_releases={},
        control_relative='_batch_pipeline/batches/' + batch_id, batch_id=batch_id,
        new_sample_ids=[r['sample_id'] for r in rows], read_only_existing_sample_ids=[])
    rebased_template = br45.rebase_config_paths(template, template_path.parent, config_dir)
    config['methods'], overrides = apply_method_sets(config['methods'], method_sets)
    rebased_template['methods'], _ = apply_method_sets(rebased_template['methods'], method_sets)
    for key in UNCHANGED_SECTIONS:
        if json.dumps(config.get(key), sort_keys=True) != json.dumps(rebased_template.get(key), sort_keys=True):
            raise AssertionError('Production setting changed during preparation: ' + key)
    config_path = config_dir / (molecule + '.json')
    count_rows = [dict(condition=c, BR=b, cells=counts[c, b]) for c in conditions for b in replicas]
    dataset = dict(molecule=molecule, config_relative=config_path.relative_to(common).as_posix(),
        data_root_relative=root.name, new_sample_ids=config['new_sample_ids'], new_cells=len(rows), old_cells=0,
        total_cells=len(rows), counts_by_condition_replica=count_rows,
        source_directories_by_condition_replica={c: {b: replica_paths[c, b] for b in replicas} for c in conditions},
        template_config_relative=TEMPLATE_RELATIVE,
        template_config_sha256=hashlib.sha256(template_path.read_bytes()).hexdigest(),
        upstream_methods_digest=br45.digest(template['methods']), upstream_parameters_unchanged=not overrides,
        method_overrides=overrides, methods_digest=br45.digest(config['methods']))
    manifest = dict(schema_version=1, batch_id=batch_id, common_root_relative='../..', path_basis='common_root',
        release_relative=release_relative, ready_for_execution=bool(release_relative),
        created_utc=datetime.now(timezone.utc).isoformat(), experiment_type='target_dataset',
        merge_with_target_datasets=False, datasets=[dataset], new_cells=len(rows), existing_cells=0,
        summary_cells=len(rows), conditions=list(conditions), replicas=list(replicas),
        require_first_complete_cell=True,
        queue_rule='Exactly the declared new_sample_ids (every configured cell of this dataset); no other dataset enters this queue',
        completion_rule='After the queue, summarize this dataset alone at cell, BR, condition and dataset levels; failed or missing cells remain explicit',
        cross_dataset_replica_pairing='unconfirmed; same acquisition dates as other proteins do not imply pairing',
        scientific_defaults=('Unchanged production methods: TrackMate LoG, annular-background Gaussian fit, raw C1 Nellie support50>=10 -> post-min32 -> holes<64, 1-SE ellipse assignment, TARDIS tau1 and summaries'
            if not overrides else 'Production methods except the declared mask parameters (' + '; '.join(o['stage'] + '.' + o['parameter'] + ' ' + json.dumps(o['template_value']) + ' -> ' + json.dumps(o['value']) for o in overrides)
            + '): TrackMate LoG, annular-background Gaussian fit, raw C1 Nellie, support counts, cleanup, 1-SE ellipse assignment, TARDIS tau1 and summaries'),
        metadata_check_scope=str(2 * len(rows)) + ' TIFF first-IFD headers and explicit ImageJ fields; unique C1/C2/ROI per cell. No pixels, full hashes, page-chain enumeration or science calls.',
        new_cells_started=0, scientific_stage_calls=0)
    return manifest, (config_path, config), dict(dataset=molecule, data_root=root.name, cells=audit)


def write_prepared(common, manifest, prepared_config, audit):
    common = Path(common).resolve(); batch = common / 'batches' / manifest['batch_id']
    if (batch / 'batch_manifest.json').exists():
        raise FileExistsError('Batch already prepared; choose a new batch ID instead of overwriting: ' + str(batch))
    if manifest['release_relative']:
        release = (common / manifest['release_relative']).resolve()
        if not release.is_relative_to(common / 'releases') or not (release / 'release_manifest.json').is_file():
            raise ValueError('An immutable common release manifest is required before execution can be enabled')
    path, config = prepared_config
    br45.atomic_json(path, config)
    br45.atomic_json(batch / 'source_inventory_audit.json', audit)
    br45.atomic_json(batch / 'batch_manifest.json', manifest)
    d = manifest['datasets'][0]
    lines = ['# ' + d['molecule'] + ' input inventory', '',
             '| Condition | ' + ' | '.join(manifest['replicas']) + ' | Total |',
             '| --- |' + ' ---: |' * (len(manifest['replicas']) + 1)]
    for condition in manifest['conditions']:
        cells = [r['cells'] for r in d['counts_by_condition_replica'] if r['condition'] == condition]
        lines.append('| ' + condition + ' | ' + ' | '.join(map(str, cells)) + ' | ' + str(sum(cells)) + ' |')
    lines += ['', 'All ' + str(d['new_cells']) + ' cells are new and queued. Inputs were not renamed, copied or modified.',
              'Checks read first TIFF IFD/ImageJ metadata only, not image pixels. No scientific stage or cell was started.',
              ('Methods, acquisition, summary, execution and runtime settings are copied unchanged from ' + d['template_config_relative'] + '.'
               if not d.get('method_overrides') else 'Acquisition, summary, execution and runtime settings are copied unchanged from ' + d['template_config_relative']
               + '. Methods are copied unchanged except these declared mask parameters: '
               + '; '.join(o['stage'] + '.' + o['parameter'] + ' ' + json.dumps(o['template_value']) + ' -> ' + json.dumps(o['value']) for o in d['method_overrides']) + '.'),
              'This dataset is summarized on its own; it is never pooled with TRAK1, TRAK2 or HaloTag-only.', '']
    (batch / 'INPUT_INVENTORY.md').write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-root', type=Path, default=COMMON_ROOT)
    parser.add_argument('--molecule', required=True)
    parser.add_argument('--dataset-dir', required=True, help='Folder next to _TRAK_pipeline, e.g. Xinran_Miro1')
    parser.add_argument('--conditions', nargs='+', default=['WT', 'TDK', 'MDK'])
    parser.add_argument('--replicas', nargs='+', required=True)
    parser.add_argument('--batch-id', required=True)
    parser.add_argument('--expected-cells', type=int)
    parser.add_argument('--release-relative', help='Frozen release relative to common root; otherwise the batch is not execution-ready')
    parser.add_argument('--method-set', action='append', default=[], metavar='STAGE.PARAMETER=JSON',
                        help='Explicit mitochondrial-mask parameter, e.g. broad.frob_thresh_division=4, mask_threshold.minimum_support=20, mask_cleanup.min_foreground_pixels=60')
    parser.add_argument('--write', action='store_true', help='Write only the new batch folder; never source data or other batches')
    args = parser.parse_args(argv)
    manifest, prepared, audit = build(args.common_root, args.molecule, args.dataset_dir, args.conditions, args.replicas,
                                      args.batch_id, args.release_relative, args.expected_cells,
                                      method_sets=parse_method_sets(args.method_set))
    if args.write:
        write_prepared(args.common_root, manifest, prepared, audit)
    d = manifest['datasets'][0]
    print(json.dumps({k: manifest[k] for k in ('batch_id', 'new_cells', 'summary_cells', 'ready_for_execution',
                                               'require_first_complete_cell', 'scientific_stage_calls')}, indent=2))
    for row in d['counts_by_condition_replica']:
        print(row['condition'], row['BR'], row['cells'], 'cells')
    for o in d['method_overrides']:
        print('method override:', o['stage'] + '.' + o['parameter'], o['template_value'], '->', o['value'])
    print('written' if args.write else 'dry run: nothing written', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
