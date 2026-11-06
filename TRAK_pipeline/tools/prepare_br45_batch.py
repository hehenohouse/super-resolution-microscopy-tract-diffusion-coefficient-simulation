#!/usr/bin/env python3
"""Prepare portable BR4/BR5-only queues and five-replica summary inventories.

Only directory entries, first TIFF IFDs and small configuration files are read.
This tool never imports or invokes a scientific stage, reads TIFF pixels,
renames inputs, modifies old inventories, or starts a cell.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import uuid

SOURCE_ROOT = Path(__file__).resolve().parents[1]
COMMON_ROOT = SOURCE_ROOT.parents[1] if SOURCE_ROOT.parent.name == 'releases' else SOURCE_ROOT
BATCH_ID = '20261102_BR4_BR5'
CONDITIONS = ('WT', 'TDK', 'MDK')
REPLICAS = ('BR1', 'BR2', 'BR3', 'BR4', 'BR5')
DATASETS = {'TRAK1': 'Xinran_TRAK1', 'TRAK2': 'Xinran_FullsetTRAK2'}
MDK_TRAK2_MAP = {'20260820_MDK_trak2_prcs': 'BR4', '20260910_MDK_TRAK2_prcs': 'BR5'}
EXPECTED_NEW = {'TRAK1': 81, 'TRAK2': 75}
EXPECTED_OLD = {'TRAK1': 105, 'TRAK2': 106}


def natural(value):
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', str(value)))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def visible_entries(path, directories=False):
    return sorted((p for p in Path(path).iterdir() if not p.name.startswith('.')
                   and (p.is_dir() if directories else p.is_file())), key=lambda p: natural(p.name))


def replica_name(molecule, condition, directory):
    name = Path(directory).name
    if molecule == 'TRAK2' and condition == 'MDK':
        if name not in MDK_TRAK2_MAP:
            raise ValueError('Unregistered TRAK2 MDK replica folder: ' + name)
        return MDK_TRAK2_MAP[name]
    match = re.search(r'_(BR[45])$', name)
    if not match:
        raise ValueError('New replica folder must explicitly end in BR4/BR5: ' + name)
    return match.group(1)


def cell_name(directory):
    match = re.search(r'_(T\d+)f?$', Path(directory).name)
    if not match:
        raise ValueError('Original cell identifier is missing: ' + str(directory))
    return match.group(1)


def unique_inputs(directory):
    files = visible_entries(directory)
    channels = [[p for p in files if p.name.startswith(prefix) and p.suffix.lower() in ('.tif', '.tiff')]
                for prefix in ('C1-', 'C2-')]
    rois = [p for p in files if p.suffix.lower() == '.roi']
    if any(len(group) != 1 for group in channels) or len(rois) != 1:
        raise ValueError('Expected exactly one C1 TIFF, C2 TIFF and ROI: ' + str(directory))
    return channels[0][0], channels[1][0], rois[0]


def tiff_metadata(path):
    """Read only classic-TIFF first-IFD metadata and ImageJ frame declarations.

    No page-chain enumeration or pixel/strip/tile payload is read. Frame count
    is the explicit ImageJ metadata count, not an independent pixel audit.
    """
    path = Path(path)
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8}
    with path.open('rb') as stream:
        header = stream.read(8)
        if len(header) != 8 or header[:2] not in (b'II', b'MM'):
            raise ValueError('Invalid TIFF header: ' + str(path))
        endian = '<' if header[:2] == b'II' else '>'
        if struct.unpack(endian + 'H', header[2:4])[0] != 42:
            raise ValueError('This metadata reader supports the new dataset classic TIFF format only: ' + str(path))
        offset = struct.unpack(endian + 'I', header[4:8])[0]
        stream.seek(offset); count_data = stream.read(2)
        if len(count_data) != 2:
            raise ValueError('Truncated first TIFF IFD: ' + str(path))
        count = struct.unpack(endian + 'H', count_data)[0]
        if not 1 <= count <= 4096:
            raise ValueError('Unreasonable first TIFF IFD count')
        directory = stream.read(count * 12)
        if len(directory) != count * 12:
            raise ValueError('Truncated TIFF directory')
        values = {}
        for i in range(count):
            entry = directory[i * 12:(i + 1) * 12]
            tag, kind, n = struct.unpack(endian + 'HHI', entry[:8])
            if tag not in (256, 257, 258, 259, 270, 274, 277, 282, 283, 296, 339):
                continue
            if kind not in sizes or n < 1 or n * sizes[kind] > 1024 * 1024:
                raise ValueError('Unsupported or excessive required TIFF metadata')
            length = n * sizes[kind]
            if length <= 4:
                data = entry[8:8 + length]
            else:
                stream.seek(struct.unpack(endian + 'I', entry[8:12])[0]); data = stream.read(length)
            if len(data) != length:
                raise ValueError('Truncated TIFF metadata payload')
            if kind == 2:
                value = data.rstrip(b'\0').decode('utf-8', errors='strict')
            elif kind == 5:
                pairs = struct.unpack(endian + 'I' * (2 * n), data)
                if any(pairs[j + 1] == 0 for j in range(0, len(pairs), 2)):
                    raise ValueError('Invalid TIFF rational denominator')
                value = [pairs[j] / pairs[j + 1] for j in range(0, len(pairs), 2)]
            else:
                fmt = {1: 'B', 3: 'H', 4: 'I', 11: 'f', 12: 'd'}[kind]
                value = list(struct.unpack(endian + fmt * n, data))
            values[tag] = value
    scalar = lambda tag, default=None: values.get(tag, [default])[0]
    description = values.get(270, '')
    imagej = dict(line.split('=', 1) for line in description.splitlines() if '=' in line)
    if 'ImageJ' not in imagej:
        raise ValueError('Explicit ImageJ acquisition metadata is required: ' + str(path))
    images = int(imagej.get('images', '0')); frames = int(imagej.get('frames', '0'))
    channels = int(imagej.get('channels', '1')); slices = int(imagej.get('slices', '1'))
    if images != frames * channels * slices or channels != 1 or slices != 1:
        raise ValueError('TIFF time/channel/Z metadata is incompatible with original TYX data')
    if scalar(258) != 16 or scalar(277, 1) != 1 or scalar(339, 1) != 1:
        raise ValueError('Expected single-channel uint16 TIFF')
    if scalar(274, 1) != 1:
        raise ValueError('Non-native TIFF orientation is not supported without a new coordinate contract')
    unit = imagej.get('unit', '').lower()
    if unit not in ('micron', 'microns', 'um', 'µm') or scalar(296, 1) != 1:
        raise ValueError('Explicit ImageJ micron calibration is required')
    rx, ry = scalar(282), scalar(283)
    if rx is None or ry is None or min(rx, ry) <= 0:
        raise ValueError('Missing positive pixel calibration')
    pixel = [1000. / rx, 1000. / ry]
    dt = float(imagej.get('finterval', 'nan'))
    shape = [frames, int(scalar(257, 0)), int(scalar(256, 0))]
    if shape != [2500, 684, 428] or not math.isclose(dt, .02, rel_tol=0, abs_tol=1e-12):
        raise ValueError('New TIFF acquisition differs from 2500x684x428 at20ms: ' + str(path))
    if any(not math.isclose(v, 117., rel_tol=0, abs_tol=.001) for v in pixel):
        raise ValueError('New TIFF pixel calibration differs from117nm: ' + str(path))
    if scalar(259, 1) == 1 and path.stat().st_size < frames * 684 * 428 * 2:
        raise ValueError('Uncompressed TIFF is smaller than its declared pixel payload')
    return dict(shape=shape, axes='TYX', dtype='uint16', declared_frames=frames,
                declared_images=images, frame_interval_s=dt, pixel_size_nm_xy=pixel,
                nominal_pixel_size_nm=117., byte_order='little' if endian == '<' else 'big',
                metadata_scope='first_IFD_and_explicit_ImageJ_fields_only; no_pixels_or_full_page_enumeration')


def old_relative(value, root):
    text = str(value).replace('\\', '/')
    marker = '/' + root.name + '/'
    if marker in text:
        text = text.rsplit(marker, 1)[1]
    elif text.startswith(root.name + '/'):
        text = text[len(root.name) + 1:]
    result = (root / text).resolve() if not Path(text).is_absolute() else Path(text).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Old inventory input is outside its dataset: ' + text)
    return result.relative_to(root).as_posix()


def load_old_inventory(config, base_path, root, molecule):
    path = root / config.get('inventory_relative', '_batch_pipeline/inventory.json')
    data = read(path)
    rows = data if isinstance(data, list) else data.get('cells', data.get('samples', []))
    if len(rows) != EXPECTED_OLD[molecule]:
        raise ValueError('Historical inventory count changed unexpectedly')
    result = []
    for original in rows:
        row = deepcopy(original)
        for field in ('source_directory', 'C1_file', 'C2_file', 'ROI_file'):
            if field not in row:
                raise ValueError('Old inventory field missing: ' + field)
            row[field] = old_relative(row[field], root)
        row['source_directory_relative'] = row['source_directory']
        if row.get('BR') not in REPLICAS[:3] or row.get('molecule') != molecule:
            raise ValueError('Historical BR/protein identity changed')
        row['batch_role'] = 'existing_completed_reference'
        result.append(row)
    return result, path


def collect_new(root, molecule, metadata_reader=tiff_metadata):
    nested = root / molecule
    result = []; counts = Counter(); audits = []; replica_paths = {}
    for condition in CONDITIONS:
        folder = nested / condition
        if not folder.is_dir():
            raise FileNotFoundError('New condition directory is missing: ' + str(folder))
        seen_replicas = set()
        for br_dir in visible_entries(folder, directories=True):
            br = replica_name(molecule, condition, br_dir)
            if br in seen_replicas:
                raise ValueError('Multiple physical folders map to the same condition/BR: ' + str(br_dir))
            seen_replicas.add(br); replica_paths[(condition, br)] = br_dir.relative_to(root).as_posix()
            for directory in visible_entries(br_dir, directories=True):
                if directory.name in ('analysis_results', 'test') or directory.name.startswith('Analysis_Figures'):
                    continue
                cell = cell_name(directory); c1, c2, roi = unique_inputs(directory)
                source_files = [c1, c2, roi]
                headers = {key: metadata_reader(path) for key, path in [('C1', c1), ('C2', c2)]}
                if headers['C1']['shape'] != headers['C2']['shape']:
                    raise ValueError('C1/C2 declared grids differ: ' + str(directory))
                acquired = datetime.strptime(directory.name[:6], '%m%d%y').date().isoformat()
                sid = '_'.join([condition, molecule, br, cell])
                result.append(dict(sample_id=sid, sample_name=' '.join([condition, molecule, br, 'cell', cell]),
                    genotype=condition, BR=br, cell=cell, molecule=molecule, acquisition_date=acquired,
                    source_directory=directory.relative_to(root).as_posix(), source_directory_relative=directory.relative_to(root).as_posix(),
                    C1_file=c1.relative_to(root).as_posix(), C2_file=c2.relative_to(root).as_posix(), ROI_file=roi.relative_to(root).as_posix(),
                    status='pending', batch_role='new_BR4_BR5', headers=headers))
                counts[condition, br] += 1
                audits.append(dict(sample_id=sid, source_directory_relative=directory.relative_to(root).as_posix(),
                    original_cell_directory_name=directory.name, parent_replica_directory=br_dir.name,
                    naming_policy='Parent BR metadata is authoritative; original filenames retained; no cross-protein pairing inferred',
                    files=[dict(relative_path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in source_files],
                    TIFF_metadata=headers))
        if seen_replicas != {'BR4', 'BR5'}:
            raise ValueError('Exactly BR4 and BR5 are required for ' + molecule + '/' + condition)
    if len(result) != EXPECTED_NEW[molecule]:
        raise ValueError('New-cell inventory count differs from the confirmed dataset')
    ids = [r['sample_id'] for r in result]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate new sample IDs; no directory/coordinate based deduplication was attempted')
    return result, counts, audits, replica_paths


def rebase_config_paths(config, old_base, new_base):
    c = deepcopy(config)
    def rebase(value):
        path = Path(value); absolute = path.resolve() if path.is_absolute() else (old_base / path).resolve()
        return os.path.relpath(absolute, new_base).replace(os.sep, '/')
    c['data_root'] = rebase(c['data_root'])
    for platform, release in c.get('legacy_releases', {}).items():
        c['legacy_releases'][platform] = rebase(release)
    for profile in c.get('runtime_profiles', {}).values():
        for key in ('reference_file', 'reference_base', 'active_override'):
            if key in profile: profile[key] = rebase(profile[key])
        if 'paths' in profile:
            profile['paths'] = {k: rebase(v) for k, v in profile['paths'].items()}
    return c


def build(common_root=COMMON_ROOT, batch_id=BATCH_ID, release_relative=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}', batch_id):
        raise ValueError('Batch ID must be a simple name without path separators')
    common = Path(common_root).resolve(); batch = (common / 'batches' / batch_id).resolve()
    if not batch.is_relative_to(common / 'batches'):
        raise ValueError('New batch output escapes the common batch root')
    config_dir = batch / 'configs'; datasets = []; outputs = []; source_audit = {}
    for molecule, dataset_name in DATASETS.items():
        base_path = common / 'configs' / (molecule + '.json'); base = read(base_path)
        root = (base_path.parent / base['data_root']).resolve()
        if root.name != dataset_name or root.parent != common.parent:
            raise ValueError('Dataset/common-root relationship differs from the portable drive layout')
        old, inventory_path = load_old_inventory(base, base_path, root, molecule)
        new, counts, audit, replica_paths = collect_new(root, molecule)
        all_rows = old + new
        if len({r['sample_id'] for r in all_rows}) != len(all_rows):
            raise ValueError('New sample IDs collide with an existing BR/cell identity')
        all_rows.sort(key=lambda r: (CONDITIONS.index(r['genotype']), natural(r['BR']), natural(r['cell'])))
        config = rebase_config_paths(base, base_path.parent, config_dir)
        if molecule == 'TRAK1':
            # This legacy release is read only for JSON metadata on either OS.
            config['legacy_releases']['mac'] = config['legacy_releases']['windows']
        config.pop('inventory_relative', None)
        config.update(samples=all_rows, replicas=list(REPLICAS),
            control_relative='_batch_pipeline/batches/' + batch_id,
            legacy_control_relative='_batch_pipeline', batch_id=batch_id,
            new_sample_ids=[r['sample_id'] for r in new],
            read_only_existing_sample_ids=[r['sample_id'] for r in old])
        if config['methods'] != base['methods'] or config['acquisition'] != base['acquisition']:
            raise AssertionError('Scientific upstream methods/acquisition were modified')
        config_path = config_dir / (molecule + '_BR4_BR5.json')
        count_rows = [dict(condition=condition, BR=br, cells=sum(r['genotype'] == condition and r['BR'] == br for r in all_rows))
                      for condition in CONDITIONS for br in REPLICAS]
        datasets.append(dict(molecule=molecule, config_relative=config_path.relative_to(common).as_posix(),
            data_root_relative=root.name, new_sample_ids=config['new_sample_ids'], new_cells=len(new),
            old_cells=len(old), total_cells=len(all_rows), counts_by_condition_replica=count_rows,
            source_directories_by_condition_replica={condition: {br: replica_paths[condition, br] for br in ('BR4', 'BR5')} for condition in CONDITIONS},
            upstream_methods_digest=digest(base['methods']), upstream_parameters_unchanged=True))
        outputs.append((config_path, config))
        source_audit[molecule] = dict(base_config_relative=base_path.relative_to(common).as_posix(),
            base_config_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
            historical_inventory_relative=inventory_path.relative_to(common.parent).as_posix(),
            historical_inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(), new_cells=audit)
    manifest = dict(schema_version=1, batch_id=batch_id, common_root_relative='../..', path_basis='common_root',
        release_relative=release_relative, ready_for_execution=bool(release_relative),
        created_utc=datetime.now(timezone.utc).isoformat(), datasets=datasets,
        new_cells=sum(d['new_cells'] for d in datasets), existing_cells=sum(d['old_cells'] for d in datasets),
        summary_cells=sum(d['total_cells'] for d in datasets), replicas=list(REPLICAS),
        queue_rule='Only exact new_sample_ids may enter cell execution; all configured samples participate in completed-only five-BR summaries',
        completion_rule='After new-cell queue, summarize both full inventories using five replicas per condition; incomplete/failed cells remain explicit',
        TRAK2_MDK_replica_mapping=dict(MDK_TRAK2_MAP), TRAK2_MDK_mapping_authorized_by_user=True,
        cross_protein_replica_pairing='unconfirmed; do not infer pairing from filenames or BR numbers',
        scientific_defaults='Original raw C1 support50>=10 -> post-min32 -> holes<64; original1SE ellipse; existing localization/QC/TARDIS/bootstrap rules',
        metadata_check_scope='312 TIFF first-IFD headers and explicit ImageJ fields; unique C1/C2/ROI per new cell. No pixels/full hashes/page-chain enumeration or science calls.',
        new_cells_started=0, scientific_stage_calls=0)
    return manifest, outputs, source_audit


def write_prepared(common, manifest, outputs, audit):
    common = Path(common).resolve(); batch = common / 'batches' / manifest['batch_id']
    if manifest['release_relative']:
        release = (common / manifest['release_relative']).resolve()
        if not release.is_relative_to(common / 'releases') or not (release / 'release_manifest.json').is_file():
            raise ValueError('An immutable common release manifest is required before execution can be enabled')
    for path, config in outputs: atomic_json(path, config)
    atomic_json(batch / 'source_inventory_audit.json', audit)
    atomic_json(batch / 'batch_manifest.json', manifest)
    lines = ['# New BR4/BR5 input inventory', '', '| Protein | Existing cells | New cells | Five-BR summary cells |', '| --- | ---: | ---: | ---: |']
    for d in manifest['datasets']: lines.append('| {molecule} | {old_cells} | {new_cells} | {total_cells} |'.format(**d))
    lines += ['', 'Inputs were not renamed, copied or modified. Original WTbr1/WTbr02 filenames are retained; parent folder BR identity is authoritative.',
        'TRAK2 MDK: 20260820_MDK_trak2_prcs = BR4; 20260910_MDK_TRAK2_prcs = BR5 (user confirmed).',
        'Checks read first TIFF IFD/ImageJ metadata only, not image pixels or every page. No scientific stages or cells were started.',
        'Execution queue contains only the 156 new cells. Completed-only summaries include all 367 configured cells across five replicas per condition.', '']
    (batch / 'INPUT_INVENTORY.md').write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-root', type=Path, default=COMMON_ROOT)
    parser.add_argument('--batch-id', default=BATCH_ID)
    parser.add_argument('--release-relative', help='Frozen release relative to common root; otherwise generated manifest is not execution-ready')
    parser.add_argument('--write', action='store_true', help='Write only new batch configs/audit, never source data or old configs')
    args = parser.parse_args(argv)
    manifest, outputs, audit = build(args.common_root, args.batch_id, args.release_relative)
    if args.write: write_prepared(args.common_root, manifest, outputs, audit)
    print(json.dumps({k:manifest[k] for k in ('batch_id','new_cells','existing_cells','summary_cells','ready_for_execution','scientific_stage_calls')}, indent=2))
    for dataset in manifest['datasets']:
        print(dataset['molecule'] + ': ' + str(dataset['new_cells']) + ' new / ' + str(dataset['total_cells']) + ' total cells', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
