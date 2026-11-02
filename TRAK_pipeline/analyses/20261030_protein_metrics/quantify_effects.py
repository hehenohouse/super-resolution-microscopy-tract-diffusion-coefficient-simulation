#!/usr/bin/env python3
"""Quantify saved TRAK1-minus-TRAK2 contrasts; no tests or upstream analyses."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

METHOD = 'saved_protein_contrast_quantification'
METHOD_VERSION = '1'
ENDPOINT_UNITS = {
    'on_percentage': 'percentage_points',
    'mean_ROI_area_um2': 'um2',
    'first_mask_ROI_area_um2': 'um2',
}
FIELDS = [
    'endpoint', 'condition', 'protein_a', 'protein_b',
    'mean_a', 'mean_b', 'difference_a_minus_b', 'effect_unit',
    'relative_percent_vs_TRAK2', 'mean_ratio_TRAK1_over_TRAK2',
    'n_BR_a', 'n_BR_b', 'n_cells_a', 'n_cells_b', 'p_raw', 'p_holm',
]


def fingerprint(path: Path) -> dict:
    data = path.read_bytes()
    return {'relative_path': str(path.name), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='',
                                     dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def quantify(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding='utf-8'))
    run = Path(config['run_dir']).resolve()
    portable_root = Path('/Volumes/My Book').resolve()
    if not run.is_relative_to(portable_root):
        raise ValueError('This analysis must write inside My Book.')
    source = run / 'data' / 'comparisons.csv'
    before = fingerprint(source)
    config_before = fingerprint(config_path)
    rows = list(csv.DictReader(source.open(encoding='utf-8', newline='')))
    selected = [row for row in rows
                if row['comparison_type'] == 'proteins_within_condition']
    expected = {(endpoint, condition) for endpoint in ENDPOINT_UNITS
                for condition in ('WT', 'TDK', 'MDK')}
    observed = [(row['endpoint'], row['condition_a']) for row in selected]
    if len(selected) != 9 or len(set(observed)) != 9 or set(observed) != expected:
        raise ValueError('Expected exactly nine unique protein contrasts.')
    out = []
    max_difference_error = 0.0
    for row in selected:
        endpoint = row['endpoint']
        if row['protein_a'] != 'TRAK1' or row['protein_b'] != 'TRAK2':
            raise ValueError('Contrast orientation must be TRAK1 minus TRAK2.')
        if row['condition_a'] != row['condition_b']:
            raise ValueError('Protein contrast mixes conditions.')
        if row['effect_unit'] != ENDPOINT_UNITS[endpoint]:
            raise ValueError('Saved contrast units differ from endpoint units.')
        mean_a, mean_b = float(row['mean_a']), float(row['mean_b'])
        saved_difference = float(row['difference_a_minus_b'])
        difference = mean_a - mean_b
        if not all(math.isfinite(x) for x in (mean_a, mean_b, difference)) or mean_b <= 0:
            raise ValueError('Nonfinite mean or nonpositive TRAK2 reference mean.')
        error = abs(saved_difference - difference)
        max_difference_error = max(error, max_difference_error)
        if not math.isclose(saved_difference, difference, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Saved difference does not agree with saved means.')
        if int(row['n_BR_a']) != 3 or int(row['n_BR_b']) != 3:
            raise ValueError('This analysis expects three independent BRs per group.')
        result = {
            'endpoint': endpoint, 'condition': row['condition_a'],
            'protein_a': 'TRAK1', 'protein_b': 'TRAK2',
            'mean_a': mean_a, 'mean_b': mean_b,
            'difference_a_minus_b': difference,
            'effect_unit': ENDPOINT_UNITS[endpoint],
            'relative_percent_vs_TRAK2': 100.0 * difference / mean_b,
            'mean_ratio_TRAK1_over_TRAK2': mean_a / mean_b,
            'n_BR_a': int(row['n_BR_a']), 'n_BR_b': int(row['n_BR_b']),
            'n_cells_a': int(row['n_cells_a']), 'n_cells_b': int(row['n_cells_b']),
            'p_raw': float(row['p_raw']), 'p_holm': float(row['p_holm']),
        }
        if not math.isclose(result['mean_ratio_TRAK1_over_TRAK2'],
                            1 + result['relative_percent_vs_TRAK2'] / 100,
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Relative and ratio effects disagree.')
        out.append(result)
    csv_buffer = io.StringIO(newline='')
    writer = csv.DictWriter(csv_buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(out)
    metadata = {
        'schema_version': 1,
        'method_id': METHOD, 'method_version': METHOD_VERSION,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'definition': {
            'mean': 'Equal-cell arithmetic mean within each BR; equal-BR condition mean.',
            'difference': 'TRAK1 condition mean minus TRAK2 condition mean.',
            'relative_percent_vs_TRAK2': '100 * (TRAK1 mean - TRAK2 mean) / TRAK2 mean.',
            'mean_ratio': 'TRAK1 condition mean / TRAK2 condition mean.',
            'interpretation': 'Observed descriptive contrasts; not standardized effect sizes, confidence intervals or significance claims.',
            'p_values': 'Copied unchanged from comparisons.csv; no statistical test rerun.',
            'on_percentage_unit': 'Absolute difference in percentage points; relative difference in percent is a separate measure.',
            'area': 'Total binary mitochondrial-mask foreground inside the original cell ROI, not single-organelle size.',
        },
        'source': before,
        'config_source': config_before,
        'records': out,
    }
    atomic_write(run / 'data' / 'effect_sizes_proteins.csv', csv_buffer.getvalue())
    atomic_write(run / 'data' / 'effect_sizes_proteins.json',
                 json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    if fingerprint(source) != before or fingerprint(config_path) != config_before:
        raise RuntimeError('Source comparisons or configuration changed during quantification.')
    validation = {
        'passed': True, 'method_id': METHOD, 'method_version': METHOD_VERSION,
        'records': len(out), 'unique_endpoint_condition_pairs': len(set(observed)),
        'contrast_orientation': 'TRAK1_minus_TRAK2',
        'maximum_saved_difference_discrepancy': max_difference_error,
        'units_checked': True, 'baseline_means_positive': True,
        'relative_and_ratio_consistency': True,
        'source_unchanged': True, 'config_unchanged': True,
        'new_statistical_test_calls': 0, 'upstream_scientific_stage_calls': 0,
        'source': before,
    }
    atomic_write(run / 'validation_effect_sizes.json',
                 json.dumps(validation, indent=2, allow_nan=False) + '\n')
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    result = quantify(args.config.resolve())
    print(json.dumps({'method_id': METHOD, 'records': result['records']}, indent=2))


if __name__ == '__main__':
    main()
