"""Reader-facing localization counts from saved cell scalar tables only.

Totals count QC-selected localization observations, including unknown labels.
Group count summaries are per-cell means; they do not change the existing On%
or area means. This module performs no detection, fitting, segmentation or TARDIS work.
"""
from collections import defaultdict
from pathlib import Path
import csv
import math
import re

CONDITIONS = ('WT', 'TDK', 'MDK')
COUNT_KEYS = ('total_localizations', 'On', 'Not_on', 'unknown', 'QC_excluded')
CELL_CSV_HEADERS = ['condition', 'BR', 'cell', 'protein', 'acquisition_date', *COUNT_KEYS]
PER_CELL_CSV_HEADERS = ['level', 'condition', 'BR', 'protein', 'cells_in_table', 'cells_with_counts', 'localizations_per_cell', 'on_per_cell', 'not_on_per_cell', 'unknown_per_cell', 'qc_excluded_per_cell']
MEAN_CSV_HEADERS = ['condition', 'BR', 'protein', 'averaging_code', 'cells', 'replicas', 'total_localizations_mean', 'on_count_mean', 'not_on_count_mean']
MEAN_KEYS = ('total_localizations', 'On', 'Not_on')
AVERAGING_LABELS = {'cell_mean_within_BR':'Cell mean within BR', 'all_cell_mean':'All-cell mean', 'equal_replica_mean':'Equal-replica mean'}


def _natural(value):
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', str(value)))


def _condition_order(value):
    return CONDITIONS.index(value) if value in CONDITIONS else len(CONDITIONS)


def _integer(value, label):
    if value in (None, ''):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed != int(parsed):
        raise ValueError('Invalid saved localization count: ' + label + '=' + str(value))
    return int(parsed)


def cell_count_records(cells):
    """Select exact saved counts; missing rows stay blank rather than becoming zero."""
    result = []
    seen = set()
    for row in cells:
        record = dict(condition=row.get('genotype', row.get('condition', '')),
                      BR=row.get('BR', ''), cell=row.get('cell', ''),
                      protein=row.get('molecule', row.get('protein', '')),
                      acquisition_date=row.get('acquisition_date', '') or None)
        identity = tuple(record[key] for key in ('condition', 'BR', 'cell', 'protein'))
        if identity in seen:
            raise ValueError('Duplicate cell in saved localization counts: ' + '/'.join(identity))
        seen.add(identity)
        values = {key:_integer(row.get(source), source) for key, source in
                  [('total_localizations', 'selected'), ('On', 'On'), ('Not_on', 'Not_on'),
                   ('unknown', 'unknown'), ('QC_excluded', 'QC_excluded')]}
        available = [value is not None for value in values.values()]
        if any(available) and not all(available):
            raise ValueError('Partially missing saved localization counts: ' + '/'.join(identity))
        if all(available):
            if values['On'] + values['Not_on'] + values['unknown'] != values['total_localizations']:
                raise ValueError('Saved localization counts do not conserve selected observations: ' + '/'.join(identity))
            candidates = _integer(row.get('ROI_candidates'), 'ROI_candidates')
            if candidates is not None and values['total_localizations'] + values['QC_excluded'] != candidates:
                raise ValueError('Saved localization counts do not conserve ROI candidates: ' + '/'.join(identity))
        record.update(values)
        result.append(record)
    return sorted(result, key=lambda row:(_condition_order(row['condition']), _natural(row['BR']), _natural(row['cell']), row['protein']))


def grouped_count_records(records, level):
    """Sum available cell counts at BR or condition level, never average counts."""
    if level not in ('BR', 'condition'):
        raise ValueError('Unsupported localization-count grouping: ' + str(level))
    groups = defaultdict(list)
    for row in records:
        groups[(row['condition'], row['BR'] if level == 'BR' else '', row['protein'])].append(row)
    result = []
    for (condition, br, protein), rows in sorted(groups.items(), key=lambda item:(_condition_order(item[0][0]), _natural(item[0][1]), item[0][2])):
        included = [row for row in rows if row['total_localizations'] is not None]
        record = dict(level=level, condition=condition, BR=br, protein=protein,
                      cells_in_table=len(rows), cells_with_counts=len(included))
        record.update({key:sum(row[key] for row in included) if included else None for key in COUNT_KEYS})
        result.append(record)
    return result


def summary_count_values(records, condition, br=None):
    """Return N and per-cell count means for an existing saved summary group."""
    level = 'condition' if br is None else 'BR'
    matched = [row for row in grouped_count_records(records, level)
               if row['condition'] == condition and (br is None or row['BR'] == br)]
    if len(matched) > 1:
        raise ValueError('Reader count group contains more than one protein')
    if not matched:
        return [0] + [None] * len(COUNT_KEYS)
    row = matched[0]
    n = row['cells_with_counts']
    return [n] + [row[key] / n if n else None for key in COUNT_KEYS]


def count_mean_records(records, node_level):
    """Count means with explicit cell-equal and replica-equal membership.

    A BR mean gives each available cell one vote. All-cell means pool the same
    available cell observations. Equal-replica means give each available BR mean
    one vote; they are deliberately not localization-count-weighted averages.
    """
    if node_level == 'cell':
        return []
    if node_level not in ('BR', 'condition', 'comparison'):
        raise ValueError('Unsupported reader localization-mean level: ' + str(node_level))
    br_groups = defaultdict(list)
    condition_groups = defaultdict(list)
    for row in records:
        br_groups[(row['condition'], row['BR'], row['protein'])].append(row)
        condition_groups[(row['condition'], row['protein'])].append(row)
    result = []
    for (condition, br, protein), rows in sorted(br_groups.items(), key=lambda item:(_condition_order(item[0][0]), _natural(item[0][1]), item[0][2])):
        included = [row for row in rows if row['total_localizations'] is not None]
        means = [sum(row[key] for row in included) / len(included) if included else None for key in MEAN_KEYS]
        result.append(dict(condition=condition, BR=br, protein=protein,
                           averaging_code='cell_mean_within_BR', cells=len(included), replicas=int(bool(included)),
                           total_localizations_mean=means[0], on_count_mean=means[1], not_on_count_mean=means[2]))
    if node_level in ('condition', 'comparison'):
        for (condition, protein), rows in sorted(condition_groups.items(), key=lambda item:(_condition_order(item[0][0]), item[0][1])):
            included = [row for row in rows if row['total_localizations'] is not None]
            br_means = [row for row in result if row['condition'] == condition and row['protein'] == protein
                        and row['averaging_code'] == 'cell_mean_within_BR' and row['cells'] > 0]
            pooled = [sum(row[key] for row in included) / len(included) if included else None for key in MEAN_KEYS]
            equal_br = [math.fsum(row[key] for row in br_means) / len(br_means) if br_means else None
                        for key in ('total_localizations_mean', 'on_count_mean', 'not_on_count_mean')]
            for code, values in [('all_cell_mean', pooled), ('equal_replica_mean', equal_br)]:
                result.append(dict(condition=condition, BR='', protein=protein, averaging_code=code,
                                   cells=len(included), replicas=len(br_means),
                                   total_localizations_mean=values[0], on_count_mean=values[1], not_on_count_mean=values[2]))
    return result


def per_cell_summary_records(records, node_level):
    """Count columns divided by the same available-cell N within each group."""
    groups = []
    if node_level in ('BR', 'condition', 'comparison'):
        groups.extend(grouped_count_records(records, 'BR'))
    if node_level in ('condition', 'comparison'):
        groups.extend(grouped_count_records(records, 'condition'))
    result = []
    output_keys = ('localizations_per_cell', 'on_per_cell', 'not_on_per_cell', 'unknown_per_cell', 'qc_excluded_per_cell')
    for group in groups:
        n = group['cells_with_counts']
        row = {key:group[key] for key in PER_CELL_CSV_HEADERS[:6]}
        row.update({output:group[source] / n if n else None for output, source in zip(output_keys, COUNT_KEYS)})
        result.append(row)
    return result


def count_csv_specs(job):
    """Return CSV specifications for the workbook exporter's atomic writer.

    Each item has filename, headers and rows. The exporter writes these files to
    job['data_dir']; existing scientific CSVs and completion records stay intact.
    Numeric counts remain full-precision integers, with unavailable values blank.
    """
    source = Path(job['data_dir']) / 'cell_metrics.csv'
    with source.open(newline='', encoding='utf-8-sig') as stream:
        cells = cell_count_records(list(csv.DictReader(stream)))
    per_cell = per_cell_summary_records(cells, job['level'])
    specifications = [dict(filename='localization_counts.csv', headers=CELL_CSV_HEADERS,
                           rows=[[row.get(key) for key in CELL_CSV_HEADERS] for row in cells])]
    if per_cell:
        specifications.append(dict(filename='localization_count_per_cell_summary.csv', headers=PER_CELL_CSV_HEADERS,
                                   rows=[[row.get(key) for key in PER_CELL_CSV_HEADERS] for row in per_cell]))
    means = count_mean_records(cells, job['level'])
    if means:
        specifications.append(dict(filename='localization_count_means.csv', headers=MEAN_CSV_HEADERS,
                                   rows=[[row.get(key) for key in MEAN_CSV_HEADERS] for row in means]))
    return specifications
