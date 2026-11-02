#!/usr/bin/env python3
"""Export reader-facing XLSX workbooks from saved scalar CSVs.

Runtime uses only the Python standard library and formatted, Artifact-authored
XLSX templates. It never recalculates scientific metrics or edits existing CSVs,
PNGs, per-node HTML indexes, or scientific completion records. Additional
localization count CSVs are written beside the original summary CSVs.
"""
from pathlib import Path
from datetime import datetime, date, timezone
import argparse
import copy
import csv
import io
import hashlib
import html
import json
import math
import os
import posixpath
import sys
import uuid
import zipfile
import xml.etree.ElementTree as ET

MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG = 'http://schemas.openxmlformats.org/package/2006/relationships'
NS = {'x': MAIN, 'r': REL}
ET.register_namespace('x', MAIN)
ET.register_namespace('r', REL)
HERE = Path(__file__).resolve().parent
SCHEMA = 2


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def q(name): return '{' + MAIN + '}' + name


def _json(path):
    with Path(path).open(encoding='utf-8') as stream: return json.load(stream)


def _guard(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not root.is_dir() or not path.is_relative_to(root):
        raise OSError(19, 'Reader-table destination is unavailable or outside the data drive: ' + str(path))
    return path


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def _column(index):
    result = ''
    while index:
        index, remain = divmod(index - 1, 26); result = chr(65 + remain) + result
    return result


def _date_serial(value):
    if isinstance(value, datetime): value = value.date()
    if not isinstance(value, date):
        text = str(value).strip()
        parsed = None
        for pattern in ('%Y-%m-%d', '%m/%d/%Y', '%Y%m%d', '%m%d%y'):
            try: parsed = datetime.strptime(text, pattern).date(); break
            except ValueError: pass
        if parsed is None: raise ValueError('Unsupported acquisition date: ' + text)
        value = parsed
    return (value - date(1899, 12, 30)).days


def _numeric(value):
    value = float(value)
    if not math.isfinite(value): raise ValueError('A reader-table numeric value is nonfinite')
    return value


def _cell(reference, value, kind, style=None):
    attrs = {'r': reference}
    if style is not None: attrs['s'] = str(style)
    cell = ET.Element(q('c'), attrs)
    if value is None or value == '': return cell
    if kind in ('text', 'string'):
        cell.set('t', 'inlineStr'); item = ET.SubElement(cell, q('is')); text = ET.SubElement(item, q('t'))
        text.text = str(value)
        if str(value) != str(value).strip(): text.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    else:
        if kind == 'date': number = _date_serial(value)
        elif kind == 'integer':
            number = _numeric(value)
            if number != int(number): raise ValueError('Reader count is not integer: ' + str(value))
            number = int(number)
        else: number = _numeric(value)
        ET.SubElement(cell, q('v')).text = repr(number)
    return cell


def _resolve_part(base, target):
    return target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join(posixpath.dirname(base), target))


def _sheet_parts(archive):
    workbook = ET.fromstring(archive.read('xl/workbook.xml'))
    relations = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
    targets = {item.get('Id'): _resolve_part('xl/workbook.xml', item.get('Target')) for item in relations}
    sheets = workbook.find(q('sheets'))
    return workbook, [(sheet, targets[sheet.get('{' + REL + '}id')]) for sheet in sheets]


def _worksheet(archive, part, spec, replacement):
    root = ET.fromstring(archive.read(part)); data = root.find(q('sheetData'))
    rows = {int(row.get('r')): row for row in data}
    if 6 not in rows or 7 not in rows: raise ValueError('Template must have header row 6 and data row 7')
    ncols = len(spec['headers']); end = max(7, 6 + len(spec['rows'])); last_col = _column(ncols)
    for number, value in ((1, spec['title']), (2, spec['subtitle']), (3, spec.get('notes', ''))):
        row = rows.get(number)
        if row is None: raise ValueError('Template title/subtitle/notes rows missing')
        original = row.find(q('c')); style = original.get('s') if original is not None else None
        old = next((cell for cell in row if cell.get('r') == 'A' + str(number)), None)
        if old is not None: row.remove(old)
        row.insert(0, _cell('A' + str(number), value, 'text', style))
    header = rows[6]
    header_styles = {cell.get('r').rstrip('0123456789'): cell.get('s') for cell in header}
    for child in list(header): header.remove(child)
    for index, value in enumerate(spec['headers'], 1):
        col = _column(index); header.append(_cell(col + '6', value, 'text', header_styles.get(col)))
    style_rows = [rows[7], rows.get(8, rows[7])]
    styles = [{cell.get('r').rstrip('0123456789'): cell.get('s') for cell in row} for row in style_rows]
    attributes = [dict(row.attrib) for row in style_rows]
    for row in list(data):
        if int(row.get('r')) >= 7: data.remove(row)
    values = spec['rows'] or [[None] * ncols]
    for offset, values_row in enumerate(values):
        number = offset + 7; attrs = dict(attributes[offset % 2]); attrs['r'] = str(number)
        attrs.pop('spans', None); row = ET.SubElement(data, q('row'), attrs)
        if len(values_row) != ncols: raise ValueError('Workbook data row/header width mismatch')
        for index, (value, kind) in enumerate(zip(values_row, spec['column_types']), 1):
            col = _column(index); row.append(_cell(col + str(number), value, kind, styles[offset % 2].get(col)))
    dimension = root.find(q('dimension'))
    if dimension is not None: dimension.set('ref', 'A1:' + last_col + str(end))
    auto = root.find(q('autoFilter'))
    if auto is not None: auto.set('ref', 'A6:' + last_col + str(end))
    # Table relationships are sheet-local; preserve table names and style metadata.
    relpart = posixpath.join(posixpath.dirname(part), '_rels', posixpath.basename(part) + '.rels')
    if relpart in archive.namelist():
        rels = ET.fromstring(archive.read(relpart))
        for rel in rels:
            if rel.get('Type', '').endswith('/table'):
                tablepart = _resolve_part(part, rel.get('Target')); table = ET.fromstring(archive.read(tablepart))
                table.set('ref', 'A6:' + last_col + str(end))
                autofilter = table.find(q('autoFilter'))
                if autofilter is not None: autofilter.set('ref', table.get('ref'))
                cols = table.find(q('tableColumns'))
                if cols is None or len(cols) != ncols: raise ValueError('Template table/header width mismatch')
                for column, text in zip(cols, spec['headers']): column.set('name', str(text))
                replacement[tablepart] = ET.tostring(table, encoding='utf-8', xml_declaration=True)
    return root


def write_workbook(template, job, destination):
    """Populate a formatted workbook template without a spreadsheet dependency."""
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name('.' + destination.stem + '.' + uuid.uuid4().hex + '.tmp.xlsx')
    try:
        with zipfile.ZipFile(template) as source:
            workbook, parts = _sheet_parts(source)
            if len(parts) != len(job['sheets']): raise ValueError('Template/sheet count mismatch')
            replacements = {}
            for (sheet, part), spec in zip(parts, job['sheets']):
                sheet.set('name', spec['name'])
                root = _worksheet(source, part, spec, replacements)
                replacements[part] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
            replacements['xl/workbook.xml'] = ET.tostring(workbook, encoding='utf-8', xml_declaration=True)
            # Shared strings left in untouched template cells remain valid.
            if 'xl/sharedStrings.xml' in source.namelist():
                strings = ET.fromstring(source.read('xl/sharedStrings.xml')); count = 0
                for _, part in parts:
                    root = ET.fromstring(replacements[part])
                    count += sum(cell.get('t') == 's' for cell in root.iter(q('c')))
                strings.set('count', str(count)); strings.set('uniqueCount', str(len(strings)))
                replacements['xl/sharedStrings.xml'] = ET.tostring(strings, encoding='utf-8', xml_declaration=True)
            with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as target:
                for info in source.infolist():
                    target.writestr(info, replacements.get(info.filename, source.read(info.filename)))
        verify_workbook(temporary, job)
        os.replace(temporary, destination)
    finally:
        if temporary.exists(): temporary.unlink()


def _read_cell(cell, strings):
    kind = cell.get('t'); value = cell.find(q('v'))
    if kind == 'inlineStr': return ''.join(cell.find(q('is')).itertext())
    if value is None: return None
    if kind == 's': return strings[int(value.text)]
    if kind in ('str', 'e'): return value.text or ''
    if kind == 'd': return float(_date_serial((value.text or '')[:10]))
    return float(value.text) if value.text else None


def verify_workbook(path, job):
    """Check saved scalar values and date/percentage cell formats; no analysis."""
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None: raise ValueError('Reader workbook failed ZIP CRC')
        workbook, parts = _sheet_parts(archive)
        if len(parts) != len(job['sheets']): raise ValueError('Reader worksheet count mismatch')
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(item.itertext()) for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        styles = ET.fromstring(archive.read('xl/styles.xml'))
        xfs = styles.find(q('cellXfs'))
        formats = {int(item.get('numFmtId')): item.get('formatCode') for item in styles.find(q('numFmts')) or []}
        formats.update({0:'General', 3:'#,##0', 9:'0%', 10:'0.00%', 14:'mm-dd-yy'})
        checked = 0
        for (sheet, part), spec in zip(parts, job['sheets']):
            if sheet.get('name') != spec['name']: raise ValueError('Reader sheet name mismatch')
            root = ET.fromstring(archive.read(part)); cells = {cell.get('r'): cell for cell in root.iter(q('c'))}
            for index, header in enumerate(spec['headers'], 1):
                ref = _column(index) + '6'
                if ref not in cells or _read_cell(cells[ref], strings) != header: raise ValueError('Reader header mismatch: ' + ref)
            for row_index, row in enumerate(spec['rows'], 7):
                for col_index, (expected, kind) in enumerate(zip(row, spec['column_types']), 1):
                    ref = _column(col_index) + str(row_index); cell = cells.get(ref)
                    actual = _read_cell(cell, strings) if cell is not None else None
                    if expected is None or expected == '':
                        if actual not in (None, ''): raise ValueError('Expected blank reader cell: ' + ref)
                    elif kind in ('text', 'string'):
                        if actual != str(expected): raise ValueError('Reader text mismatch: ' + ref)
                    else:
                        expected_number = float(_date_serial(expected)) if kind == 'date' else float(expected)
                        if actual is None or not math.isclose(float(actual), expected_number, rel_tol=1e-13, abs_tol=1e-12):
                            raise ValueError('Reader numeric mismatch: ' + ref)
                        if kind in ('date', 'percent'):
                            style = xfs[int(cell.get('s', '0'))]
                            code = formats.get(int(style.get('numFmtId', '0')), '')
                            if kind == 'date' and 'yyyy' not in code.lower(): raise ValueError('Acquisition date lacks yyyy-mm-dd formatting: ' + ref)
                            if kind == 'percent' and '%' not in code: raise ValueError('On fraction lacks percentage formatting: ' + ref)
                    checked += 1
            # A stale template row must not survive when a group loses members.
            for ref, cell in cells.items():
                rownum = int(''.join(ch for ch in ref if ch.isdigit()))
                if rownum > max(7, 6 + len(spec['rows'])) and _read_cell(cell, strings) not in (None, ''):
                    raise ValueError('Stale reader data after current rows: ' + ref)
        return dict(sheets=len(parts), data_cells_checked=checked, values_match=True)


def _signature(job, template):
    payload = {'sheets':job['sheets'], 'template_sha256':sha(template), 'exporter_sha256':sha(__file__),
               'reader_data_sha256':sha(HERE / 'reader_table_data.py'),
               'count_data_sha256':sha(HERE / 'localization_count_data.py'), 'schema_version':SCHEMA}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _descriptor(root, path):
    path = Path(path); return dict(relative_path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha(path))


def _valid_output(root, descriptor):
    if not descriptor: return False
    path = _guard(root, root / descriptor['relative_path'])
    return path.is_file() and path.stat().st_size == descriptor['bytes'] and sha(path) == descriptor['sha256']


def _write_count_tables(root, job, check_only=False):
    """Export exact integer count records without changing source scalar tables."""
    from localization_count_data import count_csv_specs
    descriptors = []
    for spec in count_csv_specs(job):
        path = _guard(root, Path(job['data_dir']) / spec['filename'])
        stream = io.StringIO(newline='')
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(spec['headers']); writer.writerows(spec['rows'])
        expected = stream.getvalue().encode('utf-8-sig')
        matches = path.is_file() and path.read_bytes() == expected
        if check_only:
            if not matches: raise ValueError('Localization count CSV is absent or stale: ' + str(path))
        elif not matches:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
            try: temporary.write_bytes(expected); os.replace(temporary, path)
            finally:
                if temporary.exists(): temporary.unlink()
        descriptors.append(dict(_descriptor(root, path), rows=len(spec['rows'])))
    return descriptors


def _write_index(root, jobs, records):
    rows = []
    for job in jobs:
        output = Path(job['output_path']); relative = output.relative_to(root).as_posix()
        data = Path(job['data_dir']).relative_to(root).as_posix()
        counts = '<a href="' + html.escape(data, quote=True) + '/localization_counts.csv">Cell localization counts</a>'
        if job['level'] != 'cell':
            counts += '<br><a href="' + html.escape(data, quote=True) + '/localization_count_per_cell_summary.csv">Counts per cell</a>'
            counts += '<br><a href="' + html.escape(data, quote=True) + '/localization_count_means.csv">Localization count means</a>'
        rows.append('<tr><td>' + html.escape(job['level']) + '</td><td>' + html.escape(job['key']) + '</td><td><a href="' + html.escape(relative, quote=True) + '">Excel workbook</a></td><td><a href="' + html.escape(data, quote=True) + '/">Saved CSVs</a><br>' + counts + '</td></tr>')
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>TRAK1 cell metrics — Excel tables</title><style>body{font:16px system-ui;margin:32px;color:#203746}table{border-collapse:collapse}td,th{padding:9px 16px;border-bottom:1px solid #ddd;text-align:left}a{color:#2166ac}</style>'
                '<h1>TRAK1 cell metrics: reader-facing Excel tables</h1><p>Total localizations = On + Not-on + Unknown (QC-selected). QC excluded is separate. On percentage = On / (On + Not-on). '
                'First-mask area describes frames 0–49; mean ROI area describes the full movie.</p>'
                '<p>Cell equal weighting within BR; BR equal weighting within condition. On SD is in percentage points. '
                'Localization Means lists within-BR cell means, pooled all-cell means, and equal-replica means. BR and condition summary count columns report means per cell. Condition count columns pool all cells; percentage and area means retain equal-BR weighting. Original source tables and scientific outputs remain unchanged.</p><table><tr><th>Level</th><th>Group</th><th>Workbook</th><th>Numerical source</th></tr>' + ''.join(rows) + '</table></html>')
    molecule=', '.join(sorted({job.get('molecule','TRAK1')for job in jobs}))or'TRAK1'
    document=document.replace('TRAK1',html.escape(molecule))
    path = _guard(root, root / 'CELL_METRICS_TABLES.html'); temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try: temporary.write_text(document, encoding='utf-8'); os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def run(control, adopt_existing=False, check_only=False):
    from reader_table_data import build_all
    control = Path(control).resolve(); root = control.parent; _guard(root, control)
    payload = build_all(control); jobs = payload['jobs']; templates = HERE / 'reader_table_templates'
    if not templates.is_dir(): templates = HERE.parent / 'templates'
    manifest_path = control / 'reader_tables_manifest.json'
    previous = _json(manifest_path) if manifest_path.is_file() else {}
    entries = dict(previous.get('workbooks', {})); actions = {'written':0, 'adopted':0, 'skipped':0, 'checked':0}
    for count, job in enumerate(jobs, 1):
        output = _guard(root, job['output_path']); key = output.relative_to(root).as_posix()
        template = templates / ('cell.xlsx' if job['level']=='cell' else 'BR.xlsx' if job['level']=='BR' else 'summary.xlsx')
        signature = _signature(job, template); old = entries.get(key, {})
        count_tables = _write_count_tables(root, job, check_only)
        if check_only:
            proof = verify_workbook(output, job); actions['checked'] += 1
        elif old.get('signature') == signature and _valid_output(root, old.get('output')):
            entries[key]['count_tables'] = count_tables
            actions['skipped'] += 1; continue
        elif adopt_existing:
            if not output.is_file(): raise ValueError('Cannot adopt absent workbook: ' + str(output))
            proof = verify_workbook(output, job); actions['adopted'] += 1
        else:
            write_workbook(template, job, output); proof = verify_workbook(output, job); actions['written'] += 1
        if not check_only:
            entries[key] = dict(key=job['key'], level=job['level'], signature=signature, output=_descriptor(root, output),
                                template_sha256=sha(template), verification=proof, count_tables=count_tables, updated_utc=utc())
        if count % 10 == 0 or count == len(jobs): print(f'Reader tables: {count}/{len(jobs)} workbooks; {actions}', flush=True)
    report = dict(schema_version=SCHEMA, updated_utc=utc(), workbooks=entries, job_count=len(jobs), actions=actions,
                  scientific_stages_run=[], new_cells_started=0, source_tables_preserved=True,
                  source_fingerprints=payload['source_fingerprints'], portable_template_runtime='Python standard library ZIP/XML')
    if not check_only:
        _atomic_json(manifest_path, report); _write_index(root, jobs, entries)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, default=HERE.parents[1])
    parser.add_argument('--completed-only', action='store_true', required=True)
    parser.add_argument('--adopt-existing', action='store_true', help='Verify and register existing workbooks without rewriting them')
    parser.add_argument('--check-only', action='store_true', help='Read-only saved-value workbook checks')
    args = parser.parse_args()
    try: report = run(args.control, args.adopt_existing, args.check_only)
    except OSError as error:
        if error.errno in (5, 6, 19, 28, 30, 57, 60):
            print('Reader-table drive/I/O failure: ' + str(error), file=sys.stderr, flush=True)
            raise SystemExit(74)
        raise
    print('Reader-facing workbooks complete: ' + str(report['job_count']) + '; ' + str(report['actions']), flush=True)


if __name__ == '__main__': main()
