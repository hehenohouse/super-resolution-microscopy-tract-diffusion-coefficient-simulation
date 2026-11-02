"""Lossless append-only ROI columns to saved CSV and HDF5 candidate tables."""
import csv
import json
import h5py
import numpy as np
import pandas as pd
from roi_core import roi_flags

NEW_FIELDS = ['roi_inside', 'roi_geometry_status', 'stats_selected_roi']


def augmented_dtype(original):
    # dtype.descr can encode h5py ASCII metadata as tuples unsupported by np.dtype.
    # Field dtype objects retain their original encoding metadata without reinterpretation.
    return np.dtype([(name,original.fields[name][0]) for name in original.names] +
        [('roi_inside','i1'),('roi_geometry_status','S14'),('stats_selected_roi','?')])


def annotated_csv(source, target, inside, status, selected, keep=None):
    count = 0
    with source.open(newline='') as a, target.open('w', newline='') as b:
        reader = csv.reader(a)
        writer = csv.writer(b, lineterminator='\n')
        fields = next(reader)
        assert not set(fields) & set(NEW_FIELDS)
        writer.writerow(fields + NEW_FIELDS)
        for i, row in enumerate(reader):
            assert len(row) == len(fields)
            if keep is None or keep[i]:
                writer.writerow(row + ['' if inside[i]<0 else str(bool(inside[i])), str(status[i]), str(bool(selected[i]))])
                count += 1
        assert i+1 == len(inside) if len(inside) else count == 0
    # Compare original field strings, not lossy reserialized floating point values.
    with source.open(newline='') as a, target.open(newline='') as b:
        old, new = csv.reader(a), csv.reader(b)
        old_header, new_header = next(old), next(new)
        assert new_header == old_header + NEW_FIELDS
        for i, row in enumerate(old):
            if keep is None or keep[i]:
                assert next(new)[:len(old_header)] == row
        assert next(new, None) is None
    return count


def annotate_group(source, out, name, vertices, metadata):
    src = source/'audit'/(name+'.csv')
    table = pd.read_csv(src, float_precision='round_trip')
    inside, status, selected = roi_flags(table, vertices)
    n = annotated_csv(src, out/'audit'/(name+'.csv'), inside, status, selected)
    assert n == len(table)
    # Preserve HDF5 native dtype and every original field bit-for-bit.
    with h5py.File(src.with_suffix('.hdf5'), 'r') as old:
        original = old['locs'][()]
        np.testing.assert_array_equal(original['candidate_id'], table.candidate_id)
        new = np.empty(len(original), dtype=augmented_dtype(original.dtype))
        for field in original.dtype.names:
            new[field] = original[field]
            if original.dtype[field].kind == 'f':
                assert np.array_equal(new[field], original[field], equal_nan=True)
            else:
                np.testing.assert_array_equal(new[field], original[field])
        # Native doubles and round-trip CSV values must lead to identical ROI membership.
        native = pd.DataFrame({'x':original['x'], 'y':original['y'],
                               'qc_unflagged_model_fit':original['qc_unflagged_model_fit']})
        a, b, s = roi_flags(native, vertices)
        np.testing.assert_array_equal(a, inside)
        np.testing.assert_array_equal(s, selected)
        new['roi_inside'], new['roi_geometry_status'], new['stats_selected_roi'] = inside, status.astype('S14'), selected
        for target, rows in [(out/'audit'/(name+'.hdf5'), new), (out/'selected'/(name+'.hdf5'), new[selected])]:
            with h5py.File(target, 'w') as dest:
                dest.create_dataset('locs', data=rows, compression='gzip', shuffle=True)
                if 'metadata' in old:
                    old.copy('metadata', dest)
                dest.create_dataset('roi_metadata', data=json.dumps(metadata), dtype=h5py.string_dtype('utf-8'))
    src_selected = source/'selected'/(name+'.csv')
    previous = pd.read_csv(src_selected, float_precision='round_trip')
    p_inside, p_status, p_keep = roi_flags(previous, vertices)
    assert previous.qc_unflagged_model_fit.all()
    np.testing.assert_array_equal(previous.loc[p_keep,'candidate_id'], table.loc[selected,'candidate_id'])
    annotated_csv(src_selected, out/'selected'/(name+'.csv'), p_inside, p_status, p_keep, keep=p_keep)
    kept = pd.read_csv(out/'selected'/(name+'.csv'))
    assert len(kept) == selected.sum() and kept.roi_inside.all()
    for field in ['frame','candidate_id','fit_state']:
        np.testing.assert_array_equal(kept[field], previous.loc[p_keep,field])
    table['roi_inside'] = inside
    table['roi_geometry_status'] = status
    table['stats_selected_roi'] = selected
    return table, kept, {'group':name,'audit_rows':len(table),'original_qc_rows':len(previous),
        'roi_qc_rows':len(kept),'invalid_centers':int((inside<0).sum()),
        'csv_original_fields_string_identical':True,'hdf_original_fields_bit_identical':True}
