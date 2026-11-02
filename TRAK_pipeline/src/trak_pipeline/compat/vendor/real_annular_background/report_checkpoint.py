"""Small, standard-library summary of one completed 250-frame checkpoint."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--end-frame',type=int,required=True)
    args=ap.parse_args();end=args.end_frame;start=end-249
    assert end in range(249,2500,250)
    source=args.output/'checkpoints'/f'fits_through_{end:04d}.csv'
    with source.open() as h:
        rows=[r for r in csv.DictReader(h) if start<=int(r['frame'])<=end]
    clean=[r for r in rows if r['qc_unflagged_model_fit']=='True']
    data=dict(frames_inclusive=[start,end],ROI_candidates=len(rows),unflagged=len(clean),
        status_counts=dict(Counter(r['audit_status'] for r in rows)),
        median_N_e=statistics.median(float(r['photons_e']) for r in clean) if clean else None,
        median_x_SE_nm=statistics.median(float(r['x_se_nm']) for r in clean) if clean else None,
        median_y_SE_nm=statistics.median(float(r['y_se_nm']) for r in clean) if clean else None,
        calibration_status='manufacturer_nominal',uncertainty_coverage_validated=False)
    reports=args.output/'progress_reports';reports.mkdir(exist_ok=True)
    (reports/f'frames{start:04d}_{end:04d}.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()
