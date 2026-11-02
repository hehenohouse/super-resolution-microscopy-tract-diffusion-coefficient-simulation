"""Replot saved real-data estimates on linear axes, without fitting."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import pandas as pd

from run_real import dump, fingerprint, plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    args = ap.parse_args()
    source = args.source.resolve()
    inputs = [source/'all_ROI_target_fits.csv', source/'summary.json']
    before = [fingerprint(p) for p in inputs]
    table = pd.read_csv(inputs[0])
    summary = json.loads(inputs[1].read_text())
    out = source/('linear_histograms_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    shown = table[table.histogram_eligible]
    counts = {}
    for field, bins in [('x_se_nm',35), ('y_se_nm',35), ('photons_e',40)]:
        values = shown[field].to_numpy()
        edges = np.linspace(0, max(float(values.max())*1.05,1), bins+1)
        counts[field] = int(np.histogram(values,bins=edges)[0].sum())
        assert counts[field] == len(shown) == summary['finite_interior_ROI']
    plots(out, table, summary)
    after = [fingerprint(p) for p in inputs]
    assert before == after
    dump(out/'render_config.json', dict(source=str(source), axes='linear X and Y',
        uncertainty_bins=35, photoelectron_bins=40, full_range_included=True,
        photoelectron_left_panel='linear detail to pooled p99; exceedances explicitly counted',
        photoelectron_right_panel='entire range on linear axes', histogram_points=len(shown),
        unflagged=int(shown.qc_unflagged_model_fit.sum()), refitting=False))
    dump(out/'validation.json', dict(cache_fingerprints_unchanged=True, full_range_counts=counts,
        original_candidate_count=len(table), histogram_points=len(shown), no_refitting=True))
    dump(out/'input_fingerprints.json', before)
    dump(out/'versions.json', dict(python=sys.version,
        packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','matplotlib']}))
    (out/'run_command.txt').write_text(shlex.join([sys.executable,'-u',str(Path(__file__).resolve()),
                                                '--source',str(source)])+'\n')
    snap = out/'source_snapshot'; snap.mkdir()
    for p in [Path(__file__),Path(__file__).with_name('run_real.py')]:
        shutil.copyfile(p,snap/p.name)
    print('OUTPUT:',out,flush=True)
    print(json.dumps({'histogram_points':len(shown),'full_range_counts':counts,'refitting':False}),flush=True)


if __name__ == '__main__':
    main()
