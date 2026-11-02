"""Presentation-only redraw of already-verified flip examples; no classification."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import tifffile
import matplotlib.pyplot as plt

from comparison_core import LOSS,GAIN
from comparison_plots import examples_page
from run_preview import HERE,dump,fingerprint

plt.style.use('default')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    assert json.loads((out/'COMPLETED.json').read_text())['all_checks_passed']
    source=json.loads((out/'source_verification.json').read_text());c=source['config']
    manifests=json.loads((out/'flip_example_manifest.json').read_text())
    ids={i for m in manifests for i in m['candidate_indices']}
    before=fingerprint(out/'changed_labels.csv')
    with (out/'changed_labels.csv').open(newline='') as stream:
        chosen={int(r['candidate_index']):r for r in csv.DictReader(stream) if int(r['candidate_index']) in ids}
    assert set(chosen)==ids
    old=np.load(Path(source['old']['mask_directory'])/'clean_full_mask.npy',mmap_mode='r')
    new=np.load(Path(source['new']['mask_directory'])/'clean_full_mask.npy',mmap_mode='r')
    vertices=np.array(source['roi_vertices'],float);preview=[]
    with tifffile.TiffFile(source['old']['C1_file']) as tf1,tifffile.TiffFile(source['old']['C2_file']) as tf2:
        for pair,m in zip([LOSS,GAIN],manifests):
            rows=[chosen[i] for i in m['candidate_indices']]
            examples_page(pair,rows,tf1,tf2,old,new,vertices,c,out)
            if rows:preview.append(rows[len(rows)//2])
        examples_page(None,preview,tf1,tf2,old,new,vertices,c,out,page_name='flip_examples_preview.png')
    assert fingerprint(out/'changed_labels.csv')==before
    dump(out/'example_presentation_validation.json',dict(all_passed=True,classification_table_unchanged=True,
        source=before,example_IDs_unchanged=sorted(ids),preview_candidate_indices=[int(r['candidate_index']) for r in preview],
        preview_selection='middle member of each existing chronologically spaced example list',no_reclassification=True))
    for name in ['comparison_plots.py',Path(__file__).name]:shutil.copyfile(HERE/name,out/'source_snapshot'/name)
    (out/'example_redraw_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),'--output',str(out)])+'\n')
    print('Redrew two six-example pages and one two-example preview; classification CSV unchanged.')


if __name__=='__main__':main()
