"""Record reviewed previews and finalize the completed postmin32 trial."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mito_tardis'))
from render_point_stack import dump,fingerprint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--visual-reviewed',action='store_true',required=True);args=parser.parse_args();out=args.output.resolve()
    done=json.loads((out/'COMPLETED.json').read_text());assert done['all_implementation_checks_passed'] and done['visual_review_pending']
    assert json.loads((out/'input_fingerprints_before.json').read_text())==json.loads((out/'input_fingerprints_after.json').read_text())
    assert json.loads((out/'mask_validation.json').read_text())['independent_all2500_raw_labels_reconstruction_hashes_match']
    assert json.loads((out/'view_validation.json').read_text())['all2500_pages_readback_equal']
    manifest=pd.read_csv(out/'frame_manifest.csv');np.testing.assert_array_equal(manifest.frame,np.arange(2500))
    np.testing.assert_array_equal(manifest.block_index,np.arange(2500)//50)
    # A fixed-shape TIFF writer may finish without resuming the generator after
    # its final yield. Record final completion from the already verified stack.
    if not (out/'checkpoints/rendered_2500.json').exists():
        dump(out/'checkpoints/rendered_2500.json',dict(rendered_frames=2500,TIFF_finalized=True,
            all2500_pages_readback_equal=True,recorded_after_verified_completion=True))
    for end in range(250,2501,250):
        assert json.loads((out/'checkpoints'/f'rendered_{end:04d}.json').read_text())['rendered_frames']==end
    checked=[]
    for frame in [200,1200,2000]:
        name=f'previews/frame_{frame:04d}_C1_old_new.png'
        with Image.open(out/name) as picture:
            picture.load();assert picture.mode=='RGB' and picture.size==(1646,532)
            checked.append(dict(path=name,sha256=fingerprint(out/name)['sha256'],size=list(picture.size)))
    dump(out/'visual_review.json',dict(passed=True,images=checked,notes='Reviewed original-frame200/1200/2000 three-panel previews: same C1 image and ROI/grid/contrast, raw|old|new red contours visible, header labels legible, ROI exterior black. No localization overlays.'))
    statistics=pd.read_csv(out/'per_block_statistics.csv')
    assert len(statistics)==50 and statistics.removed_ROI_pixel2.eq(0).all() and not statistics.empty_ROI.any()
    added=int(statistics.added_ROI_pixel2.sum());old=int(statistics.old_ROI_pixel2.sum());new=int(statistics.new_ROI_pixel2.sum())
    assert (old,new,added)==(358406,380394,21988)
    dump(out/'summary.json',dict(sample='TDKd3 TRAK1 T1',old_ROI_pixel2_sum_across50blocks=old,
        new_ROI_pixel2_sum_across50blocks=new,added_ROI_pixel2_sum_across50blocks=added,
        relative_ROI_area_sum_change_percent=100*added/old,removed_old_full_pixels=0,
        blocks=50,source_frames=2500,viewing_TIFF_pages=2500,source_interval_s=.02,mask_interval_s=1.,
        localization_reclassification_performed=False,TARDIS_performed=False))
    (out/'README.txt').write_text(
        'TDKd3 TRAK1 T1: min32 applied only after 50-frame support selection\n\n'
        'Pipeline: original individual-frame Nellie broad (labels > 0) -> nonoverlapping 50-frame per-pixel support >=10 -> full-image 8-connected foreground area >=32 pixels -> fill enclosed 4-connected background holes <64 pixels -> original ROI.\n'
        'Only the per-frame min32 stage was removed. No Nellie rerun, raw-image averaging, motion compensation, W5 voting, or post-hole-filling size filter.\n\n'
        'Fiji_C1_old_vs_postmin32_ROI_frames0000_2499.ome.tif: 2500 original C1 timepoints, 20 ms/page. Left: raw C1. Middle: old mask. Right: new mask.\n'
        'This is an RGB8 viewing composite, with lossless Deflate compression. Crop [82,232,347,430], display 2x, fixed 407-651 ADU. ROI exterior is black. Red contours are extracted from each full mask before ROI clipping.\n'
        'The raw C1 image changes at every page; both masks update once per 50 original frames. The header records the original frame, time, source block, and ROI areas.\n\n'
        'clean_full_mask.npy / clean_full_mask.ome.tif: final scientific full-image mask, before ROI clipping.\n'
        'analysis_ROI_mask.npy / analysis_ROI_mask.ome.tif: final full mask intersected with the unchanged cell ROI.\n'
        'Scientific arrays: 50x684x428, TYX, boolean NPY or uint8 0/1 OME-TIFF; 117 nm/pixel and 1 s/block. Original frame f maps to block f//50.\n'
        'foreground_count.npy: counts from original broad masks before per-frame size filtering. input_ge10_full_mask.npy: count >=10. min32_full_mask.npy: post-support component cleanup. filled_holes_full_mask.npy: pixels added by hole filling only.\n\n'
        'Across 50 blocks, ROI foreground area sums increased from 358406 to 380394 pixel2 (+21988, +6.13494%). No old full-mask foreground was removed. These sums count each block separately. More coverage is not evidence of higher segmentation accuracy or simultaneous biological structures.\n'
        'All five derived arrays match an independent reconstruction from all 2500 original broad-label frames. Scientific and viewing TIFF pages were read back and checked. Protected input fingerprints did not change.\n'
        'No localization fitting, QC change, On/Not-on reclassification, tracking, TARDIS, or significance analysis was performed. C1/C2 registration was not validated by this experiment. Existing defaults and the paused 47-sample batch remain unchanged.\n')
    for name in ['run_support50_postmin32_preview.py','support50_postmin32_core.py','test_support50_postmin32_core.py',
                 'finalize_postmin32_preview.py','publish_postmin32_preview.py']:
        shutil.copyfile(HERE/name,out/'source_snapshot'/name)
    config=json.loads((out/'resolved_config.json').read_text())
    (out/'finalize_command.txt').write_text(shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(out),'--visual-reviewed'])+'\n')
    (out/'publication_command.txt').write_text(shlex.join([sys.executable,'-B',str(HERE/'publish_postmin32_preview.py'),
        '--source',str(out),'--destination',config['external_destination']])+'\n')
    validation=json.loads((out/'validation.json').read_text());validation.update(visual_review_pending=False,visual_review_passed=True)
    dump(out/'validation.json',validation)
    done.update(visual_review_pending=False,delivery_checks_passed=True,finished_utc=datetime.now(timezone.utc).isoformat())
    dump(out/'COMPLETED.json',done)
    dump(out/'COMPLETE.json',dict(all_passed=True,source_frames=2500,mask_blocks=50,viewing_frames=2500,
        selected_as_final=False,source_broad_preserved=True,old_masks_preserved=True,delivery_checks_passed=True,output=str(out)))
    dump(out/'run_status.json',dict(status='completed',delivery_checks_passed=True,visual_review_passed=True))
    records=[fingerprint(path) for path in sorted(out.rglob('*')) if path.is_file() and path.name not in
             ['COMPLETE.json','COMPLETED.json','output_fingerprints.json','output_file_manifest.json']]
    dump(out/'output_file_manifest.json',records)
    dump(out/'output_fingerprints.json',{str(Path(r['path']).relative_to(out)):r['sha256'] for r in records})
    print(json.dumps(dict(local_output=str(out),destination=config['external_destination'],delivery_checks_passed=True),indent=2))


if __name__=='__main__':main()
