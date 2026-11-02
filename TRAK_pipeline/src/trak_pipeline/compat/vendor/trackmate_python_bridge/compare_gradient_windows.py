"""Compare frozen 7x7, 5x5 and relaxed 5x5 gradient selections, without fitting."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import yaml

from run_preview import draw_lib, dump, fingerprint, video_and_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--box7', type=Path, required=True)
    parser.add_argument('--box5', type=Path, required=True)
    parser.add_argument('--box5-relaxed', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.box7.resolve(), args.box5.resolve(), args.box5_relaxed.resolve()]
    assert all((p/'COMPLETED.json').exists() for p in paths)
    configs = [json.loads((p/'config.json').read_text()) for p in paths]
    summaries = [json.loads((p/'summary.json').read_text()) for p in paths]
    assert [c['box'] for c in configs] == [7,5,5]
    assert len({c['trackmate_preview'] for c in configs}) == 1
    assert abs(configs[2]['threshold'] - configs[1]['threshold']*.75) < 1e-10
    source = Path(configs[0]['trackmate_preview'])
    render_config = yaml.safe_load((source/'resolved_config.yaml').read_text())
    assert render_config['resolved_display_range_adu'] == [406.,554.]
    movie_path = next((source/'native_detection').glob('*/input_frames.npy'))
    protected = [source/'all_detections.csv',movie_path,source/'roi_vertices.csv']
    for p in paths: protected.extend([p/'all_detections_with_gradient.csv',p/'summary.json',p/'config.json',p/'frames_0000_0049_10fps.mp4'])
    before = [fingerprint(p) for p in protected]
    out = paths[2]/'comparison_7x7_5x5_5x5_relaxed'
    out.mkdir(exist_ok=False)
    dump(out/'source_fingerprints_before.json',before)
    dump(out/'config.json',{'source_runs':[str(p) for p in paths],'score_configs':configs,
                          'frames':[0,49],'crop_native_xyxy':[82,232,347,430],
                          'display_scale':3,'display_ADU':[406,554],'circle_radius_native_px':3,
                          'video_fps':10,'crop_is_display_only':True,'fitting':False})
    shutil.copyfile(__file__,out/'compare_gradient_windows.py')
    (out/'run_command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:]])+'\n')
    original = pd.read_csv(source/'all_detections.csv',float_precision='round_trip').sort_values('spot_id').reset_index(drop=True)
    tables = [pd.read_csv(p/'all_detections_with_gradient.csv',float_precision='round_trip').sort_values('spot_id').reset_index(drop=True) for p in paths]
    for table in tables:
        pd.testing.assert_frame_equal(original,table[original.columns],check_exact=True)
    np.testing.assert_allclose(tables[1].picasso_net_gradient_box5,tables[2].picasso_net_gradient_box5,atol=0,rtol=0,equal_nan=True)
    np.testing.assert_array_equal(tables[1].gradient_support_valid,tables[2].gradient_support_valid)
    assert not (tables[1].net_gradient_pass & ~tables[2].net_gradient_pass).any()
    membership = original.copy()
    keys = ['box7_baseline','box5_baseline','box5_factor0p75']
    for key,table in zip(keys,tables): membership[key+'_ROI_selected'] = (table.roi_inside==1)&table.net_gradient_pass
    membership.to_csv(out/'candidate_membership.csv',index=False,float_format='%.17g')
    selected = [table[(table.roi_inside==1)&table.net_gradient_pass] for table in tables]
    vertices = pd.read_csv(source/'roi_vertices.csv')[['x_px','y_px']].to_numpy()
    movie = np.load(movie_path,mmap_mode='r')
    assert movie.shape == (50,684,428)
    manifest = pd.read_csv(source/'frame_manifest.csv')
    c = dict(render_config,display_scale=3)
    native_crop = [82,232,347,430]
    crop = tuple(v*3 for v in native_crop)
    w,h = crop[2]-crop[0],crop[3]-crop[1]
    width = 3*w+56
    width += width%2
    height = h+186
    large = ImageFont.truetype(c['font_path'],28)
    small = ImageFont.truetype(c['font_path'],23)
    detail = ImageFont.truetype(c['font_path'],20)
    frames = out/'frames';frames.mkdir()
    counts = []
    for frame in range(50):
        raw = movie[frame]
        assert manifest.loc[manifest.frame==frame,'source_pixel_sha256'].iloc[0] == __import__('hashlib').sha256(raw.tobytes()).hexdigest()
        plain = Image.fromarray(draw_lib.display_gray(raw,406,554,3)).convert('RGBA')
        canvas = Image.new('RGB',(width,height),'#101317');draw=ImageDraw.Draw(canvas)
        all_count = int(((original.frame==frame)&(original.roi_inside==1)).sum())
        draw.text((14,12),f'T1 C2 | Frame {frame:04d} | Acquisition {frame*.02:.2f} s | Same cell crop | All TrackMate ROI: {all_count}',font=large,fill='white')
        numbers=[];ids=[]
        for index,(table,config) in enumerate(zip(selected,configs)):
            subset = table[table.frame==frame]
            xy = subset[['x_px','y_px']].to_numpy()
            numbers.append(len(xy));ids.append(set(subset.spot_id))
            panel = Image.alpha_composite(plain,draw_lib.overlay_layer(raw.shape,xy,vertices,c)).convert('RGB').crop(crop)
            x = 14+index*(w+14)
            factor = ' / factor 0.75' if index==2 else ' / baseline'
            draw.text((x,60),f'{config["box"]}x{config["box"]}{factor}: NG > {config["threshold"]:.2f} | {len(xy)} points',font=small,fill='white')
            canvas.paste(panel,(x,104))
        draw.text((14,h+116),'Same TrackMate floating centers and raw page. Yellow: common cell ROI. Cyan radius-3-px circles mark position, not uncertainty.',font=detail,fill='#DCE3EB')
        draw.text((14,h+144),'Display-only cell crop; fixed 406-554 ADU. Gradient score trial, no Gaussian fitting. Video 10 fps; acquisition 50 fps.',font=detail,fill='#DCE3EB')
        meta=PngImagePlugin.PngInfo();meta.add_text('original_frame',str(frame))
        for key,n in zip(keys,numbers):meta.add_text(key+'_ROI_count',str(n))
        canvas.save(frames/f'frame_{frame:04d}.png',pnginfo=meta,compress_level=5)
        with Image.open(frames/f'frame_{frame:04d}.png') as image:
            for key,n in zip(keys,numbers): assert int(image.info[key+'_ROI_count'])==n
        counts.append({'frame':frame,'time_s':frame*.02,'all_ROI_TrackMate':all_count,
                       **dict(zip(keys,numbers)), 'common_box7_box5':len(ids[0]&ids[1]),
                       'box7_only_vs_box5':len(ids[0]-ids[1]),'box5_only_vs_box7':len(ids[1]-ids[0]),
                       'added_box5_factor0p75':len(ids[2]-ids[1])})
    counts = pd.DataFrame(counts)
    for i,key in enumerate(keys):assert int(counts[key].sum())==summaries[i]['ROI_gradient_pass']
    counts.to_csv(out/'frame_counts_and_membership.csv',index=False)
    video,video_checks = video_and_checks(out,render_config)
    after = [fingerprint(p) for p in protected]
    dump(out/'source_fingerprints_after.json',after);assert before==after
    summary={'output':str(out),'ROI_TrackMate_points':int(counts.all_ROI_TrackMate.sum()),
             'selected_counts':{k:int(counts[k].sum()) for k in keys},
             'common_box7_box5':int(counts.common_box7_box5.sum()),
             'box7_only_vs_box5':int(counts.box7_only_vs_box5.sum()),
             'box5_only_vs_box7':int(counts.box5_only_vs_box7.sum()),
             'relaxed_5x5_added':int(counts.added_box5_factor0p75.sum()),
             'video':str(video),'video_checks':video_checks,
             'original_candidate_fields_preserved':True,'source_fingerprints_unchanged':True,
             'repeated_5x5_scores_exactly_equal':True,'relaxed_pass_set_contains_baseline_pass_set':True}
    dump(out/'COMPLETED.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
