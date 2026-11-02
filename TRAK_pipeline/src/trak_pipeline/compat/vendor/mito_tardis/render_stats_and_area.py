"""Annotate existing Intra curves and plot native W5/min32 mask areas.

Read-only inputs; no TARDIS bootstrap, localization, or segmentation is rerun.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summarize_intra import summarize

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
COLORS={'On':'#159C76','Not-on':'#9854CA'}


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def fingerprint(path):
    p=Path(path).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    s=p.stat()
    return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h.hexdigest())


def annotated_intra(run,stats,out):
    fig=plt.figure(figsize=(13.4,7.0))
    grid=fig.add_gridspec(2,2,height_ratios=[4,1.05],left=.07,right=.985,top=.89,bottom=.17,
                         wspace=.16,hspace=.35)
    axes=[fig.add_subplot(grid[0,j]) for j in range(2)]
    band=pd.read_csv(run/'bootstrap/paired_bands_full_L25.csv',float_precision='round_trip')
    for group,color in COLORS.items():
        d=pd.read_csv(run/'distribution'/group/'full.csv',float_precision='round_trip')
        actual=summarize(d.distance_left_um,d.distance_right_um,d.shape_density_um_inv)
        saved=stats[stats.group==group].iloc[0]
        for key in ['signed_first_moment_nm','signed_half_area_location_nm','mode_bin_center_nm']:
            np.testing.assert_allclose(actual[key],saved[key],rtol=1e-12,atol=1e-10)
        for ax,field,title in zip(axes,['intra','shape'],
            ['Signed Intra = Total - beta B','Intra normalized by signed area on 0-3 µm']):
            prefix=group.replace('-','_')+'_'+field
            ax.plot(d.distance_mid_um,d[field+'_density_um_inv'],color=color,lw=1.6,label=group)
            ax.fill_between(band.distance_mid_um,band[prefix+'_low'],band[prefix+'_high'],color=color,alpha=.16)
            ax.set(xlim=(0,.5),xlabel='Distance (µm)',ylabel='Signed density (1/µm)',title=title)
            ax.axhline(0,c='gray',lw=.6);ax.grid(alpha=.2);ax.legend(fontsize=10)
    table_ax=fig.add_subplot(grid[1,:]);table_ax.axis('off')
    rows=[]
    for group in COLORS:
        s=stats[stats.group==group].iloc[0]
        rows.append([group,f'{s.signed_first_moment_nm:.1f}',f'{s.signed_half_area_location_nm:.1f}',
                     f'{s.mode_bin_center_nm:.0f}  (bin {s.mode_bin_left_nm:.0f}–{s.mode_bin_right_nm:.0f})'])
    table=table_ax.table(cellText=rows,colLabels=['Group','Mean (signed), nm','Median*, nm','Mode / peak-bin center, nm'],
        cellLoc='center',colLoc='center',bbox=[.08,0,.84,1])
    table.auto_set_font_size(False);table.set_fontsize(11)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#D8DCE0')
        if row==0:cell.set_facecolor('#F0F2F4');cell.set_text_props(weight='bold')
        else:cell.set_text_props(color=COLORS[list(COLORS)[row-1]])
    fig.suptitle('TDKd3 | W5/min32 | On vs Not-on | Full 2500 frames, τ = 20 ms',fontsize=15,y=.98)
    notes=[
        'Statistics use the full 0–3 µm range; displayed distance range is 0–0.5 µm. All negative bins are retained.',
        'Mean is a signed first moment. Median* is the unique 50% signed-area crossing, not an ordinary probability median.',
        'Shading: exploratory paired L25 pointwise 2.5–97.5 percentiles. Mode uses the original 10 nm bins; no smoothing or peak fit.']
    for y,note in zip([.115,.080,.045],notes):fig.text(.5,y,note,ha='center',fontsize=9)
    fig.savefig(out/'intra_with_mean_median_mode.png',dpi=190)
    plt.close(fig)


def area_figures(area,summary,cell_area,out):
    fig,axes=plt.subplots(2,1,figsize=(13.4,7.6),sharex=True)
    names=[('ROI','ROI_mask_area_um2','#246B9E','Cell ROI: mitochondrial mask area'),
           ('full_FOV','full_mask_area_um2','#697F50','Full camera field: mitochondrial mask area')]
    fallback=~area.temporal_applied
    for ax,(scope,field,color,title) in zip(axes,names):
        row=next(r for r in summary if r['scope']==scope and r['frames']=='all_2500')
        ax.plot(area.frame,area[field],c=color,lw=.75,label='Per-frame W5/min32 mask area')
        ax.scatter(area.loc[fallback,'frame'],area.loc[fallback,field],s=32,c='#ED8C19',
                   edgecolors='white',lw=.5,zorder=4,label='Unsmoothed min32 fallback: 0, 1, 2498, 2499')
        ax.axhline(row['mean_um2'],c=color,lw=.8,ls='--',alpha=.7,label='Full-movie mean')
        ax.set(ylabel='Mask area (µm²)',xlim=(0,2499),
               title=f'{title}  |  mean {row["mean_um2"]:.2f} µm²; median {row["median_um2"]:.2f} µm²')
        ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=8,ncol=2,framealpha=.94)
        lo,hi=area[field].min(),area[field].max();span=hi-lo
        ax.set_ylim(lo-.30*span,hi+.18*span)
    coverage=axes[0].secondary_yaxis('right',functions=(lambda x:x/cell_area*100,lambda x:x/100*cell_area))
    coverage.set_ylabel('Cell ROI covered by mask (%)')
    seconds=axes[0].secondary_xaxis('top',functions=(lambda x:x*.02,lambda x:x/.02))
    seconds.set_xlabel('Acquisition time (s)')
    seconds.set_xticks([0,10,20,30,40,49.98])
    axes[1].set_xlabel('Original frame (0-based)')
    axes[1].set_xticks([0,500,1000,1500,2000,2499])
    fig.suptitle('TDKd3 | W5 majority vote + min32 | Native binary mask area',fontsize=15,y=.985)
    fig.subplots_adjust(left=.085,right=.91,top=.84,bottom=.17,hspace=.43)
    fig.text(.5,.065,'Area = foreground pixels × (0.117 µm)². Top: original cell ROI; bottom: complete field before ROI clipping.',
             ha='center',fontsize=9)
    fig.text(.5,.035,'No extra smoothing of area curves; no localization-error expansion. Full-field values include mask outside the selected cell ROI.',
             ha='center',fontsize=9)
    fig.savefig(out/'mask_area_by_timepoint.png',dpi=190)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--statistics',type=Path,required=True);args=parser.parse_args()
    run,statdir=args.run.resolve(),args.statistics.resolve()
    assert json.loads((run/'COMPLETED.json').read_text())['delivery_checks_passed']
    ref=json.loads((run/'source_reference.json').read_text());maskdir=Path(ref['mask_directory'])
    verification=json.loads((Path(ref['comparison_directory'])/'source_verification.json').read_text())
    c=json.loads((run/'config.json').read_text());assert c['pixel_size_nm']==117 and c['frame_interval_s']==.02
    manifest={r['path']:r['sha256'] for r in json.loads((run/'output_fingerprints.json').read_text())}
    sources=[run/'config.json',run/'per_frame_counts.csv',run/'bootstrap/paired_bands_full_L25.csv']
    sources += [run/'distribution'/g/'full.csv' for g in COLORS]
    expected={str(p):manifest[str(p)] for p in sources}
    for name in ['clean_full_mask.npy','roi_mask.npy','frame_completed.npy','original_frame_numbers.npy']:
        p=maskdir/name;expected[str(p)]=verification['expected_sha256'][str(p)]
    expected[str(statdir/'descriptive_statistics.csv')]=fingerprint(statdir/'descriptive_statistics.csv')['sha256']
    out=run.parents[1]/'mito_tardis_figures'/('T1_C2_Intra_stats_mask_area_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False);(HERE/'latest_annotated_figures_attempt.txt').write_text(str(out)+'\n')
    print('OUTPUT',out,flush=True)
    before=[]
    for p,digest in expected.items():
        print('SOURCE',Path(p).name,flush=True);value=fingerprint(p)
        assert value['sha256']==digest,p;before.append(value)
    dump(out/'input_fingerprints_before.json',before)
    mask=np.load(maskdir/'clean_full_mask.npy',mmap_mode='r');roi=np.load(maskdir/'roi_mask.npy',mmap_mode='r')
    assert mask.shape==(2500,684,428) and mask.dtype==bool and roi.shape==(684,428) and roi.dtype==bool
    assert np.load(maskdir/'frame_completed.npy').all()
    np.testing.assert_array_equal(np.load(maskdir/'original_frame_numbers.npy'),np.arange(2500))
    roi_pixels=int(np.count_nonzero(roi));assert roi_pixels==32179
    pixelsize_um=c['pixel_size_nm']/1000;pixelarea=pixelsize_um**2;cell_area=roi_pixels*pixelarea
    full=np.zeros(2500,np.int64);inside=np.zeros_like(full)
    for start in range(0,2500,50):
        stop=min(start+50,2500);block=mask[start:stop]
        full[start:stop]=np.count_nonzero(block,axis=(1,2))
        inside[start:stop]=np.count_nonzero(block&roi[None],axis=(1,2))
        if stop%250==0:print('MASK AREA VERIFIED',stop,flush=True)
    saved=pd.read_csv(run/'per_frame_counts.csv')
    np.testing.assert_array_equal(saved.frame,np.arange(2500))
    np.testing.assert_array_equal(inside,saved.mask_ROI_area_pixel2)
    np.testing.assert_array_equal(full,saved.mask_full_area_pixel2)
    np.testing.assert_array_equal(saved.loc[~saved.mask_temporal_applied,'frame'],[0,1,2498,2499])
    area=pd.DataFrame(dict(frame=np.arange(2500),time_s=np.arange(2500)*.02,
        ROI_mask_area_pixel2=inside,ROI_mask_area_um2=inside*pixelarea,
        cell_ROI_area_pixel2=roi_pixels,cell_ROI_area_um2=cell_area,
        ROI_coverage_percent=inside/roi_pixels*100,
        full_mask_area_pixel2=full,full_mask_area_um2=full*pixelarea,
        temporal_applied=saved.mask_temporal_applied.to_numpy()))
    area.to_csv(out/'mask_area_by_frame.csv',index=False)
    summaries=[]
    for scope,field in [('ROI','ROI_mask_area_um2'),('full_FOV','full_mask_area_um2')]:
        for frames,which in [('all_2500',np.ones(2500,bool)),('W5_applied',area.temporal_applied.to_numpy()),
                              ('endpoint_fallback',~area.temporal_applied.to_numpy())]:
            v=area.loc[which,field].to_numpy()
            summaries.append(dict(scope=scope,frames=frames,n=len(v),mean_um2=float(v.mean()),
                median_um2=float(np.median(v)),std_um2=float(np.std(v,ddof=1)),
                min_um2=float(v.min()),max_um2=float(v.max()),
                mean_ROI_coverage_percent=float(v.mean()/cell_area*100) if scope=='ROI' else None))
    pd.DataFrame(summaries).to_csv(out/'mask_area_summary.csv',index=False)
    stats=pd.read_csv(statdir/'descriptive_statistics.csv',float_precision='round_trip')
    annotated_intra(run,stats,out);area_figures(area,summaries,cell_area,out)
    shutil.copyfile(statdir/'descriptive_statistics.csv',out/'intra_descriptive_statistics.csv')
    for p in [Path(__file__),HERE/'summarize_intra.py']:shutil.copyfile(p,out/p.name)
    shutil.copyfile(PROJECT/'picasso_pilot/requirements.lock.txt',out/'requirements.lock.txt')
    dump(out/'parameters.json',dict(TARDIS_source=str(run),statistics_source=str(statdir),mask_source=str(maskdir),
        mask_recipe='five-frame3-of5 majority vote then8-connected area>=32pixels; endpoints unsmoothed min32',
        pixel_size_nm=117,frame_interval_s=.02,ROI_pixels=roi_pixels,ROI_area_um2=cell_area,
        area_per_pixel_um2=pixelarea,area_definition='native binary foreground pixels; no pointwise ellipse expansion',
        plot_area_series_extra_smoothing=False,intra_statistics_range_um=[0,3],intra_display_range_um=[0,.5],
        calibration_and_registration_status='unchanged; identity registration unverified'))
    dump(out/'versions.json',dict(python=sys.version,packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','matplotlib']}))
    (out/'run_command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),
        '--run',str(run),'--statistics',str(statdir)])+'\n')
    after=[fingerprint(p) for p in expected];assert before==after
    dump(out/'input_fingerprints_after.json',after)
    dump(out/'validation.json',dict(all_numerical_checks_passed=True,all2500_mask_pixel_counts_match_saved_cache=True,
        mask_ROI_pixels=roi_pixels,all_frame_numbers_retained=True,fallback_frames=[0,1,2498,2499],
        plotted_statistics_recomputed_and_equal=True,original_inputs_unchanged=True,visual_QC_pending=True))
    dump(out/'mask_area_summary.json',dict(cell_ROI_area_um2=cell_area,summaries=summaries))
    dump(out/'COMPLETED.json',dict(all_numerical_checks_passed=True,visual_QC_pending=True,output=str(out)))
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.iterdir()) if p.is_file()
        and p.name not in ['output_fingerprints.json','COMPLETED.json']])
    (HERE/'latest_annotated_figures_output.txt').write_text(str(out)+'\n')
    print('COMPLETED',json.dumps(dict(output=str(out),cell_ROI_area_um2=cell_area,summaries=summaries)),flush=True)


if __name__=='__main__':main()
