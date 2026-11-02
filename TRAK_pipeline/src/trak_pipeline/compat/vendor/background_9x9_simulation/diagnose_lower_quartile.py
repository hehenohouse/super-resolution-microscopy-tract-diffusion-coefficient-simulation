"""Read-only lower-quartile background diagnostic; no refits or parameter edits."""
from datetime import datetime,timezone
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np,pandas as pd,tifffile
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'TARDIS/path_irrelevant_python'))
from tardis_tau1.locbio import load_imagej_roi,points_in_polygon

DATA=Path('/Volumes/My Book/Xinran_ComparisonData/20251016_TDKd3_HaloTRAK1_prcs - Copy/101625_TDKd3_TRAK1_T1f')
SIM=DATA/'analysis_results/simulations/background_9x9/scene300x300_50spots_20261009T182037Z'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summary(values):
    a=np.asarray(values,float).ravel();n=int(np.floor(.25*len(a)))
    dark=np.partition(a,n-1)[:n]
    return dict(n=len(a),zero_pixels=int((a==0).sum()),mean=float(a.mean()),median=float(np.median(a)),
        std=float(a.std()),q25=float(np.percentile(a,25)),lowest_exact25_mean=float(dark.mean()),
        n_in_dark_subset=n,dark_subset_max=float(dark.max()))

def main():
    source=DATA/'C2-101625_TDKd3_TRAK1_T1.tiff';roi_path=DATA/'0063-0331-0214.roi'
    before=source.stat();protected={str(p):sha(p) for p in [SIM/'scene_components.npz',roi_path]}
    out=SIM.parent/('lower_quartile_diagnostic_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
    with np.load(SIM/'scene_components.npz') as z:
        arrays={k:z[k].copy() for k in ['background_electrons','read_noise_electrons','observed_electrons']}
    bg_noise=arrays['background_electrons']+arrays['read_noise_electrons']
    rows=[]
    for name,a in [('sim_true_background',arrays['background_electrons']),('sim_background_plus_read_noise',bg_noise),('sim_observed_with_spots',arrays['observed_electrons'])]:
        rows.append(dict(source=name,frame=None,unit='e_per_pixel',scope='full_300x300',**summary(a)))
    vertices=load_imagej_roi(roi_path);yy,xx=np.mgrid[:684,:428]
    mask=points_in_polygon(np.column_stack([xx.ravel(),yy.ravel()]),vertices).reshape(684,428)
    original=[];roi_values=None
    with tifffile.TiffFile(source) as tif:
        assert len(tif.pages)==2500
        for f in [0,1250,2000]:
            a=tif.pages[f].asarray();assert a.shape==(684,428) and a.dtype==np.uint16;original.append(a)
            for scope,v in [('full_FOV',a),('cell_ROI',a[mask]),('full_FOV_nonzero_diagnostic',a[a>0])]:
                rows.append(dict(source='real_T1_C2',frame=f,unit='ADU',scope=scope,**summary(v)))
            if f==2000:roi_values=a[mask].astype(float)
    theory=dict(background_mean_e=100,background_sd_e=20,read_sd_e=10,
        total_noise_sd_e=float(np.sqrt(500)),normal_lower25_shift_in_sd=float(norm.pdf(norm.ppf(.25))/.25),
        normal_expected_lowest25_mean_e=float(100-np.sqrt(500)*norm.pdf(norm.ppf(.25))/.25),
        definition='Conditional mean below the 25th percentile for a pure Gaussian background; not a background-mean estimator.')
    df=pd.DataFrame(rows);df.to_csv(out/'summary.csv',index=False)
    (out/'theory.json').write_text(json.dumps(theory,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for ax,v,title,unit in [(axes[0],bg_noise.ravel(),'Simulation: background + read noise only','e- / pixel'),
                           (axes[1],roi_values,'Real T1 C2: frame 2000, cell ROI','ADU')]:
        s=summary(v);bins=np.linspace(v.min(),v.max(),60)
        ax.hist(v,bins=bins,color='#abb5c0',density=True,label='All pixels')
        n=s['n_in_dark_subset'];dark=np.partition(v,n-1)[:n]
        ax.hist(dark,bins=bins,color='#3f8ec7',alpha=.7,weights=np.full(n,1/len(v)/(bins[1]-bins[0])),label='Lowest 25%')
        ax.axvline(s['mean'],color='#df7632',lw=1.8,label=f"All-pixel mean {s['mean']:.2f}")
        ax.axvline(s['lowest_exact25_mean'],color='#1e6196',lw=1.8,ls='--',label=f"Lowest-25% mean {s['lowest_exact25_mean']:.2f}")
        ax.set(title=title,xlabel=unit,ylabel='Density');ax.legend(fontsize=8)
    fig.suptitle('Selecting the dark tail lowers the mean even when no emitter is present',fontsize=12)
    fig.savefig(out/'lower25_bias.png',dpi=140);plt.close(fig)
    after=source.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert all(sha(p)==h for p,h in protected.items())
    shared_zeros=np.logical_and.reduce([a==0 for a in original])
    validation=dict(read_only=True,source_size_and_mtime_unchanged=True,ROI_and_simulation_hashes_unchanged=True,
        image_dimensions_TYX=[2500,684,428],ROI_pixel_count=int(mask.sum()),ROI_zero_pixels_frames={str(f):int((a[mask]==0).sum()) for f,a in zip([0,1250,2000],original)},
        common_zero_locations_among_three_frames=int(shared_zeros.sum()),zero_pixel_origin_not_inferred=True,
        lowest25_selection='exact floor(0.25*n) sorted values; ties arbitrary but identical intensity',no_detection_or_refitting=True)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (out/'input_fingerprints.json').write_text(json.dumps(dict(protected=protected,tiff_path=str(source),tiff_size=before.st_size,tiff_mtime_ns=before.st_mtime_ns),indent=2)+'\n')
    (out/'REPORT.txt').write_text('''This is a read-only diagnostic of the proposed lowest-25-percent mean.
Background b excludes camera offset; dark-looking display pixels do not imply zero ADU.
The dark-tail mean is downward biased even on the noise-only synthetic image.
Global cell ROI statistics do not establish each emitter's local background.
Zero pixels and their persistence are counted, but their acquisition/export origin is unknown.
Nonzero-only full-FOV statistics are a diagnostic, not a declaration that all nonzero pixels are valid.
No camera offset, fitting background, detection threshold, or localization result is modified.
''')
    shutil.copyfile(__file__,out/Path(__file__).name)
    print('OUTPUT:',out);print(df.to_string(index=False));print(json.dumps(theory,indent=2));print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
