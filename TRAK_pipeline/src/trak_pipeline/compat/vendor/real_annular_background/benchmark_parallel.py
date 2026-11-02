"""Compare execution backends on existing sample candidates, not fit models."""
import concurrent.futures
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pandas as pd
import tifffile
import yaml

import run_real as core
from parallel_fit import fit_batch


def main():
    root=Path(__file__).parent
    c=yaml.safe_load((root/'config_frames1250_1300.yaml').read_text())
    src=Path(c['source_preview']);p=yaml.safe_load((src/'resolved_config.yaml').read_text())
    camera=yaml.safe_load((root/c['nominal_camera_config']).read_text())['camera']
    all_data=pd.read_csv(src/'all_detections.csv');all_data.insert(0,'candidate_index',np.arange(len(all_data)))
    selected=all_data[all_data.frame.eq(1250)&all_data.roi_inside.eq(1)].copy()
    with tifffile.TiffFile(p['input_tiff']) as tf:raw=tf.pages[1250].asarray()
    vertices=core.roi_lib.load_imagej_roi(p['roi_file']);yy,xx=np.mgrid[:684,:428]
    roi=core.roi_lib.points_in_polygon(np.c_[xx.ravel(),yy.ravel()],vertices).reshape(684,428)
    background,masks,movie=core.background_geometry(selected,all_data[all_data.frame.eq(1250)],raw,roi,c,camera)
    items=[(int(r.candidate_index),int(r.spot_id),1250,float(r.x_px),float(r.y_px)) for r in selected.itertuples()]
    # Warm the numerical kernel before measuring the current thread backend.
    core.fit_one(items[0],raw,movie,background.iloc[0],masks[0],c)
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(core.fit_one,item,raw,movie,background.iloc[j],masks[j],c) for j,item in enumerate(items)]
        threaded=[j.result() for j in jobs]
    thread_elapsed=time.perf_counter()-start
    chunks=np.array_split(np.arange(len(items)),4)
    payloads=[([items[j] for j in group],raw,movie,background.iloc[group].reset_index(drop=True),[masks[j] for j in group],c) for group in chunks]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        initial=list(pool.map(fit_batch,payloads))
        start=time.perf_counter();processed=list(pool.map(fit_batch,payloads));process_elapsed=time.perf_counter()-start
    processed=[r for group in processed for r in group]
    expected=pd.read_csv(src.parent.parent/'annular_background_refinement/T1_C2_r0p28um_box7_frames1250_1300_20261012T174515Z/all_ROI_target_fits.csv')
    a=pd.DataFrame(threaded);b=pd.DataFrame(processed)
    fields=['photons_e','bg_e_per_pixel','fit_x_px','fit_y_px','sx_px','sy_px','x_se_nm','y_se_nm','N_se_e']
    np.testing.assert_allclose(a[fields],b[fields],rtol=1e-10,atol=1e-8,equal_nan=True)
    for name in ['fit_success','active_bound','fit_usable_interior','finite_uncertainty','suspect_overlap']:
        np.testing.assert_array_equal(a[name],b[name])
    reference=expected[expected.frame.eq(1250)].reset_index(drop=True)
    np.testing.assert_allclose(a[fields],reference[fields],rtol=1e-10,atol=1e-8,equal_nan=True)
    result=dict(candidates=len(items),thread_seconds=thread_elapsed,warm_process_seconds=process_elapsed,
        speed_ratio=thread_elapsed/process_elapsed,all_numeric_and_status_results_equal=True,
        original_saved_sample_results_reproduced=True,fit_model_and_QC_unchanged=True)
    core.dump(root/'parallel_benchmark.json',result);print(result,flush=True)


if __name__=='__main__':main()
