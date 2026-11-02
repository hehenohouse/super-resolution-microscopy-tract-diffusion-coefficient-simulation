"""Native TrackMate detector in bounded time chunks; no fits or trajectories."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import imagej
import numpy as np
import scyjava
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'trackmate_python_bridge'))
from run_detection import make_imageplus, detect


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', type=Path, required=True)
    ap.add_argument('--max-new-chunks',type=int,default=3)
    args = ap.parse_args()
    job = json.loads(args.job.read_text())
    out = Path(job['output']); out.mkdir(exist_ok=True)
    os.environ['JAVA_HOME'] = job['java_home']
    scyjava.config.set_java_constraints(fetch='never')
    scyjava.config.add_option('-Xmx2g')
    scyjava.config.add_option('-Dscijava.log.level=warn')
    scyjava.config.add_option('-Dimagej.updater.disableAutocheck=true')
    started = time.perf_counter()
    ij = imagej.init(job['fiji'], mode='headless')
    System = scyjava.jimport('java.lang.System')
    fields = ['spot_id','frame','time_s','x_px','y_px','x_um','y_um','quality_log_response','radius_setting_um']
    new_chunks=0
    try:
        with tifffile.TiffFile(job['input_tiff']) as tf:
            assert len(tf.pages) == 2500 and tf.series[0].shape == (2500,684,428)
            for start, stop in job['chunks']:
                tag = f'frames{start:04d}_{stop-1:04d}'
                completed=out/(tag+'.json')
                if completed.exists():
                    saved=json.loads(completed.read_text())
                    assert saved['csv_sha256']==hashlib.sha256((out/(tag+'.csv')).read_bytes()).hexdigest()
                    continue
                if new_chunks>=args.max_new_chunks:break
                movie = np.stack([tf.pages[f].asarray() for f in range(start,stop)])
                assert movie.shape == (stop-start,684,428) and movie.dtype == np.uint16
                hashes = [dict(frame=start+i,pixel_sha256=hashlib.sha256(plane.tobytes()).hexdigest()) for i,plane in enumerate(movie)]
                imp = make_imageplus(ij,movie,.117,.02)
                rows,counts,version = detect(imp,.28,6.5,True,start,.02)
                assert version == '7.14.0'
                p = out/(tag+'.csv')
                with p.open('w',newline='') as h:
                    writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader();writer.writerows(rows)
                report=dict(start=start,stop_exclusive=stop,trackmate_version=version,
                    radius_um=.28,threshold=6.5,median_filter=True,subpixel=True,tracking=False,
                    rows=len(rows),counts=counts,page_hashes=hashes,pixel_roundtrip_exact=True,
                    axes='TYX; C=1,Z=1',csv_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    java=str(System.getProperty('java.version')),elapsed_s=time.perf_counter()-started)
                # Completion report is published atomically after the CSV.
                tmp=out/(tag+'.json.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n')
                tmp.replace(out/(tag+'.json'))
                new_chunks+=1
                print(f'{tag}: {len(rows)} detections, {time.perf_counter()-started:.1f}s',flush=True)
                imp.close();del imp,movie,rows;gc.collect();System.gc()
        if all((out/f'frames{a:04d}_{b-1:04d}.json').exists() for a,b in job['chunks']):
            (out/'COMPLETED.json').write_text(json.dumps(dict(all_chunks_completed=True,
                 frames=sum(stop-start for start,stop in job['chunks']),elapsed_s=time.perf_counter()-started),indent=2)+'\n')
    finally:
        ij.dispose()


if __name__ == '__main__':
    main()
