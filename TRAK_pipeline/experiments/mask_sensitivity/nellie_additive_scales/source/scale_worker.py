"""Nellie worker: preserve default sigma prefix and append configured larger scales."""
import json,sys,hashlib
from pathlib import Path
import numpy as np,tifffile
from nellie.im_info.verifier import FileInfo,ImInfo
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label

root,batch=map(Path,sys.argv[1:3])
cfg=json.loads((root.parent/'config.json').read_text())
method=json.loads((batch/'method.json').read_text())
class AdditiveScaleFilter(Filter):
    def _set_default_sigmas(self):
        super()._set_default_sigmas()
        actual_default=[float(x) for x in self.sigmas]
        if actual_default!=cfg['baseline_sigmas_pixels']:
            raise ValueError('Native baseline sigma list differs from recorded baseline')
        requested=method['sigmas_pixels']
        if requested[:len(actual_default)]!=actual_default or any(b<=a for a,b in zip(requested,requested[1:])):
            raise ValueError('Explicit scale list must preserve exact default prefix and increase')
        self.sigmas=[float(x) for x in requested]
        self.sigma_min=min(self.sigmas);self.sigma_max=max(self.sigmas)
        self.halo=self._compute_halo()

fi=FileInfo(str(batch/'input.ome.tif'),output_dir=str(batch/'native'))
fi.find_metadata();fi.load_metadata()
if fi.axes!='TYX':raise ValueError('Expected original TYX axes')
for k,v in [('X',.117),('Y',.117),('T',.02)]:fi.change_dim_res(k,v)
im=ImInfo(fi);n=im.shape[0]
if im.shape!=(n,684,428) or n!=25 or not im.no_z:raise ValueError('Unexpected native input grid')
filt=AdditiveScaleFilter(im,num_t=n,min_radius_um=.25,max_radius_um=1.,remove_edges=False,device='cpu',low_memory=False)
filt.run()
if filt.low_memory or filt.device_type!='cpu':
    raise RuntimeError('A changed backend/low-memory path would confound this scale comparison')
f0=int(batch.name[6:10]);selected=[int(f) for f in cfg['sample_frames'] if f0<=f<f0+n]
with tifffile.TiffFile(im.pipeline_paths['im_preprocessed']) as saved:
    response=np.stack([saved.pages[f-f0].asarray() for f in selected])
np.savez_compressed(batch/'sampled_filter_response.npz',response=response,original_frame_ids=np.array(selected,np.int32),axes=np.array('TYX'))
lab=Label(im,num_t=n,min_radius_um=.117,otsu_thresh_intensity=False,device='cpu',low_memory=False)
records=[];native=lab._compute_frame_thresholds
def record(a,b):
    intensity,filtered=native(a,b)
    records.append({'local_frame':len(records),'intensity_threshold_ADU':None if intensity is None else float(intensity),'frangi_threshold':None if filtered is None else float(filtered)})
    return intensity,filtered
lab._compute_frame_thresholds=record
lab.run()
if lab.low_memory:raise RuntimeError('Unexpected Label low-memory path')
labels=tifffile.imread(im.pipeline_paths['im_instance_label'])
np.save(batch/'broad_mask.npy',labels>0)
(batch/'thresholds.json').write_text(json.dumps(records,indent=2)+'\n')
(batch/'actual_parameters.json').write_text(json.dumps({
 'method':method['id'],'nominal_max_radius_um':method['nominal_max_radius_um'],
 'sigmas_pixels':filt.sigmas,'baseline_sigmas_pixels':cfg['baseline_sigmas_pixels'],
 'actual_low_memory':bool(filt.low_memory),'actual_device':filt.device_type,
 'sigma_min':filt.sigma_min,'sigma_max':filt.sigma_max,'halo_pixels':filt.halo,
 'native_min_area':lab.min_area_pixels,
 'note':'Only sigma list/derived sigma bounds and halo override. Native Frangi/positive-LoG, response masking, automatic thresholds and Label formulas unchanged.'
},indent=2)+'\n')
