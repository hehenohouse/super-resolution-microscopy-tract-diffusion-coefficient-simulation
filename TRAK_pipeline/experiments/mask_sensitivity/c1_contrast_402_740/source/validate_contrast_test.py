#!/usr/bin/env python3
"""Independent sample-only audit of the isolated T5 C1 contrast experiment.

No segmentation, localization, assignment or TARDIS routines are imported or
executed. Raw pixels are decoded for ten predefined frames only. Final binary
masks are independently reconstructed using explicit pixel-component searches.
"""
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction
import argparse,csv,hashlib,json,math,os,sys,time,traceback,uuid
import numpy as np
import tifffile
import xml.etree.ElementTree as ET

FRAMES=[0,24,49,1200,1224,1249,2450,2474,2498,2499]
STARTS=[0,1200,2450];Y=684;X=428;SMALL=1024*1024


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp_'+uuid.uuid4().hex)
 tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def same(a,b,message):np.testing.assert_array_equal(a,b,err_msg=message)
def close(a,b,message):assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-10),(message,a,b)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sample_sha(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:h.update(f.read(65536));f.seek(max(0,p.stat().st_size-65536));h.update(f.read(65536))
 return h.hexdigest()
def pixel_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def exact_transform(raw,low=402,high=740):
 """Integer rational arithmetic gives an independent oracle for uint16 input."""
 assert raw.dtype==np.uint16 and low==402 and high==740
 delta=np.maximum(0,np.minimum(high-low,raw.astype(np.int64)-low))
 numerator=delta*65535;quotient,remainder=np.divmod(numerator,high-low)
 twice=2*remainder;up=(twice>high-low)|((twice==high-low)&((quotient&1)==1))
 return (quotient+up).astype(np.uint16)


def synthetic_tests():
 values=np.asarray([0,1,401,402,403,570,571,572,739,740,741,65535],np.uint16)
 expected=[]
 for value in values.tolist():
  q=Fraction(max(0,min(338,value-402))*65535,338)
  expected.append(round(q))
 same(exact_transform(values),np.asarray(expected,np.uint16),'exact scalar rounding / threshold oracle')
 f=np.rint(65535*np.clip((values.astype(np.float64)-402)/338,0,1)).astype(np.uint16)
 same(f,np.asarray(expected,np.uint16),'declared float64 formula against rational oracle')
 assert f[0]==0 and f[2]==0 and f[3]==0 and f[-1]==65535
 assert exact_transform(np.asarray([571],np.uint16))[0]==32768,'Midpoint must round to even, not truncate'
 all_values=np.arange(65536,dtype=np.uint16)
 same(exact_transform(all_values),np.rint(65535*np.clip((all_values.astype(np.float64)-402)/338,0,1)).astype(np.uint16),'all uint16 transform values')
 # Connectivity and exact-area boundaries are checked without production cleanup code.
 small=np.zeros((12,12),bool);small[1:5,1:9]=True
 assert remove_components(small,32).sum()==32 and remove_components(small,33).sum()==0
 small[1,1]=False;assert remove_components(small,32).sum()==0
 diagonal=np.eye(8,dtype=bool);assert remove_components(diagonal,8).sum()==8
 enclosed=np.ones((12,12),bool);enclosed[2:9,2:11]=False
 assert fill_holes(enclosed,64).all(),'An enclosed63pixel hole must fill'
 exact=np.ones((12,12),bool);exact[2:10,2:10]=False
 assert not fill_holes(exact,64)[3,3],'An enclosed64pixel hole must remain'
 exterior=np.ones((12,12),bool);exterior[0:3,4:6]=False
 assert not fill_holes(exterior,64)[1,4],'Exterior background must remain'
 diagonal_background=np.ones((4,4),bool);diagonal_background[0,0]=False;diagonal_background[1,1]=False
 separate=fill_holes(diagonal_background,64);assert not separate[0,0] and separate[1,1],'BG4 must not connect diagonal background pixels'
 return dict(all_passed=True,cases=['threshold clipping','uint16 subtraction without underflow','all65536uint16 values against float64 formula','exact midpoint ties-to-even','FG8 diagonal connectivity','foreground32included31removed','enclosed63filled64retained','exterior background retained'])


def components(mask,connectivity):
 """Complete independent graph traversal; no scipy/production labeling calls."""
 h,w=mask.shape;seen=np.zeros((h,w),bool)
 offsets=[(-1,0),(1,0),(0,-1),(0,1)]
 if connectivity==8:offsets += [(-1,-1),(-1,1),(1,-1),(1,1)]
 for y,x in np.argwhere(mask):
  y=int(y);x=int(x)
  if seen[y,x]:continue
  seen[y,x]=True;stack=[(y,x)];component=[];edge=False
  while stack:
   a,b=stack.pop();component.append((a,b));edge |= a==0 or b==0 or a==h-1 or b==w-1
   for dy,dx in offsets:
    u=a+dy;v=b+dx
    if 0<=u<h and 0<=v<w and mask[u,v] and not seen[u,v]:seen[u,v]=True;stack.append((u,v))
  yield component,edge


def remove_components(mask,minimum):
 out=np.zeros_like(mask,bool)
 for component,_ in components(mask,8):
  if len(component)>=minimum:
   y,x=zip(*component);out[y,x]=True
 return out


def fill_holes(mask,exclusive):
 out=mask.copy()
 for component,edge in components(~mask,4):
  if not edge and len(component)<exclusive:
   y,x=zip(*component);out[y,x]=True
 return out


def packed(path,expected_frames,decode=True):
 with np.load(path,allow_pickle=False) as z:
  assert {'packed','shape','original_frame_ids','bitorder'}.issubset(z.files),str(path)
  shape=tuple(int(x) for x in z['shape']);assert shape==(len(expected_frames),Y,X),shape
  assert z['shape'].dtype==np.int32 and z['packed'].dtype==np.uint8
  same(z['original_frame_ids'],expected_frames,'packed original frame mapping')
  assert str(z['bitorder'].item())=='little'
  data=z['packed'].copy();assert data.shape==(len(expected_frames),(Y*X+7)//8)
 if not decode:return data
 return np.unpackbits(data,axis=1,count=Y*X,bitorder='little').reshape(shape).astype(bool)


def protect(run):
 record=read(run/'source_records.json');protected=record['protected_sources'];small=0
 assert record['config_sha256']==sha(run/'config.json'),'Recorded configuration binding differs'
 if 'protected_sources_after' in record:assert record['protected_sources_after']==protected and record['protected_sources_unchanged']
 assert protected and len({x['path'] for x in protected})==len(protected)
 for item in protected:
  p=Path(item['path']);stat=p.stat();size=item.get('size_bytes',item.get('bytes'))
  assert stat.st_size==size and stat.st_mtime_ns==item['mtime_ns'],'Protected source descriptor changed: '+str(p)
  if stat.st_size<=SMALL:
   assert item.get('sha256'),'Missing small-source full fingerprint: '+str(p)
   assert sha(p)==item['sha256'],'Small source fingerprint changed: '+str(p);small+=1
  elif item.get('sampled_sha256'):assert sample_sha(p)==item['sampled_sha256'],'Sampled source fingerprint changed: '+str(p)
 return dict(protected_files=len(protected),small_full_SHA_checked=small,large_files='Size/mtime; sampled content additionally when supplied. No whole-movie content scan.')


def ome_check(tf):
 assert tf.ome_metadata,'Transformed image must retain OME calibration'
 root=ET.fromstring(tf.ome_metadata);pixels=root.find('.//{*}Pixels');assert pixels is not None
 assert int(pixels.attrib['SizeT'])==50 and int(pixels.attrib['SizeZ'])==1 and int(pixels.attrib['SizeC'])==1
 assert int(pixels.attrib['SizeY'])==Y and int(pixels.attrib['SizeX'])==X and pixels.attrib['Type']=='uint16'
 for key in ['PhysicalSizeX','PhysicalSizeY']:
  unit=pixels.attrib.get(key+'Unit','µm');scale={'µm':1000.,'um':1000.,'nm':1.}[unit];close(float(pixels.attrib[key])*scale,117,'physical pixel size')
 unit=pixels.attrib.get('TimeIncrementUnit','s');assert unit=='s';close(pixels.attrib['TimeIncrement'],.02,'frame interval')


def numerical(run,cfg):
 marker=read(run/'NUMERICAL_COMPLETE.json');assert marker.get('complete') is True
 assert cfg['sample_id']=='WT_TRAK2_BR1_T5' and cfg['block_starts']==STARTS and cfg['sample_frames']==FRAMES
 assert cfg['contrast_low_adu']==402 and cfg['contrast_high_adu']==740 and cfg['output_dtype']=='uint16'
 assert cfg['shape_TYX']==[2500,Y,X] and cfg['pixel_size_nm']==117 and cfg['frame_interval_s']==.02
 assert cfg['support_frames']==50 and cfg['support_minimum']==10 and cfg['foreground_min_pixels']==32 and cfg['hole_exclusive_max_pixels']==64
 source=Path(cfg['source_output']);roi=np.load(run/'masks/roi_mask.npy',allow_pickle=False);vertices=np.load(run/'masks/roi_vertices_xy.npy',allow_pickle=False)
 assert roi.shape==(Y,X) and roi.dtype==bool
 same(roi,np.load(source/'roi/roi_mask.npy',allow_pickle=False),'unchanged authoritative ROI pixels')
 same(vertices,np.load(source/'roi/vertices.npy',allow_pickle=False),'unchanged authoritative ROI vertices')
 mapping=rows(run/'data/frame_mapping.csv');assert len(mapping)==150
 originals=np.concatenate([np.arange(s,s+50) for s in STARTS]);same([int(r['page_index']) for r in mapping],np.arange(150),'page indices')
 same([int(r['original_frame']) for r in mapping],originals,'raw frame IDs')
 for j,r in enumerate(mapping):
  start=STARTS[j//50];assert int(r['mask_block'])==start//50 and int(r['local_block_index'])==j//50
  assert int(r['original_frame_start'])==start and int(r['original_frame_stop_exclusive'])==start+50
  close(r['time_s'],int(r['original_frame'])*.02,'original time')
 zero_packed=packed(run/'masks/original_zero_regions_packed.npz',originals,False)
 raw_samples={};transformed_samples={};copy_path=run/'inputs/C1_original.tiff';src=Path(cfg['source_C1'])
 assert copy_path.stat().st_size==src.stat().st_size and sample_sha(copy_path)==sample_sha(src),'Original copy size/sampled bytes differ'
 copy_record=read(run/'source_records.json')['copied_C1'];assert Path(copy_record['path']).resolve()==copy_path.resolve() and Path(copy_record['source_path']).resolve()==src.resolve() and copy_record['bytes']==src.stat().st_size and copy_record['copy_method']=='single_stream_byte_for_byte'
 assert marker['source_copy_sha256']==copy_record['sha256'] and marker['protected_sources_unchanged']
 with tifffile.TiffFile(src) as original,tifffile.TiffFile(copy_path) as copied:
  assert len(original.pages)==2500 and len(copied.pages)==2500
  for frame in FRAMES:
   a=original.pages[frame].asarray();b=copied.pages[frame].asarray();assert a.shape==(Y,X) and a.dtype==np.uint16
   same(a,b,'source versus byte-copy sampled frame');raw_samples[frame]=a
   page=int(np.flatnonzero(originals==frame)[0]);zeros=np.unpackbits(zero_packed[page],count=Y*X,bitorder='little').reshape(Y,X).astype(bool)
   same(zeros,a==0,'original raw zero-region provenance')
 header_count=0
 for start in STARTS:
  path=run/'inputs'/f'C1_contrast_frames{start:04d}_{start+49:04d}.ome.tif'
  with tifffile.TiffFile(path) as tf:
   assert len(tf.pages)==50 and tf.series[0].axes=='TYX' and tf.series[0].shape==(50,Y,X)
   ome_check(tf)
   for page in tf.pages:assert page.shape==(Y,X) and page.dtype==np.uint16;header_count+=1
   for frame in FRAMES:
    if not start<=frame<start+50:continue
    transformed=tf.pages[frame-start].asarray();same(transformed,exact_transform(raw_samples[frame]),'sampled transformed pixels, exact rational rounding')
    transformed_samples[frame]=transformed
 final=np.load(run/'masks/clean_full_mask.npy',allow_pickle=False);analysis=np.load(run/'masks/analysis_ROI_mask.npy',allow_pickle=False)
 baseline=np.load(run/'masks/baseline_clean_full_mask.npy',allow_pickle=False);baseline_roi=np.load(run/'masks/baseline_analysis_ROI_mask.npy',allow_pickle=False);count=np.load(run/'masks/foreground_count.npy',allow_pickle=False)
 for a in [final,analysis,baseline,baseline_roi]:assert a.shape==(3,Y,X) and a.dtype==bool
 assert count.shape==(3,Y,X) and count.dtype==np.uint8 and count.max()<=50
 same(analysis,final&roi,'new full-mask/ROI intersection');same(baseline_roi,baseline&roi,'baseline full-mask/ROI intersection')
 old=np.load(source/'masks/clean_full_mask.npy',mmap_mode='r',allow_pickle=False);oldroi=np.load(source/'masks/analysis_ROI_mask.npy',mmap_mode='r',allow_pickle=False)
 independent=[]
 for i,start in enumerate(STARTS):
  b=start//50;same(baseline[i],old[b],'unchanged baseline full mask');same(baseline_roi[i],oldroi[b],'unchanged baseline ROI mask')
  broad=packed(run/'masks/broad_packed'/f'b{b:03d}.npz',np.arange(start,start+50))
  expected_count=broad.sum(0,dtype=np.uint8);same(count[i],expected_count,'all50 broad support votes')
  voted=expected_count>=10;clean=remove_components(voted,32);filled=fill_holes(clean,64)
  same(final[i],filled,'independent FG8 min32 then enclosed BG4 holes<64')
  independent.append(dict(block_index=b,original_frame_start=start,broad_frames=50,voted_pixels=int(voted.sum()),post_min32_pixels=int(clean.sum()),final_full_pixels=int(filled.sum()),final_ROI_pixels=int((filled&roi).sum()),filled_pixels=int((filled&~clean).sum())))
 comparisons=rows(run/'data/mask_comparison.csv');assert len(comparisons)==6
 observed_keys=set()
 for r in comparisons:
  i=STARTS.index(int(r['original_frame_start']));start=STARTS[i];b=start//50;scope=r['scope'].lower();assert scope in ['full','roi']
  assert (scope,b) not in observed_keys;observed_keys.add((scope,b));assert int(r['block_index'])==b and int(r['original_frame_stop_exclusive'])==start+50
  oldmask=baseline[i] if scope=='full' else baseline_roi[i];newmask=final[i] if scope=='full' else analysis[i]
  a=int(oldmask.sum());z=int(newmask.sum());inter=int((oldmask&newmask).sum());union=int((oldmask|newmask).sum());added=int((newmask&~oldmask).sum());removed=int((oldmask&~newmask).sum())
  for field,value in dict(old_pixels=a,new_pixels=z,delta_pixels=z-a,intersection_pixels=inter,union_pixels=union,added_pixels=added,removed_pixels=removed).items():assert int(r[field])==value,(field,r[field],value)
  for field,value in dict(area_old_um2=a*.117**2,area_new_um2=z*.117**2,delta_area_um2=(z-a)*.117**2,iou=inter/union if union else 1.,dice=2*inter/(a+z) if a+z else 1.).items():close(r[field],value,field)
 assert observed_keys=={(s,b) for s in ['full','roi'] for b in [0,24,49]}
 pooled=np.concatenate([raw_samples[f][roi&(raw_samples[f]!=0)] for f in cfg['percentile_sample_frames']])
 p=np.percentile(pooled,[1,99.9]).tolist()
 statistics=rows(run/'data/transform_statistics.csv');assert len(statistics)==150;assert {int(r['original_frame']) for r in statistics}==set(originals.tolist())
 sampled_statistics=0
 for row in statistics:
  frame=int(row['original_frame']);assert int(row['block_index'])==frame//50
  for scope,total in [('full',Y*X),('ROI',int(roi.sum()))]:
   assert int(row[scope+'_pixels'])==total
   for suffix in ['original_zero_pixels','low_clipped_pixels','high_clipped_pixels','new_zero_nonzero_input_pixels']:assert 0<=int(row[scope+'_'+suffix])<=total
  if frame not in raw_samples:continue
  raw=raw_samples[frame]
  for scope,values in [('full',raw),('ROI',raw[roi])]:
   expected=dict(original_zero_pixels=np.count_nonzero(values==0),low_clipped_pixels=np.count_nonzero(values<=402),high_clipped_pixels=np.count_nonzero(values>=740),new_zero_nonzero_input_pixels=np.count_nonzero((values>0)&(values<=402)))
   for suffix,value in expected.items():assert int(row[scope+'_'+suffix])==int(value),(scope,suffix,frame)
  sampled_statistics+=1
 assert sampled_statistics==10
 report=dict(numerical_all_passed=True,transform_statistics_frames_checked=sampled_statistics,sample_frames=FRAMES,raw_source_pages_decoded=10,original_copy_pages_decoded=10,transformed_pages_decoded=10,transformed_page_headers_checked=header_count,source_copy_check='Matching size, first/last64KiB and ten original pixel pages; no complete original-movie hash scan.',original_zero_frames_checked=10,original_zero_packed_frames_mapped=150,independent_cleanup_blocks=independent,mask_comparison_rows_checked=6,frame_mapping_rows_checked=150,pooled_ROI_nonzero_sample_percentiles_P1_P99p9=p,transform='Independent exact rational integer arithmetic with ties-to-even, checked against declared float64 expression.',production_segmentation_or_analysis_calls=0)
 return report,dict(raw=raw_samples,transformed=transformed_samples,roi=roi,vertices=vertices,final=final,baseline=baseline,mapping=mapping)


def dilate(mask,iterations=1):
 out=mask.copy()
 for _ in range(iterations):
  pad=np.pad(out,1,constant_values=False);next_mask=np.zeros_like(out)
  for dy in range(3):
   for dx in range(3):next_mask |= pad[dy:dy+out.shape[0],dx:dx+out.shape[1]]
  out=next_mask
 return out


def boundary(mask):
 out=np.zeros_like(mask)
 out[1:] |= mask[1:]!=mask[:-1];out[:-1] |= mask[:-1]!=mask[1:]
 out[:,1:] |= mask[:,1:]!=mask[:,:-1];out[:,:-1] |= mask[:,:-1]!=mask[:,1:]
 return out


def render(run,cfg,reference):
 manifest=read(run/'render_manifest.json');meta=manifest['TIFF']
 assert meta['pages']==150 and meta['uniform_time_axis'] is False and meta['size_xy']==[2288,1090]
 assert manifest['crop_xyxy_px']==[20,77,392,536] and manifest['display_scale']==2
 assert manifest['display_original_C1_ADU']==[407,651] and manifest['old_and_new_panels_share_identical_original_C1_background']
 bounds=manifest['panel_image_bounds_xyxy'];assert bounds==[[12,96,756,1014],[772,96,1516,1014],[1532,96,2276,1014]]
 assert manifest['PNG_count']==7 and manifest['no_segmentation_localization_or_TARDIS_run']
 mapping=rows(run/manifest['mapping_csv']);assert len(mapping)==150
 expected_frames=np.concatenate([np.arange(s,s+50) for s in STARTS]);same([int(r['original_frame']) for r in mapping],expected_frames,'render frame mapping')
 sampled={int(r['original_frame']):r for r in manifest['sampled_pages']};assert set(sampled)==set(FRAMES)
 crop=cfg['crop_xyxy_px'];slice_y=slice(crop[1],crop[3]);slice_x=slice(crop[0],crop[2]);roi=reference['roi']
 inside=np.repeat(np.repeat(roi[slice_y,slice_x],2,axis=0),2,axis=1)
 outside_far=~dilate(inside,6);interior_far=~dilate(~inside,6)
 assert outside_far.any() and interior_far.any()
 header_checks=0;pixel_checks=0;black_pixel_checks=0;background_pixel_checks=0;difference_pixel_checks=0
 path=run/meta['path'];assert path.is_file() and path.stat().st_size==meta['bytes']
 with tifffile.TiffFile(path) as tf:
  assert tf.is_bigtiff and not tf.is_ome and len(tf.pages)==150
  for index,page in enumerate(tf.pages):
   assert page.shape==(1090,2288,3) and page.dtype==np.uint8 and int(page.photometric)==2 and int(page.compression) in [8,32946]
   first,description=page.description.split('\n',1);assert 'NON-CONTIGUOUS' in first
   record=json.loads(description);frame=int(expected_frames[index]);local=index//50;start=STARTS[local];row=mapping[index]
   assert record['page_index']==index and record['original_frame']==frame and record['uniform_time_axis'] is False
   assert int(row['page_index'])==index and int(row['block_index'])==start//50 and int(row['mask_local_index'])==local and int(row['original_mask_block'])==start//50
   for key,value in [('block_index',start//50),('mask_local_index',local),('original_mask_block',start//50),('original_frame_start',start),('original_frame_stop_exclusive',start+50)]:assert record[key]==value
   close(record['time_s'],frame*.02,'render metadata original time');close(row['time_s'],frame*.02,'viewing CSV original time')
   assert tuple(page.tags['PageNumber'].value)==(index,150);header_checks+=1
   if frame not in sampled:continue
   image=page.asarray();sample=sampled[frame];assert pixel_sha(image)==sample['pixel_SHA256'],'Sampled viewing page round-trip hash mismatch'
   raw=reference['raw'][frame];gray=np.rint(np.clip((raw[slice_y,slice_x].astype(np.float64)-407)/244,0,1)*255).astype(np.uint8)
   gray=np.repeat(np.repeat(gray,2,axis=0),2,axis=1);gray[~inside]=0;rgb=np.repeat(gray[...,None],3,axis=2)
   assert pixel_sha(rgb)==sample['shared_original_gray_SHA256'],'Independent source-page grayscale background differs'
   old=reference['baseline'][local];new=reference['final'][local]
   excluded_native=dilate(boundary(old)|boundary(new)|boundary(roi),3)
   eligible=roi[slice_y,slice_x]&~excluded_native[slice_y,slice_x]
   eligible=np.repeat(np.repeat(eligible,2,axis=0),2,axis=1)&interior_far
   assert eligible.any()
   for panel_index,(x0,y0,x1,y1) in enumerate(bounds):
    panel=image[y0:y1,x0:x1];same(panel[outside_far],np.zeros((int(outside_far.sum()),3),np.uint8),'ROI exterior must be black beyond gray outline support');black_pixel_checks+=int(outside_far.sum())
    if panel_index<2:same(panel[eligible],rgb[eligible],'same original-frame background/crop/orientation away from contours');background_pixel_checks+=int(eligible.sum())
   expected_rgb=np.zeros((Y,X,3),np.uint8)
   expected_rgb[old&new&roi]=[110,110,110];expected_rgb[new&~old&roi]=[0,220,245];expected_rgb[old&~new&roi]=[255,210,40]
   expected_rgb=np.repeat(np.repeat(expected_rgb[slice_y,slice_x],2,axis=0),2,axis=1)
   x0,y0,x1,y1=bounds[2];same(image[y0:y1,x0:x1][interior_far],expected_rgb[interior_far],'independent mask difference RGB inside ROI');difference_pixel_checks+=int(interior_far.sum());pixel_checks+=1
 assert pixel_checks==10 and header_checks==150
 assert len(manifest['pngs'])==7
 for item in manifest['pngs']:
  p=run/item['path'];assert p.is_file()
 return dict(all_passed=True,TIFF_pages=150,RGB8_BigTIFF_Deflate=True,noncontiguous_original_frame_metadata_headers_checked=header_checks,sampled_pixel_pages_checked=pixel_checks,matched_sampled_page_hashes=True,independent_original_background_hashes=True,background_pixels_checked=background_pixel_checks,ROI_exterior_black_pixels_checked=black_pixel_checks,difference_RGB_pixels_checked=difference_pixel_checks,ROI_outline_exception='Excluded six display pixels around binary ROI boundary; no header/footer pixels tested as image ROI.',PNG_files_present=7,native_PowerPoint_or_Fiji_claim=False)


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--include-render',action='store_true');args=parser.parse_args();cfg=read(args.config);run=Path(cfg['run_directory']);started=time.perf_counter()
 try:
  synthetic=synthetic_tests();before=protect(run);report,reference=numerical(run,cfg)
  if args.include_render:report['render']=render(run,cfg,reference)
  after=protect(run);assert before==after
  report.update(all_passed=True,synthetic_tests=synthetic,source_preservation=after,completed_utc=datetime.now(timezone.utc).isoformat(),include_render=args.include_render,scope='Ten original/copy/transformed pixel pages and150headers; all3final mask cleanup reconstructions. No whole-movie pixel scan.')
 except BaseException as error:report=dict(all_passed=False,error=str(error),traceback=traceback.format_exc(),completed_utc=datetime.now(timezone.utc).isoformat())
 report['elapsed_seconds']=time.perf_counter()-started;target=run/'validation/report.json';atomic(target,report)
 print(json.dumps(dict(all_passed=report['all_passed'],report=str(target),error=report.get('error'),elapsed_seconds=report['elapsed_seconds'])));return 0 if report['all_passed'] else 1
if __name__=='__main__':raise SystemExit(main())
