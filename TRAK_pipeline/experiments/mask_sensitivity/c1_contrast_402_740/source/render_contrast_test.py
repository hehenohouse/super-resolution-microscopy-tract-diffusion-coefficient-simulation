#!/usr/bin/env python3
"""Render the isolated T5 C1 contrast/mask sensitivity test.

No segmentation, localization, registration, or TARDIS is performed. The three
50-frame segments retain original frame/time labels and are not a continuous
150-frame acquisition. All controlled writes stay in this test directory.
"""
from pathlib import Path
import argparse,csv,hashlib,html,importlib.util,importlib.metadata,json,os,sys,uuid

ALLOWED_ROOT=Path('/Volumes/My Book/Xinran_FullsetTRAK2/WT/TRAK2_WT_BR1/101625_WT_TRAK2_T5f/test/C1_contrast_402_740_20261026T202633Z')


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def guarded(path):
 path=Path(path).resolve()
 if not ALLOWED_ROOT.is_dir() or not path.is_relative_to(ALLOWED_ROOT.resolve()):raise OSError(19,'Output must stay inside the isolated contrast test: '+str(path))
 return path


def atomic_json(path,value):
 path=guarded(path);temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
 try:temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temporary,path)
 finally:
  if temporary.exists():temporary.unlink()


def save_png(image,path):
 path=guarded(path);temporary=path.with_name('.'+path.stem+'.'+uuid.uuid4().hex+'.tmp.png')
 try:image.save(temporary,format='PNG');os.replace(temporary,path)
 finally:
  if temporary.exists():temporary.unlink()


def write_csv(path,rows,fields):
 path=guarded(path);temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  with temporary.open('w',encoding='utf-8',newline='') as stream:
   writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
  os.replace(temporary,path)
 finally:
  if temporary.exists():temporary.unlink()


def font_path():
 for path in ['/System/Library/Fonts/Supplemental/Arial.ttf','C:/Windows/Fonts/arial.ttf']:
  if Path(path).is_file():return path
 raise FileNotFoundError('Arial display font unavailable')


def render(config_path):
 config_path=Path(config_path).resolve();c=read_json(config_path);run=guarded(c['run_directory'])
 if not (run/'NUMERICAL_COMPLETE.json').is_file():raise RuntimeError('Segmentation is not complete; rendering must wait for NUMERICAL_COMPLETE.json')
 figures=guarded(run/'Analysis_Figures');figures.mkdir(exist_ok=True);data=guarded(run/'data');data.mkdir(exist_ok=True)
 runtime=guarded(run/'runtime/rendering');runtime.mkdir(parents=True,exist_ok=True)
 for key,folder in [('MPLCONFIGDIR','matplotlib'),('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('XDG_CACHE_HOME','xdg')]:
  path=runtime/folder;path.mkdir(exist_ok=True);os.environ[key]=str(path)
 sys.dont_write_bytecode=True
 import numpy as np
 import tifffile
 from PIL import Image,ImageDraw,ImageFont
 import matplotlib
 matplotlib.use('Agg',force=True)
 import matplotlib.pyplot as plt
 original_point_core=Path(c['source_release'])/'vendor/mito_tardis/point_stack_core.py'
 source=run/'source/point_stack_core.py'
 if not source.is_file() or sha(source)!=sha(original_point_core):raise ValueError('Frozen local point-stack core does not match its recorded original')
 spec=importlib.util.spec_from_file_location('contrast_point_stack_core',source);pc=importlib.util.module_from_spec(spec);spec.loader.exec_module(pc)
 crop=list(c['crop_xyxy_px']);scale=int(c['display_scale']);lo,hi=c['display_original_C1_ADU'];starts=list(c['block_starts'])
 if starts!=[0,1200,2450] or scale!=2:raise ValueError('Unexpected approved segment or display-scale contract')
 masks=run/'masks';full=np.load(masks/'clean_full_mask.npy',allow_pickle=False);roi_full=np.load(masks/'analysis_ROI_mask.npy',allow_pickle=False)
 old=np.load(masks/'baseline_clean_full_mask.npy',allow_pickle=False);old_roi=np.load(masks/'baseline_analysis_ROI_mask.npy',allow_pickle=False)
 roi=np.load(masks/'roi_mask.npy',allow_pickle=False);vertices=np.load(masks/'roi_vertices_xy.npy',allow_pickle=False)
 if full.shape!=(3,684,428) or old.shape!=full.shape or full.dtype!=bool or old.dtype!=bool:raise ValueError('Expected three complete boolean TYX masks')
 np.testing.assert_array_equal(roi_full,full & roi);np.testing.assert_array_equal(old_roi,old & roi)
 with (data/'mask_comparison.csv').open(encoding='utf-8',newline='') as stream:statistics=list(csv.DictReader(stream))
 stats={(row['scope'],int(row['block_index'])):row for row in statistics}
 for index in range(3):
  for scope,before,after in [('ROI',old_roi[index],roi_full[index]),('full',old[index],full[index])]:
   row=stats[scope,starts[index]//50];checks={'old_pixels':int(before.sum()),'new_pixels':int(after.sum()),'intersection_pixels':int((before&after).sum()),'union_pixels':int((before|after).sum()),'added_pixels':int((after&~before).sum()),'removed_pixels':int((before&~after).sum())}
   for key,value in checks.items():
    if int(row[key])!=value:raise ValueError('Saved mask statistics mismatch: '+scope+' '+str(index)+' '+key)
   if abs(float(row['area_old_um2'])-checks['old_pixels']*.117**2)>1e-9 or abs(float(row['area_new_um2'])-checks['new_pixels']*.117**2)>1e-9:raise ValueError('Saved mask area units mismatch')
 width=(crop[2]-crop[0])*scale;height=(crop[3]-crop[1])*scale;margin=12;gap=16;header=96;footer=76
 origins=[(margin+j*(width+gap),header) for j in range(3)];canvas_size=(3*width+2*gap+2*margin,header+height+footer)
 inside=np.repeat(np.repeat(roi[crop[1]:crop[3],crop[0]:crop[2]],scale,axis=0),scale,axis=1)
 outline=pc.roi_alpha(vertices,crop,scale);native_crop=(slice(crop[1],crop[3]),slice(crop[0],crop[2]))
 old_contours=[pc.mask_contour_alpha(mask,crop,scale)*inside for mask in old]
 new_contours=[pc.mask_contour_alpha(mask,crop,scale)*inside for mask in full]
 differences=[]
 for index in range(3):
  before,after=old[index],full[index];rgb=np.zeros((*roi.shape,3),np.uint8)
  rgb[before&after&roi]=[110,110,110];rgb[after&~before&roi]=[0,220,245];rgb[before&~after&roi]=[255,210,40]
  rgb=np.repeat(np.repeat(rgb[native_crop],scale,axis=0),scale,axis=1)
  differences.append(pc.composite(rgb,outline,pc.ROI_COLOR))
 font=font_path();fonts={size:ImageFont.truetype(font,size) for size in [15,16,17,18,20,22]}
 def text(draw,xy,label,maxwidth,size=18,fill='white'):
  for current in range(size,9,-1):
   chosen=ImageFont.truetype(font,current)
   if draw.textbbox((0,0),label,font=chosen)[2]<=maxwidth:draw.text(xy,label,font=chosen,fill=fill);return
  raise ValueError('Viewing annotation does not fit')
 def add_scale(draw,origin):
  x,y=origin;x+=18;y+=height+21;length=2000/117*scale
  draw.line([(x,y),(x+length,y)],fill='white',width=2);draw.text((x+length+9,y-10),'2 um',font=fonts[15],fill='white')
 def make_page(raw,index,frame):
  gray=np.repeat(pc.gray_crop(raw,crop,scale,[lo,hi])[...,None],3,axis=2);gray[~inside]=0
  first=pc.composite(pc.composite(gray,old_contours[index],pc.MASK_COLOR),outline,pc.ROI_COLOR)
  second=pc.composite(pc.composite(gray,new_contours[index],pc.MASK_COLOR),outline,pc.ROI_COLOR)
  canvas=Image.new('RGB',canvas_size,'black');draw=ImageDraw.Draw(canvas)
  text(draw,(margin,8),c['sample_name']+' | C1 contrast 402–740 sensitivity | NON-CONTIGUOUS SEGMENTS',canvas.width-2*margin,22)
  text(draw,(margin,37),f'Original frame {frame} | time {frame*.02:.2f} s | block {starts[index]}–{starts[index]+49} | preview segments 0–49, 1200–1249, 2450–2499',canvas.width-2*margin,18)
  titles=['Original C1 + baseline mask','Same original C1 + contrast-input mask','Mask difference: gray common; cyan added; yellow removed']
  for panel,title,origin in zip([first,second,differences[index]],titles,origins):
   canvas.paste(Image.fromarray(panel),origin);text(draw,(origin[0]+3,70),title,width-6,17);add_scale(draw,origin)
  text(draw,(margin,header+height+39),'Both image panels use the SAME original C1 pixels at 407–651 ADU. ROI exterior is black; contours are clipped only after full-mask boundary extraction.',canvas.width-2*margin,16)
  text(draw,(margin,header+height+58),'Mask only sensitivity test; source and production results unchanged. X increases right, Y increases down. Source 117 nm/pixel; display 2x.',canvas.width-2*margin,16)
  return canvas,gray
 tiff_path=guarded(figures/'C1_old_new_mask_difference_150pages.tif');temporary=tiff_path.with_name('.'+tiff_path.stem+'.'+uuid.uuid4().hex+'.tmp.tif')
 mapping=[];pngs=[];sample_pages=[];approved=set(c['sample_frames'])
 try:
  with tifffile.TiffFile(run/'inputs/C1_original.tiff') as movie,tifffile.TiffWriter(temporary,bigtiff=True,ome=False) as writer:
   if len(movie.pages)!=2500:raise ValueError('Copied original C1 must retain all 2500 pages')
   for block_index,start in enumerate(starts):
    for original_frame in range(start,start+50):
     raw=movie.pages[original_frame].asarray();page_index=len(mapping);image,gray=make_page(raw,block_index,original_frame);pixels=np.asarray(image)
     record=dict(page_index=page_index,original_frame=original_frame,time_s=original_frame*.02,block_index=start//50,mask_local_index=block_index,original_mask_block=start//50,original_frame_start=start,original_frame_stop_exclusive=start+50)
     description='TRAK2 contrast test: NON-CONTIGUOUS original frame mapping\n'+json.dumps(dict(record,uniform_time_axis=False),sort_keys=True)
     writer.write(pixels,photometric='rgb',compression='deflate',metadata=None,description=description,contiguous=False,software='TRAK2 contrast-test viewing artifact',extratags=[(297,'H',2,(page_index,150),False)])
     mapping.append(record)
     if original_frame in approved:sample_pages.append(dict(record,pixel_SHA256=hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest(),shared_original_gray_SHA256=hashlib.sha256(np.ascontiguousarray(gray).tobytes()).hexdigest()))
     if original_frame==start:
      filename=f'frame_{original_frame:04d}_old_new_mask_comparison.png';save_png(image,figures/filename);pngs.append(filename)
    print('CONTRAST_VIEW_SEGMENT_COMPLETE',start,start+49,len(mapping),flush=True)
  if len(mapping)!=150:raise ValueError('Expected exactly 150 output pages')
  os.replace(temporary,tiff_path)
 finally:
  if temporary.exists():temporary.unlink()
 write_csv(data/'viewing_frame_mapping.csv',mapping,['page_index','original_frame','time_s','block_index','mask_local_index','original_mask_block','original_frame_start','original_frame_stop_exclusive'])
 # Actual transformed inputs are shown on their uint16 range, not relabeled ADU.
 with tifffile.TiffFile(run/'inputs/C1_original.tiff') as movie:
  for start in starts:
   with tifffile.TiffFile(run/'inputs'/f'C1_contrast_frames{start:04d}_{start+49:04d}.ome.tif') as transformed_movie:transformed=transformed_movie.pages[0].asarray()
   raw=movie.pages[start].asarray();images=[]
   for array,limits in [(raw,[lo,hi]),(transformed,[0,65535])]:
    gray=np.repeat(pc.gray_crop(array,crop,scale,limits)[...,None],3,axis=2);gray[~inside]=0;images.append(pc.composite(gray,outline,pc.ROI_COLOR))
   preview=Image.new('RGB',(2*width+gap+2*margin,height+136),'black');draw=ImageDraw.Draw(preview)
   text(draw,(margin,8),f"{c['sample_name']} | actual segmentation input contrast | original frame {start}",preview.width-2*margin,22)
   for j,(panel,label) in enumerate(zip(images,['Original C1: display 407–651 ADU','Transformed input: display 0–65535 uint16 (NOT ADU)'])):
    x=margin+j*(width+gap);text(draw,(x,45),label,width,17);preview.paste(Image.fromarray(panel),(x,78))
   text(draw,(margin,height+94),'Actual input transform: rint(65535 × clip((original ADU − 402) / 338, 0, 1)). Neither display panel rescales the scientific source file.',preview.width-2*margin,16)
   filename=f'input_contrast_frame_{start:04d}.png';save_png(preview,figures/filename);pngs.append(filename)
 # Three discrete blocks; deliberately no connected temporal curve across gaps.
 fig,axes=plt.subplots(1,2,figsize=(13.5,5.7));positions=np.arange(3);barwidth=.34
 for axis,scope in zip(axes,['ROI','full']):
  rows=[stats[scope,starts[index]//50] for index in range(3)];before=np.array([float(row['area_old_um2']) for row in rows]);after=np.array([float(row['area_new_um2']) for row in rows]);maximum=max(float(before.max()),float(after.max()),1)
  old_bars=axis.bar(positions-barwidth/2,before,barwidth,label='Baseline mask',color='#607d8b');new_bars=axis.bar(positions+barwidth/2,after,barwidth,label='Contrast-input mask',color='#1b9e77')
  axis.bar_label(old_bars,labels=[f'{value:.2f}' for value in before],padding=3,fontsize=8);axis.bar_label(new_bars,labels=[f'{value:.2f}' for value in after],padding=3,fontsize=8)
  for x,row,b,z in zip(positions,rows,before,after):axis.text(x,max(b,z)+.085*maximum,f"IoU {float(row['iou']):.3f}",ha='center',fontsize=10)
  axis.set_xticks(positions,[f'{start}–{start+49}\n{start*.02:.0f}–{(start+50)*.02:.0f} s' for start in starts]);axis.set_ylim(0,maximum*1.23);axis.set_ylabel('Mask foreground area (µm²)');axis.set_title('Inside original cell ROI' if scope=='ROI' else 'Full image before ROI clipping');axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True);axis.legend(fontsize=9)
 fig.suptitle(c['sample_name']+' | three discrete 50-frame mask blocks',fontsize=16)
 fig.text(.5,.025,'Bars are separate sampled blocks, not a continuous time course. Unexpanded mask area = foreground pixels × (0.117 µm)².',ha='center',fontsize=10)
 fig.tight_layout(rect=[0,.07,1,.94]);filename='mask_area_and_IoU_three_blocks.png';p=figures/filename;temp=p.with_name('.'+p.stem+'.tmp.png')
 try:fig.savefig(temp,dpi=180,facecolor='white');os.replace(temp,p)
 finally:
  plt.close(fig)
  if temp.exists():temp.unlink()
 pngs.append(filename)
 with Image.open(figures/pngs[0]) as image:representative_size=list(image.size)
 records=[]
 for filename in pngs:
  path=figures/filename
  with Image.open(path) as image:size=list(image.size)
  records.append(dict(path='Analysis_Figures/'+filename,bytes=path.stat().st_size,sha256=sha(path),size_xy=size))
 manifest=dict(sample_id=c['sample_id'],run_id=c['run_id'],TIFF=dict(path='Analysis_Figures/'+tiff_path.name,pages=150,format='regular RGB8 BigTIFF; lossless Deflate; not OME',uniform_time_axis=False,page_description_format='one plain-text prefix line followed by a JSON object',size_xy=list(canvas_size),bytes=tiff_path.stat().st_size,photometric='RGB',page_axis='non-contiguous original-frame selection; see mappingCSV'),mapping_csv='data/viewing_frame_mapping.csv',block_starts=starts,segments_original_frames_inclusive=[[start,start+49] for start in starts],crop_xyxy_px=crop,display_scale=scale,source_pixel_size_nm=117,source_frame_interval_s=.02,display_original_C1_ADU=[lo,hi],display_transformed_uint16=[0,65535],transformed_units='uint16 contrast encoding; not raw ADU',panel_origins_xy=[list(p) for p in origins],panel_image_bounds_xyxy=[[x,y,x+width,y+height] for x,y in origins],ROI_exterior_black_rule='image panels black outside repeated ROI except gray antialiased ROI-outline pixels; all scalebars/text are below image bounds',mask_contour_rule='full native mask boundaries extracted first, then contour alpha multiplied by ROI display mask',difference_RGB=dict(common=[110,110,110],added=[0,220,245],removed=[255,210,40]),sampled_pages=sample_pages,pngs=records,PNG_count=len(records),figure_index='Analysis_Figures/INDEX.html',versions=dict(python=sys.version,python_executable=sys.executable,packages={name:importlib.metadata.version(name) for name in ['numpy','tifffile','Pillow','matplotlib']},font_path=font),renderer_source_SHA256=sha(__file__),frozen_point_stack_core=dict(local_path='source/point_stack_core.py',original_source=str(original_point_core),sha256=sha(source)),config_SHA256=sha(config_path),old_and_new_panels_share_identical_original_C1_background=True,orientation='X right; Y down',no_segmentation_localization_or_TARDIS_run=True)
 atomic_json(run/'render_manifest.json',manifest)
 body='<!doctype html><html lang="en"><meta charset="utf-8"><title>T5 C1 contrast sensitivity test</title><style>body{font:16px system-ui;margin:30px;max-width:1600px;color:#203746}img{width:100%;height:auto}a{color:#2166ac}</style><h1>T5 C1 contrast 402–740: mask comparison</h1><p>Separate test only. Original source files and production results remain unchanged.</p><p><a href="'+tiff_path.name+'">Download 150-page Fiji viewing TIFF</a> · <a href="../data/viewing_frame_mapping.csv">Original-frame mapping</a> · <a href="../data/mask_comparison.csv">Mask areas and overlap</a></p><p><strong>The 150 pages contain three separate segments: 0–49, 1200–1249, 2450–2499. They are not a continuous acquisition.</strong> Each page retains its original frame and time.</p><p>Both image panels show identical ORIGINAL C1 background at 407–651 ADU. Red contours compare baseline and newly segmented masks. Difference panels use gray common foreground, cyan added, and yellow removed. ROI exterior is black.</p>'
 for filename in pngs:body+='<h2>'+html.escape(filename)+'</h2><a href="'+filename+'"><img src="'+filename+'"></a>'
 body+='</html>';path=guarded(figures/'INDEX.html');temporary=path.with_name('.INDEX.tmp')
 try:temporary.write_text(body,encoding='utf-8');os.replace(temporary,path)
 finally:
  if temporary.exists():temporary.unlink()
 guarded(run/'INDEX.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>C1 contrast sensitivity</title><h1>C1 contrast sensitivity test</h1><p><a href="Analysis_Figures/INDEX.html">Open all viewing figures and TIFF</a></p></html>',encoding='utf-8')
 readme='C1 CONTRAST TEST — VIEWING OUTPUTS\n\nAnalysis_Figures/C1_old_new_mask_difference_150pages.tif is a lossless RGB8 Deflate BigTIFF for Fiji.\nIt contains original frames 0–49, 1200–1249, and 2450–2499 in that order.\nThis is NOT a continuous 150-frame movie. Do not assign a uniform 20 ms interval across the two gaps.\nEach page description and data/viewing_frame_mapping.csv retain the true original frame and time.\nThe first two columns use exactly the same original C1 pixels displayed at 407–651 ADU.\nThe baseline and new full-mask boundaries are extracted before ROI clipping, then restricted to ROI for display.\nThe third column shows common foreground gray, added cyan, and removed yellow.\nAll image panels have black ROI exterior apart from the gray antialiased ROI outline.\nSource coordinates are unshifted; X increases right and Y down. The crop and 2x display scale are recorded in render_manifest.json.\nInput contrast previews use the actual transformed uint16 input, displayed over 0–65535; this encoding is NOT raw ADU.\nArea plots show three independent sampled blocks, without joining them through unsampled intervals.\nThis renderer reads saved masks and input copies only. It performs no segmentation, localization, registration or TARDIS.\n'
 guarded(run/'README_RENDERING.txt').write_text(readme,encoding='utf-8')
 print('CONTRAST_RENDER_COMPLETE',len(mapping),'pages;',len(pngs),'PNGs',flush=True);return manifest


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args();render(args.config)


if __name__=='__main__':main()
