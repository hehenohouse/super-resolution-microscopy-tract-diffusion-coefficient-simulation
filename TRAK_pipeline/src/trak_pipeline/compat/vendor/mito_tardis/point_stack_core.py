"""Deterministic point-only/raw-image viewing panels on a shared native grid.

No localization, segmentation, registration, or statistical routines are used.
"""
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ON=np.array([39,221,165],float)
OFF=np.array([192,143,247],float)
ROI_COLOR=np.array([140,140,140],float)
MASK_COLOR=np.array([255,75,75],float)


def display_centers(xy,crop,scale):
    """Integer source pixel centers map to centers of replicated display pixels."""
    return (np.asarray(xy,float).reshape(-1,2)-np.asarray(crop[:2],float)+.5)*scale-.5


def gray_crop(raw,crop,scale,limits):
    x0,y0,x1,y1=crop;lo,hi=limits
    data=np.clip((raw[y0:y1,x0:x1].astype(float)-lo)/(hi-lo),0,1)
    gray=np.rint(data*255).astype(np.uint8)
    return np.repeat(np.repeat(gray,scale,axis=0),scale,axis=1)


def marker_alpha(shape,centers,radius,stroke=None):
    """Floating-center analytic distance with a one-display-pixel antialias ramp."""
    h,w=shape;alpha=np.zeros((h,w),np.float32)
    pad=radius+(.5 if stroke is None else stroke/2+.5)
    for cx,cy in np.asarray(centers,float).reshape(-1,2):
        x0=max(0,int(np.floor(cx-pad)));x1=min(w,int(np.ceil(cx+pad))+1)
        y0=max(0,int(np.floor(cy-pad)));y1=min(h,int(np.ceil(cy+pad))+1)
        if x0>=x1 or y0>=y1:continue
        yy,xx=np.ogrid[y0:y1,x0:x1];distance=np.hypot(xx-cx,yy-cy)
        coverage=np.clip(radius+.5-distance,0,1) if stroke is None else np.clip(stroke/2+.5-np.abs(distance-radius),0,1)
        alpha[y0:y1,x0:x1]=np.maximum(alpha[y0:y1,x0:x1],coverage)
    return alpha


def composite(base,alpha,color):
    return np.rint(np.asarray(base,float)*(1-alpha[...,None])+np.asarray(color,float)*alpha[...,None]).clip(0,255).astype(np.uint8)


def combined_layer(base,on,off):
    """Commutative color mixing; neither group is preferentially drawn on top."""
    total=on+off
    color=np.zeros((*on.shape,3),float)
    np.divide(on[...,None]*ON+off[...,None]*OFF,total[...,None],out=color,where=total[...,None]>0)
    alpha=np.maximum(on,off)
    return np.rint(np.asarray(base,float)*(1-alpha[...,None])+color*alpha[...,None]).clip(0,255).astype(np.uint8)


def roi_alpha(vertices,crop,scale):
    h=(crop[3]-crop[1])*scale;w=(crop[2]-crop[0])*scale;sup=4
    image=Image.new('L',(w*sup,h*sup),0)
    points=display_centers(np.asarray(vertices,float),crop,scale)
    # Convert display pixel-center coordinates to supersampled pixel centers.
    points=(points+.5)*sup-.5
    closed=np.vstack([points,points[0]])
    ImageDraw.Draw(image).line([tuple(p) for p in closed],fill=255,width=3,joint='curve')
    return np.asarray(image.resize((w,h),Image.Resampling.LANCZOS),float)/255


def mask_contour_alpha(full_mask,crop,scale):
    """Closed native pixel-square boundaries, computed BEFORE display cropping.

    Includes holes, excludes artificial crop/ROI boundaries. Supersampling only
    antialiases the viewing line and never changes the scientific binary mask.
    """
    x0,y0,x1,y1=crop;h,w=full_mask.shape
    horizontal=np.zeros((h+1,w),bool);vertical=np.zeros((h,w+1),bool)
    horizontal[0]=full_mask[0];horizontal[-1]=full_mask[-1]
    horizontal[1:-1]=full_mask[:-1]!=full_mask[1:]
    vertical[:,0]=full_mask[:,0];vertical[:,-1]=full_mask[:,-1]
    vertical[:,1:-1]=full_mask[:,:-1]!=full_mask[:,1:]
    sup=2;step=scale*sup;hh=(y1-y0)*step;ww=(x1-x0)*step
    high=np.zeros((hh,ww),np.uint8)
    yy,xx=np.nonzero(horizontal[y0:y1+1,x0:x1])
    for delta in [-1,0]:
        ry=yy*step+delta;valid=(ry>=0)&(ry<hh)
        high[ry[valid,None],xx[valid,None]*step+np.arange(step)]=255
    yy,xx=np.nonzero(vertical[y0:y1,x0:x1+1])
    for delta in [-1,0]:
        cx=xx*step+delta;valid=(cx>=0)&(cx<ww)
        high[yy[valid,None]*step+np.arange(step),cx[valid,None]]=255
    image=Image.fromarray(high).resize(((x1-x0)*scale,(y1-y0)*scale),Image.Resampling.LANCZOS)
    return np.asarray(image,float)/255


class Renderer:
    def __init__(self,vertices,c):
        self.c=c;self.crop=c['crop_xyxy_px'];self.scale=c['display_scale']
        self.w=(self.crop[2]-self.crop[0])*self.scale
        self.h=(self.crop[3]-self.crop[1])*self.scale
        self.margin=16;self.gap=16;self.header=82;self.footer=52
        self.width=6*self.w+2*self.margin+5*self.gap
        self.height=self.header+self.h+self.footer
        self.origins=[(self.margin+j*(self.w+self.gap),self.header) for j in range(6)]
        self.roi=roi_alpha(vertices,self.crop,self.scale)
        self.roi_black=composite(np.zeros((self.h,self.w,3),np.uint8),self.roi,ROI_COLOR)
        self.fonts={n:ImageFont.truetype(c['font_path'],n) for n in [14,16,17,19,21]}

    def render(self,frame,on_xy,off_xy,raw_c1,raw_c2,full_mask,fallback=False):
        assert raw_c1.shape==raw_c2.shape==full_mask.shape==(684,428)
        on=display_centers(on_xy,self.crop,self.scale);off=display_centers(off_xy,self.crop,self.scale)
        dot_on=marker_alpha((self.h,self.w),on,self.c['dot_radius_native_px']*self.scale)
        dot_off=marker_alpha((self.h,self.w),off,self.c['dot_radius_native_px']*self.scale)
        ring_on=marker_alpha((self.h,self.w),on,self.c['circle_radius_native_px']*self.scale,stroke=1.25)
        ring_off=marker_alpha((self.h,self.w),off,self.c['circle_radius_native_px']*self.scale,stroke=1.25)
        first=composite(self.roi_black,dot_on,ON)
        second=composite(self.roi_black,dot_off,OFF)
        third=combined_layer(self.roi_black,dot_on,dot_off)
        c2=np.repeat(gray_crop(raw_c2,self.crop,self.scale,self.c['display_C2_ADU'])[...,None],3,axis=2)
        fourth=combined_layer(composite(c2,self.roi,ROI_COLOR),ring_on,ring_off)
        c1=np.repeat(gray_crop(raw_c1,self.crop,self.scale,self.c['display_C1_ADU'])[...,None],3,axis=2)
        fifth=composite(composite(c1,mask_contour_alpha(full_mask,self.crop,self.scale),MASK_COLOR),self.roi,ROI_COLOR)
        # Same C1 pixels, full-mask contour and floating localization centers as
        # panels five/four. Circles stay visible above the mask contour.
        sixth=combined_layer(fifth,ring_on,ring_off)
        canvas=Image.new('RGB',(self.width,self.height),'black');draw=ImageDraw.Draw(canvas)
        heading=f'TDKd3 T1 | Original frame {frame:04d} | {frame*.02:.2f} s | W5/min32 | C1/C2 identity registration UNVERIFIED'
        draw.text((self.margin,9),heading,font=self.fonts[21],fill='#FFFFFF')
        n_on,n_off=len(on),len(off)
        labels=[f'On | {n_on} ROI points',f'Not-on | {n_off} ROI points',
                f'On + Not-on | {n_on+n_off} ROI points',f'C2 raw + circles | {n_on+n_off} points',
                'C1 raw + mask'+(' | UNSMOOTHED FALLBACK' if fallback else ''),
                f'C1 raw + mask + circles | {n_on+n_off} points']
        for i,(panel,label,origin) in enumerate(zip([first,second,third,fourth,fifth,sixth],labels,self.origins)):
            canvas.paste(Image.fromarray(panel),origin)
            draw.text((origin[0],49),label,font=self.fonts[17],fill=tuple(ON.astype(int)) if i==0 else tuple(OFF.astype(int)) if i==1 else '#FFFFFF')
        y=self.header+self.h+9
        draw.text((self.margin,y),'Dots: radius 1.5 native px; circles: radius 3 native px. Fixed sizes mark centers, not localization errors. Overlapping group markers blend.',
                  font=self.fonts[16],fill='#DADADA')
        draw.text((self.margin,y+23),'Gray: cell ROI | Red: full-mask boundary before ROI clipping | C2: 406-554 ADU; C1: 407-651 ADU | No refitting, reclassification or trajectories.',
                  font=self.fonts[16],fill='#DADADA')
        return np.asarray(canvas),dict(frame=frame,time_s=frame*.02,On=n_on,Not_on=n_off,combined=n_on+n_off,
            mask_temporal_applied=not fallback)
