"""Native-coordinate, original-page QC and unsmoothed count plots."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from comparison_core import LOSS, GAIN
from run_preview import outline, number
from geometry import pixel_square_union

OLD_COLOR='#D66A32';NEW_COLOR='#2999CB'
FLIP_COLORS={LOSS:'#E34352',GAIN:'#217BC8'}


def draw(ax,raw,geometry,vertices,limits,display,color):
    ax.imshow(raw,cmap='gray',vmin=display[0],vmax=display[1],origin='upper',interpolation='nearest')
    outline(ax,geometry,color,.7,visible_bounds=limits)
    closed=np.vstack([vertices,vertices[0]])
    ax.plot(closed[:,0],closed[:,1],color='#D4B32A',lw=.7)
    x0,y0,x1,y1=limits;ax.set(xlim=(x0,x1),ylim=(y1,y0),aspect='equal');ax.set_axis_off()


def representative(frame,C1,C2,oldmask,newmask,vertices,rows,c,out):
    x0,y0,x1,y1=c['display_crop_xyxy_px'];limits=(x0-.5,y0-.5,x1-.5,y1-.5)
    oldgeom=pixel_square_union(oldmask);newgeom=pixel_square_union(newmask)
    fig,axes=plt.subplots(1,3,figsize=(16,5.7))
    fig.subplots_adjust(left=.01,right=.99,top=.84,bottom=.14,wspace=.025)
    draw(axes[0],C1,oldgeom,vertices,limits,c['display_C1_ADU'],OLD_COLOR)
    draw(axes[1],C1,newgeom,vertices,limits,c['display_C1_ADU'],NEW_COLOR)
    draw(axes[2],C2,oldgeom,vertices,limits,c['display_C2_ADU'],OLD_COLOR)
    outline(axes[2],newgeom,NEW_COLOR,.7,visible_bounds=limits)
    losses=gains=0
    for r in rows:
        if r['spatial_selected']!='True':continue
        x,y=number(r,'fit_x_px'),number(r,'fit_y_px')
        pair=(r['pointwise_label'],r['new_pointwise_label'])
        if pair in FLIP_COLORS:
            color=FLIP_COLORS[pair];a=number(r,'x_se_nm')/117;b=number(r,'y_se_nm')/117
            axes[2].add_patch(Ellipse((x,y),2*a,2*b,fill=False,edgecolor=color,lw=1.3))
            axes[2].plot(x,y,'o',ms=5,mfc='none',mec=color,mew=1)
            losses+=pair==LOSS;gains+=pair==GAIN
        else:axes[2].plot(x,y,'.',ms=1.8,color='#8D8D8D',alpha=.55)
    axes[0].set_title('C1 raw + old min16',fontsize=10)
    axes[1].set_title('C1 raw + new W5/min32',fontsize=10)
    axes[2].set_title(f'C2 raw | On -> Not-on: {losses}; Not-on -> On: {gains}',fontsize=10)
    fig.suptitle(f'TDKd3 T1 | Original frame {frame} | {frame*.02:.2f} s | Same localizations and k=1 ellipses\n'
        'Exploratory identity registration unverified; changed labels are mask sensitivity, not molecular transfers',fontsize=11)
    handles=[Line2D([],[],color=OLD_COLOR,label='Old mask'),Line2D([],[],color=NEW_COLOR,label='New mask'),
        Line2D([],[],color='#D4B32A',label='Cell ROI'),Line2D([],[],marker='o',mfc='none',ls='none',color=FLIP_COLORS[LOSS],label='On -> Not-on'),
        Line2D([],[],marker='o',mfc='none',ls='none',color=FLIP_COLORS[GAIN],label='Not-on -> On')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.05),ncol=5,fontsize=8.5,frameon=False)
    fig.text(.5,.015,'C1: 407–651 ADU; C2: 406–554 ADU. Raw images unchanged; mask cleanup precedes ROI clipping.',ha='center',fontsize=8)
    p=out/'qc_frames'/f'frame_{frame:04d}_mask_comparison.png'
    fig.savefig(p,dpi=c['figure_dpi'],facecolor='white',metadata={'original_frame':str(frame)});plt.close(fig)
    return p


def examples_page(direction,examples,raw1,raw2,oldfull,newfull,vertices,c,out,page_name=None):
    if not examples:return None
    n=len(examples);height=3.4*n+1.3
    fig,axes=plt.subplots(n,3,figsize=(12,height),squeeze=False)
    fig.subplots_adjust(left=.035,right=.975,top=1-1.5/height,bottom=.7/height,hspace=.38,wspace=.06)
    half=c['example_context_px']//2
    for i,r in enumerate(examples):
        f=int(r['frame']);x,y=number(r,'fit_x_px'),number(r,'fit_y_px');cx,cy=int(np.floor(x+.5)),int(np.floor(y+.5))
        limits=(cx-half-.5,cy-half-.5,cx+half+.5,cy+half+.5)
        x0,y0,x1,y1=limits
        # Build display geometry only from nearby pixels; no clipped mask is
        # used by classification or the independent whole-frame checks.
        from shapely import affinity
        ox=max(0,cx-half-1);oy=max(0,cy-half-1);ex=min(428,cx+half+2);ey=min(684,cy+half+2)
        og=affinity.translate(pixel_square_union(oldfull[f,oy:ey,ox:ex]),xoff=ox,yoff=oy)
        ng=affinity.translate(pixel_square_union(newfull[f,oy:ey,ox:ex]),xoff=ox,yoff=oy)
        C1=raw1.pages[f].asarray();C2=raw2.pages[f].asarray()
        for j,g in [(0,og),(1,ng)]:draw(axes[i,j],C1,g,vertices,limits,c['display_C1_ADU'],OLD_COLOR if j==0 else NEW_COLOR)
        draw(axes[i,2],C2,og,vertices,limits,c['display_C2_ADU'],OLD_COLOR)
        outline(axes[i,2],ng,NEW_COLOR,.85,visible_bounds=limits)
        a,b=number(r,'x_se_nm')/117,number(r,'y_se_nm')/117
        for ax in axes[i]:
            pair=(r['pointwise_label'],r['new_pointwise_label'])
            ax.add_patch(Ellipse((x,y),2*a,2*b,fill=False,edgecolor=FLIP_COLORS[pair],lw=1.4))
            ax.plot(x,y,'+',color=FLIP_COLORS[pair],ms=7,mew=1)
        direction_label='On -> Not-on' if pair==LOSS else 'Not-on -> On'
        label=f'{direction_label} | frame {f}, candidate {r["candidate_index"]} | SE X/Y {number(r,"x_se_nm"):.1f}/{number(r,"y_se_nm"):.1f} nm'
        for ax,title in zip(axes[i],['C1 + old min16','C1 + new W5/min32','C2 + both masks']):ax.set_title(title,fontsize=10,pad=6)
        fig.text(.5,axes[i,1].get_position().y1+.4/height,label,ha='center',fontsize=10)
    name='on_to_not_on' if direction==LOSS else 'not_on_to_on' if direction==GAIN else 'both_directions'
    heading=name.replace('_',' ') if direction is not None else 'Both directions of same-point label changes'
    fig.suptitle(f'{heading} | Chronologically spaced examples | k=1\n'
        'Identity registration unverified; local model SE, not calibrated confidence coverage',fontsize=11,y=1-.12/height)
    fig.text(.5,.14/height,'Orange: old mask; blue: new mask. Ellipse: own X/Y model SE. C1 407–651 ADU; C2 406–554 ADU.',ha='center',fontsize=8.5)
    path=out/'flip_examples'/(page_name or f'{name}_examples.png')
    fig.savefig(path,dpi=c['figure_dpi'],facecolor='white');plt.close(fig)
    return path


def count_plots(frames,matrix,out):
    t=frames.frame
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
    for ax,field,title,color in [(axes[0],'on','On (mito-compatible)','#159C76'),(axes[1],'not_on','Not-on (non-mito-compatible)','#9854CA')]:
        ax.plot(t,frames['old_'+field],color='#555555',lw=.8,alpha=.75,label='Old min16')
        ax.plot(t,frames['new_'+field],color=color,lw=.85,label='New W5/min32')
        ax.set(title=title,ylabel='Localizations / frame');ax.legend();ax.grid(axis='y',alpha=.25)
    axes[1].set(xlabel='Original timepoint / frame (0-based)',xlim=(0,2499))
    fig.suptitle('Same points and QC | No smoothing of count curves',fontsize=12)
    fig.savefig(out/'old_new_counts_by_frame.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained')
    ax.plot(t,frames.on_to_not_on,color=FLIP_COLORS[LOSS],lw=.85,label='On -> Not-on')
    ax.plot(t,frames.not_on_to_on,color=FLIP_COLORS[GAIN],lw=.85,label='Not-on -> On')
    ax.set(xlim=(0,2499),ylim=(0,None),xlabel='Original timepoint / frame (0-based)',ylabel='Changed labels per frame',
        title='Old min16 -> new W5/min32 | Same-frame classification changes')
    ax.legend();ax.grid(axis='y',alpha=.25)
    fig.savefig(out/'flipped_labels_by_frame.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5.2),layout='constrained')
    for field,color,label in [('new_on','#159C76','On (mito-compatible)'),('new_not_on','#9854CA','Not-on (non-mito-compatible)'),('QC_excluded','#E88C1C','QC excluded')]:
        ax.plot(t,frames[field],color=color,lw=.85,label=label)
    ax.set(xlim=(0,2499),ylim=(0,None),xlabel='Original timepoint / frame (0-based)',
        ylabel='Number of localizations per frame',title='TDKd3 T1 C2 | W5/min32 mask | Pointwise k=1 classification and QC')
    ax.set_xticks([0,250,500,750,1000,1250,1500,1750,2000,2250,2499]);ax.grid(axis='y',color='#DCE0E5',lw=.6);ax.set_axisbelow(True)
    ax.legend(loc='upper right')
    fig.suptitle(f'No count-curve smoothing; identity registration unverified; unknown = {int(frames.new_unknown.sum()):,}',fontsize=10,color='#555555')
    fig.savefig(out/'new_mask_on_not_on_QC_by_timepoint.png',dpi=180);plt.close(fig)
    frames[['frame','time_s','new_on','new_not_on','QC_excluded','new_unknown','selected','ROI_candidates','temporal_applied','mask_frame_status']].to_csv(out/'new_mask_on_not_on_QC_by_timepoint.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    im=ax.imshow(matrix,cmap='Blues',vmin=0,origin='upper');fig.colorbar(im,ax=ax,label='Same localizations')
    names=['On','Not-on','Unknown'];ax.set_xticks(range(3),names);ax.set_yticks(range(3),names)
    ax.set(xlabel='New W5/min32 label',ylabel='Old min16 label',title='Paired same-point classification counts')
    maximum=max(1,matrix.max())
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',color='white' if matrix[i,j]>maximum*.5 else 'black',fontsize=13)
    fig.savefig(out/'classification_transition_matrix.png',dpi=180);plt.close(fig)
