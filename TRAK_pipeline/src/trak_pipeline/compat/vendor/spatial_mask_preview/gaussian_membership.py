"""Conditional diagonal-normal position mass over unions of native pixel squares.

This is a plug-in position model, not a calibrated posterior or binding model.
Unknown C1-zero regions and positions outside the FOV remain separate from Off.
"""
import math
import numpy as np
from scipy.special import ndtr


def normal_interval(lo,hi):
    """Stable Phi(hi)-Phi(lo), including small right-tail pixel probabilities."""
    lo=np.asarray(lo,float);hi=np.asarray(hi,float)
    return np.where(lo>=0,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo))


def position_mass(mask,zero_unknown,x,y,sigma_x,sigma_y,z=8.0):
    if mask.ndim!=2 or zero_unknown.shape!=mask.shape:
        raise ValueError('Mask and unknown region must share the native YX grid')
    if not np.isfinite([x,y,sigma_x,sigma_y,z]).all() or min(sigma_x,sigma_y,z)<=0:
        raise ValueError('Finite center and positive per-axis position SE are required')
    h,w=mask.shape
    x0=max(0,math.ceil(x-z*sigma_x-.5));x1=min(w-1,math.floor(x+z*sigma_x+.5))
    y0=max(0,math.ceil(y-z*sigma_y-.5));y1=min(h-1,math.floor(y+z*sigma_y+.5))
    outside_x=float(ndtr((-.5-x)/sigma_x)+ndtr((x-(w-.5))/sigma_x))
    outside_y=float(ndtr((-.5-y)/sigma_y)+ndtr((y-(h-.5))/sigma_y))
    outside=float(np.clip(outside_x+outside_y-outside_x*outside_y,0,1))
    on=off=unknown_inside=0.0
    if x1>=x0 and y1>=y0:
        xx=np.arange(x0,x1+1);yy=np.arange(y0,y1+1)
        wx=normal_interval((xx-.5-x)/sigma_x,(xx+.5-x)/sigma_x)
        wy=normal_interval((yy-.5-y)/sigma_y,(yy+.5-y)/sigma_y)
        weights=wy[:,None]*wx[None,:]
        mm=mask[y0:y1+1,x0:x1+1];zz=zero_unknown[y0:y1+1,x0:x1+1]
        on=float(weights[mm&~zz].sum());off=float(weights[~mm&~zz].sum())
        unknown_inside=float(weights[zz].sum())
    on=float(np.clip(on,0,1));off=float(np.clip(off,0,1));unknown=float(np.clip(unknown_inside+outside,0,1))
    residual=1-(on+off+unknown)
    bound=float(4*ndtr(-z))
    if abs(residual)>bound+1e-11:raise ArithmeticError('Normal-mass partition did not conserve probability')
    return dict(p_on_model=on,p_not_on_model=off,p_unknown_model=unknown,
        p_C1_zero_region_model=unknown_inside,p_outside_FOV_model=outside,
        probability_partition_residual=residual,numerical_tail_mass_bound=bound)
