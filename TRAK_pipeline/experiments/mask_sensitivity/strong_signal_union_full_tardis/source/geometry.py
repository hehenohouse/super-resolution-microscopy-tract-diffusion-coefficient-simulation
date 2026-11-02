"""Exact diagonal-ellipse contact with unions of closed pixel squares.

All coordinates are native pixels, with integer pixel centers. Geometry does
not establish cross-channel registration or confidence coverage.
"""
import math

import numpy as np
import shapely
from shapely import affinity

Q_TOLERANCE = 1e-12


def valid_ellipse(x, y, a, b):
    return bool(np.isfinite([x,y,a,b]).all() and a>0 and b>0)


def minimum_q(mask, x, y, a, b, tolerance=Q_TOLERANCE):
    """Minimum q among reachable foreground squares; infinity if none.

    Search bounds include the small q tolerance, so exact contacts at pixel
    boundaries are not lost to ceil/floor rounding.
    """
    if not valid_ellipse(x,y,a,b):
        raise ValueError('Ellipse center/axes must be finite; axes positive')
    if mask.ndim!=2:
        raise ValueError('Mask must be YX')
    h,w=mask.shape
    rx,ry=a*math.sqrt(1+tolerance),b*math.sqrt(1+tolerance)
    x0=max(0,math.ceil(x-rx-.5-1e-12));x1=min(w-1,math.floor(x+rx+.5+1e-12))
    y0=max(0,math.ceil(y-ry-.5-1e-12));y1=min(h-1,math.floor(y+ry+.5+1e-12))
    if x1<x0 or y1<y0:
        return math.inf
    py,px=np.nonzero(mask[y0:y1+1,x0:x1+1])
    if len(px)==0:
        return math.inf
    dx=np.maximum(np.abs(x-(px+x0))-.5,0)
    dy=np.maximum(np.abs(y-(py+y0))-.5,0)
    return float(np.min((dx/a)**2+(dy/b)**2))


def ellipse_intersects_mask(mask,x,y,a,b,tolerance=Q_TOLERANCE):
    return minimum_q(mask,x,y,a,b,tolerance)<=1+tolerance


def point_inside_mask(mask,x,y,tolerance=Q_TOLERANCE):
    if not np.isfinite([x,y]).all():
        return False
    h,w=mask.shape
    x0=max(0,math.ceil(x-.5-tolerance));x1=min(w-1,math.floor(x+.5+tolerance))
    y0=max(0,math.ceil(y-.5-tolerance));y1=min(h-1,math.floor(y+.5+tolerance))
    if x1<x0 or y1<y0:
        return False
    py,px=np.nonzero(mask[y0:y1+1,x0:x1+1])
    return bool(np.any((np.abs(x-(px+x0))<=.5+tolerance)&
                       (np.abs(y-(py+y0))<=.5+tolerance)))


def classify(mask,zero_unknown,x,y,a,b,tolerance=Q_TOLERANCE):
    if not valid_ellipse(x,y,a,b):
        return dict(label='unknown',reason='invalid_position_or_uncertainty',
            intersects_known_mask=None,intersects_zero_unknown=None,min_q=None)
    h,w=mask.shape
    if zero_unknown.shape!=mask.shape:
        raise ValueError('Unknown-region grid must match mask')
    outside=(x-a<-.5-tolerance or x+a>w-.5+tolerance or
             y-b<-.5-tolerance or y+b>h-.5+tolerance)
    q=minimum_q(mask,x,y,a,b,tolerance)
    zero=ellipse_intersects_mask(zero_unknown,x,y,a,b,tolerance)
    reasons=[]
    if outside:reasons.append('ellipse_extends_outside_closed_FOV')
    if zero:reasons.append('ellipse_touches_original_C1_zero_unknown_pixels')
    label='unknown' if reasons else ('mito-compatible' if q<=1+tolerance else 'non-mito-compatible')
    return dict(label=label,reason=';'.join(reasons),
        intersects_known_mask=q<=1+tolerance,intersects_zero_unknown=zero,
        min_q=float(q) if np.isfinite(q) else None)


def roi_boundary_flags(polygon,x,y,a,b,tolerance=Q_TOLERANCE):
    if not valid_ellipse(x,y,a,b):
        return dict(touches_ROI_boundary=None,crosses_ROI_boundary=None)
    scaled=affinity.scale(polygon,xfact=1/a,yfact=1/b,origin=(x,y))
    distance=scaled.boundary.distance(shapely.Point(x,y))
    return dict(touches_ROI_boundary=bool(distance<=1+tolerance),
        crosses_ROI_boundary=bool(distance<1-tolerance))


def pixel_square_union(mask):
    """For display only; classification uses the analytic rectangle distance."""
    y,x=np.nonzero(mask)
    if not len(x):return shapely.GeometryCollection()
    return shapely.union_all(shapely.box(x-.5,y-.5,x+.5,y+.5))


def expanded_for_display(geometry,a,b,quad_segs=64):
    if not valid_ellipse(0,0,a,b):raise ValueError('Positive buffer axes required')
    scaled=affinity.scale(geometry,xfact=1/a,yfact=1/b,origin=(0,0))
    expanded=scaled.buffer(1,quad_segs=quad_segs)
    return affinity.scale(expanded,xfact=a,yfact=b,origin=(0,0))
