"""Display-only C1 limits; no mutation or scientific-stage dependency.

A caller supplies its saved ROI and explicit representative-frame reader.
The resolved window is computed once per cell and reused by every C1 view.
"""
from __future__ import annotations
import math
import operator


def _fixed_limits(style):
    values = style.get('C1_display_ADU', [407, 651])
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError('C1_display_ADU must contain two finite increasing values')
    try:
        low, high = map(float, values)
    except (TypeError, ValueError) as error:
        raise ValueError('C1_display_ADU must contain two finite increasing values') from error
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError('C1_display_ADU must contain two finite increasing values')
    return [low, high]


def resolve_c1_display(style, frame_ids=(), frame_reader=None, roi_mask=None):
    """Return JSON-safe display metadata, never transformed image pixels.

    ``fixed`` performs no image reads. ``roi_percentile`` reads exactly the
    supplied unique original frames once, pools finite nonzero ROI pixels and
    applies explicitly linear percentiles. Empty signal uses the configured
    fixed display fallback, recorded in ``fallback_reason``. Empty ROI or
    missing/conflicting frames are errors rather than fabricated samples.
    """
    style = dict(style or {})
    mode = style.get('C1_display_mode', 'fixed')
    fixed = _fixed_limits(style)
    result = dict(mode=mode, limits_ADU=fixed, percentiles=None, sample_frames=[],
                  nonzero_ROI_pixels=0, fallback_reason=None, display_only=True)
    if mode == 'fixed':
        return result
    if mode != 'roi_percentile':
        raise ValueError('Unsupported C1_display_mode: ' + str(mode))
    values = style.get('C1_display_percentiles', [1.0, 99.9])
    try:
        if len(values) != 2:
            raise ValueError
        percentiles = list(map(float, values))
    except (TypeError, ValueError) as error:
        raise ValueError('C1_display_percentiles requires two values in increasing order') from error
    if not all(map(math.isfinite, percentiles)) or not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError('C1_display_percentiles must satisfy 0 <= low < high <= 100')
    ids = []
    for value in frame_ids:
        try:
            if isinstance(value, bool):
                raise TypeError
            frame = operator.index(value)
        except TypeError as error:
            raise ValueError('C1 display frames must be nonnegative integer original frame IDs') from error
        if frame < 0:
            raise ValueError('C1 display frames must be nonnegative integer original frame IDs')
        ids.append(int(frame))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('C1 display frames must be nonempty and unique')
    if not callable(frame_reader):
        raise ValueError('Adaptive C1 display requires an original-frame reader')
    import numpy as np
    roi = np.asarray(roi_mask)
    if roi.ndim != 2 or roi.size == 0:
        raise ValueError('Adaptive C1 display requires a nonempty two-dimensional ROI')
    if roi.dtype.kind != 'b':
        if roi.dtype.kind not in 'uif' or not np.isfinite(roi).all() or not np.isin(roi, [0, 1]).all():
            raise ValueError('C1 display ROI must be a finite binary mask')
    roi = roi.astype(bool, copy=False)
    if not roi.any():
        raise ValueError('Adaptive C1 display ROI contains no pixels')
    samples = []
    nonfinite = 0
    for frame in ids:
        image = np.asarray(frame_reader(frame))
        if image.shape != roi.shape or image.dtype.kind not in 'uif':
            raise ValueError('C1 display original frame shape or numeric type conflicts with ROI')
        pixels = image[roi]
        finite = np.isfinite(pixels)
        nonfinite += int((~finite).sum())
        pixels = pixels[finite & (pixels != 0)]
        if len(pixels):
            samples.append(pixels)
    result.update(percentiles=percentiles, sample_frames=ids,
                  nonzero_ROI_pixels=sum(len(v) for v in samples),
                  excluded_nonfinite_ROI_pixels=nonfinite, zero_pixels_excluded=True,
                  percentile_method='linear', window_scope='one_fixed_window_per_cell')
    if not samples:
        result['fallback_reason'] = 'no_finite_nonzero_ROI_pixels; configured_fixed_limits'
        return result
    pooled = np.concatenate(samples)
    low, high = map(float, np.percentile(pooled, percentiles, method='linear'))
    if low >= high:
        center = min(65535.0, max(0.0, low))
        low, high = max(0.0, center - 1.0), min(65535.0, center + 1.0)
        result['fallback_reason'] = 'degenerate_percentile_interval_expanded_within_uint16'
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError('Adaptive C1 display could not resolve finite increasing limits')
    result['limits_ADU'] = [low, high]
    return result
