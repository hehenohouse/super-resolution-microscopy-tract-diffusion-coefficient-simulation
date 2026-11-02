"""Call native TrackMate LoG detection from CPython; no trajectory linking.

Default input is a three-frame, non-square synthetic movie with a blank frame.
An explicit --input reads a bounded contiguous TIFF page range as time, not Z.
This bridge does not add a localization-uncertainty model to the LoG detector.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

import imagej
import numpy as np
import scyjava
import tifffile


def synthetic_movie():
    yy, xx = np.mgrid[:64, :80]
    truth = {
        0: [(15.3, 20.7), (50.6, 44.2)],
        1: [],
        2: [(15.8, 21.1), (51.0, 43.8)],
    }
    movie = np.full((3, 64, 80), 100.0)
    for frame, centers in truth.items():
        for x, y in centers:
            movie[frame] += 2000 * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 1.4**2))
    return np.rint(movie).astype(np.uint16), truth


def make_imageplus(ij, movie, pixel_size_um, frame_interval_s):
    """Set C=1, Z=1, T=n explicitly and verify unchanged pixels and orientation."""
    ImagePlus = scyjava.jimport('ij.ImagePlus')
    ImageStack = scyjava.jimport('ij.ImageStack')
    nframes, height, width = movie.shape
    stack = ImageStack(width, height)
    for plane in movie:
        plane_imp = ij.py.to_imageplus(np.ascontiguousarray(plane))
        processor = plane_imp.getProcessor().duplicate()
        copied = np.asarray(processor.getPixels()).view(np.uint16).reshape(height, width)
        if not np.array_equal(copied, plane):
            raise AssertionError('Pixel or X/Y conversion mismatch')
        stack.addSlice(processor)
    imp = ImagePlus('TrackMate Python detection input', stack)
    imp.setDimensions(1, 1, nframes)
    imp.setOpenAsHyperStack(nframes > 1)
    calibration = imp.getCalibration()
    calibration.pixelWidth = pixel_size_um
    calibration.pixelHeight = pixel_size_um
    calibration.pixelDepth = 1.0
    calibration.frameInterval = frame_interval_s
    calibration.setUnit('um')
    calibration.setTimeUnit('s')
    assert (imp.getWidth(), imp.getHeight(), imp.getNSlices(), imp.getNFrames()) == (
        width, height, 1, nframes)
    return imp


def detect(imp, radius_um, threshold, median_filter, original_start, frame_interval_s):
    Model = scyjava.jimport('fiji.plugin.trackmate.Model')
    Settings = scyjava.jimport('fiji.plugin.trackmate.Settings')
    TrackMate = scyjava.jimport('fiji.plugin.trackmate.TrackMate')
    LogDetectorFactory = scyjava.jimport('fiji.plugin.trackmate.detection.LogDetectorFactory')
    Double = scyjava.jimport('java.lang.Double')
    Integer = scyjava.jimport('java.lang.Integer')
    Boolean = scyjava.jimport('java.lang.Boolean')
    model = Model()
    model.setPhysicalUnits('um', 's')
    settings = Settings(imp)
    settings.detectorFactory = LogDetectorFactory()
    settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
    for key, value in {
        'TARGET_CHANNEL': Integer(1),
        'RADIUS': Double(radius_um),
        'THRESHOLD': Double(threshold),
        'DO_MEDIAN_FILTERING': Boolean(median_filter),
        'DO_SUBPIXEL_LOCALIZATION': Boolean(True),
    }.items():
        settings.detectorSettings.put(key, value)
    settings.tstart = 0
    settings.tend = imp.getNFrames() - 1
    if not settings.detectorFactory.checkSettings(settings.detectorSettings):
        raise RuntimeError(str(settings.detectorFactory.getErrorMessage()))
    tm = TrackMate(model, settings)
    tm.setNumThreads(2)
    # process() includes tracking. Call only detection to preserve all detections.
    if not tm.execDetection():
        raise RuntimeError(str(tm.getErrorMessage()))
    pixel_size_um = float(imp.getCalibration().pixelWidth)
    rows = []
    counts = [0] * imp.getNFrames()
    for spot in model.getSpots().iterable(False):
        local_frame = int(spot.getFeature('FRAME'))
        frame = original_start + local_frame
        x_um = float(spot.getFeature('POSITION_X'))
        y_um = float(spot.getFeature('POSITION_Y'))
        assert abs(float(spot.getFeature('POSITION_T')) - local_frame * frame_interval_s) < 1e-9
        rows.append({
            'spot_id': int(spot.ID()), 'frame': frame,
            'time_s': frame * frame_interval_s,
            'x_px': x_um / pixel_size_um, 'y_px': y_um / pixel_size_um,
            'x_um': x_um, 'y_um': y_um,
            'quality_log_response': float(spot.getFeature('QUALITY')),
            'radius_setting_um': float(spot.getFeature('RADIUS')),
        })
        counts[local_frame] += 1
    rows.sort(key=lambda row: (row['frame'], row['x_px'], row['y_px']))
    return rows, counts, str(TrackMate.PLUGIN_NAME_VERSION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fiji', default='/Applications/Fiji.app')
    parser.add_argument('--java-home', default='/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home')
    parser.add_argument('--input', type=Path, help='Omit to run the synthetic smoke test')
    parser.add_argument('--start', type=int, default=0, help='Original zero-based TIFF page')
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--pixel-size-um', type=float, default=0.117)
    parser.add_argument('--frame-interval-s', type=float, default=0.02)
    parser.add_argument('--radius-um', type=float, default=0.25,
                        help='Explicit working radius; historical MAT radius units are unconfirmed')
    parser.add_argument('--threshold', type=float, default=6.5)
    parser.add_argument('--no-median', action='store_true')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent / 'runs')
    args = parser.parse_args()
    if args.start < 0 or args.count < 1 or args.pixel_size_um <= 0 or args.frame_interval_s <= 0:
        parser.error('Frame range and physical scales must be valid')
    start_time = time.monotonic()
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_root / ('synthetic_' if args.input is None else 'tiff_')
    output = output.with_name(output.name + timestamp)
    output.mkdir(parents=True, exist_ok=False)
    source = {}
    if args.input is None:
        if args.start != 0:
            parser.error('--start must be zero for the synthetic test')
        movie, truth = synthetic_movie()
        source['kind'] = 'synthetic, noiseless, rounded Gaussian peaks; not a performance benchmark'
    else:
        truth = None
        before = args.input.stat()
        with tifffile.TiffFile(args.input) as tif:
            if args.start + args.count > len(tif.pages):
                raise ValueError('Requested range exceeds TIFF pages')
            movie = np.stack([tif.pages[i].asarray() for i in range(args.start, args.start + args.count)])
            source.update(path=str(args.input.resolve()), pages=len(tif.pages))
        if movie.ndim != 3 or movie.dtype != np.uint16:
            raise ValueError('This pilot bridge expects 2D uint16 TIFF pages')
        after = args.input.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        source.update(size_bytes=before.st_size, mtime_ns=before.st_mtime_ns)
    source.update(shape_TYX=list(movie.shape), dtype=str(movie.dtype),
                  selected_pixel_sha256=hashlib.sha256(movie.tobytes()).hexdigest())
    np.save(output / 'input_frames.npy', movie)
    os.environ['JAVA_HOME'] = args.java_home
    scyjava.config.set_java_constraints(fetch='never')
    scyjava.config.add_option('-Xmx2g')
    scyjava.config.add_option('-Dscijava.log.level=warn')
    scyjava.config.add_option('-Dimagej.updater.disableAutocheck=true')
    ij = imagej.init(args.fiji, mode='headless')
    try:
        imp = make_imageplus(ij, movie, args.pixel_size_um, args.frame_interval_s)
        rows, counts, tm_version = detect(imp, args.radius_um, args.threshold,
                                          not args.no_median, args.start, args.frame_interval_s)
        checks = {'pixel_roundtrip_exact': True, 'dimensions_CZT': [1, 1, len(movie)]}
        if truth is not None:
            errors = []
            for frame, centers in truth.items():
                fitted = [row for row in rows if row['frame'] == frame]
                assert len(fitted) == len(centers), (frame, len(fitted), len(centers))
                for x, y in centers:
                    error = min(np.hypot(row['x_px'] - x, row['y_px'] - y) for row in fitted)
                    assert error < 0.35, (frame, x, y, error)
                    errors.append(float(error))
            assert counts == [2, 0, 2], counts
            checks.update(expected_counts=[2, 0, 2], blank_frame_preserved=True,
                          maximum_synthetic_position_error_px=max(errors),
                          x_y_direction_and_units_verified=True)
        with (output / 'spots.csv').open('w', newline='') as handle:
            fields = ['spot_id', 'frame', 'time_s', 'x_px', 'y_px', 'x_um', 'y_um',
                      'quality_log_response', 'radius_setting_um']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        System = scyjava.jimport('java.lang.System')
        report = {
            'command': sys.argv, 'parameters': {key: str(value) if isinstance(value, Path) else value
                                               for key, value in vars(args).items()},
            'source': source, 'versions': {'python': platform.python_version(),
                'pyimagej': importlib.metadata.version('pyimagej'), 'trackmate': tm_version,
                'imagej': str(ij.getVersion()), 'java': str(System.getProperty('java.version'))},
            'per_frame_counts': [{'frame': args.start + i, 'count': count} for i, count in enumerate(counts)],
            'total_spots': len(rows), 'validation': checks, 'elapsed_s': time.monotonic() - start_time,
            'steps': ['LoG candidate detection', 'quadratic subpixel peak localization'],
            'tracking_executed': False, 'roi_applied': False,
            'uncertainty': 'No lpx/lpy or CRLB supplied by this standard LoG path',
            'historical_reproduction': 'Not verified; historical version, radius unit, and export/filter rules unknown',
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'trackmate_jar_sha256': hashlib.sha256((Path(args.fiji) / 'jars' / f'TrackMate-{tm_version}.jar').read_bytes()).hexdigest(),
        }
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'output': str(output), 'counts': counts, 'trackmate': tm_version, 'validation': checks}, indent=2))
    finally:
        ij.dispose()


if __name__ == '__main__':
    main()
