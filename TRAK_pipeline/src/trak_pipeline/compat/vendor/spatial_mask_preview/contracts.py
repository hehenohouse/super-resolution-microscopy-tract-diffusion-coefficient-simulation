"""Frame-range and count contracts shared by preview and full classification."""

LABELS = ('mito-compatible', 'non-mito-compatible', 'unknown')


def resolve_frames(config):
    n = config['expected_shape'][0]
    explicit = 'frames' in config
    interval = 'frame_start' in config or 'frame_stop_exclusive' in config
    if explicit == interval:
        raise ValueError('Specify either frames or frame_start/frame_stop_exclusive')
    if explicit:
        frames = list(config['frames'])
    else:
        start, stop = config['frame_start'], config['frame_stop_exclusive']
        if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop <= n:
            raise ValueError('Invalid original frame interval')
        frames = list(range(start, stop))
    if not frames or any(type(f) is not int or not 0 <= f < n for f in frames):
        raise ValueError('Original frames must be integers within the source movie')
    if frames != sorted(set(frames)):
        raise ValueError('Frames must be increasing and unique')
    return frames


def checkpoint_ranges(frames, size):
    if size <= 0 or frames != list(range(frames[0], frames[-1] + 1)):
        raise ValueError('Checkpoints require a positive size and a continuous frame range')
    return [frames[i:i+size] for i in range(0, len(frames), size)]


def conserved_counts(candidates, selected, excluded, counts):
    if set(counts) != set(LABELS) or any(v < 0 for v in counts.values()):
        raise ValueError('Exactly three nonnegative spatial counts are required')
    if selected + excluded != candidates or sum(counts.values()) != selected:
        raise ValueError('Spatial/QC counts do not conserve candidates')
    return True
