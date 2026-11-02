"""Fixed mapping contract for the T1 mean50 pilot; no temporal interpolation."""
import numpy as np


def frame_to_block(frame):
    if isinstance(frame,bool) or not isinstance(frame,(int,np.integer)) or not 0<=frame<2500:
        raise ValueError('Original frame must be an integer from0 through2499')
    return int(frame)//50


def block_metadata(frame):
    b=frame_to_block(frame)
    return dict(mean50_block_index=b,mean50_original_frame_start=50*b,
        mean50_original_frame_end_inclusive=50*b+49,
        mean50_time_start_s=float(b),mean50_time_end_exclusive_s=float(b+1))


def validate_mapping(records,frame_numbers):
    assert len(records)==50
    np.testing.assert_array_equal(frame_numbers,np.arange(2500).reshape(50,50))
    for b,row in enumerate(records):
        for key,value in [('block_index',b),('original_frame_start',50*b),
                          ('original_frame_end_inclusive',50*b+49),('original_frame_count',50)]:
            assert int(row[key])==value,(b,key)
        for key,value in [('time_start_s',b),('time_end_exclusive_s',b+1),
                          ('mean_original_frame_timestamp_s',b+.49),('pixel_size_um',.117)]:
            assert abs(float(row[key])-value)<=1e-12,(b,key)
        for f in range(50*b,50*b+50):assert frame_to_block(f)==b
    return True
