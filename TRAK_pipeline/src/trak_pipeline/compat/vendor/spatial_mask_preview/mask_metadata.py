"""Explicit supported-mask provenance contracts; no segmentation is performed."""


def check_recipe(complete, state, validation, parameters, config):
    recipe=config.get('mask_recipe','min16')
    assert validation['all_passed']
    assert parameters['shape']==config['expected_shape'] and parameters['axes']=='TYX'
    assert parameters['pixel_size_um']*1000==config['pixel_size_nm']
    assert parameters['time_interval_s']==config['frame_interval_s']
    assert parameters['connectivity']==8
    if recipe=='min16':
        assert complete['all_checks_passed'] and complete['completed_frames']==2500
        assert state['status']=='complete' and state['completed_frames']==2500
        assert parameters['accepted_threshold_pixel2']==16
        assert parameters['keep_rule']=='full frame component area >= 16; no boundary exemption'
    elif recipe=='W5_min32':
        assert complete['all_passed'] and complete['frames']==2500
        assert state['status']=='complete' and state['frames_complete']==2500
        assert complete['selected_recipe']=='Nellie broad + W5 majority + min32'
        assert parameters['sample']=='TDKd3'
        assert parameters['temporal_window']==5 and parameters['required_votes']==3
        assert parameters['min_keep_area_pixel2']==32 and parameters['min_keep_area_um2']==.438048
        assert parameters['boundary_exemptions'] is False and parameters['motion_compensation'] is False
        assert parameters['endpoint_span_ms']==80 and parameters['lookahead_ms']==40
        assert parameters['endpoint_rule']=='Frames 0,1,2498,2499 use unsmoothed broad followed by min32; no duplicated frames'
        assert parameters['temporal_input']=='original uncropped Nellie broad mask reconstructed as full_component_labels>0, BEFORE component cleanup'
        assert parameters['ROI_rule']=='original pixel centers inside/on polygon; cleanup before ROI intersection'
        assert parameters['no_Nellie_rerun'] is True
        assert complete['summary']['endpoint_fallback_frames']==[0,1,2498,2499]
    else:raise ValueError('Unsupported mask recipe: '+recipe)
    return recipe
