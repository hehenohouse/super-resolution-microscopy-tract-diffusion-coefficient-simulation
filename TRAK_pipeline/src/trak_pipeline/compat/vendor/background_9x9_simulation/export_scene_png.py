"""Export native 300x300 display PNGs from a stored scientific scene."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,PngImagePlugin
import tifffile
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(scene):
    scene=scene.resolve()
    inputs=[scene/'scene_components.npz',scene/'shared_added_spot_truth.csv',
            scene/'config.yaml',scene/'injected_scene_float32_ADU.tiff',scene/'output_fingerprints.json']
    before={str(p):sha(p) for p in inputs}
    config=yaml.safe_load((scene/'config.yaml').read_text())
    with np.load(scene/'scene_components.npz') as data:raw=data['observed_ADU'].copy()
    assert raw.shape==(300,300) and np.isfinite(raw).all()
    np.testing.assert_array_equal(tifffile.imread(scene/'injected_scene_float32_ADU.tiff'),raw.astype(np.float32))
    truth=pd.read_csv(scene/'shared_added_spot_truth.csv',float_precision='round_trip')
    assert len(truth)==50 and truth.spot_id.is_unique
    assert truth[['x_px','y_px']].ge(0).all().all() and truth[['x_px','y_px']].lt(300).all().all()
    low,high=config['raw_display_range_ADU']
    gray=np.rint(np.clip((raw-low)/(high-low),0,1)*255).astype(np.uint8)
    plain=Image.fromarray(gray)
    ss=4
    layer=Image.new('RGBA',(300*ss,300*ss),(0,0,0,0));d=ImageDraw.Draw(layer)
    for row in truth.itertuples():
        x,y=(np.array([row.x_px,row.y_px])+.5)*ss-.5
        radius=row.hwhm_px*ss
        d.ellipse((x-radius,y-radius,x+radius,y+radius),outline='#00E5FF',width=3)
    layer=layer.resize((300,300),Image.Resampling.LANCZOS)
    marked=Image.alpha_composite(plain.convert('RGBA'),layer).convert('RGB')
    names=['scene_300x300.png','scene_300x300_truth_circles.png']
    for name,image in zip(names,[plain,marked]):
        target=scene/name;assert not target.exists(),f'Existing export protected: {target}'
        meta=PngImagePlugin.PngInfo()
        for key,value in {'native_shape_yx':'300,300','pixel_size_nm':117,'display_low_ADU':low,'display_high_ADU':high,
                          'source':'stored empirical-background + 50 simulated-signal scene','truth_spot_count':50,
                          'annotation':'none' if image is plain else 'cyan circles: true Gaussian HWHM, not localization error'}.items():
            meta.add_text(key,str(value))
        image.save(target,pnginfo=meta)
        with Image.open(target) as check:assert check.size==(300,300)
    with Image.open(scene/names[0]) as check:np.testing.assert_array_equal(np.asarray(check),gray)
    assert before=={str(p):sha(p) for p in inputs}
    shutil.copyfile(__file__,scene/'export_scene_png.py')
    (scene/'scene_png_export_command.txt').write_text(shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--scene',str(scene)])+'\n')
    result={'all_checks_passed':True,'dimensions_xy':[300,300],'display_range_ADU':[low,high],
            'pixel_size_nm':117,'native_gray_pixels_exact':True,'TIFF_correspondence_exact':True,
            'sources_unchanged':True,'source_sha256':before,
            'exports':{name:{'path':str(scene/name),'sha256':sha(scene/name)} for name in names}}
    (scene/'scene_png_export_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    run(parser.parse_args().scene)
