"""Extract the rightmost viewing panel without resampling its pixels."""
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    here=Path(__file__).resolve().parent
    out=Path((here/'latest_point_stack_attempt.txt').read_text().strip())
    layout=json.loads((out/'panel_layout.json').read_text())
    parameters=json.loads((out/'parameters.json').read_text())
    rows=json.loads((out/'preview_manifest.json').read_text())
    assert len(layout['panel_origins'])==6
    x,y=layout['panel_origins'][5];w,h=layout['panel_width'],layout['panel_height']
    font=ImageFont.truetype(parameters['font_path'],17)
    for row in rows:
        f=row['frame']
        with Image.open(out/'previews'/f'frame_{f:04d}_six_panels.png') as image:
            panel=image.crop((x,y,x+w,y+h))
        view=Image.new('RGB',(w,h+100),'black');view.paste(panel,(0,65))
        draw=ImageDraw.Draw(view)
        draw.text((6,7),f'Frame {f} | {row["time_s"]:.2f} s | On {row["On"]} / Not-on {row["Not_on"]}',font=font,fill='white')
        draw.text((6,32),'C1 raw + red mask + green/purple localization circles',font=font,fill='white')
        draw.text((6,h+73),'C1/C2 identity registration unverified',font=font,fill='#CCCCCC')
        path=out/'previews'/f'frame_{f:04d}_merged_panel.png'
        view.save(path,compress_level=6)
        with Image.open(path) as saved:
            np.testing.assert_array_equal(np.asarray(saved)[65:65+h],np.asarray(panel))
    shutil.copyfile(__file__,out/'source_snapshot/merged_panel_previews.py')
    (out/'merged_panel_preview_validation.json').write_text(json.dumps(dict(
        source='Exact sixth-panel pixel crop of the corresponding six-panel preview',
        no_resampling=True,all_pixels_equal=True,frames=[r['frame'] for r in rows]),indent=2)+'\n')
    print(out/'previews')


if __name__=='__main__':main()
