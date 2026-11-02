"""Content-only publication of a verified trial into one fresh MyBook mask folder."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mito_tardis'))
from render_point_stack import dump,fingerprint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True);args=parser.parse_args()
    src=args.source.resolve();dst=args.destination
    config=json.loads((src/'resolved_config.json').read_text())
    assert str(dst)==config['external_destination'] and dst.parent==Path(config['C1_file']).parent/'mitochondria_masks'
    assert dst.name.startswith('Nellie_broad_support50_ge10_postmin32_holes_lt64_') and not dst.exists()
    assert json.loads((src/'COMPLETED.json').read_text())['delivery_checks_passed']
    assert json.loads((src/'COMPLETE.json').read_text())['all_passed']
    records=json.loads((src/'output_file_manifest.json').read_text())
    for r in records:
        path=Path(r['path']);assert path.is_relative_to(src) and fingerprint(path)['sha256']==r['sha256']
    dst.mkdir()
    for path in sorted(src.rglob('*')):
        target=dst/path.relative_to(src)
        if path.is_dir():target.mkdir(exist_ok=True)
        else:
            assert path.is_file() and not path.is_symlink() and not target.is_symlink()
            shutil.copyfile(path,target)
    for r in records:
        current=fingerprint(dst/Path(r['path']).relative_to(src))
        assert current['sha256']==r['sha256'] and current['bytes']==r['bytes']
    dump(dst/'publication.json',dict(local_source=str(src),destination=str(dst),copied_files_SHA256_equal=True,
        copied_verified_files=len(records),published_utc=datetime.now(timezone.utc).isoformat(),content_only_copy=True,
        original_TIFF_broad_mask_localization_files_unchanged=True))
    for name in ['COMPLETE.json','COMPLETED.json']:
        status=json.loads((dst/name).read_text());status.update(output=str(dst),local_stage=str(src),external_copy_verified=True);dump(dst/name,status)
    records=[fingerprint(path) for path in sorted(dst.rglob('*')) if path.is_file() and path.name not in
             ['COMPLETE.json','COMPLETED.json','output_fingerprints.json','output_file_manifest.json']]
    dump(dst/'output_file_manifest.json',records)
    dump(dst/'output_fingerprints.json',{str(Path(r['path']).relative_to(dst)):r['sha256'] for r in records})
    print(json.dumps(dict(output=str(dst),copied_files_SHA256_equal=True,verified_files=len(records)),indent=2))


if __name__=='__main__':main()
