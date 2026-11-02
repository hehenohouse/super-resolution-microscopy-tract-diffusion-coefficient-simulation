"""Publish already verified local QC to one fresh external-drive result folder."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mito_tardis'))
from render_point_stack import fingerprint,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--destination',type=Path,required=True);args=ap.parse_args()
    src=args.source.resolve();dst=args.destination
    done=json.loads((src/'COMPLETED.json').read_text());assert done['delivery_checks_passed']
    config=json.loads((src/'config.json').read_text());parent=json.loads((Path(config['source'])/'resolved_config.json').read_text())
    expected_parent=Path(parent['C1_file']).parent/'analysis_results/support50_not_on_cluster_qc'
    assert dst.parent==expected_parent
    if dst.exists():
        # Resume only our own incomplete publication, never another analysis.
        assert not (dst/'publication.json').exists()
        partial=json.loads((dst/'COMPLETED.json').read_text())
        assert partial['output']==str(src) and partial['delivery_checks_passed']
        assert json.loads((dst/'config.json').read_text())==config
    records=json.loads((src/'output_fingerprints.json').read_text())
    for r in records:
        path=Path(r['path']);assert path.is_relative_to(src)
        assert fingerprint(path)['sha256']==r['sha256']
    dst.mkdir(parents=True,exist_ok=True)
    # Some external filesystems reject copystat/xattr operations with EINVAL.
    # Preserve verified bytes; source metadata is already in the manifests.
    for path in sorted(src.rglob('*')):
        target=dst/path.relative_to(src)
        if path.is_dir():target.mkdir(exist_ok=True)
        else:
            assert path.is_file() and not path.is_symlink() and not target.is_symlink()
            shutil.copyfile(path,target)
    checked=[]
    for r in records:
        path=dst/Path(r['path']).relative_to(src);actual=fingerprint(path)
        assert actual['sha256']==r['sha256'] and actual['bytes']==r['bytes'];checked.append(str(path.relative_to(dst)))
    dump(dst/'publication.json',dict(local_source=str(src),destination=str(dst),copied_files_SHA256_equal=True,
        verified_files=checked,published_utc=datetime.now(timezone.utc).isoformat(),original_raw_localization_mask_files_not_modified=True,
        copy_method='Content-only copy, no unsupported filesystem metadata operations'))
    done.update(output=str(dst),local_stage=str(src),external_copy_verified=True);dump(dst/'COMPLETED.json',done)
    dump(dst/'output_fingerprints.json',[fingerprint(path) for path in sorted(dst.rglob('*')) if path.is_file() and path.name not in ['output_fingerprints.json','COMPLETED.json']])
    print(json.dumps(dict(output=str(dst),files_verified=len(checked),external_copy_verified=True),indent=2))


if __name__=='__main__':main()
