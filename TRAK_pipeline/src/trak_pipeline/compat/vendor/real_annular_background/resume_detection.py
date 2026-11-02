"""Restart native detector every three new batches to release JVM resources."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True);ap.add_argument('--python',type=Path,required=True)
    args=ap.parse_args();job=json.loads(args.job.read_text());out=Path(job['output'])
    command=[str(args.python.absolute()),'-u',str(Path(__file__).with_name('detect_batches.py').resolve()),'--job',str(args.job.resolve()),'--max-new-chunks','3']
    (out.parent/'native_detection_resume_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve())]+sys.argv[1:])+'\n')
    started=time.perf_counter()
    while True:
        done=sum((out/f'frames{a:04d}_{b-1:04d}.json').exists() for a,b in job['chunks'])
        if done==len(job['chunks']):break
        print(f'Native batch completion: {done}/{len(job["chunks"])}; fresh JVM for next <=3 chunks',flush=True)
        result=subprocess.run(command)
        if result.returncode:raise RuntimeError(f'Native child exited {result.returncode}; completed chunks retained')
        updated=sum((out/f'frames{a:04d}_{b-1:04d}.json').exists() for a,b in job['chunks'])
        if updated<=done:raise RuntimeError('No detector progress')
    (out/'COMPLETED.json').write_text(json.dumps(dict(all_chunks_completed=True,
         frames=sum(b-a for a,b in job['chunks']),JVM_restart_every_new_chunks=3,elapsed_s=time.perf_counter()-started),indent=2)+'\n')
    print('Native full-film detection complete.',flush=True)


if __name__=='__main__':main()
