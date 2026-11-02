from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trak_pipeline.stages.localization import native_main
from trak_pipeline.runtime import process_main
if __name__=='__main__':raise SystemExit(process_main(native_main))
