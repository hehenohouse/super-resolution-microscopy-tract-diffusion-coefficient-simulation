"""Explicit imports for the one frozen numerical source set."""
from pathlib import Path
import sys
COMPAT=Path(__file__).resolve().parent/'compat'
VENDOR=COMPAT/'vendor'
def paths(*names):
    for p in [COMPAT]+[VENDOR/n for n in names]:
        if str(p)not in sys.path:sys.path.insert(0,str(p))
def science():
    paths('picasso_pilot','picasso_full_sensitivity','picasso_roi_sensitivity','TARDIS/path_irrelevant_python','mito_tardis')
