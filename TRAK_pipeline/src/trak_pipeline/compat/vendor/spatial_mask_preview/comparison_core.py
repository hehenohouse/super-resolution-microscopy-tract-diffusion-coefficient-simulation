"""Paired same-point classification bookkeeping; no temporal molecular transitions."""
import numpy as np
from contracts import LABELS

LOSS=('mito-compatible','non-mito-compatible')
GAIN=('non-mito-compatible','mito-compatible')


def transition_name(old,new):
    if old not in LABELS or new not in LABELS:raise ValueError('Unknown classification label')
    return old+' -> '+new


def matrix_from_pairs(pairs):
    matrix=np.zeros((3,3),dtype=np.int64)
    for old,new in pairs:matrix[LABELS.index(old),LABELS.index(new)]+=1
    return matrix


def choose_examples(rows,n=6):
    ordered=sorted(rows,key=lambda r:(int(r['frame']),int(r['candidate_index'])))
    if len(ordered)<=n:return ordered
    indices=np.unique(np.rint(np.linspace(0,len(ordered)-1,n)).astype(int))
    return [ordered[i] for i in indices]
