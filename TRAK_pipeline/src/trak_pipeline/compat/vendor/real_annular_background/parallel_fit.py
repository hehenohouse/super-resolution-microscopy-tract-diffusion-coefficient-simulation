"""Process-parallel grouping of the identical independent per-candidate fitter."""
import run_real as core


def fit_batch(payload):
    items,raw,movie,background,masks,config=payload
    return [core.fit_one(item,raw,movie,background.iloc[j],masks[j],config)
            for j,item in enumerate(items)]
