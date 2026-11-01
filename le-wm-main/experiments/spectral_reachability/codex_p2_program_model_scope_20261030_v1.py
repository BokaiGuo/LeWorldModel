"""Instrument the real frozen model entrypoints; no CUDA startup."""
from contextlib import contextmanager


@contextmanager
def model_scope(model, meter):
    """Bind nested entrypoints, restoring exactly even on exceptions."""
    targets=[(model.action_encoder,"forward","action_encoder"),
             (model,"predict","predict"),
             (model.predictor,"forward","predictor"),
             (model.pred_proj,"forward","pred_proj")]
    restored=[]
    try:
        for owner,name,axis in targets:
            original=getattr(owner,name)
            if not callable(original):
                raise ValueError("model entrypoint is not callable: "+axis)
            own=name in vars(owner)
            old=vars(owner).get(name)
            restored.append((owner,name,own,old))
            def wrapped(*args,_original=original,_axis=axis,**kwargs):
                return meter.call(_axis,lambda:_original(*args,**kwargs))
            setattr(owner,name,wrapped)
        yield
    finally:
        for owner,name,own,old in reversed(restored):
            if own:setattr(owner,name,old)
            else:delattr(owner,name)
