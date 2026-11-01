"""Actual HOST model/source/precision guard for E3 lifecycle integration.

This module does not load a model, create an environment, or certify an
episode. Snapshots must be taken in the actual authorized math-SDPA scope.
"""
from pathlib import Path
from experiments.spectral_reachability import agent_e3_gd_host_cuda_20261029_v1 as host

need=host.need


def validate_sources(bindings, required):
    need(type(bindings) is dict and bool(bindings) and type(required) is list and bool(required), 'source closure')
    for path in required:
        need(type(path) is str and Path(path).is_absolute() and str(Path(path).resolve())==path
             and path in bindings, 'required canonical source presealed')
    for path,sha in bindings.items():
        need(type(path) is str and Path(path).is_absolute() and str(Path(path).resolve())==path
             and type(sha) is str and len(sha)==64 and all(c in '0123456789abcdef' for c in sha), 'source path/SHA')
    host.source.rehash(bindings)


def snapshot(model, torch, device, bindings, required):
    need(getattr(device,'type',None)=='cuda', 'real HOST CUDA only')
    validate_sources(bindings,required)
    torch.cuda.synchronize(device)
    host.primitives.native.loader.enc.frozen(model)
    tensor_sha=host.primitives.native.loader.enc.tensor_hash(model)
    classes=host.class_inventory(model,bindings)
    precision=host.primitives.precision(torch)
    allocated=torch.cuda.max_memory_allocated(device);reserved=torch.cuda.max_memory_reserved(device)
    need(type(allocated) is int and type(reserved) is int and 0<=allocated<=3*1024**3
         and 0<=reserved<=4*1024**3, 'actual GPU peak caps')
    return dict(tensor_sha256=tensor_sha,classes=classes,precision=precision,
                allocated=allocated,reserved=reserved)


def validate_pair(before, after, expected_model_sha):
    keys={'tensor_sha256','classes','precision','allocated','reserved'}
    need(type(before) is dict and type(after) is dict and set(before)==keys and set(after)==keys,
         'exact actual snapshot schema')
    need(expected_model_sha==host.numerical.resolver.MODEL_SHA
         and before['tensor_sha256']==after['tensor_sha256']==expected_model_sha, 'original unchanged model')
    need(type(before['classes']) is list and bool(before['classes']) and before['classes']==after['classes'],
         'unchanged loaded classes')
    need(before['precision']==after['precision']==host.primitives.precision_expected(), 'unchanged original precision')
    for value in (before,after):
        need(type(value['allocated']) is int and 0<=value['allocated']<=3*1024**3
             and type(value['reserved']) is int and 0<=value['reserved']<=4*1024**3, 'actual CUDA peaks')
    return dict(model_unchanged=True,precision_unchanged=True,scientific_admission=False)
