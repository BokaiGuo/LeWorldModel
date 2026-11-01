"""Exclusive four-slot bounded transport. No model or environment authority.

Limits are engineering ceilings, not a runtime-fit or scientific certificate.
"""
import io
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_e3_unified_transport_20261029_v1 as base
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as publication

M = 1024**2
CAP = 2*1024*M
RESERVE = 256*M
FLOOR = 512*M
WALL = 7200
LIMITS = dict(numerical_arrays=384*M, numerical_events=1200*M, numerical_metadata=32*M,
              physical_png=128*M, physical_events=128*M, physical_metadata=32*M,
              encoding_arrays=128*M, encoding_events=64*M)
ENCODER_RAW_BYTES_PER_FRAME = sum(int(np.prod(shape))*np.dtype(dtype).itemsize
                                  for _,shape,dtype in base.ENC_SHAPES)
ENCODER_16_RAW_BYTES = 16*ENCODER_RAW_BYTES_PER_FRAME
need = base.need


class Store(publication.Store):
    """Original event/array ABI with new own capacities and write-before checks."""
    def __init__(self, path, *, clock=time.monotonic):
        p = Path(path)
        need(p.is_absolute() and p.resolve() == p and not p.exists() and not p.is_symlink() and p.parent.is_dir(), 'exclusive canonical output')
        need(shutil.disk_usage(p.parent).free >= CAP+FLOOR, 'full cap and disk floor before creation')
        p.mkdir(); base.syncdir(p.parent)
        self.output = p; self.clock = clock; self.started = clock()
        self.used = {k:0 for k in LIMITS}; self.number = 0; self.index = 0
        self.previous = None; self.pending = {}; self.frames = 0
        self.sealed = False; self.failures = 0; self.final_bytes = 0

    def check(self, category, size):
        need(category in LIMITS and type(size) is int and size >= 0, 'category and size')
        need(0 <= self.clock()-self.started <= WALL, 'whole wall cap')
        need(self.used[category]+size <= LIMITS[category] and
             sum(self.used.values())+size <= CAP-RESERVE and
             shutil.disk_usage(self.output).free >= FLOOR+size, 'category, reserve and disk floor')

    def write(self, name, data, category, append=False):
        need(type(name) is str and Path(name).name == name and name not in ('','.','..') and type(data) is bytes,
             'flat literal artifact')
        p = self.output/name
        need(not p.is_symlink() and (name == 'events.jsonl' if append else not p.exists()), 'exclusive owned output')
        self.check(category, len(data))
        if append:
            need(p.exists() == (self.index > 0), 'journal owned before append')
        with p.open('ab' if append else 'xb') as f:
            need(f.write(data) == len(data), 'full write'); f.flush(); os.fsync(f.fileno())
        self.used[category] += len(data); base.syncdir(self.output)


def inspect_journal(path, expected):
    """Independent chain/intent audit; partial is retained and never promoted."""
    p = Path(path)
    blob = base.read_bytes(p, LIMITS['numerical_events']+LIMITS['physical_events']+LIMITS['encoding_events'])
    need(base.fsha(blob) == expected and blob.endswith(b'\n') and b'\n\n' not in blob, 'journal hash/single LF')
    previous = None; pending = {}; counts = {}; events = []
    for index,line in enumerate(blob.splitlines()):
        need(len(line) <= 16384, 'bounded record')
        e = base.decode(line)
        need(type(e) is dict and set(e) == {'index','previous','origin','kind','resource','token','detail','sha256'} and
             type(e['index']) is int and e['index'] == index and e['previous'] == previous and
             e['sha256'] == base.digest({k:v for k,v in e.items() if k != 'sha256'}), 'journal chain')
        origin,kind,resource,token = (e[k] for k in ('origin','kind','resource','token'))
        need(origin in base.ORIGINS and kind in ('intent','return','failure','record') and
             type(resource) is str and bool(resource) and type(token) is int and token >= 0 and
             type(e['detail']) is dict, 'event ABI')
        key = (origin,token); c = counts.setdefault(origin+':'+resource, dict(attempted=0,returned=0,failed=0))
        if kind == 'intent':
            need(token == index and key not in pending, 'unique intent'); pending[key] = resource; c['attempted'] += 1
        elif kind in ('return','failure'):
            need(pending.pop(key,None) == resource, 'matched return'); c['returned' if kind == 'return' else 'failed'] += 1
        previous = e['sha256']; events.append(e)
    return dict(events=events, counts=counts, pending=pending,
                transport_complete=not pending and all(c['failed'] == 0 for c in counts.values()),
                physical_certified=False, scientific_pass=False)
