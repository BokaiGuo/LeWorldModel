"""Exclusive durable saved-only P4 analysis and exact whole-file verifier."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from experiments.spectral_reachability import agent_p4_saved_pair_budget_20261029_v1 as base

NAME = 'agent_p4_saved_pair_budget_persist_20261029_v1'
ROOT = base.ROOT
CAP = 8*1024*1024
FLOOR = 200*1024*1024
SEALED = {
    'experiments/spectral_reachability/agent_p4_saved_pair_budget_20261029_v1.py': '97a54a9fb6b03fbeb495606261073e88826b2ee014a8eeb3b1498eb21aaa6d0d',
    'tests/test_agent_p4_saved_pair_budget_20261029_v1.py': '29208129947741c954db5662b581eb8e21fde6c4deae2669c40d03ff9908268b',
    'innerdocs/agent_p4_saved_pair_budget_20261029_v1.md': 'bc2b6c2270253440378f50988129ff8fec91ffd65a6cc390a9e47ed876af93ef',
}
OWN = (
    'experiments/spectral_reachability/'+NAME+'.py',
    'tests/test_'+NAME+'.py',
    'innerdocs/'+NAME+'.md',
)

def digest(raw): return hashlib.sha256(raw).hexdigest()

def file_bytes(path, cap=CAP):
    p = Path(path)
    base.need(p.is_file() and not p.is_symlink() and p.stat().st_size <= cap, 'regular bounded file')
    return p.read_bytes()

def closure():
    result = {}
    for path in (*SEALED, *OWN):
        value = digest(file_bytes(ROOT/path))
        if path in SEALED: base.need(value == SEALED[path], 'sealed analysis source changed')
        result[path] = value
    return result

def build():
    sources = closure()
    analysis = base.analyze(base.load_inputs())
    base.need(closure() == sources, 'source closure changed during analysis')
    return dict(schema=NAME+'.result', source_closure=sources, analysis=analysis,
                verification_scope='pinned saved receipts; not a new full strict',
                new_neural_calls=0, new_physics_calls=0, P1_admission=False, P4_admission=False)

def target(value):
    p = Path(value).absolute(); directory = ROOT/'results'
    base.need(directory.resolve() == directory and not directory.is_symlink(), 'exact results directory')
    base.need(p.parent == directory and p.name.startswith(NAME+'_') and p.suffix == '.json', 'exclusive results filename')
    return p

def write_exclusive(path, raw):
    base.need(len(raw) <= CAP, 'output cap')
    base.need(shutil.disk_usage(path.parent).free >= FLOOR+len(raw), 'disk floor')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
    except BaseException:
        # Preserve partial output for inspection; never overwrite or delete it.
        raise
    dfd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try: os.fsync(dfd)
    finally: os.close(dfd)

def verify(path, sha):
    base.need(len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'actual SHA')
    raw = file_bytes(path); base.need(digest(raw) == sha, 'saved SHA')
    base.need(raw == base.canonical(build()).encode(), 'whole saved bytes and source closure')
    return dict(schema=NAME+'.verification', status='exact_saved_analysis_verified', result_sha256=sha,
                full_strict_rerun=False, new_neural_calls=0, new_physics_calls=0)

def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('write'); p.add_argument('--output',required=True)
    p=sub.add_parser('verify'); p.add_argument('--result',required=True); p.add_argument('--sha256',required=True)
    args=parser.parse_args()
    if args.cmd == 'write':
        path=target(args.output); raw=base.canonical(build()).encode(); write_exclusive(path,raw)
        output=dict(status='saved', path=str(path), sha256=digest(raw), bytes=len(raw))
    else: output=verify(target(args.result),args.sha256)
    print(json.dumps(output,sort_keys=True,allow_nan=False))

if __name__ == '__main__': main()
