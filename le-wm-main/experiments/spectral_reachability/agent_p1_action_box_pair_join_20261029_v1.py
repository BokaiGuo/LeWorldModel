"""Saved-only, fixed 40-slot/80-pair raw-action box diagnostic."""
import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = 'agent_p1_action_box_pair_join_20261029_v1'
INPUTS = {
    'selector': ('results/agent_p1_model_library_selector_20261028_v1_MAIN01.json', 'f807d2ffe2d7ffb4d012852302f199b9edb64afc5a23cc36ac264c6adea6c1bb'),
    'evaluation': ('results/agent_p1_model_library_selector_evaluation_20261028_v1_MAIN01.json', 'd8b3bedd38e9f9a56d0156f2452d5a20f363acbdad42ea5e3ee4d91f9626ff67'),
    'margin': ('results/agent_p1_pair_margin_decomposition_20261028_v1_MAIN01.json', 'f723c5eb73ecebd0615a2ded0aa4b741c6f132ca5f6063c673f12d5e98539143'),
}
SOURCES = {
    'experiments/spectral_reachability/agent_p1_teacher_cross_library_20261028_v1.py': 'ca80a146b602406a1ef960fd2fa064afabc9a2bfd02d5396983f328d7f443b4a',
    'experiments/spectral_reachability/agent_p1_teacher_cross_library_physical_20261028_v1.py': '5367287a375cf89e45af2675997aa682ec63464d24b2b01ab413c5863882d8a4',
}
WORDS = ('zero', 'uZ', '-uZ', 'uS', '-uS')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def file_bytes(path, expected):
    b = path.read_bytes()
    require(sha(b) == expected, f'SHA mismatch: {path}')
    return b

def raw(commands, n):
    require(isinstance(commands, list) and len(commands) == n and all(isinstance(x, list) and len(x) == 2 for x in commands), 'raw shape')
    vals = [v for row in commands for v in row]
    require(all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in vals), 'raw finite')
    try:
        packed = struct.pack('<%df' % (2*n), *vals)
    except (OverflowError, struct.error) as e:
        raise ValueError('raw float32') from e
    require(all(a == b for a, b in zip(vals, struct.unpack('<%df' % (2*n), packed))), 'raw not exactly float32')
    return packed

def distances(bits):
    require(len(bits) == 80, 'raw10 byte length')
    result = []
    for k, (x,) in enumerate(struct.iter_unpack('<f', bits)):
        require(math.isfinite(x), 'raw nonfinite')
        lo, hi = x + 1.0, 1.0 - x
        result.append(dict(step=k//2+1, axis=k%2, raw=x, raw_bits_le=bits[4*k:4*k+4].hex(), lower=lo, upper=hi, nearest=min(lo, hi), exact_boundary=('-1' if x == -1 else '+1' if x == 1 else None), outside=not -1 <= x <= 1))
    return result

def segment(rows):
    return dict(min_distance=min(r['nearest'] for r in rows), exact_boundary_count=sum(r['exact_boundary'] is not None for r in rows), outside_count=sum(r['outside'] for r in rows))

def join(selector, evaluation, margin):
    candidates, rows, witnesses, slots, pairs = selector['candidates'], evaluation['rows'], evaluation['exact_reuse_witnesses'], margin['slots'], margin['pairs']
    require(tuple(map(len, (candidates, rows, witnesses, slots, pairs))) == (40, 40, 40, 40, 80), 'fixed denominator')
    out = []
    for i, (c, r, w, s) in enumerate(zip(candidates, rows, witnesses, slots)):
        p = c['original_plan']
        require(p['index'] == r['index'] == w['index'] == s['index'] == i and s['anchor_index'] == i//5 and s['word'] == p['library_label'] == WORDS[i%5] and p['library_index'] == i%5, 'slot roster')
        for key in ('anchor_id', 'task_id', 'role', 'root_slot', 'condition', 'plan_digest'):
            require(p[key] == r[key] == s[key], 'identity ' + key)
        require(r['job_digest'] == w['job_digest'] == s['job_digest'] and w['plan_digest'] == p['plan_digest'], 'execution identity')
        prefix, suffix = raw(p['prefix_commands'], 5), raw(p['suffix_commands'], 5)
        full = raw(p['commands'], 10)
        require(full == prefix + suffix and full.hex() == c['full10_raw_bits'] and suffix.hex() == c['suffix_raw_bits'] == r['suffix_raw_bits'], 'raw byte identity')
        for field, data in (('prefix_sha256', prefix), ('suffix_sha256', suffix), ('full10_sha256', full)):
            require(sha(data) == w[field] == s[field], field)
        require(w['actual20_verified'] is True and w['exact_existing_execution_reuse'] is True and w['actual_return_event_sha256'] == r['actual_return_event_sha256'] == s['actual_return_event_sha256'], 'inherited execution witness')
        d = distances(full)
        require(not any(x['outside'] for x in d), f'original raw box violation slot {i}; no clipping')
        out.append(dict(index=i, anchor_index=i//5, word=s['word'], role=s['role'], root_slot=s['root_slot'], condition=s['condition'], plan_digest=p['plan_digest'], job_digest=r['job_digest'], full10_raw_bits=full.hex(), coordinates=d, prefix=segment(d[:10]), suffix=segment(d[10:]), full=segment(d), latent_status=s['missing_reason']))
    result = []
    expected = [(5*a+i, 5*a+j) for a in range(8) for i in range(5) for j in range(i+1,5)]
    for p, (i,j) in zip(pairs, expected):
        require((p['i'], p['j'], p['anchor_index']) == (i,j,i//5), 'pair roster')
        result.append(dict(i=i, j=j, anchor_index=i//5, words=[out[i]['word'],out[j]['word']], left=out[i]['full'], right=out[j]['full'], left_suffix=out[i]['suffix'], right_suffix=out[j]['suffix'], missing_indices=p['missing_indices'], model_to_observed=p['model_to_observed'], observed_to_physical=p['observed_to_physical'], physical_lower_is_better_deltas=p['physical_lower_is_better_deltas']))
    return out, result

def build():
    objects = {k: json.loads(file_bytes(ROOT / rel, digest)) for k,(rel,digest) in INPUTS.items()}
    for rel, digest in SOURCES.items():
        file_bytes(ROOT / rel, digest)
        require(objects['evaluation']['bindings'][str(ROOT / rel)] == digest, 'source closure')
    slots, pairs = join(objects['selector'], objects['evaluation'], objects['margin'])
    return dict(schema=NAME+'.analysis', bindings={k:dict(path=str(ROOT / rel),sha256=digest) for k,(rel,digest) in INPUTS.items()}, source_bindings=SOURCES, definition='exact raw float32 chart [-1,1]; no epsilon; negative nearest is outside', evidence='execution identity inherited from sealed evaluation; no new journal or fullstrict verification', clip='release contract excludes second clip; upstream clip and physical saturation unknown', interpretation='post-hoc diagnostic; no causality, selection, gate or admission', P1_admission=False, slots=slots, pairs=pairs, counts=dict(slots=len(slots),pairs=len(pairs),latent_missing_pairs=sum(bool(p['missing_indices']) for p in pairs), exact_boundary_slots=sum(s['full']['exact_boundary_count']>0 for s in slots)))

def encoded(x):
    return (json.dumps(x, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command', choices=['analyze','verify'])
    ap.add_argument('--path', required=True, type=Path)
    args=ap.parse_args()
    require(args.path.is_absolute(), 'output path must be absolute')
    data=encoded(build())
    if args.command == 'analyze':
        require(args.path.parent.resolve() == (ROOT/'results').resolve() and args.path.name.startswith(NAME+'_'), 'own results path required')
        with args.path.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        print('wrote', args.path, sha(data))
    else:
        require(args.path.read_bytes() == data, 'whole saved report mismatch')
        print('verified', args.path, sha(data))

if __name__ == '__main__':
    main()
