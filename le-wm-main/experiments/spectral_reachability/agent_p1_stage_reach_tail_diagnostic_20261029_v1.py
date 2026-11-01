"""Independent P1 saved breakpoint analysis and ten-branch RAM tail arithmetic.

This is not a native physical driver or a receipt authority for a future tail.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import struct

NAME = 'agent_p1_stage_reach_tail_diagnostic_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT/'results/agent_p1_terminal_witness_selector_20261028_v1_MAIN01.json'
P1_SHA = 'a3a317b557da8bb6eb7a38d582f7df4d230b1b849cced00eca1add85403e63d9'
ENDPOINTS = ((0, 'C'), (0, 'Z'), (0, 'P'), (1, 'Z'), (1, 'P'))


def need(ok, why):
    if not ok: raise ValueError(why)


def number(x):
    need(type(x) in (int, float) and math.isfinite(x), 'finite literal number')
    return float(x)


def frac(x):
    return Fraction.from_float(number(x))


def ratio(x):
    return str(x.numerator)+'/'+str(x.denominator)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs(items):
    d = {}
    for k, v in items:
        need(k not in d, 'duplicate JSON key'); d[k] = v
    return d


def relation(a, b):
    if a is None or b is None: return 'null'
    return 'less' if a < b else 'greater' if a > b else 'tie'


def p1_breakpoints(doc):
    """Consume the fixed derivative, do not claim to repeat upstream validation."""
    need(doc['schema'] == 'agent_p1_terminal_witness_selector_20261028_v1.analysis'
         and doc['status'] == 'new_exploratory_model_only_selector_not_admitted'
         and doc['candidate_denominator'] == 1056 and doc['P1_admission'] is False,
         'complete saved P1 derivative')
    ps = doc['panels']; need(type(ps) is list and len(ps) == 8, 'all eight fixed panels')
    results = []
    for pnum, p in enumerate(ps):
        family, n = ('allmode', 42) if pnum < 4 else ('mixedmode', 222)
        index = pnum % 4; root = index//2; condition = ('off', 'on')[index % 2]
        need((p['family'], p['index'], p['root_slot'], p['condition'], p['candidate_denominator'])
             == (family, index, root, condition, n), 'exact family/root/condition denominator')
        rows = p['candidates']; need(len(rows) == n and [x['index'] for x in rows] == list(range(n)), 'ordered candidate support')
        need(p['status'] == 'complete_model_only' and p['terminal_available'] == n
             and p['zero_prefix_index'] == 0 and p['literal_forward_requests'] == [], 'no missing evidence')
        for x in rows:
            s, r, o, t = (number(x[k]) for k in ('stage_cost', 'own_reach_value', 'original_objective', 'nonlinear_terminal_Q'))
            need(min(s, r, t) >= 0 and o == s+r, 'native float addition and finite costs')
        winner = min(rows, key=lambda x: (x['original_objective'], x['index']))
        terminal = min(rows, key=lambda x: (x['nonlinear_terminal_Q'], x['index']))
        greedy = min(rows, key=lambda x: (x['stage_cost'], x['index']))
        need(p['original_stage_plus_ridge']['index'] == winner['index']
             and p['original_stage_plus_ridge']['score'] == winner['original_objective']
             and p['nonlinear_terminal_Q']['index'] == terminal['index']
             and p['nonlinear_terminal_Q']['score'] == terminal['nonlinear_terminal_Q']
             and p['stage_greedy_index'] == greedy['index'], 'preserve native winners/ties')
        out = []
        for x in rows:
            # Candidate minus actual native winner; exact arithmetic exposes rounding.
            ds = frac(x['stage_cost'])-frac(winner['stage_cost'])
            dr = frac(x['own_reach_value'])-frac(winner['own_reach_value'])
            native_delta = frac(x['original_objective'])-frac(winner['original_objective'])
            tdelta = frac(x['nonlinear_terminal_Q'])-frac(winner['nonlinear_terminal_Q'])
            out.append(dict(index=x['index'], stage_delta=ratio(ds), reach_delta=ratio(dr),
                exact_sum_delta=ratio(ds+dr), native_rounded_delta=ratio(native_delta),
                objective_rounding_delta=ratio(native_delta-ds-dr), terminal_delta=ratio(tdelta),
                terminal_model_better=tdelta < 0, selected=x['index'] == winner['index'],
                stage_uphill_from_initial=x['stage_cost'] > number(p['initial_Q']),
                stage_worse_than_zero=x['stage_cost'] > rows[0]['stage_cost'],
                stage_worse_than_greedy=x['stage_cost'] > greedy['stage_cost'],
                real_benefit=None))
        results.append(dict(family=family, index=index, root_slot=root, condition=condition,
            candidate_denominator=n, selected_index=winner['index'], terminal_index=terminal['index'],
            stage_greedy_index=greedy['index'], candidates=out))
    return dict(schema=NAME+'.p1_analysis', panels=results, candidate_denominator=1056,
                known_physical_roots=2, fresh_roots=0, upstream_fullstrict_rebuilt=False,
                physical_outcomes_used=False, P1_admission=False, scientific_admission=False)


def vector7(x):
    need(type(x) is list and len(x) == 7, 'native7 literal shape')
    return [number(v) for v in x]


def metrics(state, goal):
    s, g = vector7(state), vector7(goal); d = [a-b for a,b in zip(s,g,strict=True)]
    sq = [x*x for x in d]; total = math.fsum(sq)
    need(all(math.isfinite(x) for x in sq) and math.isfinite(total), 'finite squared axes')
    xy = math.sqrt(math.fsum(sq[:4])); angle = abs(d[4]); native_angle = min(angle, 2*math.pi-angle)
    return dict(native7_distance=math.sqrt(total), native7_squared=total, reward=-math.sqrt(total),
        joint_xy=xy, agent_xy=math.hypot(*d[:2]), object_xy=math.hypot(*d[2:4]),
        agent_velocity=math.hypot(*d[5:]), raw_angle=angle,
        wrapped_angle=abs(math.atan2(math.sin(d[4]), math.cos(d[4]))),
        native_success=xy < 20 and native_angle < math.pi/9)


def bits(v):
    return b''.join(struct.pack('<d', x) for x in vector7(v))


def tail_ram(rows):
    """Only projected fixtures: no authority, no environment and no self-signed guard."""
    need(type(rows) is list and len(rows) == 10, 'fixed ten branches')
    result = []
    for i, row in enumerate(rows):
        endpoint = i//2; root, role = ENDPOINTS[endpoint]; kind = ('U', 'Z2')[i%2]
        need((row['index'], row['endpoint'], row['root'], row['role'], row['kind']) ==
             (i, endpoint, root, role, kind), 'fixed ten branch identity')
        need(row['status'] in ('horizon_completed', 'native_stopped', 'engineering_failed'), 'explicit terminal')
        goal = vector7(row['goal7'])
        values = {}
        for cp in ('cp15', 'cp20'):
            state = row[cp]
            values[cp] = None if state is None else metrics(state, goal)
        need(row['status'] != 'horizon_completed' or values['cp20'] is not None, 'complete requires cp20')
        need(values['cp20'] is None or values['cp15'] is not None, 'no substituted cp15')
        result.append(dict(index=i, endpoint=endpoint, root=root, role=role, kind=kind,
                           status=row['status'], goal7=goal, **values))
    comp = []
    keys = ('native7_distance', 'native7_squared', 'joint_xy', 'agent_xy', 'object_xy',
            'agent_velocity', 'raw_angle', 'wrapped_angle')
    for i in range(5):
        u, z = result[2*i:2*i+2]
        need(bits(u['goal7']) == bits(z['goal7']), 'same goal including signed velocity bits')
        cps = {}
        for cp in ('cp15', 'cp20'):
            a, b = u[cp], z[cp]
            cps[cp] = {k: dict(Z2_minus_U=None if a is None or b is None else b[k]-a[k],
                               relation=relation(None if b is None else b[k], None if a is None else a[k]))
                       for k in keys}
        comp.append(dict(endpoint=i, root=u['root'], role=u['role'], **cps))
    return dict(schema=NAME+'.tail_RAM', rows=result, pairs=comp, branch_denominator=10,
        pair_denominator=5, logical_C=4, logical_C_comparisons=8, inherited_no_witness=3,
        source_authenticated=False, physical_run=False, original_gate_changed=False,
        P1_admission=False, P3_admission=False, scientific_admission=False)


def encode(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def build():
    need(P1.is_file() and not P1.is_symlink() and digest(P1) == P1_SHA, 'pinned full P1 derivative')
    with P1.open('r') as f: doc = json.load(f, object_pairs_hook=pairs)
    analysis = p1_breakpoints(doc)
    need(digest(P1) == P1_SHA, 'stable input across analysis')
    files = (Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md'))
    analysis['input'] = dict(path=str(P1), sha256=P1_SHA)
    analysis['sources'] = {str(p): digest(p) for p in files}
    return analysis


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('analyze', 'verify')); ap.add_argument('--output', required=True)
    arg = ap.parse_args(); p = Path(arg.output)
    need(p.is_absolute() and p.resolve() == p, 'canonical absolute output')
    data = encode(build())
    if arg.command == 'analyze':
        with p.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        fd = os.open(p.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    else: need(p.read_bytes() == data, 'complete reconstruction differs')
    print(json.dumps(dict(path=str(p), sha256=digest(p))))


if __name__ == '__main__': main()
