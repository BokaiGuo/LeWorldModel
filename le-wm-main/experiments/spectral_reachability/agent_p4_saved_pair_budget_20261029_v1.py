"""Fixed-cohort descriptive analysis of the authenticated P4 saved report."""
import argparse
import hashlib
import json
import math
from pathlib import Path

NAME = 'agent_p4_saved_pair_budget_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    'report': ('results/agent_p4_moving_multiboundary_20261028_v1_MAIN02/report.json', '60489b7db3d4eff70d1be46d47d92209e6adcf333ae1d0a0e389b259ec1f6084'),
    'prepare': ('results/agent_p4_moving_multiboundary_20261028_v1_prepare_MAIN02.json', '8528e8d7f9b688b1c93be154305ccc35b8ed6218de63e736b1ae567691c432a2'),
    'strict': ('results/consume_downstream_MAIN_20261028_v1_P4_MAIN01.json', '29c760cc10a28748789303796e38b49efad6d786fe16e39bd077227216b4e83e'),
    'native': ('results/audit_p4_native_MAIN_20261028_v1_MAIN01.json', '4f2f62029ef5e4209fcda4dc853a6404189fa820f9f8b4a35c4024ff16e9c60e'),
    'rolling': ('results/consume_zero_and_rolling_MAIN_20261028_v1_rolling_MAIN01.json', '182d5ce14a6363b287607c511d3bc4dde85769618e8c8082029f090fb19c1232'),
}
STEPS = (0, 5, 10, 15, 20)
ARMS = ('line_search_GN', 'full10_conditional_D2P')

def need(ok, message):
    if not ok: raise ValueError(message)

def pairs(items):
    result = {}
    for k, v in items:
        need(k not in result, 'duplicate JSON key'); result[k] = v
    return result

def decode(raw):
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def load_inputs():
    result = {}
    for role, (name, sha) in INPUTS.items():
        path = ROOT / name
        need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64*1024*1024, 'input file')
        raw = path.read_bytes(); need(hashlib.sha256(raw).hexdigest() == sha, 'input SHA: '+role)
        result[role] = decode(raw)
    return result

def vector(v):
    need(type(v) is list and len(v) == 7 and all(type(x) in (int, float) and math.isfinite(x) for x in v), 'finite state7')
    return v

def metrics(cp):
    s, g = vector(cp['state']), vector(cp['goal'])
    d = [a-b for a,b in zip(s,g)]
    angle = math.atan2(math.sin(d[4]), math.cos(d[4]))
    return dict(state=s, goal=g, state_minus_goal=d, agent_xy=math.hypot(*d[:2]),
                object_xy=math.hypot(*d[2:4]), joint_xy=math.sqrt(sum(x*x for x in d[:4])),
                wrapped_angle_abs=abs(angle), velocity_xy=math.hypot(*d[5:7]))

def analyze(inputs):
    report, strict, native, rolling = (inputs[k] for k in ('report','strict','native','rolling'))
    need(report['schema'] == 'agent_p4_moving_multiboundary_20261028_v1.report' and report['status'] == 'known_moving_multiboundary_recorded', 'report ABI')
    need(report['fixed_denominator'] == 8 and report['independent_tasks'] == 2 and not report['errors'] and not report['pending'], 'fixed complete report')
    need(strict['schema'] == 'agent_p4_moving_multiboundary_20261028_v1.consume' and strict['status'] == 'saved_multiboundary_verified_NOT_scientific_pass' and strict['report_sha256'] == INPUTS['report'][1] and strict['fixed_denominator'] == 8 and strict['compile_denominator'] == 24 and strict['counts'] == report['counts'], 'saved strict authority')
    need(native['schema'] == 'MAIN.p4_native_audit.v1' and native['report_sha256'] == INPUTS['report'][1], 'native audit authority')
    need(rolling['schema'] == 'agent_p4_rolling_saved_consume_20261028_v1.receipt' and rolling['status'] == 'saved_rolling_visual_seams_verified_ONLY' and rolling['fixed_denominator'] == 8, 'rolling authority')
    need(rolling['inputs'][str(ROOT / INPUTS['report'][0])] == INPUTS['report'][1] and rolling['inputs'][str(ROOT / INPUTS['prepare'][0])] == INPUTS['prepare'][1], 'rolling source join')
    rows = report['rows']; need(len(rows) == 8, 'eight slots')
    compiled = strict['compiled']; need(len(compiled) == rolling['reached_boundaries'] == 23, 'actual compile census')
    compile_keys = [(x['episode_id'],x['boundary']) for x in compiled]
    need(len(set(compile_keys)) == len(compile_keys) and all(x['arrays_verified'] is True for x in compiled), 'unique verified compile census')
    result = []
    for i in range(4):
        a,b = rows[2*i:2*i+2]; expected = (i//2, ARMS[i%2])
        for j,r in enumerate((a,b)):
            mode = ('I','R')[j]; eid = f'e{i:02}_{mode}'
            need((r['source_slot'],r['prefix_arm']) == expected and r['mode'] == mode and r['episode_id'] == eid, 'fixed identity order')
            need(r['initial_exact'] is True and len(r['executed_raw']) == r['control_returned'] and 5 <= r['control_returned'] <= 20, 'actual execution')
            cps = r['checkpoints']; times = [x['step'] for x in cps]
            need(times == [t for t in STEPS if t <= r['control_returned']], 'checkpoint missing or duplicate')
            need([x['step'] for x in r['encodings']] == [t for t in (5,10,15,20) if t <= r['control_returned']], 'encoding census')
            need({(eid,t) for t in (5,10,15) if t <= r['control_returned']} == {x for x in compile_keys if x[0] == eid}, 'compile identity census')
            need(all(cp['goal'] == cps[0]['goal'] for cp in cps), 'same goal per episode')
        need(a['checkpoints'][0]['state'] == b['checkpoints'][0]['state'] and a['checkpoints'][0]['goal'] == b['checkpoints'][0]['goal'] and a['executed_raw'][:5] == b['executed_raw'][:5], 'pair initial/prefix binding')
        lookup = [{x['step']:x for x in r['checkpoints']} for r in (a,b)]
        paired = []
        for t in STEPS:
            v = [metrics(q[t]) if t in q else None for q in lookup]
            delta = None if None in v else {k:v[1][k]-v[0][k] for k in ('agent_xy','object_xy','joint_xy','wrapped_angle_abs','velocity_xy')}
            paired.append(dict(step=t, I=v[0], R=v[1], R_minus_I=delta))
        result.append(dict(pair=i, source_slot=expected[0], prefix_arm=expected[1],
                           arms=[dict(episode_id=r['episode_id'], status=r['status'], control_returned=r['control_returned'],
                                      native_success=r['native_success'], final_metrics=r['final_metrics'],
                                      observed_encoding_records=len(r['encodings']), verified_compile_records=sum(x[0]==r['episode_id'] for x in compile_keys)) for r in (a,b)],
                           checkpoints=paired))
    need(set(compile_keys) == {(r['episode_id'],t) for r in rows for t in (5,10,15) if t <= r['control_returned']}, 'all 24 slots retained')
    return dict(schema=NAME+'.analysis', bindings={k:v[1] for k,v in INPUTS.items()}, pairs=result,
                denominator=dict(episodes=8,pairs=4,independent_known_roots=2,compile_slots=24),
                actual_counts=report['counts'], declared_caps=dict(neural=report['contract']['neural_caps'],physical=report['contract']['physical_caps']),
                unknowns=dict(FLOPs=None, device_time=None, inherited_compute=None, per_arm_metered_cost=None),
                interpretation='descriptive same-control-time, different trajectories; not causal, fresh, equal FLOPs, or P1/P4 admission',
                fresh_roots_certified=0, P1_admission=False, P4_admission=False)

def canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n'

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command', required=True)
    sub.add_parser('analyze', help='read fixed SHA-pinned saved inputs; print analysis')
    v=sub.add_parser('verify', help='recompute and compare a saved analysis'); v.add_argument('--analysis',required=True); v.add_argument('--sha256',required=True)
    a=p.parse_args(); value=analyze(load_inputs())
    if a.command == 'verify':
        path=Path(a.analysis); need(path.is_file() and not path.is_symlink() and path.stat().st_size<=8*1024*1024, 'analysis file')
        raw=path.read_bytes(); need(hashlib.sha256(raw).hexdigest()==a.sha256 and decode(raw)==value, 'saved analysis mismatch')
        print(canonical(dict(status='exact_analysis_verified', analysis_sha256=a.sha256)),end='')
    else: print(canonical(value),end='')

if __name__ == '__main__': main()
