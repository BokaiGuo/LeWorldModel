"""Fixed-support counterfactual protocol and executable RAM algebra; NOT a runner."""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import struct

NAME = 'agent_p3_nongreedy_counterfactual_contract_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
GROUPS = ((0, 0, 1), (1, 0, 3), (2, 1, 1), (3, 1, 3))
FAMILIES = (('R', 8), ('secant', 4), ('zero', 1))
TAIL_STEPS = 5  # one original packet; never selected using an endpoint outcome
ENDPOINTS = ((0, 'C'), (0, 'Z'), (0, 'P'), (1, 'Z'), (1, 'P'))
FLAGS = dict(P1_admission=False, P2_admission=False, P3_admission=False,
             full_C3_demonstrated=False, scientific_admission=False,
             freshness=False, independent_confirmation=False, physical_release=False,
             training=False, equal_FLOPs=False, equal_total_compute=False)
PINS = {
    '../docs/AGENT_P1_THEORY_GATE_ANALYSIS_20261028_v1.md': '94ef31f06f49e11195ede31cec8fe9dfe2ce807c0b37bf02c9d5de1836222246',
    'innerdocs/p1_constructive_mechanism_prerequisites_20261027_v1.md': '8817c0c9d251f24660665c430e9811cee8be98074684c8f13ef13eca1cf1d11e',
    'experiments/spectral_reachability/main_p3_secant_gate_scalar_20261029_v1.py': '675fa944c4858264b616f5eff69965cbd02b5169a4a53425ddee903cc21051db',
    'experiments/spectral_reachability/agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1.py': 'f6984725a676a79f4b4e9b32985ba6fda7957187ee4fd6f9bd1756d07afd353d',
    'experiments/spectral_reachability/agent_p3_secant_delivery_20261029_v1.py': '7b3449b27ec68d7f923ac0af93f186c651ccb97231894152cbfa9a01649f57d2',
    'experiments/spectral_reachability/agent_p3_parent_native_cuda_20261029_v1.py': 'eeea5710d4171b7c43f99d1f2b5ad2412e03a25ba14e8fc5589ea5ef73f57b3b',
    'experiments/spectral_reachability/agent_p3_same_time_capability_probe_physical_20261029_v2.py': 'f070b6afe4fe0ebba66e05d366f929467602b98805980f86f27e2344445a8344',
    '.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py': 'c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2',
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def encode(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rational(x):
    need(type(x) in (int, float) and math.isfinite(x), 'finite literal scalar')
    return Fraction(x)


def ratio(x):
    return str(x.numerator)+'/'+str(x.denominator)


def roster():
    return [dict(slot=s, root=r, horizon=h, family=f, index=i)
            for s, r, h in GROUPS for f, n in FAMILIES for i in range(n)]


def raw_bits(raw, n):
    need(type(raw) is list and len(raw) == n, 'all raw rows')
    out = bytearray()
    for row in raw:
        need(type(row) is list and len(row) == 2, 'raw shape Nx2')
        for x in row:
            rational(x)
            need(-1 <= x <= 1, 'raw box: no clip')
            b = struct.pack('<f', x)
            need(struct.unpack('<f', b)[0] == x, 'already float32 raw; no extra cast rounding')
            out.extend(b)
    return bytes(out).hex()


def analyze_bank(rows):
    """Typed RAM projection, NOT an authority bridge or upstream saved consumer.

    MAIN's future adapter must authenticate task/context/model/score/array bytes
    against complete R and secant receipts before calling this algebra.
    """
    expected = roster()
    need(type(rows) is list and len(rows) == 52, 'fixed 52 support including four paid zeros')
    fields = set(expected[0]) | {'status', 'reason', 'immediate', 'reach', 'terminal',
                                'edge', 'suffix', 'source_pointer'}
    for row, identity in zip(rows, expected, strict=True):
        need(set(row) == fields and all(type(row[k]) is type(v) and row[k] == v
             for k, v in identity.items()), 'ordered complete bank identity')
        need(type(row['source_pointer']) is str and row['source_pointer'].startswith('/'), 'explicit source JSON pointer')
        if row['status'] == 'complete':
            need(row['reason'] is None, 'complete reason')
            for k in ('immediate', 'reach', 'terminal'):
                rational(row[k]); need(row[k] >= 0, 'nonnegative original model costs')
            raw_bits(row['edge'], 5*row['horizon']); raw_bits(row['suffix'], 5)
            if row['family'] == 'zero':
                need(raw_bits(row['edge'], 5*row['horizon']) == '00000000'*(10*row['horizon']), 'paid control positive zero bits')
        else:
            need(row['status'] == 'unavailable' and type(row['reason']) is str and bool(row['reason']), 'explicit unavailable reason')
            need(all(row[k] is None for k in ('immediate', 'reach', 'terminal', 'edge', 'suffix')), 'no fake missing cost/action')
    panels = []
    for slot, root, h in GROUPS:
        group = [x for x in rows if x['slot'] == slot]
        zero = group[-1]
        for family, size in FAMILIES[:2]:
            candidates = [x for x in group if x['family'] == family]
            if zero['status'] != 'complete':
                panels.append(dict(slot=slot, root=root, horizon=h, family=family,
                    denominator=size, status='blocked_control_unavailable', rows=None,
                    original_order=None, shadows=None))
                continue
            complete = [x for x in candidates if x['status'] == 'complete']
            # Zero sorts after all original indices on exact ties in shadow panels.
            support = complete+[dict(zero, index=size)]
            orders = {k: [x['index'] for x in sorted(support, key=lambda x: (
                rational(x['immediate'])+rational(x['reach']) if k == 'stage_plus_R_beta1'
                else rational(x['reach']) if k == 'R_only' else rational(x['terminal']), x['index']))]
                for k in ('stage_plus_R_beta1', 'R_only', 'finite_terminal')}
            details = []
            for x in candidates:
                d = dict(index=x['index'], status=x['status'], reason=x['reason'], eligible=None,
                         stage_rejects=None, reach_rejects=None, model_terminal_better=None,
                         stage_plus_R_delta=None, terminal_minus_R=None,
                         physical_benefit=None)
                if x['status'] == 'complete':
                    di = rational(x['immediate'])-rational(zero['immediate'])
                    dr = rational(x['reach'])-rational(zero['reach'])
                    d.update(eligible=di > 0 and dr < 0, stage_rejects=di <= 0,
                             reach_rejects=dr >= 0,
                             model_terminal_better=x['terminal'] < zero['terminal'],
                             stage_plus_R_delta=ratio(di+dr),
                             terminal_minus_R=ratio(rational(x['terminal'])-rational(x['reach'])))
                details.append(d)
            eligible = [x for x in complete if x['immediate'] > zero['immediate'] and x['reach'] < zero['reach']]
            panels.append(dict(slot=slot, root=root, horizon=h, family=family,
                denominator=size, status='RAM_algebra_only', rows=details,
                original_order=[x['index'] for x in sorted(eligible, key=lambda x: (x['immediate'], x['index']))],
                shadows=orders))
    return dict(schema=NAME+'.RAM_analysis', panels=panels, source_authenticated=False,
                physical_benefit_verified=False, selector_deployed=False, known_roots=2,
                fresh_roots=0, **FLAGS)


def replay_schedule():
    """Future fully paid validation, not calling old delivery's restricted ABI."""
    return [dict(**r, layout=layout, raw_steps=5*r['horizon']+5,
                 predict= r['horizon']+1, metric=2)
            for r in roster() for layout in ('whole', 'packet_segmented')]


def replay_caps():
    schedule = replay_schedule()
    n = sum(x['predict'] for x in schedule)
    return dict(replay_calls=len(schedule), predict=n, action_encoder=n, predictor=n,
        pred_proj=n, AE_tokens=3*n, predictor_context_tokens=3*n, metric=2*len(schedule),
        encoder=0, projector=0, jacobian=0, VJP=0, solve=0, SVD=0,
        FLOPs=None, scalar_backward=None)


def tail_caps():
    # Warm0..10 plus control0..20, each rendered off/on/off, no deduplication.
    return dict(environments=10, constructor=10, close=10, reset=20, set_state=40,
        goal_render=20, goal_bind=10, goal_setter=30, public_step=300,
        physics_step=3040, explicit_render=960, neural=0,
        original_public_step=250, original_physics_step=2540, original_render=810,
        incremental_public_step=50, incremental_physics_step=500, incremental_render=150)


def tail_jobs():
    return [dict(index=2*i+j, endpoint=i, root=ENDPOINTS[i][0], role=ENDPOINTS[i][1], continuation=kind,
                 tail_raw5=[[0.0, 0.0] for _ in range(TAIL_STEPS)],
                 checkpoints=[0, 5, 10, 15, 20], terminal_checkpoint=20)
            for i in range(5) for j, kind in enumerate(('U', 'Z2'))]


class ReplayRAM:
    """Fixed-order whole-unit reservation model, not actual native metering."""
    def __init__(self):
        self.cursor = 0; self.units = []; self.reserved_predict = 0

    def attempt(self, request, callback):
        schedule = replay_schedule()
        need(self.cursor < len(schedule) and request == schedule[self.cursor], 'no reorder/retry/substitution')
        self.cursor += 1
        self.reserved_predict += request['predict']
        need(self.reserved_predict <= replay_caps()['predict'], 'fixed vector maximum')
        row = dict(request=deepcopy(request), status='attempted', actual=None)
        self.units.append(row)
        try:
            value = callback(deepcopy(request))
            need(type(value) is dict and value == dict(predict=request['predict'],
                action_encoder=request['predict'], predictor=request['predict'],
                pred_proj=request['predict'], metric=2), 'whole completed native vector')
            row.update(status='complete', actual=deepcopy(value))
        except BaseException:
            row['status'] = 'failed_unknown_actual'; raise
        return deepcopy(row)


class TailRAM:
    """Causal durable-sink contract, callbacks MUST be RAM fixtures, not envs.

    A write failure before guard acceptance releases no step16. A callback
    exception leaves its paid attempt and completion unknown (no refund).
    """
    def __init__(self):
        self.events = []; self.attempts = 0; self.completed = 0

    def run(self, *, job, cp15_exact, stopped, sink, callback):
        need(job in tail_jobs(), 'one fixed full job, uniform tail only')
        need(type(cp15_exact) is bool and type(stopped) is bool, 'explicit guard/native stop')
        result = dict(index=job['index'], cp20=None, status='not_started')
        if stopped:
            result['status'] = 'native_stopped'; return result
        need(cp15_exact, 'full state/snapshot/pixels/history cp15 match BEFORE tail')
        event = dict(kind='cp15_guard', index=job['index'], accepted=True)
        sink(deepcopy(event))  # writer fsync success required before local acceptance
        self.events.append(event)
        for step, raw in enumerate(job['tail_raw5'], 16):
            intent = dict(kind='public_intent', index=job['index'], step=step, raw_bits=raw_bits([raw], 1))
            sink(deepcopy(intent)); self.events.append(intent); self.attempts += 1
            try:
                returned = callback(step, deepcopy(raw))
                need(set(returned) == {'terminated', 'truncated', 'native7'}, 'RAM return ABI')
                need(type(returned['terminated']) is bool and type(returned['truncated']) is bool,
                     'native termination not success surrogate')
                need(len(returned['native7']) == 7, 'native seven axes')
                for x in returned['native7']: rational(x)
                self.completed += 1
                event = dict(kind='public_return', index=job['index'], step=step, value=deepcopy(returned))
                sink(deepcopy(event)); self.events.append(event)
            except BaseException:
                result['status'] = 'partial_unknown_callback_or_publication'; raise
            if step == 20:
                result['cp20'] = deepcopy(returned)
            if returned['terminated'] or returned['truncated']:
                result['status'] = 'native_stopped'; return result
        result['status'] = 'horizon_completed'; return result


def specification():
    bindings = {}
    for name, expected in PINS.items():
        path = (ROOT/name).resolve()
        need(path.is_file() and sha(path) == expected, 'reference source pin '+name)
        bindings[str(path)] = expected
    for path in (Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(path)] = sha(path)
    return dict(schema=NAME+'.protocol', status='RAM_contract_NOT_production_ready', bindings=bindings,
        groups=[list(x) for x in GROUPS], roster=roster(), replay_schedule=replay_schedule(),
        replay_caps=replay_caps(), tail_jobs=tail_jobs(), tail_caps=tail_caps(),
        tail_positive_zero_bits='00000000'*10,
        denominator=dict(bank=52, R_candidates=32, secant_candidates=16, paid_zero=4,
                         tail_branches=10, tail_pairs=5, logical_C=4, comparisons=8, inherited_no_witness=3),
        known_roots=2, fresh_roots=0, **FLAGS,
        tail_seconds=0.5, tail_endpoint_control=20, tail_endpoint_absolute=30,
        native_axes=['agent_x', 'agent_y', 'object_x', 'object_y', 'object_angle_mod2pi', 'agent_vx', 'agent_vy'],
        goal_velocity='retain and authenticate both original goal7 velocity components; not assumed zero',
        agent_velocity_error='distance from saved goal agent velocity, not velocity magnitude',
        per_axis_gain='error(Z2)-error(U); positive favors U, ties/null separate; no combined winner',
        original_gate_changed=False, original_cp15_replaced=False,
        authority_adapter_implemented=False, production_authorized=False,
        remaining=['source-bound full52 projection adapter', 'paid native whole/packet-segmented driver/consumer',
                   'fresh10 native replay+tail driver/consumer with causal cp15 guard',
                   'MAIN separate execution approval and actual output/wall/GPU budgets'])


def absolute(value):
    p = Path(value)
    need(p.is_absolute() and p == p.resolve(), 'canonical absolute path')
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('build', 'verify'))
    ap.add_argument('--output', required=True)
    a = ap.parse_args(); p = absolute(a.output); data = encode(specification())
    if a.command == 'build':
        with p.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        fd = os.open(p.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    else:
        need(p.read_bytes() == data, 'whole source-bound contract reconstruction')
    print(json.dumps(dict(path=str(p), sha256=sha(p))))


if __name__ == '__main__':
    main()
