"""MAIN-only missing known-root warmup collector; saved/RAM entries never simulate."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.pusht_protocol_contracts import root_fingerprint
from experiments.spectral_reachability import agent_p2_midpoint_triplet_inventory_20261028_v1 as inv

NAME = 'agent_p2_warmup_history_reconstruct_20261028_v1'
ROOT, ARMS = inv.ROOT, inv.ARMS
read, require, sha, digest = inv.read, inv.require, inv.sha, inv.digest
write, encode = inv.mid.write, inv.gap.encode
AUTH = 'MAIN_APPROVED_KNOWN4_WARMUP_RECONSTRUCTION_ONLY'
ORIGINAL = ROOT/'results/agent_p2_composition_execution_20261028_v1_prepare_MAIN01.json'
ORIGINAL_SHA = '58d0c982d550b016efaa839cf8e24abe524d830a3da637ff5bd22286c84f0b83'
CAP, FLOOR, WALL = 64*1024**2, 200*1024**2, 900
FLAGS = dict(P1_admission=False, P2_admission=False, scientific_admission=False, freshness=False)


def contract():
    return dict(root_count=4, branch_count=8, root_seed=202753010, oversample_factor=4,
                reconstruct_slots=[1,2,3], history_len=3, raw_command=[0.,0.],
                planned_attempts=dict(constructor=1, collector_reset=32, root_reset=3, warmup_step=45),
                planned_returns=None, selected_action_steps=0, neural_calls=0,
                output_cap_bytes=CAP, floor_bytes=FLOOR, wall_seconds=WALL,
                meter_scope='public factory/reset/step/render/close only; no pre-factory native hooks',
                unmeasured_native=dict(space_step=None, setter=None, goal_bind=None,
                                       reset_internal_work=None, constructor_internal_work=None),
                wall_scope='call boundaries and final wall; not preemptive watchdog', **FLAGS)


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/test_{NAME}.py', ROOT/f'innerdocs/{NAME}.md']


def build(inventory, expected):
    inv.verify(inventory, expected)
    snapshot = read(inventory)
    require(snapshot['reconstruction']['missing_root_slots'] == [1,2,3], 'fixed missing slots, no adaptive reselection')
    require(sha(ORIGINAL) == ORIGINAL_SHA, 'original physical prepare identity')
    p = read(ORIGINAL)
    physical = read(ROOT/inv.gap.INPUTS['physical'][0])
    require(sha(ROOT/inv.gap.INPUTS['physical'][0]) == inv.gap.INPUTS['physical'][1], 'original physical report')
    native = inv.gap.census(physical['rows'])
    plans = inv.gap.census(p['plans'])
    roots = []
    for i in range(4):
        a, b = [native[i,arm]['physical']['context_identity'] for arm in ARMS]
        r = plans[i,ARMS[0]]['root_sha256']
        require(a == b and r == plans[i,ARMS[1]]['root_sha256'], 'paired exact original context')
        roots.append(dict(root_slot=i, root_sha256=r, reference=p['references'][r], expected_context=a))
    bindings = {**p['bindings'], **snapshot['consumed_sha256'], **snapshot['source_files_sha256'],
                str(ORIGINAL):ORIGINAL_SHA, str(Path(inventory).absolute()):expected}
    # Includes original reset recipe, collector dependencies and installed native source.
    for path in own_files()+[Path(inv.__file__)]: bindings[str(path)] = sha(path)
    for path,h in bindings.items(): require(sha(inv.mid.regular(path)) == h, 'prepare source/input binding')
    require(p['root_seed'] == contract()['root_seed'], 'original fixed seed')
    return dict(schema=NAME+'.prepare', inventory=str(Path(inventory).absolute()), inventory_sha256=expected,
                contract=contract(), roots=roots, prior_rows=snapshot['rows'], bindings=bindings, **FLAGS)


def verify(prepared, expected):
    require(sha(inv.mid.regular(prepared)) == expected, 'external prepare SHA')
    p = read(prepared)
    require(p['schema'] == NAME+'.prepare' and p['contract'] == contract(), 'fixed collector contract')
    require(p == build(p['inventory'],p['inventory_sha256']), 'exact fresh saved reconstruction')
    return p


def state(env):
    a = np.asarray(env.unwrapped._get_obs())
    require(a.shape == (7,) and a.dtype.kind == 'f' and np.isfinite(a).all(), 'actual native state7')
    return a.tolist()


def initial_rows(p):
    return [dict(root_slot=i, arm=a, root_sha256=p['roots'][i]['root_sha256'],
                 status='already_saved' if i == 0 else 'not_started', history=None)
            for i in range(4) for a in ARMS]


class Recorder:
    """Append-before-dispatch public ledger; no internal-call estimates or retries."""
    def __init__(self, output, check):
        self.output, self.check = output, check
        self.stream = (output/'events.jsonl').open('xb')
        self.events = []

    def event(self, kind, *, _guard=True, **values):
        if _guard: self.check()
        e = dict(seq=len(self.events), kind=kind, **values)
        raw = encode(e)+b'\n'
        require(self.stream.write(raw) == len(raw), 'complete journal write')
        self.stream.flush(); os.fsync(self.stream.fileno()); self.events.append(e)
        return e

    def call(self, operation, fn, *, slot=None, cleanup=False, **details):
        require(not cleanup or operation == 'close', 'only cleanup may bypass dispatch cap')
        i = len(self.events)
        self.event('attempt', _guard=not cleanup, call_id=i, operation=operation, root_slot=slot, **details)
        result = fn()
        # A return is durable before any postprocessing, image/state checks or caps.
        e = dict(seq=len(self.events),kind='return',call_id=i,operation=operation,root_slot=slot)
        raw = encode(e)+b'\n'
        require(self.stream.write(raw) == len(raw), 'complete return write')
        self.stream.flush(); os.fsync(self.stream.fileno()); self.events.append(e)
        return result

    def image(self, name, rgb):
        from PIL import Image
        h = inv.pixel_sha(rgb)
        path = self.output/(name+'.png')
        self.check()
        with path.open('xb') as f:
            Image.fromarray(rgb).save(f,format='PNG'); f.flush(); os.fsync(f.fileno())
        return dict(path=str(path),file_sha256=sha(path),pixel_sha256=h)


class MeterEnv:
    def __init__(self, env, recorder):
        self.env, self.rec = env, recorder
        self.slot, self.step_index, self.frames, self.goal = None, 0, {}, None

    @property
    def unwrapped(self): return self.env.unwrapped

    def __getattr__(self, name): return getattr(self.env,name)

    def reset(self, *args, **kwargs):
        op = 'collector_reset' if self.slot is None else 'root_reset'
        count = sum(e['kind'] == 'attempt' and e['operation'] == op for e in self.rec.events)
        require(count < (32 if self.slot is None else 3), 'fixed public reset ceiling')
        if self.slot is not None:
            require(self.step_index == 0 and not self.frames and self.goal is None, 'one reset per missing root')
            opts = kwargs.get('options',{})
            self.rec.event('root_reset_intent',root_slot=self.slot,seed=kwargs.get('seed'),
                           state=np.asarray(opts.get('state')).tolist(),goal=np.asarray(opts.get('goal_state')).tolist())
        result = self.rec.call(op,lambda:self.env.reset(*args,**kwargs),slot=self.slot)
        if self.slot is not None:
            self.goal = self.rec.image(f'root{self.slot}.goal',np.asarray(result[1]['goal']))
            self.rec.event('reset_observed',root_slot=self.slot,state=state(self),goal=self.goal)
        return result

    def step(self, action):
        require(self.slot in (1,2,3) and self.step_index < 15, 'no collector/selected/padding steps')
        require(np.array_equal(np.asarray(action),np.zeros(2)), 'only original zero warmup commands')
        n = self.step_index+1
        out = self.rec.call('warmup_step',lambda:self.env.step(action),slot=self.slot,step=n,command=[0.,0.])
        self.step_index = n
        require(len(out) == 5 and type(out[2]) in (bool,np.bool_) and type(out[3]) in (bool,np.bool_), 'native transition ABI')
        self.rec.event('step_observed',root_slot=self.slot,step=n,state=state(self),
                       terminated=bool(out[2]),truncated=bool(out[3]))
        require(not out[2] and not out[3], 'native stop: no further step or synthetic boundary')
        return out

    def render(self, *args, **kwargs):
        rgb = self.rec.call('render',lambda:self.env.render(*args,**kwargs),slot=self.slot)
        if self.slot is not None and self.step_index in (5,10,15):
            n = self.step_index
            h = inv.pixel_sha(np.asarray(rgb)); s = state(self)
            if n not in self.frames:
                f = self.rec.image(f'root{self.slot}.warmup{n}',np.asarray(rgb))
                f.update(warmup_step=n,state=s,observation_id=str(self.rec.output/'events.jsonl')+'#'+str(len(self.rec.events)),
                         actual_observation=True)
                self.rec.event('boundary_observed',root_slot=self.slot,frame=f)
                self.frames[n] = f
            else:
                require(h == self.frames[n]['pixel_sha256'] and s == self.frames[n]['state'], 'repeated boundary render purity')
        return rgb


def collect(p, env, recorder, collector, warmup, rows):
    """Injected RAM-test seam. Production supplies the unchanged original helpers."""
    meter = MeterEnv(env,recorder)
    roots = collector(meter,count=4,seed=p['contract']['root_seed'],oversample_factor=4)
    require(accounting(recorder.events)['observed_public_calls'].get('collector_reset') == dict(attempts=32,returns=32), 'original32 collector reset returns actually observed')
    require(len(roots) == 4, 'full original four-root collector')
    require([root_fingerprint(r) for r in roots] == [r['root_sha256'] for r in p['roots']], 'all4 known identities/order before warmup')
    recorder.event('roots_collected',roots=[dict(root_slot=i,root_sha256=root_fingerprint(r),
                    state=np.asarray(r['state']).tolist(),goal=np.asarray(r['goal']).tolist()) for i,r in enumerate(roots)])
    for slot in (1,2,3):
        pair = [r for r in rows if r['root_slot'] == slot]
        for r in pair: r['status'] = 'attempted'
        meter.slot, meter.step_index, meter.frames, meter.goal = slot, 0, {}, None
        frames, raw, goal = warmup(meter,roots[slot],history_len=3)
        require(meter.step_index == 15 and set(meter.frames) == {5,10,15}, 'actual complete warmup boundaries')
        require(np.asarray(raw).dtype == np.float32 and np.array_equal(raw,np.zeros((3,2),np.float32)), 'original raw3 history')
        ref, context = p['roots'][slot]['reference'], p['roots'][slot]['expected_context']
        saved_frames = [dict(meter.frames[n],root_sha256=ref['root_sha256']) for n in (5,10,15)]
        history = inv.validate_history(ref,saved_frames,frames,goal)
        require(all(inv.pixel_sha(rgb) == f['pixel_sha256'] for rgb,f in zip(frames,saved_frames,strict=True)), 'helper returns actual observed frames, not aliases')
        require(state(meter) == context['state'] and np.asarray(roots[slot]['goal']).tolist() == context['goal'] and
                np.asarray(raw).tolist() == context['raw_history'] and inv.pixel_sha(goal) == meter.goal['pixel_sha256'], 'original state7/goal/raw context exact identity')
        history.update(state_at_warmup10=saved_frames[1]['state'],goal_frame=meter.goal,
                       state_at_warmup15=state(meter),raw_history=np.asarray(raw).tolist())
        recorder.event('root_completed',root_slot=slot,history=history)
        for r in pair: r.update(status='reconstructed_exact',history=history)


def accounting(events):
    counts = {}; opened = {}; seen = set(); warmup = {}; resets = set()
    for i,e in enumerate(events):
        require(e['seq'] == i, 'unbroken actual event sequence')
        if e['kind'] == 'attempt':
            k = e['operation']; cid = e['call_id']
            require(k in ('constructor','collector_reset','root_reset','warmup_step','render','close') and cid == i and cid not in seen, 'public call operation/identity')
            slot = e['root_slot']
            if k in ('constructor','collector_reset','close'): require(slot is None, 'global public call scope')
            if k == 'root_reset':
                require(slot in (1,2,3) and slot not in resets, 'fixed missing roots, no retry')
                resets.add(slot)
            if k == 'warmup_step':
                n = warmup.get(slot,0)+1
                require(slot in resets and n <= 15 and e['step'] == n and e['command'] == [0.,0.], 'ordered actual zero warmup, no padding')
                warmup[slot] = n
            seen.add(cid); opened[cid] = e
            c = counts.setdefault(k,dict(attempts=0,returns=0)); c['attempts'] += 1
        elif e['kind'] == 'return':
            require(e['call_id'] in opened, 'return must have actual unmatched attempt')
            before = opened.pop(e['call_id'])
            require(all(before[k] == e[k] for k in ('call_id','operation','root_slot')), 'attempt/return identity')
            counts[e['operation']]['returns'] += 1
    return dict(observed_public_calls=counts,unreturned_attempts=sorted(opened),
                never_attempted_operations=[k for k in contract()['planned_attempts'] if k not in counts],
                unmeasured_native=contract()['unmeasured_native'],scope=contract()['meter_scope'])


def run(prepared, expected, output, authority):
    require(authority == AUTH, 'explicit MAIN-only physics authority')
    p = verify(prepared,expected); output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir(), 'new exclusive output')
    require(shutil.disk_usage(output.parent).free >= FLOOR+CAP, 'preflight disk reserve')
    output.mkdir(); started = time.monotonic(); rows = initial_rows(p); errors = []; env = rec = None
    def check():
        require(time.monotonic()-started <= WALL, 'wall cap at public boundary')
        require(sum(f.stat().st_size for f in output.iterdir() if f.is_file()) < CAP-1024**2 and
                shutil.disk_usage(output).free >= FLOOR+1024**2, 'output/floor cap')
    write(output/'execution_binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected,
          authority=authority,contract=p['contract'],initial_rows=rows,**FLAGS))
    try:
        rec = Recorder(output,check)
        from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
        env = rec.call('constructor',lambda:base.gym.make('swm/PushT-v1',render_mode='rgb_array'))
        collect(p,env,rec,base.collect_nonterminal_boundary_contact_roots,base.physical._reset_and_warmup,rows)
    except BaseException as exc:
        errors.append(dict(stage='collect',type=type(exc).__name__,message=str(exc)))
    finally:
        if env is not None:
            try: rec.call('close',env.close,cleanup=True)
            except BaseException as exc: errors.append(dict(stage='close',message=str(exc)))
        if rec is not None:
            try: rec.stream.close()
            except BaseException as exc: errors.append(dict(stage='journal_close',message=str(exc)))
        for row in rows:
            if row['status'] == 'attempted': row['status'] = 'failed_partial'
        try: verify(prepared,expected); check()
        except BaseException as exc: errors.append(dict(stage='final_binding_or_cap',message=str(exc)))
        events = rec.events if rec is not None else []
        counts = accounting(events)
        complete = not errors and all(r['status'] in ('already_saved','reconstructed_exact') for r in rows)
        report = dict(schema=NAME+'.report',status='history_reconstructed' if complete else 'partial_or_failed',
                      prepare_sha256=expected,rows=rows,root_denominator=4,arm_denominator=8,errors=errors,
                      accounting=counts,wall_seconds=time.monotonic()-started,
                      artifact_sha256={f.name:sha(f) for f in output.iterdir() if f.is_file()},**FLAGS)
        write(output/'report.json',report)
    return report


def consume(prepared, expected, report_path, report_sha):
    p = verify(prepared,expected)
    require(sha(inv.mid.regular(report_path)) == report_sha, 'external terminal report SHA')
    report = read(report_path); output = Path(report_path).absolute().parent
    require(report['schema'] == NAME+'.report' and report['prepare_sha256'] == expected and
            report['root_denominator'] == 4 and report['arm_denominator'] == 8 and
            all(report[k] is False for k in FLAGS), 'own full denominator report, no admission')
    require({'execution_binding.json','events.jsonl'} <= set(report['artifact_sha256']) and
            set(report['artifact_sha256']) == {f.name for f in output.iterdir() if f.name != 'report.json'}, 'complete exclusive output manifest')
    for name,h in report['artifact_sha256'].items():
        require(Path(name).name == name and sha(inv.mid.regular(output/name)) == h, 'flat exact saved artifact')
    binding = read(output/'execution_binding.json')
    require(binding['prepare_sha256'] == expected and binding['contract'] == contract() and binding['authority'] == AUTH and
            binding['initial_rows'] == initial_rows(p), 'execution binding')
    import json
    events = [json.loads(line) for line in (output/'events.jsonl').read_bytes().splitlines()]
    require(report['accounting'] == accounting(events), 'actual attempts/returns independently recounted')
    require(type(report['wall_seconds']) in (int,float) and np.isfinite(report['wall_seconds']) and
            report['wall_seconds'] >= 0 and sum(f.stat().st_size for f in output.iterdir()) <= CAP, 'saved engineering totals')
    require([(r['root_slot'],r['arm']) for r in report['rows']] == [(i,a) for i in range(4) for a in ARMS], 'all8 canonical rows')
    completed = {e['root_slot']:e['history'] for e in events if e['kind'] == 'root_completed'}
    require(len(completed) == sum(e['kind'] == 'root_completed' for e in events), 'one completion per missing root')
    root_events = [e for e in events if e['kind'] == 'roots_collected']
    if completed:
        require(len(root_events) == 1 and len(root_events[0]['roots']) == 4, 'all4 before any completed reconstruction')
        for i,r in enumerate(root_events[0]['roots']):
            require(r['root_slot'] == i and r['root_sha256'] == p['roots'][i]['root_sha256'] ==
                    root_fingerprint({k:np.asarray(r[k]) for k in ('state','goal')}), 'saved original source roots')
    for slot,h in completed.items():
        require(slot in (1,2,3), 'only fixed missing roots')
        frames = h['frames']; rgbs = []
        for f in frames+[h['goal_frame']]:
            path = Path(f['path'])
            require(path.parent == output and path.name in report['artifact_sha256'], 'actual exclusive PNG membership')
            rgb,pixel = inv.mid.pixels(path,f['file_sha256'])
            require(pixel == f['pixel_sha256'], 'exact saved pixels')
            rgbs.append(rgb)
        base = inv.validate_history(p['roots'][slot]['reference'],frames,rgbs[:3],rgbs[3])
        observed = [e['frame'] for e in events if e['kind'] == 'boundary_observed' and e['root_slot'] == slot]
        require([{k:v for k,v in f.items() if k != 'root_sha256'} for f in frames] == observed, 'genuine distinct boundary events')
        steps = [e for e in events if e['kind'] == 'step_observed' and e['root_slot'] == slot]
        require([e['step'] for e in steps] == list(range(1,16)) and all(not e['terminated'] and not e['truncated'] for e in steps), 'full nonpadded actual15')
        reset = [e for e in events if e['kind'] == 'reset_observed' and e['root_slot'] == slot]
        require(len(reset) == 1 and reset[0]['goal'] == h['goal_frame'], 'same actual reset goal frame')
        intent = [e for e in events if e['kind'] == 'root_reset_intent' and e['root_slot'] == slot]
        root = root_events[0]['roots'][slot]
        require(len(intent) == 1 and intent[0]['seed'] == 0 and intent[0]['state'] == root['state'] and
                intent[0]['goal'] == root['goal'] == p['roots'][slot]['expected_context']['goal'], 'original reset recipe witness')
        require(h['state_at_warmup10'] == frames[1]['state'] == steps[9]['state'] and
                h['state_at_warmup15'] == frames[2]['state'] == steps[14]['state'] == p['roots'][slot]['expected_context']['state'] and
                h['raw_history'] == [[0.,0.]]*3 and base['history_rgb_sha256'] == h['history_rgb_sha256'], 'state/raw/history closure')
    for r in report['rows']:
        require(r['root_sha256'] == p['roots'][r['root_slot']]['root_sha256'], 'row root identity')
        if r['root_slot'] == 0: require(r['status'] == 'already_saved' and r['history'] is None, 'reuse root0, no fabricated return')
        elif r['root_slot'] in completed:
            require(r['status'] == 'reconstructed_exact' and r['history'] == completed[r['root_slot']], 'paired history exact completion')
        else: require(r['status'] in ('not_started','failed_partial') and r['history'] is None, 'missing remains null')
    if report['status'] == 'history_reconstructed':
        require(not report['errors'] and set(completed) == {1,2,3}, 'successful full fixed reconstruction')
        c = report['accounting']['observed_public_calls']
        require(all(c[k] == dict(attempts=n,returns=n) for k,n in contract()['planned_attempts'].items()) and
                c.get('close') == dict(attempts=1,returns=1) and report['wall_seconds'] <= WALL and
                not report['accounting']['unreturned_attempts'], 'actual complete budget, never planned returns')
    else: require(report['status'] == 'partial_or_failed' and report['errors'], 'failure not hidden')
    return dict(schema=NAME+'.consume',status=report['status'],report_sha256=report_sha,
                root_denominator=4,arm_denominator=8,rows=report['rows'],accounting=report['accounting'],
                stageB_adapter_ready=False,new_model_calls=0,new_physics_steps=0,**FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    a = sub.add_parser('prepare'); a.add_argument('--inventory',type=Path,required=True); a.add_argument('--inventory-sha',required=True); a.add_argument('--output',type=Path,required=True)
    for name in ('verify','run','consume'):
        a = sub.add_parser(name); a.add_argument('--prepared',type=Path,required=True); a.add_argument('--prepare-sha',required=True)
        if name == 'run': a.add_argument('--output',type=Path,required=True); a.add_argument('--authority',required=True)
        if name == 'consume': a.add_argument('--report',type=Path,required=True); a.add_argument('--report-sha',required=True); a.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        write(args.output,build(args.inventory,args.inventory_sha)); result = dict(output=str(args.output),sha256=sha(args.output))
    elif args.command == 'verify':
        verify(args.prepared,args.prepare_sha); result = dict(status='verified_NOT_physics_authority')
    elif args.command == 'run': result = run(args.prepared,args.prepare_sha,args.output,args.authority)
    else:
        result = consume(args.prepared,args.prepare_sha,args.report,args.report_sha); write(args.output,result)
    print(encode(result).decode())


if __name__ == '__main__': main()
