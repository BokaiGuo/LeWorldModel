"""Pure saved GN/D2P vs physical diagnostics; never import producers/models/envs.

Explicit SHA-bound JSON/task/plan/journal/PNG and optional selected native arrays.
No witness/NPZ access or fit.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import zlib
import numpy as np
from PIL import Image

NAME = 'p1_constructive_model_physical_diagnostics_20261027_v1'
SOURCE = 'p1_constructive_reachable_source_20261027_v1'
PLANNER = 'p1_constructive_blind_planners_cuda_20261027_v1'
PHYSICAL = 'p1_constructive_plan_physical_20261027_v1'
ARMS = ('line_search_GN', 'full10_conditional_D2P')
PROJECT = Path(__file__).resolve().parents[2]
PLANNER_SHA = '4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d'
SOURCE_SHA = '9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9'
PHYSICAL_SHA = 'd205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300'


def require(value, message):
    if not value: raise ValueError(message)


def canonical(x): return json.dumps(x, sort_keys=True, allow_nan=False)
def sha(b): return hashlib.sha256(b).hexdigest()
def digest(x): return sha(canonical(x).encode())


def decode(b):
    def unique(items):
        d = {}
        for k,v in items:
            require(k not in d, 'duplicate JSON key'); d[k] = v
        return d
    x = json.loads(b, object_pairs_hook=unique)
    canonical(x); return x


def finite(x, shape):
    a = np.asarray(x, dtype=np.float64)
    require(a.shape == shape and np.isfinite(a).all(), 'finite shape '+str(shape))
    return a


def equal(a,b,label): require(np.array_equal(a,b), label)
def close(a,b,label): require(np.allclose(a,b,atol=1e-12,rtol=1e-12), label)


class Reader:
    """Only explicit paths or parent-report-hash-bound children; mock byte loader."""
    def __init__(self, load=None):
        self.load = load or self.disk; self.bindings = {}; self.bytes_read = 0
    @staticmethod
    def disk(path):
        p = Path(path)
        require(p.is_file() and not p.is_symlink() and p.stat().st_size <= 8*1024**2, 'bounded regular saved input')
        return p.read_bytes()
    def raw(self,path,expected):
        p = Path(path).absolute(); require(p.suffix in ('.json','.jsonl','.gz','.png'), 'no NPZ/NPY/witness/certificate loading')
        require('certificate' not in p.name.lower() and 'witness' not in p.name.lower(), 'privileged filename forbidden')
        require(isinstance(expected,str) and len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), 'complete external SHA256 required')
        b = self.load(str(p)); require(len(b) <= 8*1024**2 and sha(b) == expected, 'saved input SHA: '+str(p))
        self.bytes_read += len(b); require(self.bytes_read <= 64*1024**2, 'total bounded saved reads')
        require(str(p) not in self.bindings or self.bindings[str(p)] == expected, 'same-path identity drift')
        self.bindings[str(p)] = expected
        return b
    def json(self,path,expected):
        b = self.raw(path,expected)
        if str(path).endswith('.gz'):
            z = zlib.decompressobj(31); b = z.decompress(b,32*1024**2+1)
            require(len(b) <= 32*1024**2 and z.eof and not z.unused_data and not z.unconsumed_tail, 'single complete bounded gzip')
        return decode(b)
    def child(self,parent,name,expected):
        require(Path(name).name == name and name not in ('','.','..'), 'flat hash-bound child')
        return self.json(Path(parent).parent/name,expected)
    def finish(self):
        for path,h in self.bindings.items(): require(sha(self.load(path)) == h, 'final saved TOCTOU: '+path)
    def native_latent(self,parent,index,entry,manifest):
        name=f'image_{index:02}.native192.npy'
        require(entry['file'] == name and entry['dtype'] == 'float32' and entry['shape'] == [192] and
            entry['sha256'] == manifest[name], 'receipt-bound native array entry')
        path=str(Path(parent).absolute().parent/name); b=self.load(path)
        require(len(b) <= 4096 and sha(b) == entry['sha256'], 'native array SHA/size')
        stream=io.BytesIO(b); z=np.load(stream,allow_pickle=False)
        require(stream.tell() == len(b) and z.dtype == np.dtype('float32') and z.shape == (192,) and np.isfinite(z).all(), 'native float32 finite192 only')
        require(path not in self.bindings or self.bindings[path] == entry['sha256'], 'native identity drift')
        self.bindings[path]=entry['sha256'];self.bytes_read+=len(b)
        require(self.bytes_read <= 64*1024**2,'total bounded saved reads')
        return z.astype(np.float64)


def metric(goal,current):
    g,s = finite(goal,(7,)),finite(current,(7,)); d = s-g
    a = float(abs(d[4])); native_angle = min(a,2*np.pi-a)
    wrapped = float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))))
    xy = float(np.linalg.norm(d[:4]))
    return dict(native_success=bool(xy < 20 and native_angle < np.pi/9),native_distance=float(np.linalg.norm(d)),
        native_angle_term=float(native_angle),negative_native_angle=bool(native_angle < 0),
        wrapped_allxy_success=bool(xy < 20 and wrapped < np.pi/9),wrapped_object_angle_error=wrapped,
        object_position_error=float(np.linalg.norm(d[2:4])),object_pose=s[2:5].tolist(),object_success=None)


def diagnostics(root,goal,current):
    s,r = finite(current,(7,)),finite(root,(7,)); initial = metric(goal,r); m = metric(goal,s)
    error=s-finite(goal,(7,))
    signed = float(np.arctan2(np.sin(s[4]-r[4]),np.cos(s[4]-r[4])))
    return dict(state=s.tolist(),metrics=m,object_displacement_from_initial=float(np.linalg.norm(s[2:4]-r[2:4])),
        native_squared_distance_components=dict(agent_xy=float(error[:2]@error[:2]),object_xy=float(error[2:4]@error[2:4]),
            raw_angle=float(error[4]**2),agent_velocity=float(error[5:]@error[5:])),
        native_allxy_error=float(np.linalg.norm(error[:4])),agent_position_error=float(np.linalg.norm(error[:2])),
        agent_velocity_error=float(np.linalg.norm(error[5:])),
        object_wrapped_rotation_signed_from_initial=signed,object_wrapped_rotation_abs_from_initial=abs(signed),
        changes_from_initial={k:m[k]-initial[k] for k in ('native_distance','object_position_error','wrapped_object_angle_error')},
        # Descriptive sign only, never a scientific margin or mechanism gate.
        native_nonprogress_descriptive=m['native_distance'] >= initial['native_distance'],
        object_position_nonprogress_descriptive=m['object_position_error'] >= initial['object_position_error'],
        rotation_scope='wrapped net change, not total path rotation')


def check_metrics(saved,goal,current):
    expected = metric(goal,current); require(set(saved) == set(expected), 'native metric schema')
    for k,v in expected.items():
        if isinstance(v,(float,list)): close(saved[k],v,'recomputed physical metric: '+k)
        else: require(type(saved[k]) is type(v) and saved[k] == v, 'physical label: '+k)


def model_cost(z,goal): return float(np.mean((finite(z,(192,))-finite(goal,(192,)))**2))


def source_identities(source):
    require(source['schema'] == SOURCE+'.sourcepack' and source['status'] == 'fixed_slots_processed' and
        source['sourcepack_complete'] is True and source['errors'] == [] and len(source['rows']) == 2, 'complete original two-source report')
    identities = []
    for i,row in enumerate(source['rows']):
        require(row['slot']['index'] == i and row['reachable'] is True, 'fixed source identity')
        # Deliberately project ONLY identities: certificate paths/contents never opened.
        identities.append(dict(root_slot=i,root_sha256=row['root_sha256'],blind_sha256=row['blind_payload_sha256']))
    require(len({x['root_sha256'] for x in identities}) == 2, 'distinct source roots')
    return identities


def read_models(reader,path,expected,identities):
    report = reader.json(path,expected)
    require(report['schema'] == PLANNER+'.report' and report['status'] == 'model_plans_sealed_not_admitted' and
        report['errors'] == [] and report['commands_released'] is True and report['source_verified'] is True and
        report['model_unchanged'] is True and report['model_hash_before'] == report['model_hash_after'] and
        report['equal_budget_claim'] is False, 'complete planner actual report')
    require([(r['order'],r['arm']) for r in report['rows']] == [(i,a) for i in range(2) for a in ARMS], 'four fixed planner rows')
    models = {}
    for row in report['rows']:
        i,arm = row['order'],row['arm']; require(row['status'] == 'model_plan_sealed', 'sealed model plan')
        plan = reader.child(path,row['plan_file'],report['artifact_sha256'][row['plan_file']])
        result = reader.child(path,row['result_file'],report['artifact_sha256'][row['result_file']])
        require(plan['schema'] == PLANNER+'.model_plan' and plan['order'] == i and plan['arm'] == arm and
            plan['with_target'] is False and plan['selection_uses_physics'] is False and plan['physical_execution'] is False and
            digest({k:v for k,v in plan.items() if k != 'selection_sha256'}) == plan['selection_sha256'] == row['selection_sha256'], 'own preselected model plan identity')
        require(Path(plan['task_export']).name.startswith('task_'), 'explicit blind task export')
        export = reader.json(plan['task_export'],plan['task_sha256']); t = export['task']
        require(export['source_slot'] == i and export['native_root_sha256'] is None and
            export['blind_sha256'] == export['blind_source_sha256'] == identities[i]['blind_sha256'] and
            export['with_target'] is False and export['task_contract']['with_target'] is False, 'source-slot/blind/new render binding')
        require(digest(t) == export['task_id'] == row['task_id'] == plan['task_id'] and
            digest(export['task_contract']) == t['task_contract_sha256'] == plan['task_contract_sha256'], 'same task/goal contract')
        equal(t['projection'],np.eye(192),'fixed identity192 Q')
        require(t['feedback'] == 'open_loop_model_only' and t['horizon'] == 2 and t['packet_steps'] == 5 and
            t['packet_logit_cap'] == float(.75*np.sqrt(5)) and t['latent_dim'] == 192, 'same native H2 task')
        history = finite(t['root_context']['latent_history'],(3,192)); raw = finite(t['root_raw_action_history'],(10,2))
        equal(raw,np.zeros((10,2)),'real zero warmup contract')
        mean,std = np.asarray(t['action_mean'],np.float32),np.asarray(t['action_std'],np.float32)
        equal(mean,np.asarray([-.0087,.0068],np.float32),'original mean'); equal(std,np.asarray([.2019,.2002],np.float32),'original std')
        tokens=(raw.astype(np.float32).reshape(2,10)-np.tile(mean,5))/np.tile(std,5)
        equal(t['root_context']['action_history'],tokens,'between-action normalization, no extra /100 or zero padding')
        commands=finite(plan['commands'],(10,2)); logits=finite(plan['logits'],(2,10))
        equal(commands,np.tanh(logits.astype(np.float32)).reshape(10,2),'exact float32 tanh time-major actions')
        require(np.all(np.linalg.norm(logits,axis=1) <= .75*np.sqrt(5)+1e-7), 'packet logit domain (roundoff only)')
        goal=finite(t['goal'],(192,)); initial=model_cost(history[-1],goal)
        if arm == ARMS[0]:
            pred=result['final_prediction']; require(len(result['steps']) == 4 and result['task_id'] == plan['task_id'], 'GN four-update evidence, not plateau')
            equal(pred['commands'],commands,'GN own chosen commands')
            prefix=finite(pred['contexts'][0]['latent_history'],(3,192))[-1]
            terminal=finite(pred['contexts'][1]['latent_history'],(3,192))[-1]
            close(pred['cost'],model_cost(terminal,goal),'GN terminal cost')
            tradeoff=None
        else:
            candidates=result['candidates']; require(len(candidates) == 10 and [c['index'] for c in candidates] == list(range(10)), 'all10 D2P candidate denominator')
            for c in candidates:
                require(c['completed'] is True, 'no failed candidate filtering')
                close(c['stage_cost'],model_cost(c['predicted_prefix'],goal),'candidate stage')
                close(c['nonlinear_terminal_cost'],model_cost(c['predicted_terminal'],goal),'candidate terminal')
                close(c['objective'],c['stage_cost']+c['own_reach_value'],'fixed beta1')
            winner=min(candidates,key=lambda c:(c['objective'],c['index']))
            require(winner == result['winner'], 'no posthoc D2P re-selection')
            equal(winner['commands'],commands,'D2P own chosen commands')
            prefix=finite(winner['predicted_prefix'],(192,)); terminal=finite(winner['predicted_terminal'],(192,))
            greedy=min(candidates,key=lambda c:(c['stage_cost'],c['index']))
            tradeoff=dict(selected_index=winner['index'],same_bank_greedy_index=greedy['index'],
                selected_is_greedy=winner['index'] == greedy['index'],
                stage_minus_same_bank_greedy=winner['stage_cost']-greedy['stage_cost'],
                own_reach_gain_vs_same_bank_greedy=greedy['own_reach_value']-winner['own_reach_value'],
                scope='model surrogate description only, no physical capability measurement')
        equal(plan['prediction']['latent_history'][-1],terminal,'plan selected terminal')
        close(plan['model_terminal_cost'],model_cost(terminal,goal),'plan terminal score')
        models[i,arm] = dict(commands=commands.tolist(),task_id=plan['task_id'],goal_pixel_sha256=t['goal_source_sha256'],
            task_export_sha256=plan['task_sha256'],task_export=plan['task_export'],selection_digest=plan['selection_sha256'],
            _vectors=dict(root=history[-1],goal=goal,prefix5=prefix,terminal10=terminal),
            initial_Q_MSE=initial,prefix5_Q_MSE=model_cost(prefix,goal),
            terminal10_Q_MSE=model_cost(terminal,goal),prefix5_Q_change=model_cost(prefix,goal)-initial,
            prefix_model_nonprogress_descriptive=model_cost(prefix,goal) >= initial,
            greedy_relative=tradeoff,resources=row['resources'],AD_API_resources=row['AD_API_resources'])
    for i in range(2):require(models[i,ARMS[0]]['task_id'] == models[i,ARMS[1]]['task_id'], 'paired task identity')
    return report,models


def checked_journal(reader,path,report):
    b=reader.raw(Path(path).parent/'journal.jsonl',report['journal_sha256'])
    events=[]; previous=None
    for index,line in enumerate(b.splitlines()):
        e=decode(line)
        require(e['index'] == index and e['previous_sha256'] == previous and
            e['event_sha256'] == digest({k:v for k,v in e.items() if k != 'event_sha256'}), 'physical event hashchain')
        events.append(e); previous=e['event_sha256']
    require(events and events[0]['kind'] == 'run_started' and events[-1]['kind'] == 'run_finished' and
        events[-1]['status'] == report['status'] and events[-1]['counts'] == report['counts'] and
        events[-1]['errors'] == report['errors'], 'physical run terminal/ledger anchor')
    counts={k:dict(attempted=0,completed=0) for k in report['counts']}; stack=[]
    for e in events:
        if e['kind'] not in ('operation_attempted','operation_completed'):continue
        resource=e['resource']; require(resource in counts,'known physical resource')
        data={k:v for k,v in e.items() if k not in ('kind','index','previous_sha256','event_sha256')}
        if e['kind'] == 'operation_attempted':
            if stack:
                allowed={'reset':('set_state','goal_render'),'set_state':('physics_step',),'public_step':('physics_step',)}
                require(resource in allowed.get(stack[-1][0]['resource'],()), 'physical API nesting')
            counts[resource]['attempted']+=1; stack.append((data,[]))
        else:
            require(stack and stack[-1][0] == data,'paired physical completion')
            _,children=stack.pop()
            grammar={'reset':['set_state','goal_render','set_state'],'set_state':['physics_step'],'public_step':['physics_step']*10}
            require(children == grammar.get(resource,[]),'physical internal census')
            if stack:stack[-1][1].append(resource)
            counts[resource]['completed']+=1
    require(counts == report['counts'],'actual attempted/completed physical ledger')
    require(all(0 <= c['completed'] <= c['attempted'] <= report['contract']['caps'][k] for k,c in counts.items()), 'physical resource cap')
    if report['status'] == 'physical_calibration_recorded':require(not stack,'no pending complete physical calls')
    return events


def pointer(value,path):
    require(isinstance(path,str) and path.startswith('/'),'explicit JSON pointer')
    for name in path[1:].split('/'):
        name=name.replace('~1','/').replace('~0','~')
        value=value[int(name)] if isinstance(value,list) else value[name]
    return value


def physical_rows(reader,path,expected,source_sha,identities,models):
    r=reader.json(path,expected)
    require(r['schema'] == PHYSICAL+'.report' and r['status'] in ('physical_calibration_recorded','partial_or_failed') and
        r['sourcepack_sha256'] == source_sha and r['new_neural_calls'] == 0 and r['P1_admission'] is False, 'physical original-source contract')
    require([(x['root_slot'],x['arm']) for x in r['rows']] == [(i,a) for i in range(2) for a in ARMS], 'fixed4 physical branches')
    events=checked_journal(reader,path,r); p=events[0]['prepare']
    require(p['sourcepack_sha256'] == source_sha and p['contract'] == r['contract'] and
        r['contract']['with_target'] is False and r['contract']['own_horizon'] == 10,'physical prepared new render/horizon')
    require(p['source_slots'] == [{k:v for k,v in x.items() if k != 'blind_sha256'} for x in identities], 'prepared original root mapping')
    output=[]; png_count=0
    for branch,row in enumerate(r['rows']):
        i,arm=row['root_slot'],row['arm']; model=models[i,arm]
        own=[e for e in events if e.get('branch') == branch]
        require(all(e['root_slot'] == i and e['arm'] == arm and e['selection_digest'] == row['selection_digest'] for e in own), 'physical branch/root/arm join')
        starts=[e for e in own if e['kind'] == 'branch_started']
        closed=[e for e in events if e['kind'] == 'branch_closed' and e['receipt']['root_slot'] == i and e['receipt']['arm'] == arm]
        require(len(closed) == 1 and closed[0]['receipt'] == row,'closed branch receipt')
        if row['status'] == 'not_started':
            require(not starts,'not-started evidence'); output.append(dict(root_slot=i,arm=arm,status='not_started',checkpoints={}));continue
        require(len(starts) == 1,'single branch start/no retry')
        plan=starts[0]['plan']; require(plan == p['plans'][branch] and
            digest({k:v for k,v in plan.items() if k != 'selection_digest'}) == row['selection_digest'], 'physical sealed selection')
        equal(plan['commands'],model['commands'],'physical commands equal own CUDA selected plan')
        # Optional format bridge is hash-bound, not trusted to reselect commands.
        bridge=reader.json(plan['compiler_report'],plan['compiler_report_SHA'])
        require(bridge['schema'] == plan['compiler_schema'] and bridge['status'] == plan['compiler_status'] and bridge['errors'] == [],'handoff bridge complete')
        require(pointer(bridge,plan['source_slot_pointer']) == i and pointer(bridge,plan['arm_pointer']) == arm and
            pointer(bridge,plan['blind_sha_pointer']) == identities[i]['blind_sha256'],'bridge source-slot/blind join')
        equal(pointer(bridge,plan['commands_pointer']),model['commands'],'bridge own commands, no witness replacement')
        frames={}; guards={}; returned={}; attached=[e for e in own if e['kind'] == 'goal_attached']
        for e in own:
            if e['kind'] == 'fresh_observation':
                obs=e['receipt']; key=(obs['phase'],obs['step']); require(key not in frames,'no duplicate fresh frame')
                require(obs['png'] == f'b{branch}_{obs["phase"]}_{obs["step"]:02}.png','own PNG scope')
                b=reader.raw(Path(path).parent/obs['png'],obs['file_sha256'])
                with Image.open(io.BytesIO(b)) as im:
                    require(im.mode == 'RGB' and im.size == (224,224),'physical RGB224'); rgb=np.array(im)
                require(sha(rgb.tobytes()) == obs['pixel_sha256'],'actual PNG pixel SHA')
                equal(obs['state'],obs['snapshot']['state'],'RGB snapshot state')
                frames[key]=(e,obs,rgb); png_count+=1
            elif e['kind'] == 'guard_receipt':
                key=(e['phase'],e['step']); require(key not in guards,'one guard per observation')
                guards[key]=e
                if key in frames:
                    _,obs,rgb=frames[key]; pixels=int(np.count_nonzero(np.all(rgb == [65,105,225],axis=-1)))
                    s=finite(obs['state'],(7,)); equal_state=np.array_equal(s,obs['observation_state'])
                    require(e['index'] > frames[key][0]['index'] and e['png'] == obs['png'] and e['file_sha256'] == obs['file_sha256'] and
                        e['canonical_agent_pixels'] == pixels and e['returned_state_equal'] == bool(equal_state),'saved fresh guard identity')
                    if e['accepted']:require(pixels > 0 and ((s[:4] >= 0)&(s[:4] <= 512)).all() and equal_state and e['render_preserved_dynamics'] is True,'accepted RGB/canvas guard')
            elif e['kind'] == 'returned_transition' and e['phase'] == 'plan':
                require(e['step'] == len(returned)+1 and e['step'] <= 10,'contiguous own returned transition')
                equal(e['command'],model['commands'][e['step']-1],'actual action equals selected own command')
                returned[e['step']]=e
        require(len(frames) == row['PNG'],'PNG row census')
        if not row['initial_exact']:
            require(row['status'] == 'failed','missing exact root cannot be successful')
            output.append(dict(root_slot=i,arm=arm,status=row['status'],checkpoints={},reason='root reconstruction failed',first_error=row['first_error']));continue
        require(len(attached) == 1,'one actual goal attachment')
        root,goal=finite(row['root_state'],(7,)),finite(row['goal'],(7,))
        require(sha(np.ascontiguousarray(np.r_[root,goal],dtype=np.float64).tobytes()) == identities[i]['root_sha256'],'actual root+goal bytes match source report')
        equal(attached[0]['goal'],goal,'actual attached goal'); equal(attached[0]['snapshot']['state'],root,'actual attached root')
        require(attached[0]['goal_rgb_sha256'] == model['goal_pixel_sha256'],'same model/physical goal RGB pixels')
        equal(frames['plan',0][1]['state'],root,'initial own plan frame'); check_metrics(row['initial_metrics'],goal,root)
        for step,e in returned.items():
            m=metric(goal,e['state']); check_metrics(e['metrics'],goal,e['state'])
            equal(e['returned_goal_state'],goal,'returned goal state binding')
            require(e['returned_goal_rgb_sha256'] == model['goal_pixel_sha256'] and
                e['actual_native_success'] == e['terminated'] == m['native_success'] and
                e['actual_native_distance'] == m['native_distance'] and e['reward'] == -m['native_distance'],'official native return/reward binding')
            if step < len(returned):require(not e['terminated'] and not e['truncated'],'no continuation after native terminal')
            if ('plan',step) in frames:equal(frames['plan',step][1]['state'],e['state'],'returned-state vs fresh PNG')
            if step < len(returned):require(guards.get(('plan',step),{}).get('accepted') is True,'no continuation after guard failure')
        require(row['own_returned'] == len(returned) and len(returned) <= row['own_attempted'] <= 10,'actual own horizon/partial count')
        checkpoints={}
        for step in (5,10):
            if step in returned:
                e=returned[step]; point=diagnostics(root,goal,e['state'])
                point.update(availability='returned_transition',fresh_guard_accepted=guards.get(('plan',step),{}).get('accepted',False))
            else:
                point=dict(availability='not_observed',reason='terminal_or_partial_before_requested_step',fresh_guard_accepted=False)
            checkpoints[str(step)]=point
        if returned:
            last=returned[max(returned)]; equal(row['final_state'],last['state'],'final own state')
            check_metrics(row['final_metrics'],goal,last['state'])
            require(row['native_success'] == metric(goal,last['state'])['native_success'] and
                row['full_horizon'] == (len(returned) == 10),'final native/horizon flags')
        expected_checks=[frames['plan',k][1] for k in (0,5,10) if ('plan',k) in frames and guards.get(('plan',k),{}).get('accepted') is True]
        require(row['checkpoints'] == expected_checks,'report checkpoints exact journal receipts')
        output.append(dict(root_slot=i,arm=arm,status=row['status'],initial=diagnostics(root,goal,root),checkpoints=checkpoints,
            official_native_success_at_last_return=row['native_success'],endpoint_scope=row.get('endpoint_scope','partial'),
            final_step=max(returned) if returned else None,full_horizon=row['full_horizon'],first_error=row['first_error'],
            native_success_is_not_object_success=True,object_success=None,capability_change='not_measured',
            strict_non_greedy_mechanism='not_established',P1_admission=False))
    if r['status'] == 'physical_calibration_recorded':
        require(r['errors'] == [] and all(x['first_error'] is None for x in r['rows']),'clean physical report')
        require(sum(x['PNG'] for x in r['rows']) == png_count,'total PNG census')
        c=r['counts'];n=len(r['rows'])
        require(all(x['attempted'] == x['completed'] for x in c.values()) and c['reset']['completed'] == 2*n and
            c['set_state']['completed'] == 4*n and c['goal_render']['completed'] == 2*n and c['goal_bind']['completed'] == n and
            c['public_step']['completed'] == sum(x['warmup_returned']+x['own_returned'] for x in r['rows']) and
            c['physics_step']['completed'] == 2*c['reset']['completed']+10*c['public_step']['completed'] and
            c['explicit_render']['completed'] == png_count,'clean actual physics/PNG census')
    require(r['native_successes'] == sum(x['status'] == 'native_success' for x in r['rows']),'native-success branch census')
    return r,output,len(events),png_count


def compare(models,rows):
    comparisons=[]
    for i in range(2):
        paired={a:next(x for x in rows if x['root_slot'] == i and x['arm'] == a) for a in ARMS}
        entry=dict(root_slot=i,model_terminal_Q_MSE_D2P_minus_GN=models[i,ARMS[1]]['terminal10_Q_MSE']-models[i,ARMS[0]]['terminal10_Q_MSE'],
            same_task_not_equal_compute=True,capability_change='not_measured',strict_non_greedy='not_established')
        for step in ('5','10'):
            pts=[paired[a]['checkpoints'].get(step,{}) for a in ARMS]
            if not all(x.get('availability') == 'returned_transition' and x['fresh_guard_accepted'] for x in pts):
                entry['step'+step]=dict(comparable=False,reason='missing or rejected fixed-time observation');continue
            diff={k:pts[1]['metrics'][k]-pts[0]['metrics'][k] for k in ('native_distance','object_position_error','wrapped_object_angle_error')}
            model_diff=(models[i,ARMS[1]]['prefix5_Q_MSE']-models[i,ARMS[0]]['prefix5_Q_MSE']) if step == '5' else entry['model_terminal_Q_MSE_D2P_minus_GN']
            entry['step'+step]=dict(comparable=True,actual_D2P_minus_GN=diff,
                model_vs_native_rank_sign_disagrees=bool(model_diff*diff['native_distance'] < 0),
                model_vs_object_position_rank_sign_disagrees=bool(model_diff*diff['object_position_error'] < 0),
                interpretation='cross-objective rank comparison, not latent prediction error or causal capability proof')
        comparisons.append(entry)
    return comparisons


def explanations():
    return [
        dict(priority=1,hypothesis='Wrong root/goal/arm or selected commands substituted',saved_test='source root+goal SHA, task/blind/source-slot, sealed plan, bridge, every actual own action',result='no discrepancy in checked saved bindings'),
        dict(priority=2,hypothesis='Action units/time-axis/history padding/normalization mismatch',saved_test='raw10 time-major, float32 tanh packet domain, states3/between-actions2, original mean/std',result='no discrepancy in checked task/command arrays; runtime tensor semantics remain receipt/source-bound'),
        dict(priority=3,hypothesis='Render/terminal/guard/angle bug or wrong evaluation',saved_test='PNG bytes/pixels, fresh guards, root/goal, independent native+wrapped metrics, termination and reward',result='checked on saved returns; negative angle cases explicitly marked'),
        dict(priority=4,hypothesis='Imagined suffix receives a different boundary than real execution',saved_test='compare fresh step5 latent/context against predicted prefix with a separately authorized true CUDA RGB encoding',result='not_tested: no new encoder or actual latent read; current runner is open-loop'),
        dict(priority=5,hypothesis='Identity latent Q objective is not native physical task error',saved_test='report native agent+object vs object-only signs separately; use independently encoded actual endpoints later',result='objective difference is explicit, not by itself a code bug or proven unique cause'),
        dict(priority=6,hypothesis='Model response/contact errors or insufficient/different optimizer coverage',saved_test='predeclared actual same-prefix probes and held-out FD/strong-budget comparisons',result='not_measured; cannot attribute to this cause or declare model/theory failure'),
    ]


def selected_arrays(reader,path,expected,planner,planner_path,planner_sha,physical,physical_path,physical_sha,models):
    r=reader.json(path,expected)
    require(r['schema'] == 'p1_constructive_selected_rgb_cuda_20261027_v1.report' and
        r['status'] == 'selected_rgb_encoded_not_admitted' and r['errors'] == [] and r['first_error'] is None and
        r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before'] == r['model_hash_after'] == planner['model_hash_after'] and
        r['new_predict'] == r['new_AD'] == r['new_physics'] == 0 and r['training'] is False and
        r['P1_admission'] is False,'complete selected RGB actual receipt/model')
    require(r['inputs'] == dict(planner_report=str(Path(planner_path).absolute()),planner_sha=planner_sha,
        physical_report=str(Path(physical_path).absolute()),physical_sha=physical_sha),'selected input report bindings')
    require(r['resources'] == {k:dict(attempted=8,completed=8) for k in ('encoder','projector')},'eight encoder/projector census')
    binding=reader.child(path,'binding.json',r['artifact_sha256']['binding.json'])
    require(binding['prepare_sha256'] == r['prepare_sha256'],'selected prepare anchor')
    prepared=reader.json(binding['prepared'],binding['prepare_sha256'])
    events=[decode(x) for x in reader.raw(Path(path).parent/'events.jsonl',r['artifact_sha256']['events.jsonl']).splitlines()]
    require(len(r['rows']) == 8 and len(events) == 48 and len(prepared['images']) == 8,'fixed eight-image denominator')
    output=[]
    for index,row in enumerate(r['rows']):
        branch=index//2;i=branch//2;arm=ARMS[branch%2];step=5 if index%2 == 0 else 10;role='prefix5' if step == 5 else 'terminal10'
        model=models[i,arm];p=row['provenance'];v=model['_vectors']
        for k,value in dict(index=index,branch=branch,root_slot=i,arm=arm,role=role,task_id=model['task_id'],
            task_export=model['task_export'],task_sha256=model['task_export_sha256']).items():
            require(row[k] == p[k] == value,'selected own identity: '+k)
        require(row['status'] == 'encoded' and p == prepared['images'][index] and p['Q'] == 'identity192' and
            p['selection_digest'] == physical['rows'][branch]['selection_digest'],'selected prepared image/selection')
        checkpoints=[c for c in physical['rows'][branch]['checkpoints'] if c['phase'] == 'plan' and c['step'] == step]
        require(len(checkpoints) == 1 and p['checkpoint'] == checkpoints[0] and
            p['png_path'] == str(Path(physical_path).absolute().parent/checkpoints[0]['png']),'fresh own physical checkpoint/PNG')
        for key in ('root','goal'):equal(p[key],v[key],'selected original '+key)
        equal(p['predicted'],v[role],'selected own prediction')
        e=events[index*6:index*6+6]
        require(e[0] == dict(kind='image_attempt',provenance=p) and e[5] == dict(kind='image_saved',receipt=row),'durable own image attempt/save')
        for j,resource in enumerate(('encoder','projector')):
            for offset,completed in ((1,False),(2,True)):
                require(e[2*j+offset] == dict(kind='module_return' if completed else 'module_attempt',
                    entry=dict(index=2*index+j,image=index,resource=resource,completed=completed)),'ordered module API accounting')
        z=reader.native_latent(path,index,row['latent'],r['artifact_sha256']);root,goal,pred=v['root'],v['goal'],v[role]
        aq,pq,rq=model_cost(z,goal),model_cost(pred,goal),model_cost(root,goal)
        metrics=dict(actual_Q=aq,predicted_Q=pq,root_Q=rq,actual_Q_change=aq-rq,predicted_Q_change=pq-rq,
            predicted_vs_real_mse=model_cost(pred,z),actual_vs_root_mse=model_cost(z,root),predicted_vs_root_mse=model_cost(pred,root),
            Q_improvement_sign_reversed=bool((aq-rq)*(pq-rq)<0))
        require(set(metrics) == set(row['metrics']),'saved latent metrics schema')
        for key,val in metrics.items():
            if isinstance(val,bool):require(row['metrics'][key] is val,'Q descriptive sign')
            else:close(row['metrics'][key],val,'independent actual latent '+key)
        output.append(dict(root_slot=i,arm=arm,step=step,metrics=metrics,latent_sha256=row['latent']['sha256'],
            pixel_sha256=checkpoints[0]['pixel_sha256']))
    paired=[]
    for i in range(2):
        for step in (5,10):
            pair=[next(x for x in output if x['root_slot'] == i and x['arm'] == a and x['step'] == step)['metrics'] for a in ARMS]
            ad=pair[1]['actual_Q']-pair[0]['actual_Q'];pd=pair[1]['predicted_Q']-pair[0]['predicted_Q']
            paired.append(dict(root_slot=i,step=step,actual_Q_D2P_minus_GN=ad,predicted_Q_D2P_minus_GN=pd,
                same_Q_rank_sign_reversed=bool(ad*pd<0)))
    return dict(rows=output,paired=paired,images_checked=8,resources=r['resources'],scope='saved encoding evidence, not a new model call or causal capability test')


def run_analysis(source_path,source_sha,planner_path,planner_sha,physical_path=None,physical_sha=None,reader=None,selected_path=None,selected_sha=None):
    reader=reader or Reader()
    out=dict(schema=NAME+'.diagnostics',status='unverified',errors=[],new_neural_calls=0,new_physics_steps=0,
        new_fits=0,P1_admission=False,capability_change='not_measured',independent_confirmation=False,
        scientific_thresholds_assigned=False,constructive_witness_success_attributed_to_model=False)
    try:
        ids=source_identities(reader.json(source_path,source_sha))
        planner,models=read_models(reader,planner_path,planner_sha,ids)
        out.update(fixed_roots=2,fixed_branches=4,models=[dict(root_slot=i,arm=a,**{k:v for k,v in models[i,a].items() if not k.startswith('_')}) for i in range(2) for a in ARMS])
        require((selected_path is None) == (selected_sha is None),'selected path and SHA together')
        require(selected_path is None or physical_path is not None,'selected requires physical report')
        if physical_path is None:
            out['status']='awaiting_explicit_physical_report_and_SHA';reader.finish()
        else:
            require(physical_sha is not None,'explicit full physical report SHA required')
            physical,rows,events,pngs=physical_rows(reader,physical_path,physical_sha,source_sha,ids,models)
            out.update(status='saved_model_physical_diagnostics_complete' if physical['status'] == 'physical_calibration_recorded' else 'saved_partial_physical_diagnostics',
                physical_rows=rows,comparisons=compare(models,rows),physical_counts=physical['counts'],physical_errors=physical['errors'],
                physical_journal_events=events,png_frames_checked=pngs,code_protocol_explanations=explanations(),
                latent_prediction_error='not_computable_without_actual_encoded_latents; Q and physical units not subtracted')
            if selected_path is not None:
                out['selected_rgb']=selected_arrays(reader,selected_path,selected_sha,planner,planner_path,planner_sha,physical,physical_path,physical_sha,models)
                out['latent_prediction_error']='independently recomputed own predicted_vs_real_mse at step5/10; no context intervention'
                out['code_protocol_explanations'][3]['result']='saved own step5/10 latent prediction MSE now measured; open-loop suffix did not consume observed step5; not a causal test'
                out['code_protocol_explanations'][4]['result']='actual Q and native/object metrics independently available; differences do not imply ranking reversal or unique cause'
            reader.finish()
    except Exception as exc:
        out['status']='unverified_or_inconsistent';out['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
        # Never retain apparently usable comparisons after a failed seal/check.
        for key in ('physical_rows','comparisons','code_protocol_explanations','selected_rgb'):out.pop(key,None)
    out['input_bindings_sha256']=reader.bindings
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-report','source-sha','planner-report','planner-sha'):parser.add_argument('--'+name,required=True)
    for name in ('physical-report','physical-sha','selected-report','selected-sha','output'):parser.add_argument('--'+name)
    a=parser.parse_args();r=run_analysis(a.source_report,a.source_sha,a.planner_report,a.planner_sha,a.physical_report,a.physical_sha,
        selected_path=a.selected_report,selected_sha=a.selected_sha)
    r['analysis_source_bindings']={str(p):sha(p.read_bytes()) for p in (Path(__file__).resolve(),PROJECT/'tests'/('test_'+NAME+'.py'),PROJECT/'innerdocs'/(NAME+'.md'))}
    b=(canonical(r)+'\n').encode()
    if a.output:
        path=Path(a.output);require(len(b) <= 200_000 and shutil.disk_usage(path.parent).free >= 200*1024**2+len(b),'small exclusive report/200MiB reserve')
        with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    print(b.decode(),end='')
    if r['status'] != 'saved_model_physical_diagnostics_complete':raise SystemExit(2)


if __name__ == '__main__':main()
