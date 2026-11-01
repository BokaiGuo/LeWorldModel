"""Independent, saved-only census of the fixed 246 physical interventions.

No producer import, simulator, model inference, new selection, or P1 admission.
Exact recorded body values do not certify contact solver warm-start caches.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as core

NAME = 'p1_same_prefix_saved_audit_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
ACTUAL = ROOT/'results/p1_same_prefix_physical_main_20261027_v1_main01'
PREPARED = ROOT/'results/p1_same_prefix_physical_main_20261027_v1_prepare01.json'
REPORT_SHA = 'd975ef03f1b7fa540b449a403fec2fa83cf5eb9a89d4dcedbaf6555efe32c4fe'
PREPARE_SHA = 'fe0d4580530cbdd10838b4eb838a202cfff5d91ab6561183cde3292a9fa46018'
ROLES = ('zero', 'bank_greedy', 'two_stage')
COUNTS = {k:dict(attempted=n, completed=n) for k,n in dict(reset=492,step=2460,render=2706).items()}
FLOOR = 200*1024**2


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(v): return hashlib.sha256(core.canonical(v).encode()).hexdigest()


def sha(path):
    p = Path(path); require(p.is_file() and not p.is_symlink(), 'regular nonsymlink input')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2), b''): h.update(b)
    return h.hexdigest()


class Reader:
    def __init__(self): self.bindings = {}
    def bind(self, path, expected=None):
        p = Path(path).absolute(); h = sha(p)
        require(expected is None or expected == h, 'input SHA mismatch: '+str(p))
        require(str(p) not in self.bindings or self.bindings[str(p)] == h, 'input drift')
        self.bindings[str(p)] = h
        return p
    def read(self, path, expected=None):
        p = self.bind(path, expected); require(p.stat().st_size < 32*1024**2, 'JSON size cap')
        data = json.loads(p.read_text()); require(sha(p) == self.bindings[str(p)], 'read drift')
        return data
    def recheck(self):
        for p,h in self.bindings.items(): require(sha(p) == h, 'final binding drift')


def vec(v, n=7):
    a = np.asarray(v); require(a.shape == (n,) and a.dtype.kind in 'fi' and np.isfinite(a).all(), 'finite vector')
    return a.astype(float)


def delta(a,b):
    d = vec(a)-vec(b); d[4] = np.arctan2(np.sin(d[4]), np.cos(d[4])); return d


def metrics(goal, state):
    """Reproduce native quirks verbatim; wrapped/object diagnostics stay separate."""
    d = vec(state)-vec(goal); absolute = float(abs(d[4]))
    native_angle = float(min(absolute, 2*np.pi-absolute))
    wrapped = float(abs(np.arctan2(np.sin(d[4]), np.cos(d[4]))))
    allxy = float(np.linalg.norm(d[:4])); native = bool(allxy < 20 and native_angle < np.pi/9)
    wd = d.copy(); wd[4] = float(np.arctan2(np.sin(d[4]), np.cos(d[4])))
    return dict(native_formula=dict(success=native, raw_angle_difference=absolute, angle_term=native_angle,
        state_distance=float(np.linalg.norm(d)), negative_angle_term=native_angle < 0),
        independent=dict(wrapped_angle_error=wrapped, all_xy_position_error=allxy,
            object_position_error=float(np.linalg.norm(d[2:4])), agent_position_error=float(np.linalg.norm(d[:2])),
            wrapped_state_delta=wd.tolist(), wrapped_state_distance=float(np.linalg.norm(wd)),
            same_allxy_threshold_wrapped_success=bool(allxy < 20 and wrapped < np.pi/9),
            label='diagnostic_only_not_official_or_admission'),
        native_vs_wrapped_success_disagreement=native != bool(allxy < 20 and wrapped < np.pi/9))


def official(d): return {k:d['native_formula'][k] for k in ('success','state_distance')}


def qcost(task, latent):
    r = np.asarray(task['projection'])@(vec(latent,192)-np.asarray(task['goal']))
    return float(np.mean(r*r))


def probes():
    values = [dict(index=0,coordinate=None,radius=0.,sign=0,logits=[0.]*10)]
    for radius in (.25,.5):
        for coordinate in range(10):
            for sign in (-1,1):
                x = np.zeros(10,np.float32); x[coordinate] = radius*sign
                values.append(dict(index=len(values),coordinate=coordinate,radius=radius,sign=sign,logits=x.tolist()))
    return values


def model_roles(task, saved):
    """Only task and saved model arrays accepted, not physical rows/analysis."""
    core.validate_task(task)
    require(saved['task_id'] == core.task_id(task) and saved['status'] == 'witness_completed_not_admitted'
            and saved['error'] is None and saved['secondary_errors'] == [], 'same complete witness task')
    cs = saved['candidates']; require(len(cs) == 64, 'all64 original model candidates')
    costs = []
    for i,c in enumerate(cs):
        require((c['index'],c['prefix_index'],c['suffix_index']) == (i,i//4,i%4), 'ordered model candidates')
        core.check_prediction(task,dict(task_id=core.task_id(task),logits=c['logits'],commands=c['commands'],
            contexts=c['contexts'],whole_replay=c['contexts'][-1],segmented_replay=c['contexts'],cost=c['cost']))
        require(c['cost'] == qcost(task,c['contexts'][-1]['latent_history'][-1]), 'original H2 Q')
        first = cs[4*(i//4)]
        require(c['commands'][:5] == first['commands'][:5] and c['contexts'][0] == first['contexts'][0], 'shared model prefix')
        if i%4 == 0: costs.append(qcost(task,c['contexts'][0]['latent_history'][-1]))
    best = min(cs,key=lambda c:(c['cost'],c['index']))
    require(saved['winner'] == best, 'stable original all64 H2 winner')
    gi = min(range(16),key=lambda i:(costs[i],i)); wi = best['prefix_index']
    return [dict(role='zero',bank_index=None,prefix=np.zeros((5,2)).tolist(),model_Q5=None),
        *[dict(role=role,bank_index=i,prefix=cs[4*i]['commands'][:5],model_Q5=costs[i])
          for role,i in (('bank_greedy',gi),('two_stage',wi))]]


def check_plan(plan, old, role, probe, task):
    commands = np.concatenate((np.asarray(role['prefix'],np.float32),np.tanh(np.asarray(probe['logits'],np.float32)).reshape(5,2)))
    require(np.linalg.norm(np.asarray(probe['logits'])) <= task['packet_logit_cap'], 'original suffix ball')
    expected = deepcopy(old)
    expected.update(arm=role['role']+'_probe_'+str(probe['index']),role=role['role'],bank_index=role['bank_index'],
        probe=probe,commands=commands.astype(float).tolist(),prediction_sha256=None,
        origin='fixed_intervention_not_a_predicted_H2_winner',parent_selection_sha256=old['selection_sha256'],
        search_cost=dict(new_neural_calls=0,inherited_search_separate=True))
    expected.pop('selection_sha256'); expected['selection_sha256'] = digest(expected)
    require(plan == expected, 'exact source-bound intervention plan')


def check_roles(saved, computed):
    require(saved == computed, 'only saved Q5 greedy/original H2 prefix roles')


def dynamics(snapshot, state):
    require(set(snapshot) == {'agent','block'}, 'both dynamic bodies')
    keys = set('position velocity unwrapped_angle angular_velocity force torque is_sleeping'.split())
    for body in snapshot.values():
        require(set(body) == keys and type(body['is_sleeping']) is bool, 'complete dynamic fields')
        for k in ('position','velocity','force'): vec(body[k],2)
        for k in ('unwrapped_angle','angular_velocity','torque'):
            require(type(body[k]) in (int,float) and np.isfinite(body[k]), 'finite dynamic scalar')
    a,b = snapshot['agent'],snapshot['block']; x = vec(state)
    obs = a['position']+b['position']+[b['unwrapped_angle']%(2*np.pi)]+a['velocity']
    require(np.array_equal(obs,x), 'dynamic body/native state7 binding')


def check_row(plan, row, packet):
    require(all(row[k] == plan[k] for k in ('arm','root_sha256','selection_sha256')), 'stable branch identity')
    require(row['status'] == 'completed' and row['full_horizon'] is True and row['endpoint_scope'] == 'full_H2'
        and row['first_error'] is None and row['secondary_errors'] == [] and row['endpoint_rgb_fresh'] is True, 'full branch only')
    require(row['initial_exact'] is True and row['initial_checks'] == dict.fromkeys(('goal','goal_rgb','rgb','root','state','state_at_render'),True), 'initial exact checks')
    require(row['attempted_steps'] == row['returned_steps'] == len(row['trace']) == 10, 'all10 actual commands')
    require([c['step'] for c in row['checkpoints']] == [0,5,10] and set(row['body_dynamics']) == {'0','5','10'}, 'all three checkpoints')
    goal = packet['goal']; initial = packet['state'].tolist()
    require(row['checkpoints'][0]['state'] == initial and row['initial_pose_diagnostics'] == metrics(goal,initial), 'source initial state/pose')
    for i,t in enumerate(row['trace'],1):
        require(t['step'] == i and t['command'] == plan['commands'][i-1] and t['attempted'] is True
            and t['returned'] is True and t['terminated'] is False and t['truncated'] is False, 'exact ordered nonterminated dispatch')
        x = vec(t['actual_state']); d = metrics(goal,x)
        require(t['observation'] == dict(state=x.tolist(),proprio=x[[0,1,5,6]].tolist()), 'returned/native state')
        info = t['official_info']
        require(info['goal_state'] == goal.tolist() and info['pos_agent'] == x[:2].tolist()
            and info['vel_agent'] == x[5:].tolist() and info['block_pose'][:2] == x[2:4].tolist()
            and info['block_pose'][2]%(2*np.pi) == x[4], 'native info/goal/body binding')
        require(t['pose_diagnostics'] == d and t['native_official_evaluation'] == official(d)
            and t['reward'] == -d['native_formula']['state_distance'], 'original native reward/eval and object pose')
        risk = dict(native_vs_wrapped_success_disagreement=d['native_vs_wrapped_success_disagreement'],
            negative_native_angle_term=d['native_formula']['negative_angle_term'], native_eval_vs_formula_disagreement=False,
            terminated_vs_eval_disagreement=d['native_formula']['success'], policy='honor_returned_terminated_truncated_no_override')
        require(t['termination_protocol_risk'] == risk, 'native termination diagnostics preserved')
    end = row['trace'][-1]; require(all(row[k] == end[v] for k,v in
        (('final_state','actual_state'),('official_info','official_info'),('pose_diagnostics','pose_diagnostics'),('official_evaluation','native_official_evaluation'))), 'final/trace binding')
    wd = vec(row['final_state'])-goal; wd[4] = (wd[4]+np.pi)%(2*np.pi)-np.pi
    require(row['wrapped_state_delta'] == wd.tolist(), 'original final wrapped delta')
    for c in row['checkpoints']:
        step = c['step']; require(c['state'] == (initial if step == 0 else row['trace'][step-1]['actual_state']), 'checkpoint/trace')
        dynamics(row['body_dynamics'][str(step)],c['state'])
    g = row['guard_receipt']; require(g['root_id'] == plan['root_sha256'] and g['abstained'] is False
        and g['P1_admission'] is False and g['primitive_commands_authorized'] == 10
        and [o['primitive_step'] for o in g['observations']] == list(range(11))
        and all(type(o['canonical_agent_pixels']) is int and o['canonical_agent_pixels'] > 0 for o in g['observations']), 'fresh visible guard chronology')


def responses(plans, rows, goals):
    require(len(plans) == len(rows) == 246, 'all246, no partial mixing')
    results, extra = [], []
    for start in range(0,246,41):
        ps,rs = plans[start:start+41],rows[start:start+41]; center = rs[0]; root = ps[0]['root_sha256']
        require([p['probe'] for p in ps] == probes() and len({p['role'] for p in ps}) == 1, 'complete role probe census')
        common = rows[(start//123)*123]; own = delta(goals[root],center['checkpoints'][1]['state'])
        shared = delta(goals[root],common['checkpoints'][1]['state']); items = []; objects = []
        for p,r in zip(ps,rs):
            require(r['status'] == 'completed' and r['full_horizon'] is True, 'full/partial must stay separate')
            require(all(r['checkpoints'][1][k] == center['checkpoints'][1][k] for k in ('step','state','rgb_sha256'))
                and r['body_dynamics']['5'] == center['body_dynamics']['5'], 'exact recorded prefix state/RGB/dynamics')
            d = delta(r['final_state'],center['final_state'])
            items.append(dict(probe_index=p['probe']['index'],probe=p['probe'],endpoint_delta_state7=d.tolist(),
                own_position_direction_response=float(own[:4]@d[:4]/max(np.linalg.norm(own[:4]),1e-12)),
                common_position_direction_response=float(shared[:4]@d[:4]/max(np.linalg.norm(shared[:4]),1e-12)),
                native_distance_change=float(r['official_evaluation']['state_distance']-center['official_evaluation']['state_distance']),
                wrapped_object_angle_change=float(r['pose_diagnostics']['independent']['wrapped_angle_error']-center['pose_diagnostics']['independent']['wrapped_angle_error'])))
            objects.append(dict(probe_index=p['probe']['index'],object_xy_delta=d[2:4].tolist(),
                object_position_error_change=float(np.linalg.norm(vec(r['final_state'])[2:4]-vec(goals[root])[2:4])-
                    np.linalg.norm(vec(center['final_state'])[2:4]-vec(goals[root])[2:4])),wrapped_object_angle_delta=float(d[4])))
        results.append(dict(root_sha256=root,role=ps[0]['role'],planned=41,completed=41,
            native_successes=sum(r['official_evaluation']['success'] for r in rs),centered_responses=items,unavailable_probe_indices=[]))
        extra.append(dict(root_sha256=root,role=ps[0]['role'],object_responses=objects))
    return dict(groups=results,P1_admission=False,fitted_controller=False,
        scope='same roots fixed interventions; directional responses are diagnostics, no posthoc action selection'),extra


class Events:
    """Stream exact hash chain and consume exact expected event payloads."""
    def __init__(self, stream): self.stream=iter(stream); self.index=0; self.previous=None
    def take(self, kind, **want):
        try: line = next(self.stream)
        except StopIteration as e: raise ValueError('missing journal event') from e
        require(len(line) < 128*1024, 'bounded journal event')
        v = json.loads(line); h = v.pop('event_sha256')
        require(v.pop('index') == self.index and v.pop('previous_sha256') == self.previous, 'journal chronology')
        sealed = dict(v,index=self.index,previous_sha256=self.previous)
        require(digest(sealed) == h, 'journal event hash')
        require(v.get('kind') == kind and all(v.get(k) == value for k,value in want.items()), 'journal payload '+kind)
        self.previous=h; self.index+=1
        return {k:x for k,x in v.items() if k != 'kind'}
    def exact(self, kind, **want): require(self.take(kind,**want) == want, 'exact journal fields '+kind)
    def finish(self): require(next(self.stream,None) is None, 'no trailing events')


def journal(stream, p, report, packets):
    e = Events(stream); counts = {k:dict(attempted=0,completed=0) for k in COUNTS}
    e.exact('run_started',prepare_sha256=PREPARE_SHA,contract=p['contract'])
    for plan,row in zip(p['inputs']['plans'],report['rows'],strict=True):
        sid = plan['selection_sha256']; root=plan['root_sha256']; packet=packets[root]
        def event(kind, **v): return e.exact(kind,selection_sha256=sid,**v)
        def call(kind):
            counts[kind]['attempted']+=1; event(kind+'_attempted',attempt=counts[kind]['attempted'])
            counts[kind]['completed']+=1; event(kind+'_returned',completed=counts[kind]['completed'])
        def capture(step):
            call('render'); x = packet['state'].tolist() if step==0 else row['trace'][step-1]['actual_state']
            if step in (0,5,10):
                c = row['checkpoints'][(0,5,10).index(step)]
                event('fresh_rgb_observed',receipt={k:c[k] for k in ('step','state','rgb_sha256')},body_dynamics=row['body_dynamics'][str(step)])
                event('checkpoint_saved',receipt=c)
            else:
                observed=e.take('fresh_rgb_observed',selection_sha256=sid)
                require(set(observed)=={'receipt','selection_sha256'} and set(observed['receipt'])=={'step','state','rgb_sha256'}
                    and observed['receipt']['step']==step and observed['receipt']['state']==x, 'fresh intermediate state')
                core.hash_value(observed['receipt']['rgb_sha256'])
            g=deepcopy(row['guard_receipt']); g['observations']=g['observations'][:step+1];g['primitive_commands_authorized']=step
            event('rgb_guard_observed',step=step,receipt=g)
        event('branch_started',plan=plan); meta=plan['source_metadata']; selected=meta['selected_attempt']
        sp=p['inputs']['source_plan']; slot=next(s for s in sp['slots'] if s['candidate_id']==plan['candidate_id'])
        seed=slot['reset_seed_span'][0]+meta['source']['proposal_index']*sp['reset_attempts_per_proposal']+selected['attempt_index']
        require(seed==selected['random_reset_seed'] and selected['placement_reset_seed']==sp['placement_reset_seed'], 'original reset recipe seed')
        call('reset'); initial=e.take('reset_observation',selection_sha256=sid,seed=seed,options=None)
        require(initial['info']['goal_state']==packet['goal'].tolist(), 'initial random-reset goal')
        request=e.take('original_placement_request',selection_sha256=sid,goal=packet['goal'].tolist())
        vec(request['state']); call('reset')
        placed=e.take('reset_observation',selection_sha256=sid,seed=sp['placement_reset_seed'],
            options=dict(state=request['state'],goal_state=packet['goal'].tolist()))
        require(placed['observation']['state']==packet['state'].tolist() and placed['info']['goal_state']==packet['goal'].tolist(), 'actual original placement')
        capture(0);event('initial_identity',checks=row['initial_checks']);event('initial_pose_diagnostics',receipt=row['initial_pose_diagnostics'])
        for t in row['trace']:
            step=t['step']; counts['step']['attempted']+=1
            event('step_attempted',receipt=dict(step=step,command=t['command'],attempted=True,returned=False))
            counts['step']['completed']+=1;event('step_returned',step=step)
            event('termination_returned',step=step,reward=t['reward'],terminated=t['terminated'],truncated=t['truncated'])
            event('returned_observation',step=step,observation=t['observation']);event('actual_successor',step=step,state=t['actual_state'])
            event('official_info_returned',step=step,info=t['official_info'])
            event('successor_evaluation',step=step,native=t['native_official_evaluation'],diagnostics=t['pose_diagnostics'])
            event('step_observation',receipt=t);capture(step)
        finished={k:v for k,v in row.items() if k!='body_dynamics'}
        event('branch_finished',receipt=finished);e.exact('branch_closed',receipt=row)
    require(counts==report['counts']==COUNTS, 'independent complete public calls')
    e.exact('run_finished',status='same_prefix_interventions_completed',counts=counts,errors=[]);e.finish()
    require(e.index==34196, 'all34196 events')
    return dict(events=e.index,chain_tail=e.previous,counts=counts)


def load_inputs(reader):
    p=reader.read(PREPARED,PREPARE_SHA);report=reader.read(ACTUAL/'report.json',REPORT_SHA)
    require(p['schema']=='p1_same_prefix_physical_main_20261027_v1.prepare' and report['schema']=='p1_same_prefix_physical_main_20261027_v1.report', 'source schemas')
    require(report['status']=='same_prefix_interventions_completed' and report['errors']==[] and report['prepare_sha256']==PREPARE_SHA
        and report['counts']==COUNTS and report['contract']==p['contract'], 'actual complete fixed protocol')
    require(report['new_neural_calls']==report['new_independent_samples']==0 and report['P1_admission'] is False, 'no new model/independence/admission')
    c=p['contract']; require(c['roles']==list(ROLES) and c['branches']==246 and c['suffixes']==41 and c['radii']==[.25,.5]
        and c['coordinates']==10 and c['signs']==[-1,1] and c['png_cap']==738 and c['neural_calls']==0
        and c['P1_admission'] is False and c['output_cap_bytes']==64*1024**2 and c['floor_bytes']==FLOOR, 'fixed operation/domain contract')
    for f,h in {**p['sources'],**p['inputs']['bindings'],**p['inputs']['source_plan']['source_closure']}.items(): reader.bind(f,h)
    parent=reader.read(ROOT/'results/p1_witness_physical_20261027_v1_prepared01/prepare.json')
    args=parent['inputs']['arguments']; wp=Path(args['witness_prepared'])
    receipt=reader.read(wp/'prepare.json',args['witness_prepare_sha']); protocol=reader.read(wp/'protocol.json',receipt['protocol_sha256'])
    for f,h in {**protocol['sources_sha256'],**protocol['inputs']['bindings'],**protocol['inputs']['model_bindings']}.items(): reader.bind(f,h)
    tasks={v['root_sha256']:v['task'] for v in protocol['inputs']['tasks']}
    olds=[v for v in parent['inputs']['plans'] if v['arm']=='two_stage']
    require(len(olds)==len(tasks)==2 and {x['root_sha256'] for x in olds}==set(tasks)
        and len(p['inputs']['plans'])==len(report['rows'])==246, 'two source roots/all246')
    expected_roles=[];packets={}
    for i,old in enumerate(olds):
        root=old['root_sha256'];task=tasks[root];saved=reader.read(old['origin'])
        require(old['task_id']==core.task_id(task), 'original task binding')
        roles=model_roles(task,saved)
        for j,role in enumerate(roles):
            expected_roles.append(dict(root_sha256=root,task_id=core.task_id(task),**role))
            for k,probe in enumerate(probes()): check_plan(p['inputs']['plans'][i*123+j*41+k],old,role,probe,task)
        f=reader.bind(old['raw_source_file'],old['raw_source_sha256'])
        with np.load(f,allow_pickle=False) as z: packet={k:z[k].copy() for k in z.files}
        metadata=json.loads(str(packet.pop('metadata_json')))
        require(metadata==old['source_metadata'] and metadata['plan_id']==digest(p['inputs']['source_plan']), 'raw source metadata')
        actualroot=hashlib.sha256(np.ascontiguousarray(np.concatenate((vec(packet['state']),vec(packet['goal']))),dtype=np.float64).tobytes()).hexdigest()
        require(actualroot==root and np.array_equal(packet['rgb_history'],np.stack([packet['rgb_history'][0]]*3)), 'actual root and explicit causal padding')
        packets[root]=packet
    check_roles(p['inputs']['roles'],expected_roles)
    return p,report,packets


def audit():
    reader=Reader(); p,report,packets=load_inputs(reader); files={'report.json','analysis.json','journal.jsonl'}
    from PIL import Image
    for plan,row in zip(p['inputs']['plans'],report['rows'],strict=True):
        packet=packets[plan['root_sha256']];check_row(plan,row,packet)
        for c in row['checkpoints']:
            name=f"{plan['candidate_id']}_{plan['arm']}_{c['step']}.png";files.add(name)
            require(Path(c['path'])==ACTUAL/name, 'same-run exact PNG path')
            path=reader.bind(c['path'],c['file_sha256'])
            with Image.open(path) as im: image=np.asarray(im)
            require(image.shape==(224,224,3) and image.dtype==np.uint8 and hashlib.sha256(image.tobytes()).hexdigest()==c['rgb_sha256'], 'saved RGB content hash')
            count=int(np.count_nonzero(np.all(image==[65,105,225],axis=-1)))
            require(count==row['guard_receipt']['observations'][c['step']]['canonical_agent_pixels'], 'saved RGB guard evidence')
            if c['step']==0: require(np.array_equal(image,packet['rgb_history'][0]), 'actual source initial RGB')
    require({f.name for f in ACTUAL.iterdir()}==files and len(files)==741, 'complete exclusive artifact census')
    require(sum(f.stat().st_size for f in ACTUAL.iterdir())<=64*1024**2, 'actual complete output cap')
    jpath=reader.bind(ACTUAL/'journal.jsonl',report['journal_sha256'])
    with jpath.open() as f: work=journal(f,p,report,packets)
    analysis,objects=responses(p['inputs']['plans'],report['rows'],{r:x['goal'] for r,x in packets.items()})
    require(analysis==reader.read(ACTUAL/'analysis.json'), 'independent all246 response arithmetic')
    groups=[]
    for a,b in zip(analysis['groups'],objects):
        values=b['object_responses']; changes=[x['object_position_error_change'] for x in values]
        angles=[x['wrapped_object_angle_change'] for x in a['centered_responses']]
        groups.append(dict(root_sha256=a['root_sha256'],role=a['role'],planned=41,fullH2=41,partial=0,
            native_successes=a['native_successes'],object_position_error_change_range=[min(changes),max(changes)],
            wrapped_object_angle_error_change_range=[min(angles),max(angles)],
            native_responses_sha256=digest(a['centered_responses']),object_responses_sha256=digest(values)))
    reader.recheck()
    return dict(schema=NAME+'.report',status='saved_data_audited_not_admitted',actual_report_sha256=REPORT_SHA,
        prepare_sha256=PREPARE_SHA,resources=work,branches=246,PNGs=738,groups=groups,model_roles=p['inputs']['roles'],
        all_responses_sha256=digest(dict(original=analysis,object_only=objects)),bindings_sha256=digest(reader.bindings),
        bound_files=len(reader.bindings),new_neural_calls=0,new_physics_steps=0,P1_admission=False,
        limitations=['two reused padded calibration roots; not independent confirmation',
            'recorded body fields exact; solver warm-start/internal cache identity not certified',
            'intermediate RGBs are receipts, only step0/5/10 pixels retained',
            'native distance mixes agent/object/angle/velocity; object responses separate',
            'finite probe census is not a controller or global controllability certificate',
            'saved trajectory/receipt consistency, not an independent simulator rerun'],
        audit_sources={str(f):sha(f) for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'))})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    require(not a.output.exists(), 'exclusive new report path'); result=audit(); raw=(core.canonical(result)+'\n').encode()
    require(len(raw)<=64*1024 and shutil.disk_usage(a.output.parent).free>=FLOOR+len(raw), 'small report plus200MiB floor')
    with a.output.open('xb') as f: f.write(raw);f.flush();os.fsync(f.fileno())
    print(core.canonical(dict(status=result['status'],output=str(a.output),report_sha256=sha(a.output),bytes=len(raw))))


if __name__=='__main__': main()
