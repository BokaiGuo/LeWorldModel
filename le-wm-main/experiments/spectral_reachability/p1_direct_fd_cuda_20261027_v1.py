"""MAIN-only full192 finite differences of saved direct-H2 zero-logit J.

Prepare/verify are saved-array work. Only explicit run imports torch; no new
images, AD, optimization, physics, fitted pass threshold or scientific admission.
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

from experiments.spectral_reachability import p1_direct_cuda_calibration_main_20261027_v2 as original
from experiments.spectral_reachability import p1_conditional_direct_screen_20261027_v1 as native

NAME = 'p1_direct_fd_cuda_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
SAVED = ROOT/'results/p1_direct_cuda_calibration_main_20261027_v2_cuda_main01'
OLD_PREPARED = ROOT/'results/p1_direct_cuda_calibration_main_20261027_v2_prepared01'
OLD_REPORT_SHA = 'fad2c2ff4f409762fa5925f142f9b4d3edc371645fc4c114bfa69c996483f3d4'
OLD_PREPARE_SHA = '24eaf8b02873b3d93ebeeadbead9ca56c2caa079cea8c30b308357ddd8a3d2e6'
EPS = (1e-3, 1e-2)
CAP = 4*1024**2
FLOOR = 200*1024**2
FINAL_RESERVE = 128*1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular source file')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for raw in iter(lambda: stream.read(1024**2), b''):
            h.update(raw)
    return h.hexdigest()


def read(path, expected=None):
    path = Path(path)
    require(path.stat().st_size < 16*1024**2, 'bounded saved JSON')
    if expected is not None:
        require(sha(path) == expected, 'external/bound source hash: '+str(path))
    return json.loads(path.read_text())


def task_id(task):
    # Original direct-task identity uses stdlib default separators, not ours.
    return hashlib.sha256(json.dumps(task, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array(value, shape):
    a = np.asarray(value)
    require(a.dtype.kind in 'fi' and a.shape == shape and np.isfinite(a).all(), 'finite real array '+str(shape))
    return a.astype(np.float64)


def directions():
    a = np.zeros(20); a[0] = 1
    b = np.zeros(20); b[10] = 1
    random = np.random.Generator(np.random.PCG64(20261027)).standard_normal(20)
    random /= np.linalg.norm(random)
    return [dict(id=name, vector=v.astype(np.float32).astype(float).tolist()) for name, v in (
        ('packet0_e0', a), ('packet1_e0', b), ('equal_packet_e0', (a+b)/np.sqrt(2)), ('seed20261027_unitrandom', random))]


def perturb(vector, epsilon, sign):
    require(epsilon in EPS and sign in (-1, 1), 'fixed epsilon and side')
    v = array(vector, (20,)).astype(np.float32)
    logits = (v*np.float32(epsilon)*np.float32(sign)).reshape(2, 10)
    raw = np.tanh(logits).reshape(10, 2)
    require(raw.dtype == np.float32 and np.isfinite(raw).all() and np.all(np.abs(raw) <= 1), 'NumPy float32 tanh raw commands')
    return dict(logits=logits.astype(float).tolist(), commands=raw.astype(float).tolist(),
                epsilon_requested=epsilon, epsilon_float32=float(np.float32(epsilon)))


def tokens(task, raw):
    a = np.asarray(raw, np.float32).reshape(-1, 10)
    return (a-np.tile(np.asarray(task['action_mean'], np.float32), 5))/np.tile(np.asarray(task['action_std'], np.float32), 5)


def check_path(task, commands, path):
    require(len(path) == 2, 'full wholeH2 only')
    before = task['root_context']
    for i, c in enumerate(path):
        require({k:c[k] for k in ('layout','context_length','latent_dim','action_dim')} ==
            dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10), 'native layout')
        z = array(c['latent_history'], (3,192))
        require(np.array_equal(z[:2], np.asarray(before['latent_history'],np.float32)[1:]), 'causal retained history')
        expected = np.concatenate((np.asarray(before['action_history'],np.float32), tokens(task, commands)[i:i+1]))[-2:]
        require(np.array_equal(array(c['action_history'],(2,10)), expected), 'full normalized token contents/order')
        before = c


def extract_zero(task, component, events):
    """Bind the first GN derivative to raw rows/forward, NOT old check flags."""
    require(events[0]['kind'] == 'direct_started' and events[0]['task'] == task and
        events[0]['configuration'] == component['configuration'], 'original direct-start task/config')
    require(component['configuration']['method'] == 'full_H2_GN' and len(component['direct_runs']) == 1, 'one GN restart')
    run = component['direct_runs'][0]; zero = run[0]['incumbent']; d = run[1]['derivative']
    require(run[0]['iteration'] == 0 and run[0]['derivative'] is None and
        np.array_equal(array(zero['logits'], (2,10)), np.zeros((2,10))) and
        np.array_equal(array(zero['commands'],(10,2)), np.zeros((10,2))), 'actual iteration0 zero logits/commands')
    require(d['task_id'] == zero['task_id'] == component['task_id'] == task_id(task) and
        d['at_logits'] == zero['logits'] and d['at_terminal'] == zero['contexts'][-1]['latent_history'][-1], 'first derivative AT iteration0, not iteration1 proposal')
    check_path(task, zero['commands'], zero['contexts'])
    stop = next(i for i,e in enumerate(events) if e['kind'] == 'joint_logit_jacobian')
    first = events[:stop+1]
    rows = [e for e in first if e['kind'] == 'jacobian_row']
    require([e['row'] for e in rows] == list(range(192)), 'all original192 raw AD rows ordered')
    raw = array([e['raw_control_derivative'] for e in rows], (192,20))
    j = array(d['raw_terminal_jacobian'], (192,20))  # legacy field name; actually logit-J
    require(np.array_equal(raw*(1-np.asarray(zero['commands']).reshape(20)**2)[None], j), 'raw-to-logit tanh chain exactly once')
    require(events[stop]['jacobian'] == j.tolist() and events[stop]['terminal'] == d['at_terminal'], 'original joint logit-J event')
    forwards = [e for e in first if e['kind'] == 'jacobian_forward_complete']
    packets = [e for e in first if e['kind'] == 'jacobian_forward_packet']
    require(len(forwards) == 1 and forwards[0]['terminal'] == d['at_terminal'] and
        forwards[0]['logits'] == zero['logits'], 'original AD forward point')
    require([e['packet'] for e in packets] == [0,1] and [e['context'] for e in packets] == zero['contexts'], 'both AD forward packets bind original zero rollout')
    require(next(e['derivative'] for e in events if e['kind']=='derivative_complete') == d, 'raw derivative/report identity')
    return dict(logit_jacobian=j.tolist(), nominal_terminal=array(d['at_terminal'],(192,)).tolist(),
        derivative_location='direct_runs[0][1].derivative, evaluated at direct_runs[0][0] zero incumbent',
        raw_rows=192, original_ad_forward_packets=2, chain_at_zero_is_identity=True)


def material():
    bindings = {}
    def bound(path, expected=None):
        value = read(path, expected); bindings[str(path)] = sha(path); return value
    top = bound(SAVED/'report.json', OLD_REPORT_SHA)
    pr = bound(OLD_PREPARED/'prepare.json', OLD_PREPARE_SHA)
    p = bound(OLD_PREPARED/'protocol.json', pr['protocol_sha256'])
    require(top['prepare_sha256'] == OLD_PREPARE_SHA and top['status'] == 'direct_calibration_completed' and
        top['errors'] == [] and top['model_hash_before'] == top['model_hash_after'] and top['model_hash_before'] is not None,
        'successful source direct execution binding, not scientific admission')
    # Independently hash original derivative implementation, source input bytes
    # and frozen model. No checker/auditor result is consumed.
    closure = {**p['sources_sha256'], **p['source_plan']['source_closure'], **p['bindings']}
    for path, h in closure.items():
        require(sha(path) == h, 'original derivative/source path drift: '+path)
    require(str(Path(native.__file__).resolve()) in p['sources_sha256'], 'actual original NativeCuda implementation is bound')
    for path,h in p['model_bindings'].items():
        require(sha(path) == h, 'original model bytes')
    images_path = SAVED/'image.events.jsonl'; bindings[str(images_path)] = sha(images_path)
    images = [json.loads(x) for x in images_path.read_bytes().splitlines()]
    encoded = [e for e in images if e['kind']=='native_image_completed']
    require(len(encoded) == 8, 'original eight native context/goal images')
    roots = []
    for i in range(2):
        e = bound(SAVED/f'task_{i}.json'); task = e['task']
        require(task_id(task) == e['task_id'] and e['history_semantics'] == p['source_history'] == 'causal_padding_not_observed_transitions', 'actual new direct task identity/history caveat')
        require(task['horizon'] == 2 and task['packet_steps'] == 5 and task['control_dim'] == 2 and
            task['latent_dim'] == 192 and task['representation'] == 'native_projected_cls', 'full native H2 task')
        require(task['action_mean'] == p['normalization']['mean'] and task['action_std'] == p['normalization']['std'] and
            np.all(array(task['action_std'],(2,)) > 0), 'same source normalizer')
        require(task['root_context']['latent_history']+[task['goal']] == [x['latent'] for x in encoded[4*i:4*i+4]] and
            task['goal_source_sha256'] == encoded[4*i+3]['rgb_sha256'], 'original saved encoder context and independent goal')
        require(np.array_equal(tokens(task,task['root_raw_action_history']),task['root_context']['action_history']), 'original normalized action history')
        for filename, field in (('weights.pt','model_sha256'),('config.json','model_config_sha256')):
            require(task[field] == next(h for path,h in p['model_bindings'].items() if Path(path).name==filename), 'task frozen model identity')
        reference = next(x for x in top['direct'] if x['root_sha256']==e['root_sha256'] and x['method']=='full_H2_GN')
        directory = SAVED/f'root_{i}_full_H2_GN'
        require(reference['report_path'] == str(directory/'report.json'), 'own GN path')
        component = bound(directory/'report.json', reference['report_sha256'])
        journal = directory/'direct_journal.jsonl'; require(journal.stat().st_size < 12*1024**2, 'bounded original journal')
        bindings[str(journal)] = sha(journal)
        events = [json.loads(x) for x in journal.read_bytes().splitlines()]
        witness = extract_zero(task, component, events)
        roots.append(dict(root_index=i, root_sha256=e['root_sha256'], task_id=e['task_id'], task=task,
                          original_component=str(directory/'report.json'), original_journal=str(journal), **witness))
    require(len({r['root_sha256'] for r in roots}) == 2, 'two distinct NEW direct roots, not old four calibration roots')
    sources = set(original.io.source_files()+[Path(__file__).resolve(),Path(original.__file__).resolve(),Path(native.__file__).resolve(),
        Path(native.core.__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')])
    return dict(schema=NAME+'.protocol', roots=roots, directions=directions(), epsilons=list(EPS), signs=[-1,1],
        evaluation_order='root,direction,epsilon,sign; wholeH2 packets0,1 without boundary reload',
        policy=p['policy'], model_bindings=p['model_bindings'], expected_model_tensor_sha256=top['model_hash_before'],
        input_bindings=bindings, original_derivative_source_bindings=closure,
        sources_sha256={str(f):sha(f) for f in sources}, predict_cap=64, action_encoder_cap=64,
        eval_count=32, image_cap=0, AD_cap=0, physics_cap=0, output_cap_bytes=CAP,disk_floor_bytes=FLOOR,
        inherited_direct_resources=top['actual_direct_resources'], inherited_image_resources=top['actual_image_resources'],
        relative_error_denominator='norm(savedJ@stored_float32_direction); null if zero, no fitted numerical floor',
        scientific_pass_threshold=None, new_Q=False,training=False,P1_admission=False,
        scope='calibration derivative/chain diagnostic only; no physical, control, or P1 effectiveness claim')


class Storage:
    def __init__(self, folder):
        self.folder = Path(folder).absolute()
        require(not self.folder.exists() and shutil.disk_usage(self.folder.parent).free > FLOOR+CAP, 'exclusive output + 4MiB +200MiB reserve')
        self.folder.mkdir(); self.written = 0

    def put(self, file, value, final=False):
        raw = canonical(value)
        self._check(len(raw), final)
        with (self.folder/file).open('xb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        self.written += len(raw)

    def _check(self, n, final=False):
        require(self.written+n <= CAP-(0 if final else FINAL_RESERVE), 'whole-output byte cap incl final reserve')
        require(shutil.disk_usage(self.folder).free >= FLOOR+n, '200MiB floor')

    def event(self, event):
        raw = canonical(event); self._check(len(raw))
        with (self.folder/'fd.events.jsonl').open('ab') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        self.written += len(raw)


def prepare(output):
    p = material(); storage = Storage(output)
    storage.put('protocol.json',p)
    storage.put('prepare.json',dict(schema=NAME+'.prepared',protocol_sha256=sha(storage.folder/'protocol.json'),
        status='saved_inputs_bound_not_executed',new_neural_calls=0,new_physical_steps=0))
    return dict(prepare_sha256=sha(storage.folder/'prepare.json'), evaluations=32, predict_cap=64,action_encoder_cap=64)


def verify(folder, expected):
    folder = Path(folder).absolute(); r = read(folder/'prepare.json', expected)
    p = read(folder/'protocol.json',r['protocol_sha256'])
    require(p == material(), 'full task/J/raw derivative/source reconstruction')
    return p


def error_metrics(fd, reference):
    f, j = array(fd,(192,)), array(reference,(192,)); d = f-j
    n, fn = float(np.linalg.norm(j)), float(np.linalg.norm(f))
    return dict(absolute_l2=float(np.linalg.norm(d)), absolute_linf=float(np.max(np.abs(d))),
        absolute_rmse=float(np.sqrt(np.mean(d*d))), saved_jv_l2=n, fd_l2=fn,
        relative_l2=float(np.linalg.norm(d)/n) if n else None,
        symmetric_relative_l2=float(np.linalg.norm(d)/max(n,fn)) if max(n,fn) else None,
        cosine=float(np.dot(f,j)/(fn*n)) if fn and n else None)


def compare(j, nominal, vector, epsilon, minus, plus):
    j = array(j,(192,20)); v = array(vector,(20,)); m, p = array(minus,(192,)),array(plus,(192,))
    fd = (p-m)/(2*float(np.float32(epsilon))); expected = j@v
    return dict(epsilon_requested=epsilon,epsilon_float32=float(np.float32(epsilon)),
        direction_norm_float32=float(np.linalg.norm(v)), fd=fd.tolist(),saved_jv=expected.tolist(),
        midpoint_drift_from_saved_nominal_l2=float(np.linalg.norm((p+m)/2-array(nominal,(192,)))),
        **error_metrics(fd,expected))


def analyze(protocol, evaluations):
    require(len(evaluations)==32, 'complete census for final FD comparison')
    expected_keys=[(r['root_index'],d['id'],eps,sign) for r in protocol['roots'] for d in protocol['directions'] for eps in EPS for sign in (-1,1)]
    require([(e['root_index'],e['direction'],e['epsilon_requested'],e['sign']) for e in evaluations] == expected_keys, 'all ordered signs/roots/directions/epsilon bindings')
    by_key={k:e for k,e in zip(expected_keys,evaluations)}; rows=[]
    for r in protocol['roots']:
        for direction in protocol['directions']:
            results=[]
            for eps in EPS:
                sides=[]
                for sign in (-1,1):
                    e=by_key[r['root_index'],direction['id'],eps,sign]
                    wanted=perturb(direction['vector'],eps,sign)
                    require(e['task_id']==r['task_id'] and e['root_sha256']==r['root_sha256'] and
                        e['order']==expected_keys.index((r['root_index'],direction['id'],eps,sign)) and
                        all(e[k]==v for k,v in wanted.items()), 'actual task/root/order/float32 perturbation binding')
                    check_path(r['task'], e['commands'],e['contexts'])
                    sides.append(e['contexts'][-1]['latent_history'][-1])
                results.append(compare(r['logit_jacobian'],r['nominal_terminal'],direction['vector'],eps,*sides))
            stability=error_metrics(np.asarray(results[0]['fd'])-results[1]['fd'],np.zeros(192))
            n=results[0]['saved_jv_l2']
            rows.append(dict(root_index=r['root_index'],root_sha256=r['root_sha256'],task_id=r['task_id'],direction=direction['id'],
                epsilons=results,epsilon_stability_l2=stability['absolute_l2'],
                epsilon_stability_relative_to_saved_jv=stability['absolute_l2']/n if n else None))
    return dict(schema=NAME+'.analysis',rows=rows,comparisons=16,epsilon_stability_pairs=8,
        scientific_pass_threshold=None,scientific_verdict='not_assigned',P1_admission=False,new_Q=False)


class Meter:
    """NativeCuda-compatible dispatch meter:64 of EACH module, no retry refund."""
    def __init__(self, storage, sync=lambda:None, no_grad=lambda:True):
        self.storage,self.sync,self.no_grad=storage,sync,no_grad
        self.events=[]; self.evaluation=None; self.packet=0; self.paths=[]

    def start(self, evaluation):
        require(self.evaluation is None or self.packet==2, 'previous wholeH2 incomplete')
        require(len(self.events)==evaluation['order']*4 and all(e['completed'] for e in self.events), 'no retry after partial dispatch')
        self.evaluation=evaluation;self.packet=0;self.paths=[]
        self.storage.event(dict(kind='evaluation_attempt',evaluation=deepcopy(evaluation)))

    def begin(self, resource):
        require(self.evaluation is not None and self.packet<2 and self.no_grad(), 'only no-grad wholeH2 dispatch')
        offset=len(self.events)-self.evaluation['order']*4
        require(resource==('action_encoder','predict')[offset%2] and offset//2==self.packet and
            all(e['completed'] for e in self.events), 'exact module/packet order')
        require(sum(e['resource']==resource for e in self.events)<64,'predispatch hard cap')
        e=dict(index=len(self.events),resource=resource,samples=1,completed=False,
            evaluation_order=self.evaluation['order'],packet=self.packet)
        self.events.append(e);self.storage.event(dict(kind='module_attempt',entry=deepcopy(e)))
        return e

    def end(self, token):
        require(token is self.events[-1] and not token['completed'],'matched unique return')
        self.sync();token['completed']=True
        self.storage.event(dict(kind='module_return',entry=deepcopy(token)))

    def emit(self, kind, **value):
        require(kind=='native_packet' and value['mode']=='whole_replay' and value['packet']==self.packet and
            len(self.events)==self.evaluation['order']*4+2*(self.packet+1) and all(e['completed'] for e in self.events), 'durable native packet chronology')
        self.storage.event(dict(kind='native_packet',evaluation_order=self.evaluation['order'],**deepcopy(value)))
        self.paths.append(deepcopy(value['context']));self.packet+=1

    def counts(self):
        return {k:dict(attempted=sum(e['resource']==k for e in self.events),
            completed=sum(e['resource']==k and e['completed'] for e in self.events)) for k in ('predict','action_encoder')}


def failure(stage, exc):
    return dict(stage=stage,type=type(exc).__name__,message=str(exc)[:1500],
                notes=[str(x)[:1500] for x in getattr(exc,'__notes__',[])])


def run(folder, expected, output):
    p=verify(folder,expected); storage=Storage(output); rows=[];errors=[];model=None;before=after=None;source_ok=False;host=None
    storage.put('execution_binding.json',dict(prepared=str(Path(folder).absolute()),prepare_sha256=expected))
    meter=Meter(storage);analysis_sha=None
    try:
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0');torch.backends.cuda.matmul.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        original.io.frozen(model);before=original.io.tensor_hash(model)
        require(before==p['expected_model_tensor_sha256'],'same original direct tensor state')
        meter.sync=lambda:torch.cuda.synchronize(device);meter.no_grad=lambda:not torch.is_grad_enabled()
        with sdpa_kernel(SDPBackend.MATH):
            for r in p['roots']:
                backend=native.NativeCuda(model,device,r['task'])
                with backend.guard(meter):
                    for d in p['directions']:
                        for eps in EPS:
                            for sign in (-1,1):
                                e=dict(order=len(rows),root_index=r['root_index'],root_sha256=r['root_sha256'],task_id=r['task_id'],
                                    direction=d['id'],sign=sign,**perturb(d['vector'],eps,sign))
                                meter.start(e)
                                contexts=backend.rollout(r['task'],np.asarray(e['commands'],np.float32),meter,'whole_replay')
                                require(contexts==meter.paths,'durable packet/path identity')
                                check_path(r['task'],e['commands'],contexts)
                                result=dict(**e,contexts=contexts)
                                storage.put(f'eval_{e["order"]:02d}.json',result)
                                rows.append(result);storage.event(dict(kind='evaluation_saved',order=e['order'],file=f'eval_{e["order"]:02d}.json',sha256=sha(storage.folder/f'eval_{e["order"]:02d}.json')))
        result=analyze(p,rows);storage.put('analysis.json',result);analysis_sha=sha(storage.folder/'analysis.json')
    except BaseException as exc:
        errors.append(failure('execution',exc))
        try:storage.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary:errors.append(failure('error_journal',secondary))
    finally:
        if model is not None:
            try:original.io.frozen(model);after=original.io.tensor_hash(model)
            except BaseException as exc:errors.append(failure('final_model',exc))
        try:verify(folder,expected);source_ok=True
        except BaseException as exc:errors.append(failure('final_source',exc))
        counts=meter.counts()
        complete=not errors and len(rows)==32 and counts=={k:dict(attempted=64,completed=64) for k in counts} and before is not None and before==after and source_ok and analysis_sha is not None
        journal=storage.folder/'fd.events.jsonl'
        report=dict(schema=NAME+'.receipt',status='cuda_fd_completed' if complete else 'partial_or_failed',
            first_error=errors[0] if errors else None,errors=errors,host=host,prepare_sha256=expected,
            resources=counts,evaluations_completed=len(rows),module_entries=meter.events,
            model_hash_before=before,model_hash_after=after,model_unchanged=(before==after if before is not None else None),
            source_verified=source_ok,journal_sha256=sha(journal) if journal.exists() else None,analysis_sha256=analysis_sha,
            encoder_calls=0,projector_calls=0,AD_calls=0,physics_steps=0,training=False,new_Q=False,
            scientific_pass_threshold=None,scientific_verdict='not_assigned',P1_admission=False)
        try:storage.put('report.json',report,final=True)
        except BaseException as exc:
            # Never replace the original execution fault with a cleanup/report fault.
            errors.append(failure('final_report',exc))
            print(canonical(dict(first_error=errors[0],errors=errors)).decode(),flush=True)
    require(complete and not errors,'FD incomplete; preserve partial journal/evaluations, no automatic retry: '+str(errors))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha');parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    result=prepare(a.output) if a.command=='prepare' else (dict(status='verified_not_executed',roots=len(verify(a.prepared,a.prepare_sha)['roots'])) if a.command=='verify' else run(a.prepared,a.prepare_sha,a.output))
    print(canonical({k:v for k,v in result.items() if k!='module_entries'}).decode().strip())


if __name__=='__main__':main()
