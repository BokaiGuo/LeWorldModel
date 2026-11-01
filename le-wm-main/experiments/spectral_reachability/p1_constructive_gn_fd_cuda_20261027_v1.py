"""MAIN-only constructive first-zero full-H2 GN finite differences.

Saved approved blind tasks + actual first J only. No outcomes, new AD, image
encoding, physics, optimization, threshold fitting, or scientific admission.
"""
from contextlib import ExitStack
from copy import deepcopy
import argparse
import ast
import os
from pathlib import Path
import shutil
import numpy as np

from experiments.spectral_reachability import p1_line_search_gn_fd_cuda_20261027_v1 as fd
from experiments.spectral_reachability import p1_constructive_blind_planners_cuda_20261027_v1 as planner
from experiments.spectral_reachability import p1_constructive_blind_planners_saved_audit_20261027_v1 as saved

NAME = 'p1_constructive_gn_fd_cuda_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/'results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01/report.json'
INPUT_SHA = '4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d'
INPUT_PREPARE_SHA = '30b58155a59f73fdc8cc132ff069e645ed6c2bbff9434f474b1f7b9c412f7eb4'
EPS, SIGNS = fd.EPS, fd.SIGNS
CAP, FLOOR, FINAL = fd.CAP, fd.FLOOR, fd.FINAL
PREPARE_CAP = 256*1024
require, sha, canonical = planner.require, planner.sha, planner.canonical
screen, io = fd.screen, fd.io
directions, perturb, compare, errors = fd.directions, fd.perturb, fd.compare, fd.errors
Store, failure = fd.Store, fd.failure


def budget():
    return dict(points=2, directions=3, epsilons=3, central_comparisons=18,
        signed_H2_evaluations=36, nominal_H2_evaluations=2, total_H2_evaluations=38,
        predict=76, action_encoder=76, FD_predict=72, nominal_predict=4,
        AD_calls=0, new_jacobians=0, new_vjp_directions=0, encoder=0, projector=0,
        physics_steps=0, SVD=0, solve=0, selection=0,
        inherited_GN_predict=156, inherited_GN_action_encoder=156,
        inherited_GN_AD_forward_predict_inside156=16, inherited_GN_jacobians=8,
        inherited_GN_mathematical_vjp_directions=1536,
        inherited_D2P_predict=132, inherited_D2P_jacobians=22,
        inherited_D2P_mathematical_vjp_directions=4224,
        inherited_encoding='separate already-paid source cost, not zero total-pipeline cost',
        equal_total_compute_claim=False)


def extract_point(task, slot, events):
    """No selected costs, later iterations, D2P values, or physical rows used."""
    group = 'iter_1_AD'
    ds = [e for e in events if e['kind']=='derivative_complete' and e['evaluation_id']==group]
    require(len(ds)==1, 'unique first GN derivative')
    d = ds[0]['derivative']; j = saved.arr(d['raw_terminal_jacobian'], (192,20))
    require(d['task_id']==slot['task_id'] and not saved.arr(d['at_logits'],(2,10)).any(), 'same task first J at zero')
    path = [e['context'] for e in events if e['kind']=='jacobian_forward_packet' and e['evaluation_id']==group]
    saved.path_check(task, np.zeros((10,2)), path)
    require(path[-1]['latent_history'][-1]==d['at_terminal'], 'AD path terminal binding')
    rows = [e for e in events if e['kind']=='jacobian_row' and e['evaluation_id']==group]
    require([e['row'] for e in rows]==list(range(192)), 'all192 ordered raw rows')
    require(np.array_equal(saved.arr([e['raw_control_derivative'] for e in rows],(192,20)),j), 'zero tanh chain equals one')
    joint = [e for e in events if e['kind']=='joint_logit_jacobian' and e['evaluation_id']==group]
    forward = [e for e in events if e['kind']=='jacobian_forward_complete' and e['evaluation_id']==group]
    require(len(joint)==len(forward)==1 and joint[0]['jacobian']==j.tolist() and
        joint[0]['terminal']==forward[0]['terminal']==d['at_terminal'] and
        forward[0]['logits']==d['at_logits'], 'actual AD forward and joint J')
    initial = [e for e in events if e['kind']=='evaluation_complete' and e['evaluation_id']=='initial']
    require(len(initial)==3 and all(e['prediction']['contexts']==path and e['prediction']['logits']==d['at_logits'] for e in initial), 'zero initial triple equals AD anchors')
    point = dict(point_order=slot['order'], root_index=slot['order'],
        root_sha256=None, root_identity='approved opaque task_id; no invented native root hash',
        task_id=slot['task_id'], task_path=slot['task_export'], task_sha256=slot['task_sha256'],
        iteration=1, evaluation_id=group, at_logits=d['at_logits'], at_terminal=d['at_terminal'],
        saved_context_sha256=[screen.core.digest(c) for c in path],
        logit_J_sha256=screen.core.digest(j.tolist()),
        saved_Jv={v['id']:(j@np.asarray(v['vector'])).tolist() for v in directions()})
    return point, j


def collect():
    r = saved.read(INPUT, INPUT_SHA)
    require(r['schema']==planner.NAME+'.report' and r['prepare_sha256']==INPUT_PREPARE_SHA and
        r['status']=='model_plans_sealed_not_admitted' and r['errors']==[] and
        r['source_verified'] is True and r['commands_released'] is True and r['model_unchanged'] is True,
        'complete externally bound actual planner, not partial')
    p = saved.read(r['prepared'], INPUT_PREPARE_SHA)
    require(p['schema']==planner.NAME+'.prepare' and len(p['slots'])==2 and
        r['contract']==p['contract'] and p['contract']['numerical']==dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False) and
        r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'], 'two tasks same frozen numerical model')
    require(sha(planner.__file__)==saved.PLANNER_SHA, 'approved frozen planner source')
    bindings = deepcopy(p['bindings_sha256'])
    # Bind authenticated artifacts by hash; do not interpret selected/later losses.
    for name,h in r['artifact_sha256'].items():
        require(Path(name).name==name and name not in ('','.','..'), 'flat actual artifact')
        bindings[str(INPUT.parent/name)] = h
    bindings.update({str(INPUT):INPUT_SHA, str(Path(r['prepared'])):INPUT_PREPARE_SHA})
    task_paths = {s['task_export'] for s in p['slots']}
    for path,h in p['bindings_sha256'].items():
        f=Path(path)
        require(path in task_paths or path in planner.encode.MODELS or
            (f.suffix in ('.py','.md') and 'results' not in f.parts), 'no privileged sourcepack/certificate closure')
    raw = saved.load(INPUT.parent/'events.jsonl.gz', r['artifact_sha256']['events.jsonl.gz'])
    require(raw.endswith(b'\n'), 'complete journal')
    wrapped = [planner.decode(line) for line in raw.splitlines()]
    tasks, matrices, points = {}, {}, []
    require([s['order'] for s in p['slots']]==[0,1] and len({s['task_id'] for s in p['slots']})==2, 'ordered unique approved tasks')
    for slot in p['slots']:
        k = slot['order']; approved = planner.load_export(slot['task_export'],slot['task_sha256'])
        require(approved['task_id']==slot['task_id'], 'slot and approved export exact task identity')
        task = approved['gn_task']
        group = [e for e in wrapped if e['root_order']==k and e['arm']==planner.ARMS[0]]
        require(group and all(e['task_id']==slot['task_id'] for e in group), 'GN opaque task join')
        events = [e['event'] for e in group]
        require(group[0]['event']==dict(kind='arm_started',task_sha256=slot['task_sha256']), 'approved task entry')
        point,j = extract_point(task,slot,events)
        tasks[k]=task; matrices[k]=j; points.append(point)
    config = planner.read(planner.encode.POLICY/'config.json')
    require(config['encoder'].get('pretrained') is False, 'no pretrained download')
    files = [Path(__file__),Path(fd.__file__),Path(saved.__file__),Path(saved.io.__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    # Static local import closure, including inherited FD arithmetic/runtime.
    seen=set(); pending=list(files)
    while pending:
        f=Path(pending.pop()).resolve()
        if f in seen: continue
        seen.add(f); bindings[str(f)]=sha(f)
        if f.suffix!='.py' or not f.is_relative_to(ROOT/'experiments'): continue
        for node in ast.walk(ast.parse(f.read_text())):
            names=[]
            if isinstance(node,ast.Import): names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('experiments'):
                names=[node.module]+[node.module+'.'+x.name for x in node.names]
            for name in names:
                child=ROOT/Path(*name.split('.')).with_suffix('.py')
                if name.startswith('experiments.') and child.is_file() and child not in seen: pending.append(child)
    for path,h in bindings.items(): require(sha(path)==h, 'source/artifact changed: '+path)
    result = dict(schema=NAME+'.prepare',input_report=str(INPUT),input_report_sha256=INPUT_SHA,
        input_prepare_sha256=INPUT_PREPARE_SHA,points=points,directions=directions(),epsilons=list(EPS),signs=list(SIGNS),
        budget=budget(),policy=p['policy'],expected_model_hash=p['expected_model_hash'],bindings_sha256=bindings,
        numerical=p['contract']['numerical'],order='root0 then root1; nominal then direction/epsilon/minus-plus',
        perturbation='float64 zero +/- eps unit vector -> float32 once -> NumPy float32 tanh once; no projection',
        reference='saved first fullH2 logitJ, including both latent and action-history chains; effective float32 direction also reported',
        scientific_pass_threshold=None,derivative_verdict='not_assigned',P1_admission=False,
        source_access='approved task exports + saved planner journal; no physical outcomes, certificate, blind source payload',
        output_cap_bytes=CAP,floor_bytes=FLOOR,final_reserve_bytes=FINAL,
        inherited_planner_resources=r['resources'],inherited_encoding='separate original source receipts, not reread or charged as zero')
    return result,tasks,matrices


def prepare(output):
    p,_,_=collect(); raw=canonical(p); output=Path(output).absolute()
    require(not output.exists() and len(raw)<PREPARE_CAP, 'NEW small prepare')
    require(shutil.disk_usage(output.parent).free>=FLOOR+len(raw)+4096, '200MiB prepare reserve')
    with output.open('xb') as f:
        require(f.write(raw)==len(raw), 'short prepare'); f.flush(); os.fsync(f.fileno())
    io.fsync_directory(output.parent)
    return dict(prepared=str(output),prepare_sha256=sha(output),bytes=len(raw),budget=budget())


def verify(path, expected):
    screen.core.hash_value(expected)
    require(Path(path).stat().st_size<PREPARE_CAP and sha(path)==expected, 'external prepare SHA')
    p=planner.read(path,PREPARE_CAP); current,tasks,matrices=collect()
    require(p==current, 'complete fresh-source reconstruction')
    return p,tasks,matrices


def requests(p):
    require(len(p['points'])==2 and [x['point_order'] for x in p['points']]==[0,1] and
        p['directions']==directions(), 'two fixed points/directions')
    rows=[]
    for point in p['points']:
        require(not saved.arr(point['at_logits'],(2,10)).any(), 'zero only')
        common={k:point[k] for k in ('point_order','root_index','root_sha256','task_id','iteration')}
        rows.append(dict(common,order=len(rows),mode='nominal',direction=None,epsilon=None,sign=0,
            logits=point['at_logits'],commands=np.zeros((10,2)).tolist()))
        for d in directions():
            for eps in EPS:
                for sign in SIGNS:
                    rows.append(dict(common,order=len(rows),mode='central_side',direction=d['id'],epsilon=eps,sign=sign,
                        **perturb(point,d['vector'],eps,sign)))
    require(len(rows)==38, 'exact38 fullH2 evaluations')
    return rows


class Meter(fd.Meter):
    def start(self, request):
        require((self.request is None or self.packet==2) and request['order']<38 and
            len(self.events)==request['order']*4 and all(e['completed'] for e in self.events), 'once-only ordered fullH2')
        self.request=request; self.packet=0; self.paths=[]
        self.emit_event(dict(kind='evaluation_attempt',request=deepcopy(request)))
    def begin(self, resource):
        require(self.request is not None and self.packet<2 and self.no_grad(), 'no-grad fullH2')
        offset=len(self.events)-self.request['order']*4
        require(resource==('action_encoder','predict')[offset%2] and offset//2==self.packet and
            all(e['completed'] for e in self.events), 'module/packet order, no failed retry')
        require(sum(e['resource']==resource for e in self.events)<76, 'hard predispatch76 cap')
        e=dict(index=len(self.events),resource=resource,samples=1,completed=False,
            evaluation_order=self.request['order'],packet=self.packet)
        self.events.append(e); self.emit_event(dict(kind='module_attempt',entry=deepcopy(e)))
        return e


def analyze(p, rows, matrices):
    wanted=requests(p)
    require(len(rows)==38 and all({k:e[k] for k in w}==w for e,w in zip(rows,wanted,strict=True)), 'all38 exact ordered requests')
    result=[]; stability=[]; offset=0
    for point in p['points']:
        n=rows[offset]; offset+=1
        require(n['terminal']==point['at_terminal'] and n['context_sha256']==point['saved_context_sha256'], 'nominal EXACT saved AD path')
        for d in directions():
            group=[]
            for eps in EPS:
                group.append(compare(point,matrices[point['point_order']],d,eps,rows[offset],rows[offset+1])); offset+=2
            result.extend(group)
            stability.append(dict(point_order=point['point_order'],direction=d['id'],
                adjacent_epsilon_fd_l2=[float(np.linalg.norm(np.asarray(a['fd'])-b['fd'])) for a,b in zip(group,group[1:])]))
    require(offset==38 and len(result)==18, 'all18 signed comparisons')
    return dict(schema=NAME+'.analysis',comparisons=result,epsilon_stability=stability,nominal_anchors=2,
        scientific_pass_threshold=None,derivative_verdict='not_assigned',P1_admission=False)


def evaluate(p,tasks,backend_for_root,meter,sink,rows=None):
    rows=[] if rows is None else rows; current=None
    with ExitStack() as active:
        for request in requests(p):
            k=request['root_index']; backend=backend_for_root(k)
            if k!=current:
                active.close(); active.enter_context(backend.guard(meter)); current=k
            meter.start(request)
            path=backend.rollout(tasks[k],np.asarray(request['commands'],np.float32),meter,'whole_replay')
            require(path==meter.paths, 'actual packet evidence')
            saved.path_check(tasks[k],request['commands'],path)
            row=dict(request,terminal=path[-1]['latent_history'][-1],context_sha256=[screen.core.digest(c) for c in path])
            # Save raw result BEFORE checking exact anchor: mismatch is evidence.
            sink(dict(kind='evaluation_complete',evaluation=row)); rows.append(row)
            if request['mode']=='nominal':
                point=p['points'][request['point_order']]
                require(row['terminal']==point['at_terminal'] and row['context_sha256']==point['saved_context_sha256'], 'nominal EXACT saved AD path; no retry')
    return rows


def run(prepared,expected,output):
    p,tasks,matrices=verify(prepared,expected)
    store=Store(output); problems=[]; forbidden=[]; hooks=io.Hooks()
    model=before=after=host=analysis_sha=None; source_ok=False; rows=[]
    meter=Meter(store.event); validations=dict(attempted=1,completed=1)
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        io.frozen(model); before=io.tensor_hash(model)
        require(before==p['expected_model_hash'], 'same original frozen numerical model')
        meter.sync=lambda:torch.cuda.synchronize(device); meter.no_grad=lambda:not torch.is_grad_enabled()
        targets=[(torch.autograd,k) for k in ('grad','backward','_engine_run_backward')]
        targets += [(torch.autograd.functional,k) for k in ('jacobian','hessian','jvp','vjp','hvp','vhp')]
        targets += [(torch.Tensor,'backward')]
        targets += [(torch.func,k) for k in ('jacrev','jacfwd','hessian','jvp','vjp','grad','grad_and_value')]
        for obj,key in targets:
            if not hasattr(obj,key): continue
            def denied(*a,_key=key,**kw):
                forbidden.append(_key); store.event(dict(kind='forbidden_AD',entry=_key))
                raise ValueError('FD prohibits new AD: '+_key)
            hooks.install(obj,key,denied)
        backends={k:screen.NativeCuda(model,device,t) for k,t in tasks.items()}
        with sdpa_kernel(SDPBackend.MATH): evaluate(p,tasks,lambda k:backends[k],meter,store.event,rows)
        result=analyze(p,rows,matrices); store.put('analysis.json.gz',result)
        analysis_sha=sha(store.output/'analysis.json.gz')
    except BaseException as exc:
        problems.append(failure('execution',exc))
        try: store.event(dict(kind='execution_error',error=problems[0]))
        except BaseException as e: problems.append(failure('error_journal',e))
    finally:
        hooks.restore(problems)
        if model is not None:
            try: io.frozen(model); after=io.tensor_hash(model)
            except BaseException as e: problems.append(failure('final_model',e))
        try:
            validations['attempted']+=1; verify(prepared,expected)
            validations['completed']+=1; source_ok=True
        except BaseException as e: problems.append(failure('final_source',e))
        try: store.close()
        except BaseException as e: problems.append(failure('journal_finalize',e))
        counts=meter.counts()
        complete=not problems and not forbidden and source_ok and before is not None and before==after and len(rows)==38 and analysis_sha is not None and counts=={k:dict(attempted=76,completed=76) for k in ('predict','action_encoder')}
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try: hashes[f.name]=sha(f)
                except BaseException as e: problems.append(failure('artifact_hash',e)); complete=False
        report=dict(schema=NAME+'.report',status='cuda_fd_completed' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,input_report_sha256=INPUT_SHA,
            errors=problems,first_error=problems[0] if problems else None,forbidden_AD=forbidden,
            host=host,numerical=p['numerical'],resources=counts,budget=budget(),evaluations_completed=len(rows),
            inherited_planner_resources=p['inherited_planner_resources'],
            saved_validation_calls=validations,saved_validation_AD_SVD_solves=0,
            model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,artifact_sha256=hashes,analysis_sha256=analysis_sha,
            output_cap_bytes=CAP,floor_bytes=FLOOR,new_AD_calls=0 if complete else None,
            new_images=0,new_physics=0,new_Q=False,training=False,independent_sample_increment=0,
            scientific_pass_threshold=None,derivative_verdict='not_assigned',P1_admission=False)
        try: store.put('report.json',report,final=True)
        except BaseException as e:
            problems.append(failure('final_report',e)); complete=False
            print(canonical(dict(first_error=problems[0],errors=problems)).decode(),flush=True)
    require(complete and not problems, 'partial FD preserved; no automatic retry: '+str(problems))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run'))
    for key in ('prepared','prepare-sha','output'): parser.add_argument('--'+key)
    args=parser.parse_args()
    result=prepare(args.output) if args.command=='prepare' else (
        dict(status='verified_saved_arrays_only',points=len(verify(args.prepared,args.prepare_sha)[0]['points']),budget=budget())
        if args.command=='verify' else run(args.prepared,args.prepare_sha,args.output))
    print(canonical(result).decode().strip())


if __name__=='__main__': main()
