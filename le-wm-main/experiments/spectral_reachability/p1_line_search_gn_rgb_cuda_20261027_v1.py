"""Six actual line-search GN RGB checkpoints -> raw native192 NPY, MAIN CUDA only.

No actual physical receipt assumed at authoring. prepare requires explicit
physical report/prepare SHA; no GPU, model or physics is imported by tests.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p1_line_search_gn_saved_audit_20261027_v1 as saved
from experiments.spectral_reachability import p3_joint_prefix_cuda_encoding_20261027_v1 as imageio

NAME = 'p1_line_search_gn_rgb_cuda_20261027_v1'
ROOT, sha, read, require, canonical = saved.ROOT, saved.sha, saved.read, saved.require, saved.canonical
ARM = 'line_search_GN'
CAP, FLOOR, FINAL = 2*1024**2, 200*1024**2, 128*1024
PHYSICAL_REPORT = ROOT/'results/p1_line_search_gn_physical_20261027_v1_main01/report.json'
PHYSICAL_SHA = '3a41fe83f375e5bb3f390e9f3a0e8e17dd70c2d49a7ebe3a99079f5774de85c1'
PHYSICAL_PREPARED = ROOT/'results/p1_line_search_gn_physical_20261027_v1_prepare02.json'
PHYSICAL_PREPARE_SHA = 'f36f915ccfc2687da056810d1241da433190eff430bb54fa7b7b35a6af3818d5'


def contract():
    return dict(roots=2, arm=ARM, checkpoints=[0,5,10], frames=6,
        order='source-slot root0 step0/5/10 then root1 step0/5/10; retain duplicate pixels',
        encoder_cap=6, projector_cap=6, batch=1, native_shape=[192], native_dtype='float32',
        image_shape=[224,224,3], predict_cap=0, actionencoder_cap=0, AD_cap=0, physics_cap=0,
        no_cpu_neural_fallback=True, no_retries=True, training=False, new_Q=False, P1_admission=False,
        output_cap_bytes=CAP, reserve_bytes=FLOOR,
        transform='preflight_pusht_d2p_p0._image_tensor + original image_transform',
        representation='model.encode pixels-only -> frozen native encoder -> existing projector, no Q applied to NPY')


def frames_from_rows(rows, plans, tasks, predictions, folder, check_pixels=imageio.pixels):
    """Pure fixture-testable contract. Does not infer terminal from old outputs."""
    require(len(rows) == len(plans) == len(tasks) == len(predictions) == 2, 'complete two-root census')
    require(len({p['root_sha256'] for p in plans}) == 2, 'two distinct source roots')
    frames = []; folder = Path(folder).resolve()
    for index, (row, plan, task, pred) in enumerate(zip(rows, plans, tasks, predictions)):
        require(plan['arm'] == row['arm'] == ARM and all(row[k] == plan[k] for k in ('root_sha256','selection_sha256')),
                'physical selected root/arm/authorization pairing')
        require(row['status'] == 'completed' and row['full_horizon'] is True and row['initial_exact'] is True and
                row['first_error'] is None and row['secondary_errors'] == [] and row['returned_steps'] == row['attempted_steps'] == 10,
                'require own full ten-step path; do not substitute partial/old endpoint')
        require([c['step'] for c in row['checkpoints']] == [0,5,10] and len(row['trace']) == 10, 'all three actual RGB checkpoints')
        require(pred['commands'] == plan['commands'] and len(pred['contexts']) == 2, 'same chosen actual action plan')
        require(pred['task_id'] == plan['task_id'], 'same original goal/Q/context task')
        require(hashlib_task(task) == plan['task_id'], 'original complete task identity')
        for step, entry in enumerate(row['trace'], 1):
            require(entry['step'] == step and entry['attempted'] is True and entry['returned'] is True and
                    entry['command'] == pred['commands'][step-1], 'actual trace complete exact selected raw commands')
        predicted = [task['root_context']['latent_history'][-1]]+[c['latent_history'][-1] for c in pred['contexts']]
        for ci, checkpoint in enumerate(row['checkpoints']):
            step = [0,5,10][ci]
            path = Path(checkpoint['path']); path = path if path.is_absolute() else ROOT/path
            path = path.absolute()
            require(path.parent == folder and not path.is_symlink(), 'own new physical PNG, no old endpoint')
            state = checkpoint['state'] if step == 0 else row['trace'][step-1]['actual_state']
            require(checkpoint['state'] == state, 'actual checkpoint state chronology')
            check_pixels(path, checkpoint['file_sha256'], checkpoint['rgb_sha256'])
            frames.append(dict(order=len(frames), root_index=index, root_sha256=plan['root_sha256'], arm=ARM,
                step=step, path=str(path), file_sha256=checkpoint['file_sha256'], pixel_sha256=checkpoint['rgb_sha256'],
                prediction_native192=predicted[ci], goal=task['goal'], q=task['projection'], task_id=plan['task_id']))
    return frames


def hashlib_task(task):
    import hashlib
    return hashlib.sha256(json.dumps(task, sort_keys=True, allow_nan=False).encode()).hexdigest()


def collect(physical_report=PHYSICAL_REPORT, physical_report_sha=PHYSICAL_SHA,
            physical_prepared=PHYSICAL_PREPARED, physical_prepare_sha=PHYSICAL_PREPARE_SHA):
    # Only after MAIN supplies actual hashes. These validators perform saved reads.
    from experiments.spectral_reachability import p1_line_search_gn_physical_20261027_v1 as physical
    from experiments.spectral_reachability import p1_direct_cuda_calibration_main_20261027_v2 as native
    arguments = dict(physical_report=str(Path(physical_report).resolve()), physical_report_sha=physical_report_sha,
                     physical_prepared=str(Path(physical_prepared).resolve()), physical_prepare_sha=physical_prepare_sha)
    require(physical_report_sha == PHYSICAL_SHA and physical_prepare_sha == PHYSICAL_PREPARE_SHA, 'fixed MAIN actual physical pair, no substitutions')
    require(sha(arguments['physical_report']) == physical_report_sha, 'MAIN actual physical report SHA')
    p = physical.verify(arguments['physical_prepared'], physical_prepare_sha)
    r = read(arguments['physical_report']); folder = Path(arguments['physical_report']).parent
    require(r['schema'] == physical.NAME+'.report' and r['status'] == 'physical_calibration_recorded' and
            r['prepare_sha256'] == physical_prepare_sha and r['errors'] == [] and r['neural_calls'] == 0 and
            r['counts'] == {k:dict(attempted=v, completed=v) for k,v in dict(reset=4,render=22,step=20).items()}, 'actual physical complete six-frame acquisition')
    journal = folder/'journal.jsonl'; require(sha(journal) == r['journal_sha256'], 'actual physical journal seal')
    # Reconstruct selected predictions/observed tasks using source-bound planner, not B8 endpoint.
    data = physical.collect(**p['arguments'])
    require(data['plans'] == p['plans'], 'physical prepare selected plans reproduced')
    tasks = [read(plan['task_file'])['task'] for plan in p['plans']]
    predictions = [read(plan['origin'])['final_prediction'] for plan in p['plans']]
    for plan, pred in zip(p['plans'], predictions):
        require(physical.source.digest(pred) == plan['prediction_sha256'], 'own final selected prediction hash')
    # Hash-chain journal must carry all complete branch receipts and checkpoint exports.
    require(journal.stat().st_size <= 16*1024**2, 'bounded physical journal')
    events = [json.loads(line) for line in journal.read_bytes().splitlines()]; previous = None
    for i,e in enumerate(events):
        copy = dict(e); h = copy.pop('event_sha256')
        require(e['index'] == i and e['previous_sha256'] == previous and physical.source.digest(copy) == h, 'physical journal hash chain')
        previous = h
    require([e['receipt'] for e in events if e['kind'] == 'branch_closed'] == r['rows'], 'all complete rows from actual journal')
    checkpoints = [e for e in events if e['kind'] == 'checkpoint_saved']
    require(len(checkpoints) == 6, 'all six actual saved checkpoint events')
    for row in r['rows']:
        require([e['receipt'] for e in checkpoints if e['selection_sha256'] == row['selection_sha256']] == row['checkpoints'], 'PNG exports from own selected physical branch')
    frames = frames_from_rows(r['rows'], p['plans'], tasks, predictions, folder)
    root_metrics = [dict(root_index=i,root_sha256=p['plans'][i]['root_sha256'],task_id=p['plans'][i]['task_id'],
                         q=t['projection'],goal=t['goal']) for i,t in enumerate(tasks)]
    for frame in frames:
        del frame['q']; del frame['goal']  # one original Q/goal per root, not six large duplicate matrices
    model, normalization = native.native_metadata()
    require(all(t['action_mean'] == normalization['mean'] and t['action_std'] == normalization['std'] for t in tasks), 'original normalizer')
    for t in tasks:
        for filename,key in (('weights.pt','model_sha256'),('config.json','model_config_sha256')):
            require(t[key] == next(h for path,h in model['model_bindings'].items() if Path(path).name == filename), 'original model identity')
    search_report = read(p['arguments']['search_report'])
    sources = set(imageio.source_files()+[Path(__file__).resolve(),Path(physical.__file__).resolve(),Path(native.__file__).resolve(),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),Path(saved.__file__).resolve()])
    bindings = {arguments['physical_report']:physical_report_sha, arguments['physical_prepared']:physical_prepare_sha,
                str(journal):r['journal_sha256'], **p['sources_sha256'], **data['bindings_sha256'], **model['model_bindings']}
    bindings.update({f['path']:f['file_sha256'] for f in frames})
    bindings.update({str(path):sha(path) for path in sources})
    for path,h in bindings.items(): require(sha(path) == h, 'full source/input TOCTOU: '+path)
    return dict(schema=NAME+'.prepare', arguments=arguments, contract=contract(), frames=frames,root_metrics=root_metrics,
        bindings_sha256=bindings, policy=model['policy'], model_bindings=model['model_bindings'],
        expected_model_hash=search_report['model_hash_after'], inherited_physical_counts=r['counts'],
        inherited_planner_resources=search_report['resources'], scientific_pass_threshold=None)


def prepare(output, **arguments):
    p = collect(**arguments); data = canonical(p); output = Path(output).absolute()
    require(len(data) < 512*1024 and not output.exists(), 'exclusive small hash-bound prepare')
    require(shutil.disk_usage(output.parent).free >= FLOOR+len(data), 'prepare 200MiB floor')
    with output.open('xb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return dict(prepared=str(output), prepare_sha256=sha(output), bytes=len(data))


def verify(prepared, expected):
    require(sha(prepared) == expected, 'external prepare SHA')
    p = read(prepared); require(p == collect(**p['arguments']), 'full physical/model/Q/order/source reconstruction')
    return p


class Store:
    def __init__(self, output):
        self.output = Path(output).absolute()
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free >= FLOOR+CAP, 'exclusive output/full reserve')
        self.output.mkdir()

    def put(self, name, raw, append=False, final=False):
        require(Path(name).name == name, 'flat local file')
        used = sum(f.stat().st_size for f in self.output.iterdir() if f.is_file())
        require(used+len(raw) <= CAP-(0 if final else FINAL), '2MiB cap including failed writes/final reserve')
        require(shutil.disk_usage(self.output).free >= FLOOR+len(raw), '200MiB floor')
        with (self.output/name).open('ab' if append else 'xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())

    def event(self, value): self.put('image.events.jsonl', canonical(value), append=True)
    def json(self, name, value, final=False): self.put(name, canonical(value), final=final)
    def latent(self, frame, z):
        native(z); target = BytesIO(); np.save(target, z, allow_pickle=False)
        name = f'frame_{frame["order"]:02d}.native192.npy'; self.put(name, target.getvalue())
        path = self.output/name
        with path.open('rb') as stream: require(np.array_equal(np.load(stream,allow_pickle=False),z), 'NPY exact readback')
        return dict(path=str(path), sha256=sha(path), dtype='float32', shape=[192])


def native(z):
    require(isinstance(z,np.ndarray) and z.dtype == np.float32 and z.shape == (192,) and np.isfinite(z).all(), 'raw finite float32 native192, no casting')


class Ledger:
    """Actual module entry/return separately; no duplicate-input elision or retry."""
    def __init__(self, sink, sync=lambda: None, no_grad=lambda: True):
        self.sink,self.sync,self.no_grad = sink,sync,no_grad; self.entries=[]; self.frame=None; self.blocked=[]

    def dispatch(self, kind, tensor, callback, *args, **kwargs):
        expected = [(i,k) for i in range(6) for k in ('encoder','projector')]
        require(self.frame is not None and len(self.entries) < 12 and (self.frame,kind) == expected[len(self.entries)], 'exact six-frame module order/cap')
        require(all(e['completed'] for e in self.entries) and self.no_grad(), 'failed work cannot retry; inference only')
        require(getattr(getattr(tensor,'device',None),'type',None) == 'cuda', 'CUDA only, no CPU neural')
        require(tuple(tensor.shape) == ((1,3,224,224) if kind == 'encoder' else (1,192)), 'batch1 native image/module shape')
        entry = dict(index=len(self.entries),frame=self.frame,resource=kind,completed=False)
        self.entries.append(entry); self.sink(dict(kind='module_attempt', entry=deepcopy(entry)))
        try:
            result = callback(*args, **kwargs); self.sync(); entry['completed'] = True
            self.sink(dict(kind='module_return',entry=deepcopy(entry))); return result
        except BaseException as exc:
            entry['error'] = error('module',exc)
            try: self.sink(dict(kind='module_error',entry=deepcopy(entry)))
            except BaseException as secondary: exc.add_note('module error journal: '+str(secondary))
            raise

    def forbid(self, name):
        def denied(*args, **kwargs):
            self.blocked.append(name); self.sink(dict(kind='forbidden',resource=name,frame=self.frame))
            raise ValueError('forbidden '+name)
        return denied

    def counts(self):
        return {k:dict(attempted=sum(e['resource']==k for e in self.entries), completed=sum(e['resource']==k and e['completed'] for e in self.entries)) for k in ('encoder','projector')}


def analyze(frames, values, root_metrics):
    require(len(frames) == len(values) == 6, 'all six, no dedup')
    require(len(root_metrics) == 2, 'two original root Q/goal contracts')
    result=[]
    for i,(f,z) in enumerate(zip(frames,values)):
        native(z); require((f['order'],f['root_index'],f['step']) == (i,i//3,[0,5,10][i%3]), 'fixed chronological frame order')
        metric=root_metrics[f['root_index']]
        require(all(metric[k] == f[k] for k in ('root_index','root_sha256','task_id')), 'own original Q/goal, no cross-root metrics')
        q = saved.array(metric['q'],(192,192)); goal = saved.array(metric['goal'],(192,)); pred = saved.array(f['prediction_native192'],(192,))
        result.append(dict(order=i,root_sha256=f['root_sha256'],step=f['step'],task_id=f['task_id'],
            actual_Q_mse=float(np.mean((q@(z.astype(float)-goal))**2)),
            predicted_Q_mse=float(np.mean((q@(pred-goal))**2)),native_prediction_mse=float(np.mean((z.astype(float)-pred)**2))))
    return dict(rows=result,new_Q=False,training=False,P1_admission=False,scope='original-Q descriptive calibration, not causal efficacy')


def encode_frames(p, encode, store, ledger, rows, values):
    for f in p['frames']:
        ledger.frame=f['order']; row=dict(order=f['order'],root_sha256=f['root_sha256'],step=f['step'],status='attempted')
        rows.append(row); store.event(dict(kind='frame_attempt',row=deepcopy(row)))
        try:
            z=encode(f); native(z)
            require([(e['resource'],e['completed']) for e in ledger.entries if e['frame']==f['order']] == [('encoder',True),('projector',True)], 'actual two module returns per frame')
            row['latent']=store.latent(f,z); row['status']='saved'; values.append(z.copy())
            store.event(dict(kind='frame_saved',row=deepcopy(row)))
        except BaseException as exc:
            row.update(status='failed',error=error('frame',exc)); raise
        finally: ledger.frame=None


def error(stage, exc):
    return dict(stage=stage,type=type(exc).__name__,message=str(exc)[:1500],notes=list(getattr(exc,'__notes__',[])))


@contextmanager
def guard(model, torch, ledger, errors):
    undo=[]
    def patch(obj,name,new):
        old=getattr(obj,name); undo.append((obj,name,old)); setattr(obj,name,new)
    try:
        for kind,module in (('encoder',model.encoder),('projector',model.projector)):
            old=module.forward
            def counted(*args,_kind=kind,_old=old,**kwargs): return ledger.dispatch(_kind,args[0],_old,*args,**kwargs)
            patch(module,'forward',counted)
        for obj,name in ((model,'predict'),(model.predictor,'forward'),(model.action_encoder,'forward'),(model.pred_proj,'forward'),
                         (torch.autograd,'grad'),(torch.autograd,'backward'),(torch.Tensor,'backward')):
            patch(obj,name,ledger.forbid(name))
        for obj,names in ((torch.autograd.functional,('jacobian','hessian','jvp','vjp','hvp','vhp')),
                          (torch.func,('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value'))):
            for name in names: patch(obj,name,ledger.forbid(name))
        yield
    finally:
        for obj,name,old in reversed(undo):
            try: setattr(obj,name,old)
            except BaseException as exc: errors.append(error('restore_'+name,exc))


def run(prepared, expected, output):
    p=verify(prepared,expected); store=Store(output); errors=[]; rows=[]; values=[]
    ledger=Ledger(store.event); model=None; before=after=host=None; source_ok=False
    store.json('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
    try:
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        imageio.frozen(model); before=imageio.tensor_hash(model)
        require(before==p['expected_model_hash'],'same planner frozen model tensor state')
        transform=p0.image_transform(); ledger.sync=lambda:torch.cuda.synchronize(device); ledger.no_grad=lambda:not torch.is_grad_enabled()
        with guard(model,torch,ledger,errors):
            def encode(f):
                rgb,_=imageio.pixels(f['path'],f['file_sha256'],f['pixel_sha256'])
                with torch.inference_mode():
                    x=p0._image_tensor(rgb,transform=transform,device=device)
                    require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'native input CUDA shape')
                    z=model.encode({'pixels':x})['emb']
                    require(z.device.type=='cuda' and z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'raw native CUDA float32 output')
                    return z[0,0].detach().cpu().numpy().copy()
            encode_frames(p,encode,store,ledger,rows,values)
        store.json('analysis.json',analyze(p['frames'],values,p['root_metrics']))
    except BaseException as exc:
        errors.insert(0,error('execution',exc))  # cleanup failures never replace primary
        try: store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary: errors.append(error('error_journal',secondary))
    finally:
        if model is not None:
            try: imageio.frozen(model); after=imageio.tensor_hash(model)
            except BaseException as exc: errors.append(error('final_model',exc))
        try: verify(prepared,expected); source_ok=True
        except BaseException as exc: errors.append(error('final_source',exc))
        for f in p['frames'][len(rows):]: rows.append(dict(order=f['order'],root_sha256=f['root_sha256'],step=f['step'],status='blocked_not_attempted'))
        counts=ledger.counts()
        complete=(not errors and not ledger.blocked and len(values)==6 and source_ok and before is not None and before==after and
                  counts=={k:dict(attempted=6,completed=6) for k in counts})
        hashes={}
        for path in store.output.iterdir():
            if path.is_file():
                try: hashes[path.name]=sha(path)
                except BaseException as exc: errors.append(error('artifact_hash',exc)); complete=False
        receipt=dict(schema=NAME+'.report',status='six_actual_rgb_latents_saved' if complete else 'partial_or_failed',
            prepare_sha256=expected,rows=rows,errors=errors,first_error=errors[0] if errors else None,resources=counts,
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
            host=host,artifact_sha256=hashes,new_predict=0,new_actionencoder=0,new_AD=0,new_physics=0,new_Q=False,training=False,P1_admission=False)
        try: store.json('report.json',receipt,final=True)
        except BaseException as exc:
            errors.append(error('final_report',exc)); complete=False; print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial encoding retained; no automatic retry: '+str(errors))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','verify','run'))
    for name in ('prepared','prepare-sha','output'): parser.add_argument('--'+name)
    for name,default in (('physical-report',str(PHYSICAL_REPORT)),('physical-report-sha',PHYSICAL_SHA),
                         ('physical-prepared',str(PHYSICAL_PREPARED)),('physical-prepare-sha',PHYSICAL_PREPARE_SHA)):
        parser.add_argument('--'+name,default=default)
    a=parser.parse_args()
    result=prepare(a.output,**{k:getattr(a,k) for k in ('physical_report','physical_report_sha','physical_prepared','physical_prepare_sha')}) if a.command=='prepare' else (
        dict(status='verified_no_execution',frames=len(verify(a.prepared,a.prepare_sha)['frames'])) if a.command=='verify' else run(a.prepared,a.prepare_sha,a.output))
    print(canonical(result).decode().strip())
