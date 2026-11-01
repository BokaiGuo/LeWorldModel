"""P1 fixed40 literal-input public rollout versus cached suffix; MAIN CUDA only."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
from types import ModuleType
import numpy as np
from experiments.spectral_reachability import agent_p1_teacher_library40_history_localization_20261028_v1 as history
from experiments.spectral_reachability import agent_p2_stageb_triplet_cuda_20261028_v1 as stageb

NAME = 'agent_p1_teacher_library40_api_equivalence_20261028_v1'
ROOT, rgb, teacher, mid = history.ROOT, history.rgb, history.teacher, stageb.mid
require, sha, digest, canonical = history.require, history.sha, history.digest, history.canonical
LOCAL = ROOT/'results/agent_p1_teacher_library40_history_localization_20261028_v1_analysis_NASH01.json'
LOCAL_SHA = '7d171fea5ad553541dfd6393e4bf2530992ef916c92fcc39295590de7337d3a2'
AUTH = 'MAIN_APPROVED_P1_FIXED40_LITERAL_API_HOST_CUDA_ONLY'
ROUTES = ('public_rollout', 'cached_suffix')
CAP, FINAL, FLOOR, WALL = 512*1024**2, 8*1024**2, 200*1024**2, 900
FLAGS = dict(P1_admission=False, scientific_admission=False, freshness=False, full40_released=False,
             physical_benefit_claim=False, winner_selection=False, goal_or_Q_changed=False)


def contract():
    return dict(stage='compare', known_roots=2, independent_roots=0, fixed_slots=40, route_slots=80, routes=list(ROUTES), missing_actual=[12],
        order='fixed4task x2role x5word; each public then cached; no retry/dedup/selection',
        literal_input='original saved z5/z10/z15, outgoing raw6..20 tokens; genuine same-condition PNG5/10/15; explicit cached emb on both routes',
        public_ABI=dict(pixels=[1,1,3,3,224,224], emb=[1,1,3,192], actions=[1,1,3,10], history_size=3,
                        n_steps=0, predict_loop_iterations=1, predicted_emb=[1,1,4,192]),
        planned_calls=dict(encode=0,encoder=0,projector=0,rollout=40,action_encoder=80,predict=80,predictor=80,pred_proj=80),
        planned_endpoint_predictions=80, planned_batch_samples=80, planned_prediction_temporal_rows=240,
        planned_returns=None, equal_FLOPs=None, equal_total_compute=None,
        AD=0, physics=0, training=False, Qfit=False, CPU_neural_fallback=False, retries=False,
        device='cuda:0', precision=mid.precision_contract(), model_hash_algorithm='original teacher.enc.tensor_hash',
        output_cap_bytes=CAP, final_reserve_bytes=FINAL, disk_floor_bytes=FLOOR, wall_seconds=WALL,
        wall_scope='operation boundaries/final, not preemptive; nested timings are inclusive and nonadditive', **FLAGS)


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/test_{NAME}.py', ROOT/f'innerdocs/{NAME}.md']


def sources():
    # Bind reused ledger and its actual helper closure without importing torch/model packages.
    seen = {}; todo = [history, stageb]
    while todo:
        m = todo.pop()
        if m.__name__ in seen: continue
        require(len(seen) < 192, 'bounded experiment dependency graph')
        seen[m.__name__] = Path(m.__file__).resolve()
        todo += [x for x in vars(m).values() if isinstance(x,ModuleType) and x.__name__.startswith('experiments.') and getattr(x,'__file__',None)]
    return sorted(set(own_files()+list(seen.values())+mid.source_files()))


def validate_jobs(jobs):
    require(len(jobs) == 40, 'entire fixed40; missing terminal is not missing input')
    for i,j in enumerate(jobs):
        task, rem = divmod(i,10); role, word = divmod(rem,5)
        require((j['index'],j['root_slot'],j['condition'],j['role'],j['library_index'],j['library_label'],j['job_id']) ==
                (i,task//2,('off','on')[task%2],('Z','S')[role],word,teacher.LIBRARY[word],f'slot{i:02d}'), 'literal canonical P1 roster, not P2 arm denominator')
        require(j['job_binding_sha256'] == digest({k:v for k,v in j.items() if k != 'job_binding_sha256'}) and
                (j['actual192'] is None) == (i == 12) and len(j['frames']) == 3, 'job/full denominator/missing12 identity')


def build():
    a = rgb.read(LOCAL,LOCAL_SHA)
    require(a == history.build(), 'complete actual fixed40 saved localization rebuild')
    p = rgb.read(history.TEACHER_PREP,history.TEACHER_PREP_SHA)
    folder = ROOT/Path(rgb.INPUTS['compiler'][0]).parent
    c = rgb.read(folder/'report.json',rgb.INPUTS['compiler'][1])
    original_rank = rgb.read(history.RANK,history.RANK_SHA)
    physical_base = ROOT/Path(rgb.INPUTS['physical'][0]).parent
    bindings = {**a['bindings'], str(LOCAL):LOCAL_SHA}
    jobs = []
    for row, old in zip(a['rows'],original_rank['rows'],strict=True):
        i = row['index']; ai = i//5; cr = c['rows'][ai]; anchor = p['anchors'][ai]
        frames = [dict(path=str(physical_base/f['png']), file_sha256=f['file_sha256'], pixel_sha256=f['pixel_sha256']) for f in row['RGB_history_frames']]
        suffix = np.frombuffer(bytes.fromhex(old['suffix_raw_bits']),dtype='<f4').reshape(5,2)
        incoming = teacher.incoming(cr['blind_input']['context'],suffix)
        z = teacher.load_array(folder,cr['arrays']['latents'])
        require(history.bits(z) == history.bits(incoming['latent_history']), 'literal original saved full3, no re-encoding')
        actual_receipt = old['array_receipts']['actual']
        actual = None if actual_receipt is None else teacher.load_array(history.rank.REPORT.parent,actual_receipt).tolist()
        goal = teacher.load_array(folder,cr['arrays']['goal'])
        j = dict(index=i,job_id=f'slot{i:02d}',root_slot=row['root_slot'],condition=row['condition'],role=row['role'],
            library_index=old['library_index'],library_label=old['library_label'],anchor_id=row['anchor_id'],task_id=old['task_id'],
            blind_sha256=old['blind_sha256'],plan_digest=row['plan_digest'],goal_binding=deepcopy(old['goal_binding']),
            context_sha256=digest(cr['blind_input']['context']), latent_receipt=deepcopy(cr['arrays']['latents']),compiler_folder=str(folder),
            full3=z.tolist(),outgoing_tokens=deepcopy(incoming['action_history']),
            raw_prefix_bits=row['history']['prefix_raw_bits'], raw_suffix_bits=old['suffix_raw_bits'], raw_history_bits=row['history']['teacher_raw_bits'],
            frames=frames,goal192=goal.tolist(),actual192=actual,actual_receipt=deepcopy(actual_receipt),
            old_prediction_receipt=deepcopy(old['array_receipts']['prediction']),original_predicted_Q=old['predicted_Q'],actual_Q=old['actual_Q'],
            action_mean=teacher.kernel.ACTION_MEAN.astype(float).tolist(),action_std=teacher.kernel.ACTION_STD.astype(float).tolist(),
            missing_reason=old['missing_reason'],target_contract=row['target'])
        require(anchor['task_id'] == j['task_id'] and j['goal192'] == old['goal192'], 'same teacher input/target')
        j['job_binding_sha256'] = digest(j); jobs.append(j)
    validate_jobs(jobs)
    config = teacher.overlay.read(Path(p['policy'])/'config.json')
    require(mid.encoder_abi(config)['tokens'] == 257, 'original checkpoint config, no arbitrary shapes')
    abi = stageb.suffix.native_abi(config)
    stats = mid.image_statistics(); source_shas = {str(f):sha(mid.regular(f)) for f in sources()}
    for path,h in {**bindings,**source_shas,**teacher.enc.MODELS}.items(): require(sha(mid.regular(path)) == h,'full input/source/model closure '+path)
    return dict(schema=NAME+'.prepare',contract=contract(),jobs=jobs,bindings=bindings,sources_sha256=source_shas,
                policy=p['policy'],model_bindings=teacher.enc.MODELS,expected_model_state_sha256=p['expected_model_hash'],
                prediction_abi=abi,image_statistics=stats,input_analysis_sha256=LOCAL_SHA,rank_report_sha256=history.RANK_SHA,
                inherited_resources=dict(compiler=c['resources'],actual_RGB_encoding=history.rank.rgb.read(history.rank.REPORT,history.rank.REPORT_SHA)['counts']),**FLAGS)


def verify(prepared,expected):
    p = rgb.read(prepared,expected)
    require(p['schema'] == NAME+'.prepare' and p == build(), 'complete deterministic own prepare/closure')
    return p


def plans(p):
    return [dict(job=j,route=route,run_id=j['job_id']+'.'+route) for j in p['jobs'] for route in ROUTES]


def kinds(route):
    require(route in ROUTES,'registered P1 API route')
    return (('rollout',) if route == 'public_rollout' else ())+('action_encoder','predict','predictor','pred_proj')


class Ledger(stageb.Ledger):
    def __init__(self,p,*args,**kwargs):
        # Initialize storage/topology machinery only. Never call P2 collect/prepare/run or patch globals.
        super().__init__(dict(jobs=[],contract=dict(stage='compare')),*args,**kwargs)
        self.order = [(q['run_id'],k) for q in plans(p) for k in kinds(q['route'])]


class Journal(mid.Journal):
    def __call__(self,value):
        event = dict(index=self.index,previous_sha256=self.previous,payload=deepcopy(value))
        event['event_sha256'] = digest(event); raw = canonical(event)+b'\n'
        require(self.stream.tell()+len(raw) < 16*1024**2,'bounded fixed40 durable journal')
        self.stream.write(raw.decode()); self.stream.flush(); os.fsync(self.stream.fileno())
        self.index += 1; self.previous = event['event_sha256']


def inputs_for(j,stats):
    frames = [teacher.rgb(f['path'],f['file_sha256'],f['pixel_sha256']) for f in j['frames']]
    z = teacher.load_array(j['compiler_folder'],j['latent_receipt'])
    history.equal_bits(z,j['full3'],'exact original latent cache bytes')
    raw = np.frombuffer(bytes.fromhex(j['raw_history_bits']),dtype='<f4').reshape(10,2)
    suffix = np.frombuffer(bytes.fromhex(j['raw_suffix_bits']),dtype='<f4').reshape(5,2)
    tokens = np.concatenate((teacher.kernel.normalized_history(raw),teacher.kernel.normalized_history(suffix)))
    history.equal_bits(tokens,j['outgoing_tokens'],'same full3 outgoing original tokens')
    require(z.dtype == np.float32 and z.shape == (3,192) and tokens.shape == (3,10),'fixed original prediction ABI')
    return dict(rgb=np.stack(frames),normalized_pixels=np.concatenate([mid.normalized_rgb(x,stats) for x in frames],axis=1),
                full3=z[None].copy(),outgoing_tokens=tokens[None],goal192=np.asarray(j['goal192'],np.float32))


def process(p,folder,adapter,ledger,journal,rows,errors):
    for q in plans(p):
        j = q['job']; rid = q['run_id']
        row = dict(run_id=rid,index=j['index'],route=q['route'],job_binding_sha256=j['job_binding_sha256'],status='blocked_not_attempted',arrays={})
        if not errors:
            ledger.active = rid; row['status'] = 'attempted'
            try:
                ledger.check(); journal(dict(event='row_attempt',run_id=rid,binding=j['job_binding_sha256']))
                b = inputs_for(j,p['image_statistics'])
                for key,x in b.items(): row['arrays'][key] = mid.save_array(folder,rid+'.'+key,x)
                for key,x in adapter(q,b).items(): row['arrays'][key] = mid.save_array(folder,rid+'.'+key,x)
                require([e['kind'] for e in ledger.entries if e['run_id'] == rid and e['saved']] == list(kinds(q['route'])),'complete independently executed native route')
                row['status'] = 'saved'; journal(dict(event='row_saved',run_id=rid,arrays=deepcopy(row['arrays'])))
            except BaseException as exc:
                errors.append(mid.failure(rid,exc)); row.update(status='failed',error=errors[-1])
            finally: ledger.active = None
        rows.append(row)
        try: mid.write(folder/(rid+'.row.json'),row)
        except BaseException as exc: errors.append(mid.failure('row_persistence',exc))


def run_cuda(prepared,expected,output,authority):
    require(authority == AUTH,'explicit MAIN authority before torch/model imports')
    p = verify(prepared,expected); output = Path(output).absolute()
    require(output.resolve() == output and not output.exists() and output.name.startswith(NAME+'_') and
            output.parent == ROOT/'results' and shutil.disk_usage(output.parent).free > CAP+FLOOR,'exclusive own output and full storage reserve')
    mid.write(Path(str(Path(prepared).absolute())+'.launch.json'),dict(output=str(output),prepare_sha256=expected))
    output.mkdir(); started = time.monotonic(); mid.write(output/'execution_binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected,authority=authority,**FLAGS))
    rows,errors,hooks,contexts = [],[],mid.Hooks(),ExitStack()
    model = host = before = after = precision_before = precision_after = tf32 = None
    source_ok = False
    def check():
        require(time.monotonic()-started < WALL and mid.used(output) < CAP-FINAL and shutil.disk_usage(output).free > FLOOR,'bounded wall/output/floor at dispatch boundaries')
    with (output/'events.jsonl').open('x') as stream:
        journal = Journal(stream)
        def persist(name,t):
            require(t.device.type == 'cuda' and str(t.dtype) == 'torch.float32','actual CUDA float32 native tensors only')
            return mid.save_array(output,name,t.detach().cpu().numpy().copy())
        ledger = Ledger(p,journal,persist,check)
        try:
            import torch
            from torch.nn.attention import sdpa_kernel, SDPBackend
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
            host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
            tf32 = torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            contexts.enter_context(sdpa_kernel(SDPBackend.MATH)); contexts.enter_context(torch.autocast(device_type='cuda',enabled=False))
            precision_before = mid.checked_precision(torch); ledger.precision = lambda:mid.checked_precision(torch)
            ledger.sync = lambda:torch.cuda.synchronize(device)
            model = load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
            mid.frozen(model); before = teacher.enc.tensor_hash(model)
            require(before == p['expected_model_state_sha256'],'original P1 frozen tensor-hash algorithm and state')
            modules = dict(encode=(model,'encode'),encoder=(model.encoder,'forward'),projector=(model.projector,'forward'),
                rollout=(model,'rollout'),action_encoder=(model.action_encoder,'forward'),predict=(model,'predict'),
                predictor=(model.predictor,'forward'),pred_proj=(model.pred_proj,'forward'))
            for kind,(obj,key) in modules.items():
                original = getattr(obj,key)
                def counted(*args,_kind=kind,_original=original,**kwargs): return ledger.dispatch(_kind,_original,args,kwargs)
                hooks.install(obj,key,counted if contract()['planned_calls'][kind] else ledger.forbid(kind))
            for obj,keys in ((torch.autograd,('grad','backward')),(torch.Tensor,('backward',)),
                (torch.autograd.functional,('jacobian','hessian','jvp','vjp','hvp','vhp')),
                (torch.func,('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value'))):
                for key in keys: hooks.install(obj,key,ledger.forbid('AD.'+key))
            transform = p0.image_transform()
            def adapter(q,b):
                with torch.inference_mode():
                    x = torch.cat([p0._image_tensor(frame,transform=transform,device=device) for frame in b['rgb']],dim=1)
                    stageb.equal(x.detach().cpu().numpy(),b['normalized_pixels'],'original image transform versus saved arithmetic')
                    lat = torch.tensor(b['full3'],dtype=torch.float32,device=device)
                    actions = torch.tensor(b['outgoing_tokens'],dtype=torch.float32,device=device)
                    if q['route'] == 'public_rollout':
                        out = model.rollout({'pixels':x[:,None],'emb':lat[:,None]},actions[:,None],history_size=3)
                        return dict(public_predicted_emb=out['predicted_emb'].detach().cpu().numpy().copy())
                    out = model.predict(lat,model.action_encoder(actions))
                    return dict(cached_predictions=out.detach().cpu().numpy().copy())
            process(p,output,adapter,ledger,journal,rows,errors)
        except BaseException as exc: errors.append(mid.failure('setup_or_execution',exc))
        finally:
            hooks.restore(errors)
            if model is not None:
                try: mid.frozen(model); after = teacher.enc.tensor_hash(model)
                except BaseException as exc: errors.append(mid.failure('final_model',exc))
            if precision_before is not None:
                try: precision_after = mid.checked_precision(torch)
                except BaseException as exc: errors.append(mid.failure('final_precision',exc))
            try:
                contexts.close()
                if tf32 is not None: torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32 = tf32
            except BaseException as exc: errors.append(mid.failure('precision_restore',exc))
            try: verify(prepared,expected); check(); source_ok = True
            except BaseException as exc: errors.append(mid.failure('final_sources_wall',exc))
            for q in plans(p)[len(rows):]:
                row = dict(run_id=q['run_id'],index=q['job']['index'],route=q['route'],job_binding_sha256=q['job']['job_binding_sha256'],status='blocked_not_attempted',arrays={})
                rows.append(row)
                try: mid.write(output/(q['run_id']+'.row.json'),row)
                except BaseException as exc: errors.append(mid.failure('pending_row_persistence',exc))
            complete = (not errors and not ledger.blocked and not ledger.journal_errors and source_ok and before is not None and
                before == after and precision_before == precision_after == mid.precision_contract() and len(rows) == 80 and
                all(r['status'] == 'saved' for r in rows) and len(ledger.entries) == len(ledger.order) == 360)
            if not complete and not errors: errors.append(dict(stage='census',message='incomplete actual native closure'))
            r = dict(schema=NAME+'.report',status='completed' if complete else 'partial_or_failed',contract=contract(),prepare_sha256=expected,
                fixed_slots=40,route_slots=80,rows=rows,entries=ledger.entries,counts=ledger.counts(),errors=errors,
                blocked=ledger.blocked,journal_errors=ledger.journal_errors,host=host,model_hash_before=before,model_hash_after=after,
                precision_before=precision_before,precision_after=precision_after,source_verified=source_ok,model_bindings=p['model_bindings'],
                total_wall_seconds=time.monotonic()-started,journal_sha256=sha(output/'events.jsonl'),journal_tail_sha256=journal.previous,
                artifact_sha256={f.name:sha(f) for f in output.iterdir() if f.is_file()},new_AD_calls=0,new_physics_steps=0,
                equal_FLOPs=None,equal_total_compute=None,**FLAGS)
            mid.write(output/'report.json',r)
    require(complete,'partial/failed; fixed80 rows and spent counts retained, no retry')
    return dict(report=str(output/'report.json'),report_sha256=sha(output/'report.json'),status=r['status'])


def same(a,b,message):
    require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),message)


def lane_buffers(route,source,saved,native):
    """Actual full-array plumbing; numerical route inequality is retained data."""
    outkey = 'public_predicted_emb' if route == 'public_rollout' else 'cached_predictions'
    require(set(saved) == set(source)|{outkey},'entire source/return row buffers')
    for k,x in source.items(): same(saved[k],x,'literal row input '+k)
    def fields(kind,names): require(set(native[kind]) == set(names.split()),'native complete keys '+kind)
    for k in ('action_encoder','predict','predictor','pred_proj'):
        fields(k,'input_0 input_1 output_output' if k in ('predict','predictor') else 'input_0 output_output')
    ae,api,pr,pp = [native[k] for k in ('action_encoder','predict','predictor','pred_proj')]
    require(ae['input_0'].shape == (1,3,10) and ae['output_output'].shape == (1,3,192) and
            all(x.shape == (1,3,192) for x in list(api.values())+list(pr.values())) and
            all(x.shape == (3,192) for x in pp.values()),'strict original module call shapes')
    same(ae['input_0'],source['outgoing_tokens'],'raw6..20 original tokens to AE')
    same(api['input_0'],source['full3'],'literal cached latent to predict')
    same(api['input_1'],ae['output_output'],'AE to predict')
    for n in (0,1): same(pr['input_'+str(n)],api['input_'+str(n)],'predictor input bridge')
    same(pp['input_0'],pr['output_output'][0],'predictor to pred_proj')
    same(api['output_output'],pp['output_output'][None],'pred_proj to predict')
    if route == 'public_rollout':
        fields('rollout','pixels emb action_sequence output_pixels output_emb output_action output_predicted_emb')
        public = native['rollout']
        same(public['pixels'],source['normalized_pixels'][:,None],'public genuine RGB six-dimensional ABI')
        same(public['emb'],source['full3'][:,None],'public same literal cache')
        same(public['action_sequence'],source['outgoing_tokens'][:,None],'public same literal actions')
        for k in ('pixels','emb'): same(public['output_'+k],public[k],'public preserved '+k)
        same(public['output_action'],public['action_sequence'],'n_steps0 public action slice')
        prediction = public['output_predicted_emb']
        require(prediction.shape == (1,1,4,192),'one future5 endpoint, not three futures')
        same(prediction[:,:,:3],public['emb'],'three unchanged public history latents')
        same(prediction[0,0,-1],api['output_output'][0,-1],'public actual paid terminal')
        same(saved[outkey],prediction,'public native return saved')
    else: same(saved[outkey],api['output_output'],'cached native return saved')
    return api['output_output']


def validate_payload(p,r,payloads,load,source_for):
    validate_jobs(p['jobs']); qs = plans(p)
    require(r['schema'] == NAME+'.report' and r['status'] == 'completed' and r['contract'] == contract() and
        r['errors'] == r['blocked'] == r['journal_errors'] == [] and r['host']['device'] == 'cuda:0' and
        r['source_verified'] is True and r['model_hash_before'] == r['model_hash_after'] == p['expected_model_state_sha256'] and
        r['model_bindings'] == p['model_bindings'] and r['precision_before'] == r['precision_after'] == mid.precision_contract(),'complete original source/model/CUDA/precision closure')
    require(all(r[k] is False for k in FLAGS) and r['new_AD_calls'] == r['new_physics_steps'] == 0 and
        r['fixed_slots'] == 40 and r['route_slots'] == len(r['rows']) == 80 and len(r['entries']) == 360 and
        r['equal_FLOPs'] is r['equal_total_compute'] is None and 0 <= r['total_wall_seconds'] < WALL,'full fixed40/80 actual denominator and no invented equal-FLOPs')
    expected,outputs,ix = [],[],0
    for q,row in zip(qs,r['rows'],strict=True):
        j,rid,route = q['job'],q['run_id'],q['route']
        require(all(row[k] == v for k,v in dict(run_id=rid,index=j['index'],route=route,job_binding_sha256=j['job_binding_sha256'],status='saved').items()),'literal row and route order')
        expected.append(dict(event='row_attempt',run_id=rid,binding=j['job_binding_sha256']))
        es = r['entries'][ix:ix+len(kinds(route))]; native = {}
        for kind,e in zip(kinds(route),es,strict=True):
            require(e['index'] == ix and e['run_id'] == rid and e['kind'] == kind and
                    e['native_entered'] is e['native_returned'] is e['saved'] is True and
                    e['precision'] == mid.precision_contract() and 0 <= e['wall_seconds'] < WALL and 'error' not in e,'actual native attempt/entry/return, no RAM execution claim')
            native[kind] = {}
            for k,receipt in e['arrays'].items():
                require(receipt['file'] == rid+'.'+kind+'.'+k+'.npy','exact native durable array path')
                x = load(receipt); require(x.dtype == np.float32,'actual native float32')
                native[kind][k] = x
            ix += 1
        def begin(e): return [dict(event=k,index=e['index'],run_id=rid,kind=e['kind']) for k in ('attempt','enter')]
        def end(e): return dict(event='return_or_error',entry=e)
        off = int(route == 'public_rollout')
        if off: expected += begin(es[0])
        ae,api,pr,pp = es[off:]
        expected += begin(ae)+[end(ae)]+begin(api)+begin(pr)+[end(pr)]+begin(pp)+[end(pp),end(api)]
        if off: expected += [end(es[0])]
        saved = {}
        for k,receipt in row['arrays'].items():
            require(receipt['file'] == rid+'.'+k+'.npy','exact row durable array path'); saved[k] = load(receipt)
        source = source_for(j,p['image_statistics'])
        prediction = lane_buffers(route,source,saved,native)
        outputs.append(dict(predictions=prediction,native=native,source=source))
        expected.append(dict(event='row_saved',run_id=rid,arrays=row['arrays']))
    require(payloads == expected,'entire durable row/native callback topology, not log hashes alone')
    require(set(r['counts']) == set(contract()['planned_calls']),'exact count axes')
    for k,n in contract()['planned_calls'].items():
        es = [e for e in r['entries'] if e['kind'] == k]; counts = stageb.counted(es,k)
        require(r['counts'][k] == counts and all(counts[f] == n for f in ('attempted','entered','returned','saved')) and
                counts['input_axis0_samples'] == n*(3 if k == 'pred_proj' else 1) and counts['input_temporal_rows'] == 3*n,'actual call/shape/axis resource reconciliation '+k)
    return outputs


def summarize(p,outputs):
    rows = []; rank_rows = {route:[] for route in ROUTES}
    for i,j in enumerate(p['jobs']):
        pub,cache = outputs[2*i:2*i+2]
        for k in pub['source']: same(pub['source'][k],cache['source'][k],'same literal source across both routes')
        differences = {kind+'.'+key:stageb.discrepancy(x,cache['native'][kind][key])
            for kind in ('action_encoder','predict','predictor','pred_proj') for key,x in pub['native'][kind].items()}
        old = teacher.load_array(j['compiler_folder'],j['old_prediction_receipt'])
        item = dict(index=i,root_slot=j['root_slot'],condition=j['condition'],role=j['role'],library_label=j['library_label'],
            native_comparison=differences,all_native_buffers_exact_equal=all(v['exact_equal'] for v in differences.values()),
            first_unequal_buffer=next((k for k,v in differences.items() if not v['exact_equal']),None),missing_reason=j['missing_reason'],routes={})
        for route,out in zip(ROUTES,(pub,cache),strict=True):
            terms = history.rank.scalar_terms(out['predictions'][0,-1],j['actual192'],j['goal192'])
            require(terms['actual_Q'] == j['actual_Q'],'original39 actual Q; missing12 remains null')
            item['routes'][route] = dict(**terms,vs_original_prediction=stageb.discrepancy(out['predictions'][0],old))
            rr = {k:deepcopy(j[k]) for k in ('index','root_slot','condition','role','library_index','library_label','anchor_id','goal192','goal_binding')}
            rr.update(suffix_raw_bits=j['raw_suffix_bits'],**terms); rank_rows[route].append(rr)
        rows.append(item)
    ranks = {route:history.rank.assemble(members) for route,members in rank_rows.items()}
    return dict(rows=rows,route_rank_errors=ranks,all40_routes_exact_equal=all(r['all_native_buffers_exact_equal'] for r in rows))


def consume(report,expected):
    r = rgb.read(report,expected); folder = Path(report).absolute().parent
    binding = mid.read(folder/'execution_binding.json'); p = verify(binding['prepared'],binding['prepare_sha256'])
    require(binding == dict(prepared=binding['prepared'],prepare_sha256=r['prepare_sha256'],authority=AUTH,**FLAGS),'exact own launch/prepare/authority')
    require(set(r['artifact_sha256']) == {x.name for x in folder.iterdir() if x.name != 'report.json'} and
            {'execution_binding.json','events.jsonl'} <= set(r['artifact_sha256']) and mid.used(folder) < CAP,'bounded exact whole-run artifact census')
    for name,h in r['artifact_sha256'].items():
        require(Path(name).name == name and sha(mid.regular(folder/name)) == h,'original durable artifact SHA')
    events = folder/'events.jsonl'; require(events.stat().st_size < 16*1024**2 and sha(events) == r['journal_sha256'],'whole bounded journal SHA')
    payloads = []; previous = None
    for i,line in enumerate(events.read_bytes().splitlines(keepends=True)):
        e = rgb.decode(line); h = e['event_sha256']
        require(e['index'] == i and e['previous_sha256'] == previous and line == canonical(e)+b'\n' and
                digest({k:v for k,v in e.items() if k != 'event_sha256'}) == h,'canonical complete durable journal chain')
        previous = h; payloads.append(e['payload'])
    require(previous == r['journal_tail_sha256'],'whole terminal journal tail')
    for row in r['rows']: require(mid.read(folder/(row['run_id']+'.row.json')) == row,'full durable row receipt')
    outputs = validate_payload(p,r,payloads,lambda receipt:mid.load_array(folder,receipt),inputs_for)
    return dict(schema=NAME+'.consume',status='fixed40_both_API_saved_verified_not_admitted',report_sha256=expected,
        prepare_sha256=binding['prepare_sha256'],fixed_slots=40,route_slots=80,missing_actual=[12],counts=r['counts'],
        known_roots=2,independent_roots=0,equal_FLOPs=None,equal_total_compute=None,new_neural_calls=0,new_physics_steps=0,
        **summarize(p,outputs),**FLAGS)


def exclusive_metadata(path,value):
    p = Path(path).absolute()
    require(p.resolve() == p and p.parent == ROOT/'results' and p.name.startswith(NAME+'_') and p.suffix == '.json','own new exclusive metadata only')
    mid.write(p,value)


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='command',required=True)
    a = sub.add_parser('prepare'); a.add_argument('--output',required=True)
    for cmd in ('verify','run-cuda'):
        a = sub.add_parser(cmd); a.add_argument('--prepared',required=True); a.add_argument('--prepare-sha',required=True)
        if cmd == 'run-cuda': a.add_argument('--output',required=True); a.add_argument('--authority',required=True)
    a = sub.add_parser('consume'); a.add_argument('--report',required=True); a.add_argument('--report-sha',required=True); a.add_argument('--output',required=True)
    args = ap.parse_args()
    if args.command == 'prepare':
        p = build(); exclusive_metadata(args.output,p); result = dict(prepared=str(Path(args.output).absolute()),prepare_sha256=sha(args.output),fixed_slots=40,route_slots=80)
    elif args.command == 'verify': verify(args.prepared,args.prepare_sha); result = dict(verified=True,not_execution_authority=True)
    elif args.command == 'run-cuda': result = run_cuda(args.prepared,args.prepare_sha,args.output,args.authority)
    else: result = consume(args.report,args.report_sha); exclusive_metadata(args.output,result)
    print(canonical(result).decode())


if __name__ == '__main__': main()
