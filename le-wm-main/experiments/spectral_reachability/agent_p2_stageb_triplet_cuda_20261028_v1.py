"""Actual all8 triplet cache prerequisite, then 16 public/manual predictions.

Only MAIN calls run-cuda. All preparation and consumption are saved NumPy/PIL.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p2_warmup_history_reconstruct_20261028_v1 as reconstruction

NAME = 'agent_p2_stageb_triplet_cuda_20261028_v1'
inv = reconstruction.inv
suffix, mid = inv.suffix, inv.mid
ROOT, ARMS = inv.ROOT, inv.ARMS
read, sha, require, digest, write = mid.read, mid.sha, mid.require, mid.digest, mid.write
CAP, FLOOR, WALL = 128*1024**2, mid.FLOOR, mid.WALL
FLAGS = dict(P1_admission=False,P2_admission=False,scientific_admission=False,freshness=False)
AUTH = 'MAIN_APPROVED_KNOWN8_STAGEB_HOST_CUDA_ONLY'
RECON_PREP = ROOT/'results/agent_p2_warmup_history_reconstruct_20261028_v1_prepare_MAIN01.json'
RECON_PREP_SHA = '376adbeebb4a9f749269448121ed477d16705392fc3bb9fe212b57f3b21a9a8c'
RECON_REPORT = ROOT/'results/agent_p2_warmup_history_reconstruct_20261028_v1_physical_MAIN01/report.json'
RECON_REPORT_SHA = '89d545d538a75f8ae7778d3d4761651ff5b7866ad2041712b26915b82e303c1f'
RECON_STRICT = ROOT/'results/agent_p2_warmup_history_reconstruct_20261028_v1_consume_MAIN01.json'
RECON_STRICT_SHA = '151e9ec6a36a0bd189476298f32381efd32d21c98781c0bedac06ac660579330'
ROUTES = ('public_rollout','manual')


def contract(stage):
    require(stage in ('cache','compare'), 'explicit cache prerequisite or comparison')
    return dict(stage=stage,roots=4,arms=list(ARMS),branches=8,order='root/arm/route; all8, no outcome selection',
        representation='actual warmup10,warmup15,actualstep5; native full3 original encoder/projector',
        planned_calls=(dict(encode=8,encoder=8,projector=8,predict=0,action_encoder=0,predictor=0,pred_proj=0,rollout=0)
                       if stage == 'cache' else dict(encode=0,encoder=0,projector=0,predict=16,action_encoder=16,predictor=16,pred_proj=16,rollout=8)),
        encoder_samples=24 if stage == 'cache' else 0,projector_samples=24 if stage == 'cache' else 0,
        prediction_batch_samples=16 if stage == 'compare' else 0,prediction_temporal_rows=48 if stage == 'compare' else 0,
        public_ABI=dict(pixels=[1,1,3,3,224,224],emb=[1,1,3,192],actions=[1,1,3,10],history_size=3,
                        n_steps=0,predict_loop_iterations=1,predicted_emb=[1,1,4,192]),
        planned_returns=None,AD=0,physics=0,training=False,CPU_neural_fallback=False,retries=False,
        device='cuda:0',precision=mid.precision_contract(),original_goal_Q_actions_normalizer_unchanged=True,
        output_cap_bytes=CAP,floor_bytes=FLOOR,wall_seconds=WALL,
        wall_scope='operation boundaries/final; inclusive nested times not additive; no preemptive watchdog',**FLAGS)


def own_files():
    return [Path(__file__).resolve(),ROOT/f'tests/test_{NAME}.py',ROOT/f'innerdocs/{NAME}.md']


def actual_triplets():
    """Reconsume complete MAIN reconstruction, not just trust a success string."""
    require(sha(RECON_STRICT) == RECON_STRICT_SHA, 'MAIN strict actual history SHA')
    strict = reconstruction.consume(RECON_PREP,RECON_PREP_SHA,RECON_REPORT,RECON_REPORT_SHA)
    require(strict == read(RECON_STRICT) and strict['status'] == 'history_reconstructed', 'exact full reconstruction consumer equality')
    rp = reconstruction.verify(RECON_PREP,RECON_PREP_SHA)
    snapshot = read(rp['inventory']); previous = inv.gap.census(snapshot['rows'])
    histories = inv.gap.census(strict['rows'])
    original = suffix.collect()
    jobs, bindings = [], {**rp['bindings'],**original['bindings']}
    for path,h in ((RECON_PREP,RECON_PREP_SHA),(RECON_REPORT,RECON_REPORT_SHA),(RECON_STRICT,RECON_STRICT_SHA)):
        bindings[str(path)] = h
    for name,h in read(RECON_REPORT)['artifact_sha256'].items(): bindings[str(RECON_REPORT.parent/name)] = h
    for old in original['jobs']:
        slot,arm = old['root_slot'],old['arm']; row = previous[slot,arm]
        require(row['root_sha256'] == old['root_sha256'] == histories[slot,arm]['root_sha256'], 'all8 exact root join')
        frames = deepcopy(row['frames'])
        if slot:
            h = histories[slot,arm]['history']
            require(h['frames'][2]['pixel_sha256'] == frames[1]['pixel_sha256'] and
                    h['state_at_warmup15'] == frames[1]['state'], 'reconstructed original root15 identity')
            frames[0] = deepcopy(h['frames'][1])
        require(len(frames) == 3 and all(f is not None and f['actual_observation'] is True and
                f['root_sha256'] == old['root_sha256'] for f in frames), 'genuine complete actual triplet')
        require(len({f['path'] for f in frames}) == len({f['observation_id'] for f in frames}) == 3, 'distinct temporal observation witnesses, no same-frame aliases')
        require(frames[-1]['pixel_sha256'] == old['midpoint_pixel_sha256'], 'same actual z5 frame/actions/goal chain')
        for f in frames:
            _,pixel = mid.pixels(f['path'],f['file_sha256'])
            require(pixel == f['pixel_sha256'], 'all actual triplet pixels')
            bindings[f['path']] = f['file_sha256']
        job = deepcopy(old)
        job['source_suffix_job_binding_sha256'] = job.pop('job_binding_sha256')
        job['job_id'] = f'root{slot}_{arm}'
        job['frames'] = frames
        job['frame_roles'] = ['warmup10','warmup15','actualstep5']
        job['cached_teacher_provenance'] = 'old observed slots1/2 plus actual singleton z5; not a native actual-triplet full3 encode receipt'
        job['job_binding_sha256'] = digest(job)
        jobs.append(job)
    require([(j['root_slot'],j['arm']) for j in jobs] == [(i,a) for i in range(4) for a in ARMS], 'complete canonical8')
    return dict(jobs=jobs,bindings=bindings,policy=original['policy'],model_bindings=original['model_bindings'],
        expected_model_state_sha256=original['expected_model_state_sha256'],image_statistics=mid.image_statistics(),
        encoder_abi=mid.encoder_abi(read(Path(original['policy'])/'config.json')),
        prediction_abi=suffix.native_abi(read(Path(original['policy'])/'config.json')),
        inherited_resources=dict(original=original['inherited_resources'],midpoint_encoding=original['inherited_midpoint_encoding_counts'],
                                 reconstruction=strict['accounting']),cached_actual_triplet_full3_receipt_available_in_bound_prior_assets=False)


def build(stage,cache_report=None,cache_sha=None,cache_strict=None,cache_strict_sha=None):
    p = actual_triplets(); p.update(schema=NAME+'.protocol',contract=contract(stage))
    if stage == 'compare':
        require(all(x is not None for x in (cache_report,cache_sha,cache_strict,cache_strict_sha)), 'missing native full3 triplet cache prerequisite: first run cache stage, not 0encoder fallback')
        cache_report,cache_strict = Path(cache_report).absolute(),Path(cache_strict).absolute()
        require(sha(mid.regular(cache_strict)) == cache_strict_sha, 'external cache strict SHA')
        receipt = consume(cache_report,cache_sha)
        require(receipt == read(cache_strict) and receipt['stage'] == 'cache' and receipt['status'] == 'all8_saved_verified_not_admitted', 'exact native full3 cache strict')
        launch = read(Path(cache_report).parent/'execution_binding.json')
        cp = verify(launch['prepared'],launch['prepare_sha256'])
        require(cp['jobs'] == p['jobs'] and cp['model_bindings'] == p['model_bindings'], 'same all8 contexts/model/actions/Q cache')
        r = read(cache_report)
        p['cache_source'] = dict(report=str(Path(cache_report).absolute()),report_sha256=cache_sha,
            strict=str(Path(cache_strict).absolute()),strict_sha256=cache_strict_sha)
        for job,row in zip(p['jobs'],r['rows'],strict=True):
            job['cache_full3'] = dict(folder=str(Path(cache_report).absolute().parent),receipt=row['arrays']['full3'])
            job['job_binding_sha256'] = digest({k:v for k,v in job.items() if k != 'job_binding_sha256'})
        p['bindings'].update(cp['bindings']); p['bindings'].update(cp['sources_sha256'])
        for path in (cache_report,cache_strict,Path(launch['prepared']),Path(cache_report).parent/'execution_binding.json',Path(cache_report).parent/'events.jsonl'):
            p['bindings'][str(Path(path).absolute())] = sha(path)
        for path,h in r['artifact_sha256'].items(): p['bindings'][str(Path(cache_report).parent/path)] = h
        p['inherited_resources']['native_triplet_cache'] = r['counts']
    for path,h in {**p['bindings'],**p['model_bindings']}.items(): require(sha(mid.regular(path)) == h,'bound saved/source/model drift')
    p['sources_sha256'] = {str(path):sha(path) for path in own_files()+[Path(reconstruction.__file__),Path(inv.__file__),Path(suffix.__file__),Path(mid.__file__)]}
    return p


def verify(prepared, expected):
    require(sha(mid.regular(prepared)) == expected,'external prepare SHA')
    p = read(prepared); require(p['schema'] == NAME+'.protocol','own prepare schema')
    source = p.get('cache_source',{})
    rebuilt = build(p['contract']['stage'],source.get('report'),source.get('report_sha256'),source.get('strict'),source.get('strict_sha256'))
    require(p == rebuilt,'exact all8 stage/input/receipt reconstruction')
    return p


def plans(p):
    return [dict(job_id=j['job_id'],job=j,route=route,run_id=j['job_id']+'.'+route)
            for j in p['jobs'] for route in (('cache',) if p['contract']['stage'] == 'cache' else ROUTES)]


def kinds(route):
    return ('encode','encoder','projector') if route == 'cache' else (
        ('rollout',) if route == 'public_rollout' else ()) + ('action_encoder','predict','predictor','pred_proj')


def counted(es,kind):
    axis0 = temporal = 0; times = []
    for e in es:
        if not e['native_entered']: continue
        shape = e['arrays']['pixels' if kind in ('encode','rollout') else 'input_0']['shape']
        axis0 += shape[0]
        temporal += shape[0] * (shape[1]*shape[2] if kind == 'rollout' else
                               1 if kind in ('encoder','projector','pred_proj') else shape[1])
        times.append(e['wall_seconds'])
    return dict(attempted=len(es),entered=sum(e['native_entered'] for e in es),returned=sum(e['native_returned'] for e in es),
                saved=sum(e['saved'] for e in es),input_axis0_samples=axis0,input_temporal_rows=temporal,
                inclusive_wall_seconds=None if any(t is None for t in times) else sum(times))


def inputs_for(job,stats,stage):
    rgbs = [mid.pixels(f['path'],f['file_sha256'])[0] for f in job['frames']]
    rgb = np.stack(rgbs)
    pixels = np.concatenate([mid.normalized_rgb(r,stats) for r in rgbs],axis=1)
    out = dict(rgb=rgb,normalized_pixels=pixels)
    if stage == 'compare':
        cache = job['cache_full3']; lat = mid.load_array(cache['folder'],cache['receipt'])
        require(lat.dtype == np.float32 and lat.shape == (1,3,192),'matching native cached actual triplet full3 required')
        _,b = suffix.intervention(job['paid_mid_context'],job['teacher_mid_context']['latent_history'][-1],job['raw10'],job['action_normalization'])
        raw = suffix.f32(job['raw10'],(10,2))
        equal(b['outgoing_tokens'][1],((raw[:5]-b['action_mean'])/b['action_std']).reshape(10),
              'paid A3 incoming token must match actual sealed prefix5; no stale history')
        out.update(full3=lat,outgoing_tokens=b['outgoing_tokens'][None],raw10=suffix.f32(job['raw10'],(10,2)),
                   action_mean=b['action_mean'],action_std=b['action_std'])
    return out


class Ledger:
    """Exact native call tree including public wrapper; buffers persisted immediately."""
    def __init__(self,p,journal,persist,check=lambda:None,sync=lambda:None,precision=lambda:None):
        self.order = [(q['run_id'],k) for q in plans(p) for k in kinds(q['route'])]
        self.entries, self.stack, self.blocked, self.journal_errors = [], [], [], []
        self.active, self.journal, self.persist, self.check, self.sync, self.precision = None,journal,persist,check,sync,precision

    def emit(self,event):
        try: self.journal(event)
        except BaseException as exc:
            self.journal_errors.append(mid.failure('journal',exc)); raise

    def dispatch(self,kind,callback,args,kwargs):
        try:
            self.check()
            require(not self.blocked and not self.journal_errors and len(self.entries) < len(self.order) and
                    (self.active,kind) == self.order[len(self.entries)],'fixed paid order/cap/no retry')
            parent = [self.entries[n]['kind'] for n in self.stack]
            allowed = dict(encode=[[]],encoder=[['encode']],projector=[['encode']],rollout=[[]],
                action_encoder=[[],['rollout']],predict=[[],['rollout']],predictor=[['predict'],['rollout','predict']],
                pred_proj=[['predict'],['rollout','predict']])
            require(parent in allowed[kind] and all(e['saved'] or e['index'] in self.stack for e in self.entries),'actual nested native topology')
            if kind == 'encode':
                require(not kwargs and len(args) == 1 and set(args[0]) == {'pixels'},'native encode ABI')
                tensors = {'pixels':args[0]['pixels']}
            elif kind == 'rollout':
                require(kwargs == dict(history_size=3) and len(args) == 2 and set(args[0]) == {'pixels','emb'},'public cached rollout ABI, no hidden goal encoding')
                tensors = dict(args[0],action_sequence=args[1])
            else:
                require((kind == 'encoder' and kwargs == dict(interpolate_pos_encoding=True)) or
                        (kind != 'encoder' and not kwargs),'native module kwargs')
                tensors = {'input_'+str(i):v for i,v in enumerate(args)}
            require(all(getattr(v.device,'type',None) == 'cuda' and str(v.dtype) == 'torch.float32' for v in tensors.values()),'CUDA float32 only')
        except BaseException as exc:
            self.blocked.append(mid.failure('predispatch',exc)); raise
        e = dict(index=len(self.entries),run_id=self.active,kind=kind,arrays={},native_entered=False,native_returned=False,saved=False,
                 precision=None,wall_seconds=None)
        self.entries.append(e); begun = None
        try:
            self.emit(dict(event='attempt',index=e['index'],run_id=self.active,kind=kind))
            for key,t in tensors.items(): e['arrays'][key] = self.persist(self.active+'.'+kind+'.'+key,t)
            e['precision'] = self.precision(); self.check()
            self.emit(dict(event='enter',index=e['index'],run_id=self.active,kind=kind))
            self.stack.append(e['index']); e['native_entered'] = True; begun = time.monotonic()
            result = callback(*args,**kwargs); e['native_returned'] = True
            self.sync(); e['wall_seconds'] = time.monotonic()-begun
            outputs = result if kind in ('encode','rollout') else {'output':result.last_hidden_state if kind == 'encoder' else result}
            for key,t in outputs.items(): e['arrays']['output_'+key] = self.persist(self.active+'.'+kind+'.output_'+key,t)
            e['saved'] = True
            return result
        except BaseException as exc:
            e['error'] = mid.failure('native',exc)
            if begun is not None and e['wall_seconds'] is None: e['wall_seconds'] = time.monotonic()-begun
            raise
        finally:
            if self.stack and self.stack[-1] == e['index']: self.stack.pop()
            self.emit(dict(event='return_or_error',entry=deepcopy(e)))

    def forbid(self,kind):
        def denied(*args,**kwargs):
            self.blocked.append(mid.failure('forbidden',RuntimeError(kind)))
            self.emit(dict(event='forbidden',run_id=self.active,kind=kind)); raise RuntimeError('forbidden '+kind)
        return denied

    def counts(self):
        result = {}
        for kind in ('encode','encoder','projector','rollout','action_encoder','predict','predictor','pred_proj'):
            es = [e for e in self.entries if e['kind'] == kind]
            result[kind] = counted(es,kind)
        return result


def process(p,folder,adapter,ledger,journal,rows,errors):
    for q in plans(p):
        j = q['job']; row = dict(run_id=q['run_id'],root_slot=j['root_slot'],arm=j['arm'],route=q['route'],
            job_binding_sha256=j['job_binding_sha256'],status='blocked_not_attempted',arrays={})
        if not errors:
            ledger.active = q['run_id']; row['status'] = 'attempted'
            try:
                ledger.check(); journal(dict(event='row_attempt',run_id=q['run_id'],binding=j['job_binding_sha256']))
                buffers = inputs_for(j,p['image_statistics'],p['contract']['stage'])
                for key,a in buffers.items(): row['arrays'][key] = mid.save_array(folder,q['run_id']+'.'+key,a)
                result = adapter(q,buffers)
                for key,a in result.items(): row['arrays'][key] = mid.save_array(folder,q['run_id']+'.'+key,a)
                require([e['kind'] for e in ledger.entries if e['run_id'] == q['run_id'] and e['saved']] == list(kinds(q['route'])),'complete paid native entries')
                row['status'] = 'saved'; journal(dict(event='row_saved',run_id=q['run_id'],arrays=deepcopy(row['arrays'])))
            except BaseException as exc:
                errors.append(mid.failure(q['run_id'],exc)); row.update(status='failed',error=errors[-1])
            finally: ledger.active = None
        rows.append(row)
        try: write(folder/(q['run_id']+'.row.json'),row)
        except BaseException as exc: errors.append(mid.failure('row_persistence',exc))


def run_cuda(prepared,expected,output,authority):
    require(authority == AUTH,'explicit MAIN HOST CUDA authority before runtime import')
    p = verify(prepared,expected); output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and shutil.disk_usage(output.parent).free > CAP+FLOOR,'new exclusive CUDA run/reserve')
    # Exclusive launch claim prevents silently repeating the same preparation.
    write(Path(str(Path(prepared).absolute())+'.launch.json'),dict(output=str(output),prepare_sha256=expected))
    output.mkdir(); started = time.monotonic()
    write(output/'execution_binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected,authority=authority,**FLAGS))
    errors,rows,hooks,contexts = [],[],mid.Hooks(),ExitStack()
    model = host = before = after = precision_before = precision_after = tf32 = None
    source_ok = False
    def check():
        require(time.monotonic()-started < WALL and mid.used(output) < CAP-1024**2 and shutil.disk_usage(output).free > FLOOR,'wall/output/floor boundary')
    with (output/'events.jsonl').open('x') as stream:
        journal = mid.Journal(stream)
        def persist(name,t):
            require(t.device.type == 'cuda' and str(t.dtype) == 'torch.float32','native original float32 CUDA arrays')
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
            precision_before = mid.checked_precision(torch)
            ledger.precision = lambda:mid.checked_precision(torch); ledger.sync = lambda:torch.cuda.synchronize(device)
            model = load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
            mid.frozen(model); before = mid.model_hash(model)
            require(before == p['expected_model_state_sha256'],'same original frozen model state')
            stage = p['contract']['stage']
            modules = dict(encode=(model,'encode'),encoder=(model.encoder,'forward'),projector=(model.projector,'forward'),
                rollout=(model,'rollout'),action_encoder=(model.action_encoder,'forward'),predict=(model,'predict'),
                predictor=(model.predictor,'forward'),pred_proj=(model.pred_proj,'forward'))
            for kind,(obj,attr) in modules.items():
                original = getattr(obj,attr)
                def counted(*args,_kind=kind,_original=original,**kwargs): return ledger.dispatch(_kind,_original,args,kwargs)
                hooks.install(obj,attr,counted if contract(stage)['planned_calls'][kind] else ledger.forbid(kind))
            for obj,key,kind in ((torch.autograd,'grad','AD.grad'),(torch.autograd,'backward','AD.backward'),(torch.Tensor,'backward','AD.Tensor.backward')):
                hooks.install(obj,key,ledger.forbid(kind))
            for obj,keys in ((torch.autograd.functional,('jacobian','hessian','jvp','vjp','hvp','vhp')),
                             (torch.func,('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value'))):
                for key in keys: hooks.install(obj,key,ledger.forbid('AD.'+key))
            transform = p0.image_transform()
            def adapter(q,b):
                with torch.inference_mode():
                    if stage == 'cache':
                        # Original transform on genuine RGB, not treating latent vectors as images.
                        x = torch.cat([p0._image_tensor(rgb,transform=transform,device=device) for rgb in b['rgb']],dim=1)
                        require(np.array_equal(x.detach().cpu().numpy(),b['normalized_pixels']),'original transform actual triplet pixels')
                        full = model.encode({'pixels':x})['emb']
                        return dict(full3=full.detach().cpu().numpy().copy())
                    x = torch.tensor(b['normalized_pixels'],dtype=torch.float32,device=device)
                    lat = torch.tensor(b['full3'],dtype=torch.float32,device=device)
                    actions = torch.tensor(b['outgoing_tokens'],dtype=torch.float32,device=device)
                    if q['route'] == 'public_rollout':
                        result = model.rollout({'pixels':x[:,None],'emb':lat[:,None]},actions[:,None],history_size=3)
                        return dict(public_predicted_emb=result['predicted_emb'].detach().cpu().numpy().copy())
                    result = model.predict(lat,model.action_encoder(actions))
                    return dict(manual_predictions=result.detach().cpu().numpy().copy())
            process(p,output,adapter,ledger,journal,rows,errors)
        except BaseException as exc: errors.append(mid.failure('setup_or_execution',exc))
        finally:
            hooks.restore(errors)
            if model is not None:
                try: mid.frozen(model); after = mid.model_hash(model)
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
            if len(rows) < len(plans(p)):
                missing = plans(p)[len(rows):]
                for q in missing:
                    row = dict(run_id=q['run_id'],root_slot=q['job']['root_slot'],arm=q['job']['arm'],route=q['route'],
                        job_binding_sha256=q['job']['job_binding_sha256'],status='blocked_not_attempted',arrays={})
                    rows.append(row)
                    try: write(output/(q['run_id']+'.row.json'),row)
                    except BaseException as exc: errors.append(mid.failure('missing_row_persistence',exc))
            complete = (not errors and not ledger.blocked and not ledger.journal_errors and source_ok and before is not None and
                before == after and precision_before == precision_after == mid.precision_contract() and
                all(r['status'] == 'saved' for r in rows) and len(ledger.entries) == len(ledger.order))
            if not complete and not errors: errors.append(dict(stage='census',message='incomplete resource or integrity closure'))
            r = dict(schema=NAME+'.report',stage=p['contract']['stage'],status='completed' if complete else 'partial_or_failed',
                prepare_sha256=expected,contract=p['contract'],rows=rows,entries=ledger.entries,counts=ledger.counts(),errors=errors,
                blocked=ledger.blocked,journal_errors=ledger.journal_errors,host=host,model_hash_before=before,model_hash_after=after,
                source_verified=source_ok,model_bindings=p['model_bindings'],precision_before=precision_before,precision_after=precision_after,
                root_denominator=4,arm_denominator=8,route_denominator=len(plans(p)),total_wall_seconds=time.monotonic()-started,
                journal_sha256=sha(output/'events.jsonl'),journal_tail_sha256=journal.previous,
                artifact_sha256={f.name:sha(f) for f in output.iterdir() if f.is_file()},new_AD_calls=0,new_physics_steps=0,**FLAGS)
            write(output/'report.json',r)
    if not complete: raise RuntimeError('partial/failed stageB; full denominator retained, no retry')
    return dict(report=str(output/'report.json'),report_sha256=sha(output/'report.json'),status=r['status'])


def equal(a,b,message):
    require(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a,b),message)


def validate_run(folder,r,p):
    """Strict resource/ABI/plumbing closure; route mismatches are retained data."""
    folder = Path(folder).absolute(); stage = p['contract']['stage']; qs = plans(p)
    require(r['schema'] == NAME+'.report' and r['stage'] == stage and r['status'] == 'completed' and
        r['contract'] == contract(stage) and r['errors'] == r['blocked'] == r['journal_errors'] == [] and
        r['host']['device'] == 'cuda:0' and r['source_verified'] is True and
        r['model_hash_before'] is not None and r['model_hash_before'] == r['model_hash_after'] == p['expected_model_state_sha256'] and
        r['model_bindings'] == p['model_bindings'] and r['precision_before'] == r['precision_after'] == mid.precision_contract(),
        'completed original model/source/precision receipt')
    require(all(r[k] is False for k in FLAGS) and r['new_AD_calls'] == r['new_physics_steps'] == 0 and
        r['root_denominator'] == 4 and r['arm_denominator'] == 8 and r['route_denominator'] == len(qs) and
        len(r['rows']) == len(qs) and len(r['entries']) == sum(len(kinds(q['route'])) for q in qs) and
        0 <= r['total_wall_seconds'] < WALL, 'complete fixed denominators and scope')
    require(set(r['artifact_sha256']) == {f.name for f in folder.iterdir() if f.name != 'report.json'} and
        {'execution_binding.json','events.jsonl'} <= set(r['artifact_sha256']) and
        sum(f.stat().st_size for f in folder.iterdir()) < CAP, 'bounded complete own manifest')
    for name,h in r['artifact_sha256'].items():
        require(Path(name).name == name and sha(mid.regular(folder/name)) == h,'exact flat artifact')
    binding = read(folder/'execution_binding.json')
    require(binding['prepare_sha256'] == r['prepare_sha256'] and binding['authority'] == AUTH and
        all(binding[k] is False for k in FLAGS),'execution binding')
    require(sha(folder/'events.jsonl') == r['journal_sha256'],'journal SHA')
    payloads, previous = [],None
    for i,line in enumerate((folder/'events.jsonl').read_bytes().splitlines()):
        e = json.loads(line); h = e.pop('event_sha256')
        require(e['index'] == i and e['previous_sha256'] == previous and digest(e) == h,'unbroken actual ledger chain')
        previous = h; payloads.append(e['payload'])
    require(previous == r['journal_tail_sha256'],'journal tail')
    expected,outputs,entry_index = [],[],0
    for q,row in zip(qs,r['rows'],strict=True):
        j,rid,route = q['job'],q['run_id'],q['route']
        require(digest({k:v for k,v in j.items() if k != 'job_binding_sha256'}) == j['job_binding_sha256'] and
            all(row[k] == v for k,v in dict(run_id=rid,root_slot=j['root_slot'],arm=j['arm'],route=route,
                    job_binding_sha256=j['job_binding_sha256'],status='saved').items()) and
            read(folder/(rid+'.row.json')) == row,'ordered full denominator row/source identity')
        expected.append(dict(event='row_attempt',run_id=rid,binding=j['job_binding_sha256']))
        ks = kinds(route); es = r['entries'][entry_index:entry_index+len(ks)]; native = {}
        for kind,e in zip(ks,es,strict=True):
            require(e['index'] == entry_index and e['run_id'] == rid and e['kind'] == kind and
                e['native_entered'] is e['native_returned'] is e['saved'] is True and
                e['precision'] == mid.precision_contract() and 0 <= e['wall_seconds'] < WALL,'exact native paid entry')
            native[kind] = {}
            for key,receipt in e['arrays'].items():
                require(receipt['file'] == rid+'.'+kind+'.'+key+'.npy','native input/output filename')
                a = mid.load_array(folder,receipt); require(a.dtype == np.float32,'original native float32')
                native[kind][key] = a
            entry_index += 1
        # Rebuild actual native call tree independently of report count summaries.
        def begin(e): return [dict(event=k,index=e['index'],run_id=rid,kind=e['kind']) for k in ('attempt','enter')]
        def end(e): return dict(event='return_or_error',entry=e)
        if route == 'cache':
            expected += begin(es[0])+begin(es[1])+[end(es[1])]+begin(es[2])+[end(es[2]),end(es[0])]
        else:
            off = int(route == 'public_rollout')
            if off: expected += begin(es[0])
            a,b,c,d = es[off:]
            expected += begin(a)+[end(a)]+begin(b)+begin(c)+[end(c)]+begin(d)+[end(d),end(b)]
            if off: expected += [end(es[0])]
        source = inputs_for(j,p['image_statistics'],stage); saved = {}
        require(set(row['arrays']) == set(source) | ({'full3'} if route == 'cache' else
                {'public_predicted_emb'} if route == 'public_rollout' else {'manual_predictions'}),'all source and native row buffers')
        for key,receipt in row['arrays'].items():
            require(receipt['file'] == rid+'.'+key+'.npy','row buffer filename')
            saved[key] = mid.load_array(folder,receipt)
        for key,a in source.items(): equal(saved[key],a,'exact actual RGB/normalizer/action/cache buffer '+key)
        def fields(kind,names): require(set(native[kind]) == set(names.split()),'exact native buffer keys '+kind)
        if route == 'cache':
            fields('encode','pixels output_pixels output_emb'); fields('encoder','input_0 output_output'); fields('projector','input_0 output_output')
            enc,proj,api = native['encoder'],native['projector'],native['encode']
            require(enc['input_0'].shape == (3,3,224,224) and enc['output_output'].shape ==
                (3,p['encoder_abi']['tokens'],p['encoder_abi']['encoder_dim']) and
                proj['input_0'].shape == proj['output_output'].shape == (3,192) and saved['full3'].shape == (1,3,192),'derived native full3 encoding ABI')
            equal(api['pixels'],source['normalized_pixels'],'original actual triplet transform')
            equal(api['output_pixels'],api['pixels'],'encode returned same pixels')
            equal(enc['input_0'],api['pixels'][0],'encode time-flatten input')
            equal(proj['input_0'],enc['output_output'][:,0],'original CLS projector input')
            equal(saved['full3'],proj['output_output'][None],'native triplet full3')
            equal(api['output_emb'],saved['full3'],'public encode output cache')
            outputs.append(dict(full3=saved['full3'],row=row,native=native))
        else:
            fields('action_encoder','input_0 output_output'); fields('predict','input_0 input_1 output_output')
            fields('predictor','input_0 input_1 output_output'); fields('pred_proj','input_0 output_output')
            ae,api,pr,pp = [native[k] for k in ('action_encoder','predict','predictor','pred_proj')]
            require(ae['input_0'].shape == (1,3,10) and ae['output_output'].shape == (1,3,192) and
                all(a.shape == (1,3,192) for a in api.values()) and all(a.shape == (1,3,192) for a in pr.values()) and
                all(a.shape == (3,192) for a in pp.values()),'native all full3 prediction buffers')
            equal(ae['input_0'],source['outgoing_tokens'],'original three outgoing tokens, no extra bootstrap')
            equal(api['input_0'],source['full3'],'matching cached actual triplet')
            equal(api['input_1'],ae['output_output'],'AE->predict')
            equal(pr['input_0'],api['input_0'],'predict->predictor latents')
            equal(pr['input_1'],api['input_1'],'predict->predictor actions')
            equal(pp['input_0'],pr['output_output'][0],'predictor->pred_proj')
            equal(api['output_output'],pp['output_output'][None],'pred_proj->predict full3')
            if route == 'public_rollout':
                fields('rollout','pixels emb action_sequence output_pixels output_emb output_action output_predicted_emb')
                public = native['rollout']
                equal(public['pixels'],source['normalized_pixels'][:,None],'genuine public six-dimensional pixel ABI')
                equal(public['emb'],source['full3'][:,None],'public full3 cache ABI')
                equal(public['action_sequence'],source['outgoing_tokens'][:,None],'public action ABI')
                for key in ('pixels','emb'): equal(public['output_'+key],public[key],'public preserved '+key)
                equal(public['output_action'],public['action_sequence'],'n_steps0 public action slice')
                prediction = public['output_predicted_emb']
                require(prediction.shape == (1,1,4,192),'exactly one packet; H3 is history, not three futures')
                equal(prediction[:,:,:3],public['emb'],'public preserves original three history latents')
                equal(prediction[0,0,-1],api['output_output'][0,-1],'public endpoint uses actual paid API terminal')
                equal(saved['public_predicted_emb'],prediction,'public returned native output')
            else: equal(saved['manual_predictions'],api['output_output'],'manual independent output')
            outputs.append(dict(predictions=api['output_output'],native=native,row=row))
        expected.append(dict(event='row_saved',run_id=rid,arrays=row['arrays']))
    require(payloads == expected,'complete raw ledger topology and full rows, not just summary counters')
    planned = dict(encode=0,encoder=0,projector=0,rollout=0,action_encoder=0,predict=0,predictor=0,pred_proj=0)
    planned.update(contract(stage)['planned_calls'])
    for k,n in planned.items():
        es = [e for e in r['entries'] if e['kind'] == k]
        expected_counts = counted(es,k)
        require(r['counts'][k] == expected_counts and all(expected_counts[key] == n for key in ('attempted','entered','returned','saved')) and
                expected_counts['input_axis0_samples'] == n*(3 if k in ('encoder','projector','pred_proj') else 1) and
                expected_counts['input_temporal_rows'] == 3*n,'actual complete paid per-axis budget '+k)
    return outputs


def discrepancy(a,b):
    require(a.shape == b.shape,'same-shape diagnostic, never broadcast')
    return dict(exact_equal=np.array_equal(a,b),max_abs_difference=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),
                mean_squared_difference=float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2)))


def summary(p,outputs):
    rows = []; stage = p['contract']['stage']
    for i,j in enumerate(p['jobs']):
        row = dict(root_slot=j['root_slot'],arm=j['arm'],root_sha256=j['root_sha256'],
                   job_binding_sha256=j['job_binding_sha256'],goal_sha256=digest(j['goal']),metric_id=j['metric_id'])
        old = suffix.f32(j['teacher_mid_context']['latent_history'],(3,192))[None]
        if stage == 'cache':
            new = outputs[i]['full3']
            row.update(cached_full3=outputs[i]['row']['arrays']['full3'],
                actual_triplet_vs_old_teacher_full3=discrepancy(new,old),older2=discrepancy(new[:,:2],old[:,:2]),
                actualstep5=discrepancy(new[:,-1],old[:,-1]))
        else:
            pub,manual = outputs[2*i:2*i+2]
            checks = {}
            for kind in ('action_encoder','predictor','pred_proj','predict'):
                for key,a in pub['native'][kind].items(): checks[kind+'.'+key] = discrepancy(a,manual['native'][kind][key])
            zpub,zmanual = pub['predictions'][0,-1],manual['predictions'][0,-1]
            row.update(native_route_comparison=checks,first_unequal_buffer=next((k for k,v in checks.items() if not v['exact_equal']),None),
                all_native_buffers_exact_equal=all(v['exact_equal'] for v in checks.values()),
                public_terminal=mid.numeric(j['actual_terminal_z'],zpub.tolist(),j['goal'],j['projection']),
                manual_terminal=mid.numeric(j['actual_terminal_z'],zmanual.tolist(),j['goal'],j['projection']))
        rows.append(row)
    return rows


def consume(report,expected):
    require(sha(mid.regular(report)) == expected,'external actual run report SHA')
    folder = Path(report).absolute().parent; r = read(report); binding = read(folder/'execution_binding.json')
    require(binding['prepare_sha256'] == r['prepare_sha256'],'same externally sealed preparation')
    p = verify(binding['prepared'],binding['prepare_sha256'])
    outputs = validate_run(folder,r,p)
    return dict(schema=NAME+'.consume',status='all8_saved_verified_not_admitted',stage=p['contract']['stage'],report_sha256=expected,
        prepare_sha256=binding['prepare_sha256'],root_denominator=4,arm_denominator=8,route_denominator=len(plans(p)),
        rows=summary(p,outputs),counts=r['counts'],sample_axes=contract(p['contract']['stage']),
        inherited_resources=p['inherited_resources'],new_AD_calls=0,new_physics_steps=0,
        limitation='known4 only; ABI equality does not prove shared protocol correctness, physical task success or P2 mechanism',**FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    a = sub.add_parser('prepare'); a.add_argument('--stage',choices=('cache','compare'),required=True); a.add_argument('--output',type=Path,required=True)
    for arg in ('cache-report','cache-sha','cache-strict','cache-strict-sha'): a.add_argument('--'+arg)
    for name in ('verify','run-cuda'):
        a = sub.add_parser(name); a.add_argument('--prepared',type=Path,required=True); a.add_argument('--prepare-sha',required=True)
        if name == 'run-cuda': a.add_argument('--output',type=Path,required=True); a.add_argument('--authority',required=True)
    a = sub.add_parser('consume'); a.add_argument('--report',type=Path,required=True); a.add_argument('--report-sha',required=True); a.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        p = build(args.stage,args.cache_report,args.cache_sha,args.cache_strict,args.cache_strict_sha); write(args.output,p)
        result = dict(prepared=str(args.output.absolute()),prepare_sha256=sha(args.output),stage=args.stage)
    elif args.command == 'verify':
        p = verify(args.prepared,args.prepare_sha); result = dict(status='saved_verified_not_CUDA_authority',stage=p['contract']['stage'])
    elif args.command == 'run-cuda': result = run_cuda(args.prepared,args.prepare_sha,args.output,args.authority)
    else: result = consume(args.report,args.report_sha); write(args.output,result)
    print(mid.gap.encode(result).decode())


if __name__ == '__main__': main()
