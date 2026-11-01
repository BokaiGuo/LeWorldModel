"""Predeclared two-stage P1 witness PRODUCER, not P1 admission.

16 fixed prefix slots, a fresh own-context suffix J for EVERY slot, four
nonlinear suffix evaluations each; independent full-H2 winner verification.
Imports/prepare/verify are model-free. Only explicit run --authorize-host-cuda
loads the existing model, never a simulator. No threshold/efficacy defaults.
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

from experiments.spectral_reachability import p1_conditional_direct_screen_20261027_v1 as screen
from experiments.spectral_reachability import p1_direct_cuda_calibration_main_20261027_v2 as upstream

core, source, io = screen.core, upstream.source, upstream.io
NAME = 'p1_two_stage_witness_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
RESERVE = 200*1024**2
OUTPUT_LIMIT = 64*1024**2
require = core.require
# Engineering protocol fixed BEFORE reading new witness gains. Not attributed
# to user parameter choices, and not an independent threshold authority.
PROTOCOL = dict(prefix_slots=16, suffix_slots=4, candidates=64, damping=.002,
    ridge_objective='mean_square_linear_residual_plus_damping_times_step_norm_squared',
    scales=[.25, .5, 1.], spectral_modes=3, radius_fractions=[1/3, 1.], signs=[-1, 1],
    prefix_order='zero; root_H2_prefix_DLS_3scales; radius/mode/sign',
    suffix_order='zero; own_context_DLS_3scales',
    root_compile='full192x20_H2_at_zero_both_packets; prefix_response_first10_columns',
    suffix_compile='fresh192x10_at_zero_suffix_in_each_actual_full_predicted_prefix_context',
    selection='all64_original_Q_nonlinear_terminal_MSE; stable_first_tie',
    domain='independent_original_per_packet_logit_L2_balls; NumPy_float32_tanh_raw_box',
    reuse='prefix_transition_shared_only_among_its_4_suffixes; no_context_or_derivative_dedup',
    verification='winner_new_nominalH2_plus_new_wholeH2_plus_new_segmentedH2',
    physical_future_labels_read=False, P1_admission=False, model_task_eligible=False)


def budgets():
    return dict(producer=dict(predict=120, action_encoder=120, jacobian=17, svd=17,
        vjp_directions=3264, linear_solve=17, prefix_generation=1, suffix_proposal=64, selection=64),
        witness=dict(predict=6, action_encoder=6, jacobian=0, svd=0,
        vjp_directions=0, linear_solve=0, prefix_generation=0, suffix_proposal=0, selection=0))


def counts(events):
    return {phase: {r: dict(attempted=sum(e['samples'] for e in events if e['phase']==phase and e['resource']==r),
        completed=sum(e['samples'] for e in events if e['phase']==phase and e['resource']==r and e['completed']))
        for r in caps} for phase, caps in budgets().items()}


class Meter(screen.Meter):
    def __init__(self, sink):
        super().__init__({r: sum(v[r] for v in budgets().values()) for r in budgets()['producer']}, sink)
        self.phase, self.stage = 'producer', 'root_nominal'

    def emit(self, kind, **values):
        self.sink(dict(kind=kind, stage=self.stage, **deepcopy(values)))

    def begin(self, resource):
        require(resource in budgets()[self.phase], 'known scoped resource')
        used = sum(e['samples'] for e in self.extended if e['phase']==self.phase and e['resource']==resource)
        require(used < budgets()[self.phase][resource], 'predispatch '+self.phase+' '+resource+' cap')
        e = dict(index=len(self.extended), phase=self.phase, evaluation_id=self.phase,
                 resource=resource, samples=1, completed=False)
        self.extended.append(e)
        idx = None
        if resource in core.RESOURCES:
            idx = len(self.events); self.events.append(dict(e, index=idx))
        self.emit('operation_attempted', event=e)
        return e, idx


def edge(task, before, logits, after):
    core.context(before); core.context(after)
    a = core.array(logits, (10,))
    require(np.linalg.norm(a) <= task['packet_logit_cap']+1e-12, 'per-packet logit cap')
    raw = np.tanh(a.astype(np.float32)).reshape(5, 2)
    token = core.normalized_tokens(task, raw)[0]
    require(after['latent_history'][:2] == before['latent_history'][1:], 'packet latent chronology')
    require(np.array_equal(after['action_history'], np.concatenate((np.asarray(before['action_history']), token[None]))[-2:]),
            'exact own-context normalized incoming token history')
    return raw.tolist()


def feasible(x, task):
    return screen.projected_logits(np.stack((core.array(x, (10,)), np.zeros(10))), task['packet_logit_cap'])[0]


def compile_response(task, jacobian, terminal, cap, meter):
    """One SVD, one bounded ridge solve; same SVD reused for spectral slots.

    Nonnegative L2 multiplier bisection is CPU arithmetic, not more SVDs.
    Every coefficient evaluation and iteration count is saved in the receipt.
    """
    j = core.array(jacobian, (192, 10)); z = core.array(terminal, (192,))
    q = np.asarray(task['projection']); a = (q@j)/np.sqrt(192)
    residual = (q@(np.asarray(task['goal'])-z))/np.sqrt(192)
    _, s, vh = meter.call('svd', lambda: np.linalg.svd(a, full_matrices=False))
    for v in vh:
        if v[np.argmax(np.abs(v))] < 0: v *= -1
    # Normal-equation RHS in V coordinates preserves canonicalized signs and
    # avoids division by possibly zero/tiny singular values entirely.
    target = vh@a.T@residual
    arithmetic = dict(coefficient_evaluations=0, bracket_doublings=0, bisection_iterations=0)
    def solve():
        def at(multiplier):
            arithmetic['coefficient_evaluations'] += 1
            return vh.T@(target/(s*s+PROTOCOL['damping']+multiplier))
        value = at(0.); multiplier = 0.
        if np.linalg.norm(value) > cap:
            lo, hi = 0., 1.
            while np.linalg.norm(at(hi)) > cap:
                require(arithmetic['bracket_doublings'] < 1024, 'finite ridge bracket')
                hi *= 2; arithmetic['bracket_doublings'] += 1
            for _ in range(70):
                mid = (lo+hi)/2
                if np.linalg.norm(at(mid)) > cap: lo = mid
                else: hi = mid
                arithmetic['bisection_iterations'] += 1
            multiplier = hi; value = at(hi)
        return value, multiplier
    step, multiplier = meter.call('linear_solve', solve)
    receipt = dict(step=feasible(step, task).tolist(), right_vectors=vh.tolist(), singular_values=s.tolist(),
                   ball_multiplier=multiplier, radius=cap, arithmetic=arithmetic,
                   objective=PROTOCOL['ridge_objective'])
    meter.emit('compiled_response', receipt=receipt)
    return receipt


def prefix_slots(task, compilation, meter):
    def generate():
        values = [np.zeros(10)]
        values += [feasible(scale*np.asarray(compilation['step']), task) for scale in PROTOCOL['scales']]
        for radius in PROTOCOL['radius_fractions']:
            for mode in range(PROTOCOL['spectral_modes']):
                for sign in PROTOCOL['signs']:
                    values.append(feasible(sign*radius*task['packet_logit_cap']*np.asarray(compilation['right_vectors'][mode]), task))
        require(len(values)==16, 'fixed prefix census')
        return [np.asarray(v).tolist() for v in values]
    slots = meter.call('prefix_generation', generate)
    meter.emit('prefix_slots_frozen', logits=slots)
    return slots


class NativeCuda(screen.NativeCuda):
    def packet(self, task, context, logits, meter, label):
        require(core.task_id(task)==self.bound_id, 'same native task')
        core.context(context)
        a = core.array(logits, (10,)); require(np.linalg.norm(a)<=task['packet_logit_cap']+1e-12, 'packet ball')
        raw = np.tanh(a.astype(np.float32)).reshape(5, 2)
        with self.torch.no_grad():
            z, h = self._step(task, self._tensor(context['latent_history'])[None],
                               self._tensor(context['action_history'])[None], self._tensor(raw))
            saved = self._saved(z, h)
        edge(task, context, a, saved)
        meter.emit('packet_completed', label=label, input_context_sha256=core.digest(context),
                   logits=a.tolist(), commands=raw.tolist(), context=saved)
        return saved

    def suffix_jacobian(self, task, context, nominal, meter):
        require(core.task_id(task)==self.bound_id, 'same suffix task')
        core.context(context); core.context(nominal)
        torch = self.torch
        with torch.enable_grad():
            # A fresh local graph in THIS prefix context, not slices of root J.
            raw = self._tensor(np.zeros((5, 2))).requires_grad_(True)
            z, h = self._step(task, self._tensor(context['latent_history'])[None],
                self._tensor(context['action_history'])[None], raw)
            saved = self._saved(z, h)
            meter.emit('suffix_AD_forward', input_context_sha256=core.digest(context), context=saved)
            require(saved==nominal, 'suffix AD forward differs from paid zero-suffix nominal')
            terminal = z[0, -1]; rows = []
            for i in range(192):
                row = meter.call('vjp_directions', lambda: torch.autograd.grad(
                    terminal[i], raw, retain_graph=i<191, create_graph=False)[0])
                rows.append(row.detach().cpu().numpy().reshape(10).astype(float))
                meter.emit('suffix_J_row', row=i, derivative=rows[-1].tolist())
            result = dict(input_context_sha256=core.digest(context), task_id=self.bound_id,
                at_logits=[0.]*10, terminal=saved['latent_history'][-1], jacobian=np.stack(rows).tolist())
            # tanh'(0)=1: normalization already in graph. No extra chain/divisor.
            meter.emit('suffix_J_complete', derivative=result)
            return result


def run_task(task, backend, sink):
    """Backend is native CUDA in MAIN, pure arrays in mock tests only."""
    core.validate_task(task)
    require(task['packet_logit_cap']==float(.75*np.sqrt(5)), 'original P1 per-packet cap')
    t = deepcopy(task); meter = Meter(sink)
    report = dict(schema=NAME+'.root', status='started', task_id=core.task_id(t),
        protocol_sha256=core.digest(PROTOCOL), candidates=[], prefix_compilations=[], winner=None, witness=None,
        journal=meter.events, extended_journal=meter.extended, budget=budgets(), error=None, secondary_errors=[],
        P1_admission=False, model_task_eligible=False, physical_future_labels_read=False,
        direct_plateau_established_by_this_producer=False,
        missing=['direct finite-budget stall evidence', 'independent threshold authority and confirmation'])
    try:
        meter.emit('producer_started', task_id=report['task_id'], protocol=PROTOCOL, budgets=budgets())
        with backend.guard(meter):
            nominal = screen.evaluate(t, np.zeros((2, 10)), backend, meter)
            meter.stage = 'root_fullH2_J'
            z, joint = meter.call('jacobian', lambda: backend.jacobian(t, nominal, meter))
            joint = core.array(joint, (192, 20))
            require(np.array_equal(z, nominal['contexts'][-1]['latent_history'][-1]), 'root fullH2 AD nominal')
            meter.emit('root_fullH2_J_complete', terminal=np.asarray(z).tolist(), jacobian=joint.tolist())
            compiled = compile_response(t, joint[:, :10], z, t['packet_logit_cap'], meter)
            prefixes = prefix_slots(t, compiled, meter)
            best = None
            for i, prefix in enumerate(prefixes):
                meter.stage = f'prefix_{i}'
                mid = backend.packet(t, t['root_context'], prefix, meter, 'actual_prefix')
                prefix_raw = edge(t, t['root_context'], prefix, mid)
                meter.emit('actual_prefix', index=i, logits=prefix, context=mid)
                zero_end = backend.packet(t, mid, [0.]*10, meter, 'suffix_nominal')
                edge(t, mid, [0.]*10, zero_end)
                derivative = meter.call('jacobian', lambda: backend.suffix_jacobian(t, mid, zero_end, meter))
                require(derivative['input_context_sha256']==core.digest(mid) and derivative['task_id']==report['task_id'] and
                    derivative['at_logits']==[0.]*10 and derivative['terminal']==zero_end['latent_history'][-1],
                    'fresh suffix derivative must bind actual full prefix context / same Q goal task')
                local = compile_response(t, derivative['jacobian'], derivative['terminal'], t['packet_logit_cap'], meter)
                report['prefix_compilations'].append(dict(index=i, input_context_sha256=core.digest(mid),
                    derivative=derivative, compilation=local))
                for j, scale in enumerate([0.]+PROTOCOL['scales']):
                    suffix = meter.call('suffix_proposal', lambda: feasible(scale*np.asarray(local['step']), t))
                    end = backend.packet(t, mid, suffix, meter, f'candidate_{i}_{j}')
                    suffix_raw = edge(t, mid, suffix, end)
                    candidate = dict(index=4*i+j, prefix_index=i, suffix_index=j,
                        logits=[prefix, suffix.tolist()], commands=prefix_raw+suffix_raw,
                        contexts=[deepcopy(mid), deepcopy(end)])
                    report['candidates'].append(candidate)
                    def select():
                        candidate['cost'] = core.model_cost(t, end['latent_history'][-1])
                        return candidate if best is None or candidate['cost']<best['cost'] else best
                    best = meter.call('selection', select)
                    meter.emit('candidate_complete', candidate=candidate, incumbent_index=best['index'])
            report['winner'] = deepcopy(best)
            meter.phase, meter.stage = 'witness', 'independent_winner_H2_replays'
            witness = screen.evaluate(t, best['logits'], backend, meter)
            require(witness['contexts']==best['contexts'] and witness['commands']==best['commands'] and witness['cost']==best['cost'],
                    'independent winner H2/whole/segmented exact witness')
            report['witness'] = witness
        work = counts(meter.extended)
        require(all(work[p][r]==dict(attempted=n, completed=n) for p, caps in budgets().items() for r,n in caps.items()),
                'all stages complete exact budget, no refunded failed work')
        report['status'] = 'witness_completed_not_admitted'
    except BaseException as exc:
        report['status']='failed'; report['error']=dict(type=type(exc).__name__, message=str(exc))
    report['resources'] = counts(meter.extended)
    try:
        meter.emit('producer_finished', status=report['status'], error=report['error'], resources=report['resources'], witness=report['witness'])
    except BaseException as exc:
        err = dict(type=type(exc).__name__, message=str(exc))
        if report['error'] is None: report['error']=err; report['status']='failed'
        else: report['secondary_errors'].append(err)
    return report


class Store:
    """One global output quota +200MiB floor including reports, no overwrites."""
    def __init__(self, output):
        self.output=Path(output).absolute()
        require(not self.output.exists(), 'NEW output only')
        require(shutil.disk_usage(self.output.parent).free >= RESERVE+OUTPUT_LIMIT, 'global output budget plus reserve')
        self.output.mkdir(); self.used=0; self.journal_sizes={}

    def put(self, name, value, *, append=False):
        path=self.output/name
        require(path.parent==self.output and '..' not in Path(name).parts, 'flat local artifacts only')
        raw=(core.canonical(value)+'\n').encode()
        require(len(raw)<=8*1024**2 and self.used+len(raw)<=OUTPUT_LIMIT, 'bounded artifact / global output quota')
        require(shutil.disk_usage(self.output).free>=RESERVE+len(raw), '200MiB before every write')
        if append:
            require(self.journal_sizes.get(name,0)+len(raw)<=16*1024**2, 'per journal quota')
        with path.open('ab' if append else 'xb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        self.used+=len(raw)
        if append: self.journal_sizes[name]=self.journal_sizes.get(name,0)+len(raw)

    def sink(self, name):
        self.put(name, dict(kind='journal_created'))
        return lambda e:self.put(name,e,append=True)


def collect(direct_prepared, direct_prepare_sha, task_directory):
    """Only task/input/provenance/accounting metadata; NEVER direct loss curves.

    Reads aggregate upstream report (no loss fields), native image receipts and
    tasks, not per-method report/trajectory/candidate outcomes.
    """
    prepared=Path(direct_prepared).absolute(); directory=Path(task_directory).absolute()
    p=upstream.verify(prepared,direct_prepare_sha)
    report_path=directory/'report.json'; r=io.read(report_path)
    require(r['prepare_sha256']==direct_prepare_sha and r['source_verified'] is True and
        r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after'] and
        r['actual_image_resources']=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')},
        'actual native task acquisition/model/source completion required, not direct efficacy')
    events_path=directory/'image.events.jsonl'
    require(events_path.stat().st_size<2*1024**2,'bounded native image journal')
    events=[io.io.decode(x) for x in events_path.read_bytes().splitlines()]
    require(len(events)==24, 'all eight original native image encodings')
    for i in range(8):
        a,b,c=events[3*i:3*i+3]
        require(a==dict(kind='native_image_attempted',resource='encoder',samples=1) and
                b==dict(kind='native_image_attempted',resource='projector',samples=1) and
                c['kind']=='native_image_completed' and c['counts']==dict(encoder=1,projector=1), 'native image entry receipts')
    bindings={str(report_path):io.sha(report_path),str(events_path):io.sha(events_path)}
    sourcepack=io.read(Path(p['sourcepack'])/'sourcepack.json'); tasks=[]; mapping={}
    model_sha=next(h for f,h in p['model_bindings'].items() if Path(f).name=='weights.pt')
    config_sha=next(h for f,h in p['model_bindings'].items() if Path(f).name=='config.json')
    for i,row in enumerate(sourcepack['slots']):
        path=directory/f'task_{row["slot"]["candidate_id"]}.json'; saved=io.read(path); t=saved['task']
        raw=source.load_raw_source(Path(p['sourcepack'])/row['raw_source_file'],p['source_plan'])
        core.validate_task(t)
        require(core.task_id(t)==saved['task_id'] and saved['root_sha256']==row['metadata']['root_sha256'] and
                saved['raw_source_sha256']==row['raw_source_sha256'] and saved['plan_id']==source.digest(p['source_plan']) and
                saved['source_purpose']=='calibration_exploration' and saved['freshness_certified'] is False and
                saved['history_semantics']=='causal_padding_not_observed_transitions', 'same actual task/source receipt')
        require(t['task_contract_sha256']==io.digest(p) and t['model_sha256']==model_sha and t['model_config_sha256']==config_sha and
                t['metric_source_sha256']==io.digest(p['metric']) and np.array_equal(t['projection'],np.eye(192)) and
                t['action_mean']==p['normalization']['mean'] and t['action_std']==p['normalization']['std'] and
                t['packet_logit_cap']==p['packet_logit_cap'] and t['environment']==p['source_plan']['environment'], 'original frozen Q/model/normalizer/action scale')
        frames=[*raw['rgb_history'],raw['goal_rgb']]; completed=[events[3*(4*i+k)+2] for k in range(4)]
        require(all(c['rgb_sha256']==core.sha(np.ascontiguousarray(rgb).tobytes()) for c,rgb in zip(completed,frames)), 'native RGB input binding')
        require(t['root_context']['latent_history']==[c['latent'] for c in completed[:3]] and t['goal']==completed[3]['latent'] and
                t['goal_source_sha256']==core.sha(raw['goal_rgb'].tobytes()) and
                np.array_equal(t['root_raw_action_history'],raw['raw_action_history']), 'native encoded goal/history exact receipts')
        bindings[str(path)]=io.sha(path); tasks.append(dict(root_sha256=saved['root_sha256'],task_id=saved['task_id'],task=t))
        require(saved['root_sha256'] not in mapping,'distinct roots, no replacement'); mapping[saved['root_sha256']]=saved['task_id']
    scan=source.to_admission_scan(p['source_plan'],p['exclusions'],sourcepack,Path(p['sourcepack']),mapping)
    require(scan==io.read(directory/'source_scan.json') and len(tasks)==2,'complete source/task scan, no omitted roots')
    bindings[str(directory/'source_scan.json')]=io.sha(directory/'source_scan.json')
    return dict(schema=NAME+'.inputs',direct_prepared=str(prepared),direct_prepare_sha256=direct_prepare_sha,
        task_directory=str(directory),bindings=bindings,tasks=tasks,policy=p['policy'],model_bindings=p['model_bindings'],
        inherited=dict(source_resources=r['source_resources'],source_reset_internal_physics_steps=r['source_reset_internal_physics_steps'],
                       direct_resources=r['actual_direct_resources'],native_image_resources=r['actual_image_resources'],
                       attribution='cached acquisition once, separately inherited; not free or rerun'),
        source_purpose=p['source_plan']['purpose'],history_semantics=p['source_history'],P1_admission=False)


def prepare(direct_prepared, direct_prepare_sha, task_directory, output):
    output=Path(output).absolute(); require(not output.exists(),'NEW prepared path')
    require(shutil.disk_usage(output.parent).free>=RESERVE+8*1024**2,'prepare reserve')
    inputs=collect(direct_prepared,direct_prepare_sha,task_directory)
    files=set(io.source_files()+[Path(__file__).resolve(),Path(screen.__file__).resolve(),Path(core.__file__).resolve(),
        Path(upstream.__file__).resolve(),Path(source.__file__).resolve(),
        ROOT/'experiments/spectral_reachability/p1_conditional_source_metadata_20261027_v1.py',
        ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')])
    p=dict(schema=NAME+'.protocol',algorithm=deepcopy(PROTOCOL),budget_per_root=budgets(),inputs=inputs,
           sources_sha256={str(f):io.sha(f) for f in sorted(files)},output_limit_bytes=OUTPUT_LIMIT,reserve_bytes=RESERVE)
    raw=(core.canonical(p)+'\n').encode()
    require(shutil.disk_usage(output.parent).free>=RESERVE+sum(f.stat().st_size for f in files)+len(raw)+1024**2,
            'complete source snapshot and metadata plus finalization reserve')
    output.mkdir()
    with (output/'protocol.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    seal=io.create_seal(files=sorted(files),args_json=core.canonical(dict(protocol_sha256=io.sha(output/'protocol.json'))),output_dir=output/'source_seal')
    receipt=dict(schema=NAME+'.prepare',status='prepared_not_executed',protocol_sha256=io.sha(output/'protocol.json'),source_seal=seal)
    with (output/'prepare.json').open('xb') as f:f.write((core.canonical(receipt)+'\n').encode());f.flush();os.fsync(f.fileno())
    return receipt


def verify(folder, expected_sha):
    folder=Path(folder).absolute(); require(io.sha(folder/'prepare.json')==expected_sha,'external prepare SHA')
    r=io.read(folder/'prepare.json'); require(io.sha(folder/'protocol.json')==r['protocol_sha256'],'protocol SHA')
    p=io.read(folder/'protocol.json'); s=r['source_seal']
    io.verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    require(p['schema']==NAME+'.protocol' and p['algorithm']==PROTOCOL and p['budget_per_root']==budgets() and
            p['output_limit_bytes']==OUTPUT_LIMIT and p['reserve_bytes']==RESERVE,'unchanged predeclared protocol')
    for f,h in {**p['sources_sha256'],**p['inputs']['bindings'],**p['inputs']['model_bindings']}.items(): require(io.sha(f)==h,'source/input/model drift')
    old=p['inputs']; require(collect(old['direct_prepared'],old['direct_prepare_sha256'],old['task_directory'])==old,'actual input reconstruction')
    return p


def run(folder, expected_sha, output, *, authorize_host_cuda=False):
    require(authorize_host_cuda is True,'explicit MAIN host CUDA authorization; no automatic neural execution')
    p=verify(folder,expected_sha); store=Store(output)
    errors=[]; roots=[]; model=None; before=after=None; source_ok=False; torch=None
    store.put('launch.json',dict(prepared=str(Path(folder).absolute()),prepare_sha256=expected_sha,algorithm_sha256=core.digest(PROTOCOL)))
    try:
        import torch as torch_runtime
        torch=torch_runtime
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False
        model=load_compatible_pretrained(p['inputs']['policy'],device=device).eval().requires_grad_(False)
        io.frozen(model); before=io.tensor_hash(model)
        with sdpa_kernel(SDPBackend.MATH):
            for i,item in enumerate(p['inputs']['tasks']):
                sink=store.sink(f'root_{i}.events.jsonl')
                result=run_task(item['task'],NativeCuda(model,device,item['task']),sink)
                roots.append(dict(root_sha256=item['root_sha256'],task_id=item['task_id'],status=result['status'],
                                  resources=result['resources'],error=result['error'],report_path=str(store.output/f'root_{i}.json')))
                store.put(f'root_{i}.json',result)
                roots[-1].update(report_sha256=io.sha(store.output/f'root_{i}.json'),
                    journal_path=str(store.output/f'root_{i}.events.jsonl'),
                    journal_sha256=io.sha(store.output/f'root_{i}.events.jsonl'))
                require(result['status']=='witness_completed_not_admitted' and result['error'] is None,'producer failed; preserve partials, no retry')
    except BaseException as exc:errors.append(io.error('execution',exc))
    finally:
        if torch is not None:
            try:torch.cuda.synchronize()
            except BaseException as exc:errors.append(io.error('synchronize',exc))
        if model is not None:
            try:io.frozen(model);after=io.tensor_hash(model)
            except BaseException as exc:errors.append(io.error('final_model',exc))
        try:verify(folder,expected_sha);source_ok=True
        except BaseException as exc:errors.append(io.error('final_sources',exc))
        if before is not None and before!=after:errors.append(dict(stage='model_identity',type='ValueError',message='model state changed or unavailable'))
        complete=not errors and len(roots)==2 and all(r['status']=='witness_completed_not_admitted' for r in roots) and before is not None and before==after and source_ok
        result=dict(schema=NAME+'.report',status='cuda_completed_not_admitted' if complete else 'partial_or_failed',
            prepare_sha256=expected_sha,algorithm_sha256=core.digest(PROTOCOL),roots=roots,errors=errors,
            model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,inherited=p['inputs']['inherited'],source_purpose=p['inputs']['source_purpose'],
            history_semantics=p['inputs']['history_semantics'],new_image_encoder_calls=0,new_physical_steps=0,
            P1_admission=False,model_task_eligible=False,physical_future_labels_read=False,
            direct_plateau_established_by_this_producer=False,
            missing=['direct finite-budget stall evidence', 'independent threshold/calibration authority and confirmation',
                     'fresh confirmation / observed-transition history', 'physical non-greedy mechanism and consequence'])
        try:store.put('report.json',result)
        except BaseException as exc:
            errors.append(io.error('final_report',exc)); result['status']='partial_or_failed'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run'))
    parser.add_argument('--direct-prepared',type=Path);parser.add_argument('--direct-prepare-sha')
    parser.add_argument('--task-directory',type=Path);parser.add_argument('--prepared',type=Path)
    parser.add_argument('--prepare-sha');parser.add_argument('--output',type=Path)
    parser.add_argument('--authorize-host-cuda',action='store_true');a=parser.parse_args()
    if a.command=='prepare':
        prepare(a.direct_prepared,a.direct_prepare_sha,a.task_directory,a.output); print(io.sha(a.output/'prepare.json'))
    elif a.command=='verify': print(verify(a.prepared,a.prepare_sha)['schema'])
    else:
        result=run(a.prepared,a.prepare_sha,a.output,authorize_host_cuda=a.authorize_host_cuda)
        print(core.canonical(result))
        if result['status']!='cuda_completed_not_admitted':raise SystemExit(2)


if __name__=='__main__':main()
