"""Saved P3 -> root-specific P4 prepare bridge. No neural or physical execution.

This is deliberately NOT the old 123-probe physical capability schema, nor an
executable response. A new MAIN driver must compile at the observed context.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p3_joint_prefix_cuda_encoding_20261027_v1 as saved
from experiments.spectral_reachability import p3_prefix_teacher_forcing_20261027_v1 as teacher

NAME = 'p4_feedback_root_sourcepack_20261027_v1'
REPO = saved.REPO
RESERVE = 200 * 1024**2
OUTPUT_CAP = 4 * 1024**2
require = saved.require


def budget(requests):
    """Proposed H1 compiler budget, not counts inherited from an actual run.

    One nominal forward, one full192x10 Jacobian forward/192 VJP directions,
    ten independently scored candidates, one independent winner replay.
    Engine backward entries are implementation-dependent: no invented count.
    """
    require(type(requests) is int and requests > 0, 'positive planning requests')
    return dict(requests=requests, horizon_tokens=1, native_context_tokens=3,
        candidate_count_per_request=10, predict_samples_cap=13*requests,
        action_encoder_calls_cap=13*requests, processed_context_tokens_cap=39*requests,
        AD_jacobian_api_calls_cap=requests, jacobian_shape=[192,10],
        mathematical_VJP_directions_cap=192*requests, backward_engine_entries=None,
        backward_engine_policy='must instrument and prospectively cap in NEW driver before launch',
        new_encoder_calls_cap=0, existing_projector_calls_cap=0, physics_steps_cap=0,
        device='cuda:0', cpu_neural_fallback=False, retries=False,
        training=False, fit=False, finite_difference_calls_cap=0,
        walltime_cap_seconds=None, peak_cuda_bytes_cap=None,
        launch_blockers=['unimplemented reviewed compiler/driver', 'unsealed resource hooks',
            'walltime/VRAM/backward-engine caps require MAIN preflight'])


def collect(prefix_dir, prefix_sha, teacher_prepared, teacher_prepare_sha,
            teacher_dir, teacher_report_sha, prior_report=None, prior_report_sha=None):
    """Actual saved bytes only; reuse frozen read-only source reconstruction."""
    prefix_dir, teacher_dir = Path(prefix_dir).resolve(), Path(teacher_dir).resolve()
    require(saved.sha(prefix_dir/'receipt.json') == prefix_sha, 'external NEW17 receipt SHA')
    r = saved.read(prefix_dir/'receipt.json')
    b = saved.read(prefix_dir/'execution_binding.json')
    p = saved.verify(b['prepared'], b['prepare_sha256'])
    require(r['status']=='all_17_prefix_latents_saved' and not r['errors'] and not r['blocked'] and
        not r['journal_errors'] and r['source_verified'] is r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after'] and bool(r['model_hash_before']) and
        r['model_bindings']==p['model_bindings'] and r['prepare_sha256']==b['prepare_sha256'], 'actual NEW17 completion/model')
    require(r['counts']=={k:dict(attempted=17,completed=17) for k in ('encoder','existing_projector')}, 'NEW17 exact counts')
    require(r['schema']==saved.NAME+'.derived_receipt' and r['first_error'] is None and
        r['physical_report_sha256']==p['physical_report_sha256'] and r['planner_report_sha256']==p['planner_report_sha256'] and
        r['goal_q_sha256']==p['goal_q_sha256'] and
        r['new_predict_calls']==r['new_AD_calls']==r['new_physical_steps']==0 and
        all(r[k] is False for k in ('training','P1_admission','P3_admission','efficacy_claim')), 'NEW17 scope/source edges')
    require(len(r['rows'])==len(p['frames'])==17 and
        [(x['frame_id'],x['kind']) for x in r['entries']]==[(x['frame_id'],k) for x in p['frames'] for k in ('encoder','existing_projector')] and
        all(x['completed'] is True for x in r['entries']), 'NEW17 ordered module entries')
    journal=prefix_dir/'encoding.events.jsonl'
    require(saved.sha(journal)==r['journal_sha256'], 'NEW17 journal SHA')
    events=[saved.io.decode(line) for line in journal.read_bytes().splitlines()]
    expected_events=[]
    bindings = {str(prefix_dir/'receipt.json'):prefix_sha,
        str(prefix_dir/'execution_binding.json'):saved.sha(prefix_dir/'execution_binding.json'),
        str(journal):r['journal_sha256'], **p['bindings'], **p['model_bindings'], **p['sources_sha256']}
    frames=[]
    for f, row in zip(p['frames'],r['rows'],strict=True):
        require(all(row[k]==v for k,v in f.items() if k!='predicted_vector') and row['status']=='encoded_and_saved', 'exact source frame identity')
        for suffix in ('.result.json','.latent.json'):
            path=prefix_dir/(f['frame_id']+suffix)
            require(saved.same(saved.read(path),row),'durable prefix receipt identity');bindings[str(path)]=saved.sha(path)
        attempt=prefix_dir/(f['frame_id']+'.attempt.json')
        require(saved.same(saved.read(attempt),{k:v for k,v in dict(row,status='attempted',latent_sha256=None).items() if k not in ('latent_dtype','latent_shape')}),'durable original prefix attempt')
        bindings[str(attempt)]=saved.sha(attempt)
        path=prefix_dir/(f['frame_id']+'.native192.npy')
        require(row['latent_path']==str(path) and saved.sha(path)==row['latent_sha256'], 'native NPY SHA/path')
        a=np.load(path,allow_pickle=False)
        require(a.shape==(192,) and a.dtype==np.float32 and np.isfinite(a).all(), 'native float32 prefix192')
        saved.pixels(f['rgb_path'],f['rgb_file_sha256'],f['pixel_sha256'])
        bindings[str(path)]=row['latent_sha256']
        frames.append(dict(**f,latent_path=str(path),latent_sha256=row['latent_sha256'],actual_vector=a.tolist()))
        expected_events.append(dict(event='endpoint_attempt',frame_id=f['frame_id']))
        for kind in ('encoder','existing_projector'):
            index=(len(frames)-1)*2+(kind=='existing_projector')
            entry=dict(kind=kind,frame_id=f['frame_id'],index=index,completed=True)
            require(r['entries'][index]==entry,'exact ordered encoding entry')
            expected_events.extend((dict(event='attempt',**dict(entry,completed=False)),dict(event='return_or_error',**entry)))
        expected_events.append(dict(event='endpoint_saved',frame_id=f['frame_id'],latent_path=str(path),latent_sha256=row['latent_sha256']))
    require(saved.same(events,expected_events),'complete encoding event accounting')
    tp=teacher.verify(teacher_prepared,teacher_prepare_sha)
    require(saved.sha(teacher_dir/'report.json')==teacher_report_sha, 'external teacher report SHA')
    tr=saved.read(teacher_dir/'report.json')
    require(tr['status']=='teacher_forcing_completed' and not tr['errors'] and tr['source_verified'] is tr['model_unchanged'] is True and
        tr['model_hash_before']==tr['model_hash_after']==r['model_hash_before'] and
        tr['prepare_sha256']==teacher_prepare_sha and tr['new_predict_calls']==17 and
        tr['new_encoder_calls']==tr['new_AD_calls']==tr['new_physical_steps']==0, 'teacher diagnostic completion only')
    require(len(tr['rows'])==17 and [(x['task_id'],x['kind']) for x in tr['entries']]==[(x['task_id'],k) for x in tp['tasks'] for k in ('action_encoder','predict')] and
        all(x['completed'] is True for x in tr['entries']), 'teacher ordered entries')
    require(tr['training'] is tr['P1_admission'] is tr['P3_admission'] is False,'teacher diagnostic scope')
    require(saved.sha(teacher_dir/'events.jsonl')==tr['journal_sha256'], 'teacher journal SHA')
    for task,row in zip(tp['tasks'],tr['rows'],strict=True):
        require(all(row[k]==task[k] for k in ('task_id','root_sha256','arm','boundary_step','target_step')), 'teacher identity')
        path=teacher_dir/(task['task_id']+'.context.json'); saved.context(saved.read(path)); bindings[str(path)]=saved.sha(path)
        result=teacher_dir/(task['task_id']+'.result.json'); require(saved.same(saved.read(result),row), 'teacher row identity'); bindings[str(result)]=saved.sha(result)
    for path in (Path(teacher_prepared)/'prepare.json',Path(teacher_prepared)/'protocol.json',teacher_dir/'report.json',teacher_dir/'events.jsonl'):
        bindings[str(path.resolve())]=saved.sha(path)
    for path in (Path(b['prepared'])/'prepare.json',Path(b['prepared'])/'protocol.json'):
        bindings[str(path.resolve())]=saved.sha(path)
    require(prior_report is not None and saved.sha(prior_report)==prior_report_sha,'pinned actual first-batch report')
    prior=saved.read(prior_report)
    require(prior['schema']=='p4_multiround_feedback_runtime_20261027_v1' and prior['status']=='calibration_recorded' and
        not prior['failures'] and len(prior['rows'])==2 and {x['arm'] for x in prior['rows']}=={'imagined','actual'} and
        all(x['status']=='completed' and x['completed_packets']==4 and x['selected_returns']==20 for x in prior['rows']) and
        len({x['root_sha256'] for x in prior['rows']})==1,'actual first batch is one root with two complete arms')
    bindings[str(Path(prior_report).resolve())]=prior_report_sha
    launch=saved.read(saved.PLANNER/'execution_protocol.json')
    cp=saved.read(Path(launch['prepared'])/'protocol.json')
    planner=saved.read(p['planner_report']); physics=saved.read(p['goal_q_source'])
    compilation_refs={}
    for entry in planner['roots']:
        require(set(entry['compilation']['raw'])=={'2','3'},'original H2/H3 compilation census')
        jacobians={}
        for h in (2,3):
            j=saved.array(entry['compilation']['raw'][str(h)]['jacobian'],(192,10*h))
            jacobians[str(h)]=dict(shape=list(j.shape),content_sha256=saved.digest(j.tolist()))
        compilation_refs[entry['root_sha256']]=dict(report=p['planner_report'],report_sha256=p['planner_report_sha256'],
            root_sha256=entry['root_sha256'],compilation_sha256=saved.digest(entry['compilation']),
            initial_only_jacobians=jacobians,may_reuse_as_observed_prefix_response=False)
    for path,h in bindings.items():require(saved.sha(path)==h,'saved source changed during reconstruction: '+path)
    return dict(records=cp['records'],normalization=cp['normalization'],frames=frames,
        roots=planner['roots'],plans=physics['plans'],root_metrics=physics['root_metrics'],
        compilation_refs=compilation_refs,
        prior_roots=sorted({x['root_sha256'] for x in prior['rows']}),
        bindings=bindings,model_bindings=p['model_bindings'],teacher_tasks=tp['tasks'],
        census=dict(known_roots=4,physical_branches=16,real_prefix_frames=17,independent_new_roots=0))


def reconstruct(bundle, root, source_arm, metric, prior_roots):
    """Pure format bridge; only observed prefixes and original actions enter."""
    require(prior_roots and len(prior_roots)==len(set(prior_roots)) and root not in prior_roots, 'explicit prior roots exclude same-root batch')
    require(set(prior_roots)==set(bundle['prior_roots']),'prior roster must match pinned actual batch, not invented exclusions')
    require(metric in ('old_Q','goal_patch_Q') and source_arm in saved.ARMS and source_arm.startswith(metric+'__'), 'fixed arm/metric; no outcome switching')
    records={x['packet']['root_sha256']:x for x in bundle['records']}
    require(len(records)==len(bundle['records'])==4 and root in records, 'four unique known roots')
    require(set(prior_roots)<=set(records), 'prior roots in audited roster')
    record=records[root]; initial=deepcopy(record['packet']['root_context']); saved.context(initial)
    require(record['packet']['input_sha256']==saved.digest({k:record['packet'][k] for k in ('root_sha256','root_context','goal','projection')}), 'compiled input packet SHA')
    frame=[x for x in bundle['frames'] if (x['root_sha256'],x['arm'],x['primitive_step'])==(root,source_arm,5)]
    plan=[x for x in bundle['plans'] if (x['root_sha256'],x['method'])==(root,source_arm)]
    source=[x for x in bundle['roots'] if x['root_sha256']==root]
    require(len(frame)==len(plan)==len(source)==1,'unique root/arm prefix source')
    frame,plan,source=frame[0],plan[0],source[0]['arms'][source_arm]
    require(frame['selection_sha256']==plan['selection_sha256'] and frame['source_input_sha256']==record['packet']['input_sha256'] and
        plan['predictor_context']==initial and plan['commands']==source['winner']['actions'], 'plan/prefix/compiled identity')
    predicted=deepcopy(source['segments'][0]); saved.history(initial,predicted,plan['commands'][:5],bundle['normalization'])
    require(predicted['latent_history'][-1]==frame['predicted_vector'], 'exact imagined prefix')
    observed=deepcopy(predicted); observed['latent_history'][-1]=frame['actual_vector']; saved.context(observed)
    task=[x for x in bundle['teacher_tasks'] if (x['root_sha256'],x['arm'],x['boundary_step'])==(root,source_arm,5)]
    require(len(task)==1 and task[0]['context']==observed, 'causal observed history matches real teacher input')
    metric_source=bundle['root_metrics'][root]
    require(metric_source==dict(goal=record['packet']['goal'],metrics=record['metrics']) and
        frame['active_metric_id']==record['metrics'][metric]['active_metric_id'], 'original goal/Q identity')
    require(root in bundle['compilation_refs'] and bundle['compilation_refs'][root]['root_sha256']==root and
        bundle['compilation_refs'][root]['may_reuse_as_observed_prefix_response'] is False,'root-specific original compilation reference')
    return dict(schema=NAME+'.protocol',status='prepared_missing_actual_response',root_sha256=root,
        prior_root_sha256s=list(prior_roots),source_arm=source_arm,metric=metric,
        source_capability_kind='P3_saved_compilation_and_observed_prefix_NOT_123probe_physical_capability',
        saved_capability=dict(status='saved_evidence_reconstructed_not_physical_probe_qualified',
            compilation_ref=bundle['compilation_refs'][root],prefix_frame_id=frame['frame_id']),
        source_input_sha256=record['packet']['input_sha256'],initial_context=initial,
        prefix_commands=plan['commands'][:5],prefix_evidence={k:v for k,v in frame.items() if k not in ('predicted_vector','actual_vector')},
        response_requests=[dict(arm=arm,boundary_step=5,context=ctx,
            context_sha256=saved.digest(ctx),must_recompile=True,can_reuse_initial_jacobian=False)
            for arm,ctx in (('imagined',predicted),('actual',observed))],
        goal=record['packet']['goal'],q=record['metrics'][metric]['projection'],
        active_metric_id=record['metrics'][metric]['active_metric_id'],normalization=bundle['normalization'],
        bindings=bundle['bindings'],model_bindings=bundle['model_bindings'],census=bundle['census'],
        offline_response_budget=budget(2),
        future_runtime_budget=runtime_budget(),
        missing=['actual root/arm/context-specific CUDA AD Jacobian and ten-candidate response/replay',
            'NEW reviewed bridge/driver; frozen runtime v1 rejects this schema by design',
            'root reset RGB/raw-history identity and guarded live real-prefix consumption',
            'fresh recompile at each actual step5/10/15 before durable selection and dispatch'],
        runtime_admissible=False,response_completed=False,known_calibration_root=True,
        independent_confirmation=False,P1_admission=False,P4_admission=False,
        actual_new_calls=dict(encoder=0,projector=0,predict=0,AD=0,physics=0),
        teacher_forcing_scope='fixed actions only; MSE reduction is not ranking repair or control improvement')


def runtime_budget():
    b=budget(6)
    b.update(physics_steps_cap=40,new_encoder_calls_cap=8,existing_projector_calls_cap=8,
        encoded_frames_cap=20,terminal_evaluation_calls=2,
        feedback_rounds_per_arm=3,arms=2,per_arm_encoder_call_cap=4,
        prefix_reset_steps_included=10,terminal_purpose='evaluation_only_never_feedback',
        output_bytes_cap=OUTPUT_CAP,remaining_disk_reserve_bytes=RESERVE,
        inherited_acquisition_cost='not free; retain original P3/NEW17 accounting by reference')
    return b


def actual_sources():
    return dict(prefix_dir=str(teacher.PREFIX),prefix_sha=teacher.PREFIX_SHA,
        teacher_prepared=str(REPO/'results/p3_prefix_teacher_forcing_20261027_v1_prepared01'),
        teacher_prepare_sha='1018bc326957644c168ddea1985dfa74f495264dc57b934960d61460a70efcd9',
        teacher_dir=str(REPO/'results/p3_prefix_teacher_forcing_20261027_v1_cuda_main01'),
        teacher_report_sha='c5bd1f15a4a8aa7dd60050f8ce5a681aa20c1966e1715cdc3e1df013e9fba499',
        prior_report=str(REPO/'results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1/report.json'),
        prior_report_sha='f97ac090645cc417ea63ede4f9f883e8dba3891125a8d9f14fabe454c95f059e')


def material(sources,root,source_arm,metric,prior_roots):
    return reconstruct(collect(**sources),root,source_arm,metric,prior_roots)


def prepare(output, sources, root, source_arm, metric, prior_roots):
    output=Path(output).absolute()
    require(not output.exists() and shutil.disk_usage(output.parent).free>=RESERVE+OUTPUT_CAP,'exclusive NEW output / 200MiB reserve')
    p=material(sources,root,source_arm,metric,prior_roots)
    sources_to_seal=[Path(__file__).resolve(),REPO/'tests'/('test_'+NAME+'.py'),REPO/'innerdocs'/(NAME+'.md'),
        Path(saved.__file__).resolve(),Path(teacher.__file__).resolve(),Path(saved.io.__file__).resolve(),
        REPO/'experiments/spectral_reachability/d2p_prerun_source_seal.py']
    p['producer_sources']={str(x):saved.sha(x) for x in sources_to_seal}
    raw=(saved.io.canonical(p)+'\n').encode()
    require(len(raw)+sum(x.stat().st_size for x in sources_to_seal)+65536<OUTPUT_CAP,'bounded small sourcepack; no large protocol copy')
    output.mkdir(exist_ok=False)
    with (output/'protocol.json').open('xb') as stream:
        stream.write(raw);stream.flush();saved.os.fsync(stream.fileno())
    seal=saved.create_seal(files=sources_to_seal,args_json=saved.io.canonical(dict(protocol_sha256=saved.sha(output/'protocol.json'))),output_dir=output/'source_seal')
    receipt=dict(schema=NAME+'.prepare',status=p['status'],protocol_sha256=saved.sha(output/'protocol.json'),source_seal=seal,
        sources=sources,root=root,source_arm=source_arm,metric=metric,prior_roots=list(prior_roots))
    saved.write(output/'prepare.json',receipt)
    return receipt


def verify(output,expected):
    output=Path(output).resolve();require(saved.sha(output/'prepare.json')==expected,'external prepare SHA')
    r=saved.read(output/'prepare.json');p=saved.read(output/'protocol.json')
    require(saved.sha(output/'protocol.json')==r['protocol_sha256'],'protocol SHA')
    saved.verify_seal(output_dir=r['source_seal']['output_dir'],expected_manifest_sha256=r['source_seal']['manifest_sha256'])
    for path,h in p['producer_sources'].items():require(saved.sha(path)==h,'producer source drift')
    rebuilt=material(r['sources'],r['root'],r['source_arm'],r['metric'],r['prior_roots']);rebuilt['producer_sources']=p['producer_sources']
    require(saved.same(p,rebuilt),'sourcepack reconstruction drift')
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('audit','prepare','verify'))
    parser.add_argument('--root');parser.add_argument('--source-arm',choices=saved.ARMS)
    parser.add_argument('--metric',choices=('old_Q','goal_patch_Q'))
    parser.add_argument('--prior-roots',nargs='+');parser.add_argument('--output',type=Path)
    parser.add_argument('--prepare-sha');a=parser.parse_args()
    if a.command=='verify':p=verify(a.output,a.prepare_sha)
    elif a.command=='prepare':
        prepare(a.output,actual_sources(),a.root,a.source_arm,a.metric,a.prior_roots)
        print(saved.sha(a.output/'prepare.json'));return
    else:p=material(actual_sources(),a.root,a.source_arm,a.metric,a.prior_roots)
    print(json.dumps({k:p[k] for k in ('status','root_sha256','census','response_completed','runtime_admissible','missing','offline_response_budget','future_runtime_budget')},sort_keys=True))


if __name__=='__main__':main()
