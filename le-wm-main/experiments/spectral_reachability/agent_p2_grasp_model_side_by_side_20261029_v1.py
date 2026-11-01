"""Pinned saved model plans only; no physical outcomes or neural dispatch.

Reuses completed strict receipts, NOT their expensive consumers. No matched
budget, official GRASP reproduction, physical performance or admission claim.
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
from experiments.spectral_reachability import agent_p2_grasp_raw15_prepare_20261029_v1 as inputs
from experiments.spectral_reachability import agent_p2_wallcapped_cuda_20261029_v1 as wall
from experiments.spectral_reachability import agent_p2_history_joint30_grasp_saved_consumer_20261029_v2 as grasp

NAME = 'agent_p2_grasp_model_side_by_side_20261029_v1'
ROOT = inputs.ROOT
CAP = 16*1024**2
PINS = {
 'grasp_receipt': ('results/agent_p2_history_joint30_grasp_saved_consumer_20261029_v2_consume_MAIN01.json', '55c192a908835e6f8c05160c92c4d5a2d46a0ed81ce4985dbf1de3c3c2f4fe2e'),
 'grasp_report': ('results/agent_p2_history_joint30_grasp_cuda_20261029_v1_cuda_MAIN01/report.json', 'd92e24d9807a20f296811d4321b64cc87f2c8c73c04677a350f136debc1ce14a'),
 'grasp_prepare': ('results/agent_p2_history_joint30_grasp_cuda_20261029_v1_prepare_MAIN01.json', '545dc34a593f464977deab8123a3f44d2f395ee8d1b16702527e8f58b4dcd4d3'),
 'common_receipt': ('results/agent_p2_wallcapped_cuda_20261029_v1_consume_MAIN01.json', '7b3eba575a9cf39a565c1956a26a937828509b313be7e4aecb9f95e7f0b67ac1'),
 'common_report': ('results/agent_p2_wallcapped_cuda_20261029_v1_cuda_MAIN01/report.json', '1359439a994f91f8eb18e0128fd53e0b0aea800ff281876c6cbf84e222f6e3ba'),
 'common_prepare': ('results/agent_p2_wallcapped_cuda_20261029_v1_prepare_MAIN01.json', 'd634d9c8da6b065d7a013bd28d80c91a4f087ad96103e88da3f40c3ffbbd7f3a')}
ORDER = [[0,'joint'],[0,'composition'],[1,'composition'],[1,'joint']]
need = inputs.need


def equal(a,b,message):
    need(inputs.encoded(a) == inputs.encoded(b), message)


def load_pinned(path, expected):
    p=Path(path)
    need(p.is_file() and not p.is_symlink() and p.stat().st_size<=CAP, 'bounded regular JSON')
    b=p.read_bytes();need(len(b)<=CAP and inputs.sha(b)==expected, 'pinned saved JSON SHA')
    return inputs.parse(b)


def authenticated(plan, digest):
    p=deepcopy(plan);expected=p.pop('selection_digest')
    need(digest(p)==expected, 'producer-specific selection digest')


def closed(value):
    need(all(value[k] is False for k in ('P1_admission','P2_admission','physical_release')), 'admission stays closed')


def array(value, shape):
    a=np.asarray(value,np.float64)
    need(a.shape==shape and np.isfinite(a).all(), 'finite exact shape')
    return a


def checked_q(context,goal,reported):
    inputs.core.context(context)
    z=array(context['latent_history'],(3,192))[-1];g=array(goal,(192,))
    q=float(np.mean((z-g)**2))
    need(type(reported) is float and np.isfinite(reported) and q==reported, 'raw terminal model Q independent float64 MSE')
    return q


def checked_actions(request, logits, commands, bits=None, sha=None):
    # Original normalization/chart validator; never projects or repairs a plan.
    d=inputs.raw15_inputs(request,logits)
    equal(commands,d['raw15'],'literal tanh(float64)->float32 raw15')
    a=np.asarray(commands,'<f4')
    if bits is not None:need(a.tobytes().hex()==bits,'literal raw float32 bits')
    if sha is not None:need(inputs.sha(a.tobytes())==sha,'literal raw SHA')
    return dict(logit_L2=float(np.linalg.norm(np.asarray(logits,np.float64))),
                raw_float32_sha256=inputs.sha(a.tobytes()),
                normalized_packets_sha256=inputs.sha(np.asarray(d['normalized_packets'],'<f4').tobytes()))


def compare(d, tasks):
    """Pure RAM validation; production entry supplies only pinned saved authorities."""
    gr,wr,gp,wp,gre,wre=(d[k] for k in ('grasp_receipt','common_receipt','grasp_prepare','common_prepare','grasp_report','common_report'))
    need(gr['schema']==grasp.NAME+'.consume' and gr['status']=='history_joint30_variant_verified_not_admitted' and gr['errors']==[], 'complete fixed GRASP strict authority')
    need(wr['schema']==wall.NAME+'.consume' and wr['status']=='wallcapped_records_verified_not_admitted' and wr['fixed_denominator']==4, 'complete fixed common strict authority')
    need(gr['planner_report_sha256']==PINS['grasp_report'][1] and gr['prepare_sha256']==PINS['grasp_prepare'][1] and wr['report_sha256']==PINS['common_report'][1], 'receipt report/prepare joins')
    for key,report in (('grasp',gre),('common',wre)):
        need(report['prepare_sha256']==PINS[key+'_prepare'][1] and report['errors']==[], 'actual prepared report closure')
        need(report['model_before']==report['model_after'], 'unchanged saved model')
        closed(report)
    equal(gr['resources'],gp['contract']['total'],'entire paid GRASP totals')
    equal(gp['contract'],grasp.contract(),'frozen GRASP domain/algorithm/resources')
    need(gp['policy']==wp['policy'] and gp['expected_model_hash']==wp['expected_model_hash']==inputs.TENSORS, 'same frozen model tensors/policy')
    need(gp['source_report_sha256']==inputs.SOURCE, 'original source identity')
    need(gre['model_before']['original_state_dict']==inputs.TENSORS and wre['model_before']==inputs.TENSORS, 'original model report hash')
    equal(wp['root_arm_order'],ORDER,'fixed serial four-arm order')
    need(len(tasks)==len(gp['slots'])==len(wp['slots'])==len(gr['plans'])==2 and len(wr['plans'])==len(wr['accounts'])==4, 'full 2+4 denominator')
    for authority in (gr,wr):closed(authority)
    rows=[];identities=[]
    warm=np.concatenate((np.tile(np.asarray([.1,0],np.float32),(5,1)),np.tile(np.asarray([0,.1],np.float32),(5,1)))).tolist()
    for root,(task,gs,ws,gplan) in enumerate(zip(tasks,gp['slots'],wp['slots'],gr['plans'],strict=True)):
        need(gs['root_slot']==ws['root_slot']==gplan['root_slot']==root and gs['condition']==ws['condition']==gplan['condition']=='off', 'known2 off root identity')
        for k in ('task_file','task_sha256','task_id'):
            equal(gs[k],ws[k],'same original '+k);equal(gs[k],gplan[k],'GRASP plan '+k)
        need(gs['task_sha256']==inputs.TASK_SHAS[root] and inputs.core.task_id(task)==gs['task_id'], 'original task content identity')
        equal(gs['warm_raw10'],warm,'literal warmup raw10');equal(ws['warm_raw10'],warm,'common warmup')
        request=dict(root_context=task['root_context'],goal=task['goal'],q=task['projection'],normalization=inputs.normalization())
        equal(gs['model_input'],request,'original GRASP full input');equal(ws['packet'],dict(root_context=task['root_context'],goal=task['goal']),'original common full input')
        equal(ws['q'],task['projection'],'common identity Q');equal(task['projection'],np.eye(192).tolist(),'original identity192')
        need(gs['target_raw_steps']==15 and gs['target_future_packets']==3 and gs['source_export_horizon']==task['horizon']==2,'source H2 distinct from actual H3')
        equal(gplan['original_context'],task['root_context'],'plan original full3 context');equal(gplan['goal'],task['goal'],'unchanged goal');equal(gplan['Q'],task['projection'],'unchanged Q');equal(gplan['normalization'],inputs.normalization(),'original normalizer')
        equal(gplan['blind_pointer'],gs['blind_pointer'],'unopened source blind identity')
        need(gs['blind_pointer']['opened'] is False and gs['blind_pointer']['path']==ws['blind_path'] and gs['blind_pointer']['sha256']==ws['blind_sha256'],'blind pointer only; never opened')
        authenticated(gplan,grasp.digest);closed(gplan)
        need(gplan['model_sha256']==inputs.TENSORS and gplan['source_report_sha256']==inputs.SOURCE and gplan['planner_report_sha256']==PINS['grasp_report'][1] and gplan['raw_steps']==15,'GRASP model/source/length')
        aq=checked_q(gplan['terminal'],task['goal'],gplan['model_terminal_Q'])
        evidence=checked_actions(request,gplan['logits'],gplan['commands'],sha=gplan['raw_float32_sha256'])
        identities.append(dict(root_slot=root,condition='off',task_id=gs['task_id'],task_sha256=gs['task_sha256'],task_file=gs['task_file'],
                               context_sha256=inputs.sha(inputs.encoded(task['root_context'])),goal_sha256=inputs.sha(inputs.encoded(task['goal'])),
                               Q_sha256=inputs.sha(inputs.encoded(task['projection'])),blind_pointer=gs['blind_pointer'],identity_domain_equal=True))
        resources=deepcopy(gp['contract']['per_root'])
        rows.append(dict(root_slot=root,method='history_aware_joint30_GRASP_variant',reported_model_Q=gplan['model_terminal_Q'],recomputed_model_Q=aq,
                         selection_digest=gplan['selection_digest'],resources_attempted=resources,resources_returned=deepcopy(resources),
                         search_wall_seconds=None,whole_two_root_stage_wall_seconds=gre['wall_seconds'],whole_stage_wall_cap_seconds=gp['contract']['wall_seconds'],
                         wall_scope='verify/load + both roots + final hash; per-root search wall not separately certified',
                         total_FLOPs=None,engine_internal_backward_calls=None,**evidence))
        for index,(rr,arm) in enumerate(ORDER):
            if rr!=root:continue
            plan=wr['plans'][index];account=wr['accounts'][index]
            need(plan is not None and plan['root_slot']==account['root']==root and plan['arm']==account['arm']==arm,'fixed plan/account/root join')
            authenticated(plan,wall.old.digest)
            need(plan['P1_admission'] is plan['P2_admission'] is False,'common plan not admitted; release controlled by whole receipt')
            for k in ('task_id','task_sha256'):equal(plan[k],gs[k],'common plan original '+k)
            equal(plan['root_context'],task['root_context'],'common full history');equal(plan['goal'],task['goal'],'common goal');equal(plan['warm_raw10'],warm,'plan warmup')
            equal(account['binding'],ws,'account full original input identity')
            need(plan['model_hash']==inputs.TENSORS and plan['report_sha256']==PINS['common_report'][1] and plan['selection_uses_physics'] is False,'model-only plan authority')
            need(account['stop']=='natural_completion' and account['export_complete'] is True and account['errors']==[] and account['sealed'] is True,'complete saved account')
            equal(account['attempted'],account['completed'],'no refunded/unfinished calls')
            ap=account['attempted'];npredict=1902 if arm=='joint' else 1062
            expected=dict(AD_rollout_steps=255 if arm=='joint' else 129,action_encoder=npredict,encoder=0,jacobian=85,
                jacobian_elements=489600 if arm=='joint' else 247680,mathematical_VJP=16320,physics=0,
                pred_proj=npredict,predict=npredict,predictor=npredict,projector=0,root_SVD=2,root_ridge=1,training=0)
            equal(ap,expected,'all actual common resource axes')
            equal(account['caps'],wp['resource_caps_per_root_arm'],'paid common caps')
            witness=account['incumbent']['witness'];equal(plan['contexts'],witness['contexts'],'incumbent prediction witness')
            need(len(plan['contexts'])==3,'all3 predicted packets')
            q=checked_q(plan['contexts'][-1],task['goal'],plan['model_Q'])
            need(account['incumbent']['cost']==q,'incumbent objective')
            ev=checked_actions(request,witness['logits'],plan['commands'],bits=plan['raw_float32_bits'])
            resources=dict(predict_API=ap['predict'],predict_samples=ap['predict'],action_encoder=ap['action_encoder'],predictor=ap['predictor'],pred_proj=ap['pred_proj'],
                predictor_context_tokens=3*ap['predict'],pred_proj_rows=3*ap['pred_proj'],scalar_backward=0,Jacobian=ap['jacobian'],
                mathematical_VJP=ap['mathematical_VJP'],jacobian_elements=ap['jacobian_elements'],AD_rollout_steps=ap['AD_rollout_steps'],
                SVD=ap['root_SVD'],ridge=ap['root_ridge'],encoder=ap['encoder'],projector=ap['projector'],physics=ap['physics'],training=ap['training'])
            rows.append(dict(root_slot=root,method=arm,reported_model_Q=plan['model_Q'],recomputed_model_Q=q,selection_digest=plan['selection_digest'],
                resources_attempted=resources,resources_returned=deepcopy(resources),search_wall_seconds=account['elapsed_seconds'],
                search_wall_cap_seconds=account['wall_cap_seconds'],whole_two_root_stage_wall_seconds=wre['wall_seconds'],
                wall_scope='single root-arm search; whole stage includes four serial root-arms',total_FLOPs=None,engine_internal_backward_calls=None,**ev))
    return dict(schema=NAME+'.analysis',status='saved_model_identity_domain_checked_not_matched',identities=identities,rows=rows,
        known_roots=2,fresh_roots=0,fixed_plan_denominator=6,matched=False,equal_budget=False,equal_FLOPs=False,
        scope='model-only predicted endpoint MSE, NOT native success or physical gain',official_GRASP_reproduction=False,
        authority='pinned completed strict receipts reused; no full consumer rerun',
        new_neural_calls=0,new_physics_calls=0,P1_admission=False,P2_admission=False,physical_release=False)


def build():
    d={k:load_pinned(ROOT/p,h) for k,(p,h) in PINS.items()}
    # Only exact original task exports; do not open blind NPZ, images, weights,
    # physical receipts, or any path discovered by a directory walk.
    tasks=[];task_pins={}
    for s in d['grasp_prepare']['slots']:
        need(s['task_sha256'] in inputs.TASK_SHAS,'only original known task exports')
        t=inputs.parse(inputs.read_bound(s['task_file'],s['task_sha256'],compressed=True))['task']
        tasks.append(t);task_pins[s['task_file']]=s['task_sha256']
    result=compare(d,tasks)
    # Hash source references only, never open model tensors/NPZ/physical results.
    sources={}
    for group in (d['grasp_receipt']['consumer_bindings'],d['common_prepare']['bindings_sha256']):
        for p,h in group.items():
            if Path(p).suffix in ('.py','.md'):
                need(grasp.filehash(p)==h,'sealed referenced source bytes');sources[p]=h
    for key,(p,h) in PINS.items():need(grasp.filehash(ROOT/p)==h,'saved input unchanged')
    for p,h in task_pins.items():need(grasp.filehash(p)==h,'task bytes unchanged')
    result['bindings']={str(ROOT/p):h for p,h in PINS.values()}|task_pins|sources
    result['summary_bundle']={str(p):grasp.filehash(p) for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'))}
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output');a=parser.parse_args(argv)
    result=build();raw=inputs.encoded(result)
    if a.output:
        p=Path(a.output).absolute()
        need(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and p.suffix=='.json' and not p.exists() and not p.is_symlink(),'new own exclusive result only')
        need(len(raw)<=CAP and shutil.disk_usage(p.parent).free>=200*1024**2+len(raw),'bounded new output/floor')
        with p.open('xb') as f:need(f.write(raw)==len(raw),'whole write');f.flush();os.fsync(f.fileno())
        print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest())))
    else:print(raw.decode(),end='')
    return 0


if __name__=='__main__':raise SystemExit(main())
