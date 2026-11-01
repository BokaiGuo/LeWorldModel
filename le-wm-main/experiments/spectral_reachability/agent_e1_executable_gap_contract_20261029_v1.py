"""E1 saved evidence join and executable-gap contract, never an experiment runner."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

NAME='agent_e1_executable_gap_contract_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
CTRL='agent_experiment1_common_controller_20261028_v1'
PHY='agent_experiment1_common_controller_physical_20261028_v1'
RGB='agent_experiment1_common_controller_rgb_20261028_v1'
ARMS=('arbitrary_latent_waypoint','existing_learned_prior','dynamics_compiled_witness')
ACTIVE=(ARMS[0],ARMS[2])
CAP=16*1024**2
CATALOG={
 'controller_prepare':(CTRL+'_prepared_MAIN01.json','8c472485d44c40ce3d95f8043d2c58f9ec2694fef74786a840d1699d5a1a86bd'),
 'controller_report':(CTRL+'_cuda_MAIN01/report.json','bc3cd74c85f878cdfc7b170d0b5b555827e0dc4a59cdaea6fb5781e71f03767f'),
 'controller_receipt':(CTRL+'_consume_MAIN01.json','b4b72dea9d782c6d2e1b93f50523a1089e7f3d894e2e2c1171fc13f9b8cd425d'),
 'physical_prepare':(PHY+'_prepare_MAIN02.json','160517a999680baf8bc79809e36d01e0d5ed8ade69b54706ff48edb6b5786586'),
 'physical_receipt':(PHY+'_consume_MAIN02.json','3926373a79e5c0eb9a6295dadd4ce1403dfca13d9baad51639cd7012c913c12b'),
 'physical_report':(PHY+'_physics_MAIN02/report.json','a75d97bfd5e6c19bf942ce5510ad7689ad3fd79f7bb3539b2ca96a6cda9fec6c'),
 'rgb_receipt':(RGB+'_consume_MAIN01.json','4d5d8e9b5014487e7f4a932fe19a92a5d3a011c4db874cb9cd5ca14b47f04dc7'),
 'rgb_report':(RGB+'_cuda_MAIN01/report.json','3f64f2b3f74be98fdd0f0eda36fb0c309990ae42d046a71ce4ca8d7289d654c9'),
 'hi_delivery':('agent_hilewm_rgb_high_low_20261028_v1_consume_MAIN01.json','40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0'),
 'hi_catalog':('agent_p2_hilewm_pairing_catalog_20261029_v1_join_MAIN01.json','1523183330336bd3e66884a15fa96e7a15161abd8a343d1f50dba7a0427494b8'),
 'hi_parity':('agent_p2_hilewm_computation_parity_20261029_v1_consume_MAIN01.json','3555696c02329062084d791c0c40958084d7352e6f7680938bade0068afe9480'),
 'hi_tracker':('agent_p2_history3_tracker_cuda_20261029_v1_consume_MAIN01.json','080fe1173895621ea4c0565671509a267ba456135fedfac313ddac7894f98138'),
 'prior_protocol':('agent_p2_subgoal_gap_protocol_20261029_v1_prepare_MAIN01.json','624c369764ba75776c9649de67a84f99a4004490808b626c68def405c512b788'),
 'sage_metadata':('p2_learned_subgoal_asset_metadata_20261027_v1/report.json','4234e1deff0d45c62417c34498ae1cbb3b627a39c4fd23f26f4690e34e8559c6'),
}
READ_SOURCES=(CTRL,PHY,'agent_experiment1_common_controller_physical_consume_20261028_v1',RGB,
    'agent_subgoal_gap_contract_20261028_v1','agent_subgoal_gap_runner_20261028_v1',
    'agent_subgoal_gap_assets_20261028_v1','agent_p2_subgoal_gap_protocol_20261029_v1',
    'agent_hilewm_history3_target_tracker_20261029_v1')
SHARED=('task','root','history','goal','loaded_model','encoder_preprocess','normalizer','q','raw_domain',
        'horizon_feedback','controller_source','controller_budget','information_permissions','native_evaluator')


def need(ok,message):
    if not ok:raise ValueError(message)


def raw(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def equal(a,b):return raw(a)==raw(b)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(raw(x)).hexdigest()


def absolute(path):
    p=Path(path);need(p.is_absolute() and str(p.resolve())==str(p) and not p.is_symlink(),'canonical absolute path');return p


def read(path,h):
    p=absolute(path);need(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'external SHA256')
    need(p.is_file() and p.stat().st_size<=CAP,'bounded regular metadata');b=p.read_bytes()
    need(hashlib.sha256(b).hexdigest()==h,'exact input bytes')
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    def bad(_):raise ValueError('nonfinite JSON')
    x=json.loads(b,object_pairs_hook=pairs,parse_constant=bad);raw(x);return x


def pointer(x,p):
    need(type(p) is str and (p=='' or p.startswith('/')),'JSON pointer')
    if p=='':return x
    for key in p[1:].split('/'):
        key=key.replace('~1','/').replace('~0','~')
        if type(x) is list:
            need(key.isdigit() and str(int(key))==key and int(key)<len(x),'exact pointer index');x=x[int(key)]
        else:need(type(x) is dict and key in x,'existing pointer field');x=x[key]
    return x


def vector(x,n):need(type(x) is list and len(x)==n and all(type(v) in (int,float) and math.isfinite(v) for v in x),'finite literal vector')


def contract():
    return dict(roots=[0,1],arms=list(ARMS),scientific_slots=6,historical_dispatches=4,checkpoints=[5,10],
        raw_steps=10,scope='existing known2 calibration; not fresh/full E1/P1',shared_fields=list(SHARED),
        proposals=dict(arbitrary='frozen finite20 latent search, not all possible latent methods',
            learned='authenticated frozen prior actual forward, delivered exact target, no fallback, no training',
            compiled='target from paid action-program causal rollout; witness is a model claim, not real arrival'),
        controller='only target differs; same original context/source/config/domain/cap; no witness warmstart privilege',
        proof_levels=['proposal_origin','model_action_witness','common_controller_delivery','native_execution','observed_tracking','predeclared_arrival'],
        arrival='continuous error is not arrival; missing preregistered threshold -> null; never use native_success as latent reach',
        resources='proposal/inherited/controller/replay/encoder/AD/native axes separate; cap match is not actual/equal FLOPs',
        old_controller=dict(predict_per_dispatch=144,total_predict=576,new_replay=0,AD=0),
        extension='third arm at same cost would add144/root=288 low predicts, not authorization; prior overhead still unknown',
        conditional_H10_extension_bound=dict(branches=2,predict_API=288,action_encoder=288,pred_proj=288,
            public_steps=40,physics_steps=408,explicit_PNG=44,endpoint_encodes=4,endpoint_projectors=4,
            scope='old recipe no-early-stop bound only; new runtime metering/review needed; learned generation cost unknown'),
        minimum_before_true_three_arm=['same-H10 actual learned delivery with exact common-coordinate/time/permission receipts',
            'or separately approved all-arm new horizon/backbone; never transplant a single Hi arm',
            'presealed proposal budgets and source/config/loaded-bytes identities, controller replay accounting',
            'presealed arrival criterion and full six-slot native/encoding dispatch protocol'],
        training=False,production_ready=False,dispatch_allowed=False,P1_admission=False,scientific_admission=False,total_FLOPs=None)


def field_matrix(rows):
    """Typed coordinate comparison, usable on future packets without authorizing a run."""
    need(len(rows)==6 and [(r['root_slot'],r['arm']) for r in rows]==[(r,a) for r in (0,1) for a in ARMS],'fixed all6 matrix')
    out=[]
    for root in (0,1):
        group=rows[root*3:root*3+3]
        for field in SHARED:
            v=[None if r['shared'] is None else r['shared'].get(field) for r in group]
            # Compare arbitrary/compiled without pretending the missing prior matched.
            known=[x for x in v if x is not None]
            state='mismatch' if any(not equal(x,known[0]) for x in known[1:]) else 'unknown' if len(known)<3 else 'match'
            out.append(dict(root_slot=root,field=field,status=state,available=len(known),
                two_active_match=len(known)>=2 and all(equal(x,known[0]) for x in known[1:])))
    return out


def joined(data,refs):
    cp,cr,cu=data['controller_prepare'],data['controller_report'],data['controller_receipt']
    pp,pr,pu=data['physical_prepare'],data['physical_report'],data['physical_receipt']
    rr,ru=data['rgb_report'],data['rgb_receipt']
    need(cp['schema']==CTRL+'.prepare' and cu['schema']==CTRL+'.consume' and cu['status']=='common_controller_calibration_verified'
        and cr['prepare_sha256']==refs['controller_prepare']['sha256'] and cr['prepared']==refs['controller_prepare']['path']
        and equal(cr['contract'],cp['contract']) and equal(cr['rows'],cu['rows'])
        and cr['model_hash_before']==cr['model_hash_after']==cp['expected_model_hash']
        and cu['scientific_slots']==6 and cu['dispatch_denominator']==4 and cu['learned_unknown_slots']==2,
        'actual controller full receipt and prepare/report closure')
    need(pu['schema']=='agent_experiment1_common_controller_physical_consume_20261028_v1.consume'
        and pu['physical_report_sha256']==refs['physical_report']['sha256'] and equal(pu['rows'],pr['rows'])
        and pr['prepare_sha256']==refs['physical_prepare']['sha256'] and pr['prepared']==refs['physical_prepare']['path']
        and pp['schema']==PHY+'.prepare' and pu['fixed_branch_denominator']==4 and pu['scientific_slot_denominator']==6
        and equal(pu['counts'],pr['counts']),'actual native full receipt/report/prepare')
    need(ru['schema']==RGB+'.consume' and rr['schema']==RGB+'.report'
        and rr['status']=='encoded_sealed' and rr['errors']==[] and equal(rr['rows'],ru['rows'])
        and rr['input_binding']==dict(physical_report=refs['physical_report']['path'],physical_sha=refs['physical_report']['sha256'])
        and ru['fixed_branch_denominator']==4 and ru['scientific_slot_denominator']==6 and ru['learned_unknown_slots']==2
        and rr['meta']['model_hash_before']==rr['meta']['model_hash_after']==cp['expected_model_hash'],
        'actual RGB complete receipt + sealed report, not a rerun')
    for obj in (cu,pu,ru):need(obj['P1_admission'] is False,'old gate preserved')
    need(len(cu['plans'])==len(pu['rows'])==len(ru['rows'])==len(pp['plans'])==4 and len(cp['slots'])==2,'all two roots, four dispatches')
    need([(r['source_slot'],r['arm']) for r in cu['rows']]==[(r,a) for r in (0,1) for a in ARMS], 'six scientific slots not shortened')
    for obj in (pu,ru):
        need([(r['root_slot'],r['arm']) for r in obj['scientific_rows']]==[(r,a) for r in (0,1) for a in ARMS], 'native/RGB scientific six slots retained')
        for r in obj['scientific_rows']:
            if r['arm']==ARMS[1]:need(r['status']=='unknown' and r['commands'] is r['resources'] is r['native_success'] is None,'learned unknown never zero/fallback')
    rows=[]
    for root in (0,1):
        slot=cp['slots'][root];need(slot['source_slot']==root and cp['targets'][root]['source_slot']==root,'fixed original root identity')
        for arm in ARMS:
            idx=root*3+ARMS.index(arm);u=cu['rows'][idx]
            need(u['task_id']==slot['task_id'],'all arms original task')
            if arm==ARMS[1]:
                need(u['status']=='unknown' and u['commands'] is u['resources'] is u['predictor_samples'] is None,
                    'no invented learned baseline or zero score')
                rows.append(dict(root_slot=root,arm=arm,shared=None,status='missing_same_E1_learned_delivery',
                    target=None,proposal_action_witness=None,controller_action_witness=None,native_executed=None,
                    observed_tracking=None,arrival=None,native_success=None,evidence=evidence(refs,'controller_receipt',f'/rows/{idx}',u)))
                continue
            branch=root*2+ACTIVE.index(arm);p=cu['plans'][branch];n=pu['rows'][branch];z=ru['rows'][branch];s=n['selected_plan']
            need((p['source_slot'],p['arm'])==(root,arm) and (n['root_slot'],n['arm'])==(root,arm)
                and (z['root_slot'],z['arm'])==(root,arm) and p['task_id']==z['task_id']==s['task_id']==slot['task_id']
                and equal(p['commands'],u['commands']) and equal(p['commands'],s['commands']) and equal(s,pp['plans'][branch])
                and equal(z['commands'],p['commands']) and z['selection_digest']==n['selection_digest']==s['selection_digest']
                and s['compiler_selection_sha256']==p['selection_sha256']==u['selection_sha256']
                and p['whole_run_report_sha256']==refs['controller_report']['sha256']
                and equal(p['target'],s['target']) and equal(p['target'],cp['targets'][root]['targets'][arm]),
                'unique selected target/action/compiler/raw/native/RGB join')
            need(len(p['commands'])==10 and all(type(a) is list and len(a)==2 for a in p['commands']),'literal raw10')
            for a in p['commands']:vector(a,2);need(all(abs(x)<=1 for x in a),'raw action box')
            need([c['step'] for c in z['checkpoints']]==[5,10]
                and n['own_returned']==10 and n['initial_exact'] is True
                and z['native_success'] is n['native_success'] and z['physical_status']==n['status'],
                'actual full10 and exact initial guard; no model-success substitution')
            if arm==ARMS[2]:
                w=cp['compiled'][root]
                need(w['source_slot']==root and equal(w['target'],p['target'])
                    and equal(w['prefix_context']['latent_history'][-1],p['target']), 'compiled target has its own original model action witness')
                witness=evidence(refs,'controller_prepare',f'/compiled/{root}',w)
            else:witness=None  # Does not mean unreachable, only no action-derived proposal.
            shared=dict(task={k:slot[k] for k in ('task_id','task_sha256','task_contract_sha256','blind_sha256')},
                root=n['root_state'],history=dict(contract=cp['contract']['history'],task_export=slot['task_export'],task_sha256=slot['task_sha256']),
                goal=n['goal'],loaded_model=dict(saved_tensor_hash=cp['expected_model_hash'],original_loaded_bytes_certified_here=False),
                encoder_preprocess=dict(report=cp['inherited_encoding_report_sha256']),
                normalizer=dict(contract=cp['contract']['action_map'],original_task_ref=slot['task_sha256']),q=cp['contract']['Q'],raw_domain=cp['contract']['packet_logit_cap'],
                horizon_feedback=dict(raw_steps=10,checkpoints=[5,10],feedback='open_loop'),
                controller_source='frozen '+CTRL,controller_budget=cp['contract']['per_group'],
                information_permissions='original blind full-history192, no privileged witness warmstart',native_evaluator='original native7 goal/success')
            # Some common fields are inherited contract descriptions, NOT loaded-byte
            # certification. Evidence grading below preserves that distinction.
            rows.append(dict(root_slot=root,arm=arm,status='existing_two_arm_calibration',shared=shared,
                target=evidence(refs,'controller_receipt',f'/plans/{branch}/target',p['target']),proposal_action_witness=witness,
                controller_action_witness=evidence(refs,'controller_receipt',f'/plans/{branch}',p),native_executed=True,
                observed_tracking=[dict(step=c['step'],MSE=c['actual_tracking_MSE'],
                    evidence=evidence(refs,'rgb_receipt',f'/rows/{branch}/checkpoints/{j}',c)) for j,c in enumerate(z['checkpoints'])],
                arrival=None,native_success=n['native_success'],native_evidence=evidence(refs,'physical_receipt',f'/rows/{branch}',n)))
    return rows


def evidence(refs,role,p,value):return dict(**refs[role],pointer=p,value_sha256=digest(value),scope='saved_metadata_MAIN_receipt_not_fullstrict_rebuilt')


def build():
    data={};refs={};missing=[]
    for role,(rel,h) in CATALOG.items():
        p=ROOT/'results'/rel
        if not p.exists():missing.append(dict(role=role,path=str(p),expected_sha256=h));continue
        data[role]=read(p,h);refs[role]=dict(path=str(p),sha256=h)
    required=('controller_prepare','controller_report','controller_receipt','physical_prepare','physical_report','physical_receipt','rgb_report','rgb_receipt')
    rows=joined(data,refs) if all(k in data for k in required) else [dict(root_slot=r,arm=a,shared=None,status='missing_saved_chain',arrival=None,native_success=None) for r in (0,1) for a in ARMS]
    prior={}
    for role in ('hi_delivery','hi_catalog','hi_parity','hi_tracker','prior_protocol','sage_metadata'):
        if role not in data:prior[role]=None;continue
        x=data[role];prior[role]=dict(evidence=evidence(refs,role,'',x),schema=x['schema'],saved_status=x.get('status'))
    if 'hi_delivery' in data:
        x=data['hi_delivery'];need(x['schema']=='agent_hilewm_rgb_high_low_20261028_v1.consume' and x['physical_release'] is False,'actual Hi integration not a fake prior')
        prior['hi_delivery'].update(frozen_learned_integration_exists=True,usable_as_old_E1_third_arm=False)
    if 'hi_parity' in data:
        x=data['hi_parity'];need(x['schema']=='agent_p2_hilewm_computation_parity_20261029_v1.receipt' and x['all_values_exact'] is True,'actual bounded computation parity')
        prior['hi_parity'].update(bounded_computation_pairing=True,original_raw_scaler_certified=x['diagnostics']['normalizer_pairing'])
    if 'hi_tracker' in data:
        x=data['hi_tracker'];need(x['schema']=='agent_p2_history3_tracker_cuda_20261029_v1.receipt' and x['raw_inverse'] is False,'normalized tracker not missing/not raw deployment')
        prior['hi_tracker'].update(normalized_history3_controller_exists=True,raw_inverse=False)
    bindings={r['path']:r['sha256'] for r in refs.values()}
    for path in own_files()+[ROOT/'experiments/spectral_reachability'/(s+'.py') for s in READ_SOURCES]:
        need(path.is_file() and not path.is_symlink() and path.stat().st_size<=CAP,'existing bounded source only');bindings[str(path)]=sha(path)
    need(all(sha(p)==h for p,h in bindings.items()),'locally read closure unchanged')
    return dict(schema=NAME+'.prepare',status='evidence_joined_three_arm_not_complete' if not missing else 'missing_catalog_evidence',
        contract=contract(),rows=rows,comparability=field_matrix(rows),existing_prior_evidence=prior,
        evidence_grading=dict(matrix='saved contract field comparison, not full source/load/raw-byte provenance certification',
            witness='model predicted postcondition only',native='actual raw10 dispatch saved; no latent threshold implied',
            tracking='existing RGB continuous measurement, no rerun/re-score'),
        missing=missing,remaining=['same_E1_learned_target_delivery','presealed_arrival_threshold',
            'all_three_proposal_cost_contract','original_Hi_raw_scaler_if_official_route','fresh_independent_cohort_for_confirmation'],
        bindings=bindings,new_neural_calls=0,new_physics_calls=0,new_fullstrict_calls=0,training=False,
        production_ready=False,dispatch_allowed=False,three_arm_comparison_complete=False,P1_admission=False,scientific_admission=False)


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def write(path,value):
    p=absolute(path);b=raw(value);need(p.name.startswith(NAME+'_') and not p.exists() and len(b)<=CAP,'own bounded exclusive output')
    with p.open('xb') as f:need(f.write(b)==len(b),'full write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(p),sha256=sha(p),status=value['status'])


def verify(path,h):
    x=read(path,h);need(x['schema']==NAME+'.prepare' and equal(x,build()),'whole deterministic offline contract rebuild')
    return dict(verified=True,sha256=h,dispatch_allowed=False,three_arm_comparison_complete=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    a=sub.add_parser('verify');a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
    a=p.parse_args(argv);v=write(a.output,build()) if a.command=='prepare' else verify(a.prepared,a.prepare_sha)
    print(raw(v).decode(),end='')


if __name__=='__main__':main()
