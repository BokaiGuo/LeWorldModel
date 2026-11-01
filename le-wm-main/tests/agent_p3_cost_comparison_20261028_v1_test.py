"""Saved real metadata and RAM negative tests; no neural/physics/real CLI writes."""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_cost_comparison_20261028_v1 as m


@lru_cache(maxsize=1)
def fixture():
    cuda_path=m.ROOT/'results/agent_p3_fixed_execution_cuda_20261028_v1_cuda_MAIN01/report.json'
    physical_path=m.ROOT/'results/agent_p3_fixed_execution_physical_20261028_v2_MAIN01/report.json'
    receipt=m.ROOT/'results/consume_p3_native_saved_MAIN_20261028_v1_MAIN01.json'
    assert m.sha(cuda_path)==m.CUDA_SHA and m.sha(physical_path)==m.PHYSICAL_SHA
    assert m.sha(receipt)=='0ece90fd652a171cb6dda72ebd7385f99698d73faf126471da6b37f9bc0e4214'
    cu=m.read(cuda_path,256*1024**2);ph=m.read(physical_path,32*1024**2);v=m.read(receipt,32*1024**2)
    assert m.sha(cu['prepared'])==cu['prepare_sha256']==v['bindings'][cu['prepared']]
    p=m.read(cu['prepared'],16*1024**2);signals={}
    for mode,value in p['panel']['base']['inputs'].items():
        path=Path(value['output'])/'report.json'
        assert m.sha(path)==value['report_sha256']==v['bindings'][str(path)]==p['sources'][str(path)]
        signals[mode]=m.read(path,256*1024**2)
    return cu,ph,v,p,signals


@lru_cache(maxsize=1)
def completed():return m.assemble(*fixture())


def test_real_saved_exact_all4_budget_with_paid_signal_and_prefix():
    result=completed()
    assert result['fixed_denominator']==4 and result['paired_denominator']==result['known_roots']==2
    assert [(r['slot'],r['mode']) for r in result['rows']]==m.ORDER
    assert [r['candidate_quota'] for r in result['rows']]==[12,4,12,4]
    for r in result['rows']:
        x=r['resources']
        assert [x[k]['completed']['predict_API'] for k in ('signal','search','selected_prefix_verification','new_model')]==[8,24,1,25]
        assert x['total_completed']['predict_API']==x['total_completed']['predict_samples']==33
        assert x['total_completed']['jacobian_API']==2 and x['total_completed']['VJP_directions']==384
        assert x['physical']['public_step']==dict(attempted=15,completed=15)
        assert x['physical']['physics_step']==dict(attempted=154,completed=154)
    assert result['cost_evidence']['forward_AD_count_saving'] is False
    assert result['cost_evidence']['lower_count_axes']==['metric_calls','replay_calls'] or set(result['cost_evidence']['lower_count_axes'])=={'metric_calls','replay_calls'}
    assert result['cost_evidence']['FLOPs_saving'] is result['cost_evidence']['whole_job_wall_saving'] is None
    assert all(result[k] is False for k in m.FLAGS)


def test_real_native7_joint_object_columns_are_independently_recomputed():
    cu,ph,v,p,signals=fixture();result=completed()
    for raw,row in zip(ph['rows'],result['rows'],strict=True):
        d=np.asarray(raw['final_state'])-np.asarray(raw['goal'])
        assert row['final']['native7_distance']==float(np.linalg.norm(d))
        assert row['final']['joint_xy_error']==float(np.linalg.norm(d[:4]))
        assert row['final']['object_xy_error']==float(np.linalg.norm(d[2:4]))
        assert row['final']['native_success'] is False and row['final']['object_success'] is None
        assert row['returned_steps']==5 and row['native_earlystop'] is False
    assert result['benefit_evidence']['native_success_gain_observed'] is False
    assert result['pairs'][0]['final_error_delta_adaptive_minus_fixed']['object_xy_error']==0
    assert result['rows'][0]['final']['object_xy_error']==result['rows'][0]['initial']['object_xy_error']
    assert len(result['missing_for_C3'])>=6 and result['complete_C3_tree'] is False


def test_native_success_not_native7_or_object_only_threshold():
    g=[0.]*7;s=[1.,1.,1.,1.,.01,10000.,10000.]
    value=m.native_errors(g,s)
    assert value['native_success'] is True and value['native7_distance']>10000
    assert value['joint_xy_error']==2 and value['object_success'] is None
    s=[100.,100.,0.,0.,0.,0.,0.]
    value=m.native_errors(g,s)
    assert value['object_xy_error']==0 and value['native_success'] is False


@pytest.mark.parametrize('fault',['denominator','duplicate','boolslot','rowmode','request','raw','selection','context','goal','quota',
    'signal_count','prefix_count','sample','J','VJP','times','null_as_zero','gate','nested_consume','physical_count','signal_sha'])
def test_full_saved_join_and_ledger_mutations_failclosed(fault):
    cu,ph,v,p,signals=deepcopy(fixture())
    if fault=='denominator':cu['per_branch'].pop()
    elif fault=='duplicate':cu['winners'][1]=deepcopy(cu['winners'][0])
    elif fault=='boolslot':ph['rows'][0]['slot']=False
    elif fault=='rowmode':ph['rows'][0]['mode']='adaptive'
    elif fault=='request':ph['rows'][0]['request']['binding']['task_id']='1'*64
    elif fault=='raw':ph['rows'][0]['commands'][0][0]=.5
    elif fault=='selection':ph['rows'][0]['selection_digest']='1'*64
    elif fault=='context':v['producer_consume']['root_contexts'][0]['context']['latent_history'][0][0]+=1
    elif fault=='goal':ph['rows'][1]['goal'][0]+=1
    elif fault=='quota':p['panel']['base']['branches'][0]['candidates']=1
    elif fault=='signal_count':cu['per_branch'][0]['signal_resources']['predict_API']=0
    elif fault=='prefix_count':next(e for e in cu['resource_receipt']['events'] if e['scope'].get('operation') and e['resource']=='predict')['delta']['predict_API']=0
    elif fault=='sample':next(e for e in cu['resource_receipt']['events'] if e['resource']=='predict')['delta']['predict_samples']=2
    elif fault=='J':cu['per_branch'][0]['total_resources']['jacobian_API']=0
    elif fault=='VJP':next(e for e in signals['fixed']['resource_receipt']['events'] if e['delta']['VJP_directions'])['delta']['VJP_directions']+=1
    elif fault=='times':cu['resource_receipt']['events'][0]['ended_ns']=-1
    elif fault=='null_as_zero':cu['resource_receipt']['FLOPs']=0
    elif fault=='gate':ph['P3_admission']=True
    elif fault=='nested_consume':v['producer_consume']['status']='unchecked'
    elif fault=='physical_count':ph['counts']['public_step']['completed']-=1
    else:p['panel']['base']['branches'][0]['signal_report_sha256']='1'*64
    # Mimic an already authenticated object boundary for semantic negative tests:
    # synchronize mirrored fields so a deeper ledger/join check must still reject.
    v['rows']=deepcopy(ph['rows']);v['counts']=deepcopy(ph['counts'])
    for key in ('per_branch','prefixes','winners'):v['producer_consume'][key]=deepcopy(cu[key])
    with pytest.raises((ValueError,KeyError,StopIteration)):m.assemble(cu,ph,v,p,signals)


def test_unknown_failed_endpoint_keeps_full_denominator_not_zero():
    cu,ph,v,p,signals=deepcopy(fixture());r=ph['rows'][0]
    r.update(status='engineering_failed',first_error={'saved':'failure'},physical_receipt=None)
    v['rows']=deepcopy(ph['rows']);ph['errors']=v['errors']=[{'saved':'failure'}]
    result=m.assemble(cu,ph,v,p,signals)
    assert len(result['rows'])==4 and result['rows'][0]['final'] is None
    summary=result['modes']['fixed']['final']['native7_distance']
    assert summary['denominator']==2 and summary['known']==1 and summary['unknown']==1 and summary['full_denominator_mean'] is None
    assert result['pairs'][0]['final_error_delta_adaptive_minus_fixed'] is None
    assert result['benefit_evidence']['native_success_gain_observed'] is None
    assert result['rows'][0]['resources']['physical']['public_step']['completed']==15


def test_wall_scopes_never_summed_or_used_as_per_method_speed():
    result=completed();cu,ph,_,p,signals=fixture();w=result['wall']
    assert w['cuda_report_wall_seconds']==cu['wall_seconds'] and w['cuda_new_ledger_wall_ns']==cu['resource_receipt']['wall_ns']
    assert w['physical_execute_wall_seconds']==ph['wall_seconds']
    assert w['exclusive_branch_wall_seconds'] is w['whole_pipeline_wall_seconds'] is None
    assert set(w['signal_panels'])=={'fixed','adaptive'}
    for mode in ('fixed','adaptive'):assert w['signal_panels'][mode]['wall_seconds']==signals[mode]['wall_seconds']
    assert all(r['resources']['signal']['exclusive_branch_wall_seconds'] is None for r in result['rows'])


def boundary(monkeypatch):
    cu,ph,v,p,signals=fixture();calls=[]
    args=dict(cuda_output=str(m.ROOT/'results/agent_p3_fixed_execution_cuda_20261028_v1_cuda_MAIN01'),cuda_report_sha256=m.CUDA_SHA,
        physical_output=str(m.ROOT/'results/agent_p3_fixed_execution_physical_20261028_v2_MAIN01'),physical_report_sha256=m.PHYSICAL_SHA)
    docs={str(Path(args['cuda_output'])/'report.json'):cu,str(Path(args['physical_output'])/'report.json'):ph,cu['prepared']:p}
    for mode,value in p['panel']['base']['inputs'].items():docs[str(Path(value['output'])/'report.json')]=signals[mode]
    def consume(*a):calls.append(a);return deepcopy(v)
    monkeypatch.setattr(m,'consume_saved',consume)
    monkeypatch.setattr(m,'read',lambda path,cap:deepcopy(docs[str(path)]))
    return args,calls


def test_material_full_saved_consumer_boundary_retains_exact_binding(monkeypatch):
    args,calls=boundary(monkeypatch);result=m.material(args)
    assert calls==[(args['physical_output'],m.PHYSICAL_SHA)]
    assert result['upstream_consume']['artifact_authenticity_verified'] is True
    assert result['upstream_consume']['errors']==[] and len(result['rows'])==4
    assert all(str(p) in result['bindings'] for p in m.own_files())


@pytest.mark.parametrize('fault',['source','test_source','doc_source','cuda_sha','physical_sha','future','consume','signal_sha','drift'])
def test_prechecks_consume_failures_and_hash_drift(monkeypatch,fault):
    args,calls=boundary(monkeypatch);original=m.sha;counter=[]
    if fault in ('source','test_source','doc_source'):
        index={'source':0,'test_source':1,'doc_source':2}[fault];target=str(m.saved.own_files()[index])
    elif fault=='cuda_sha':target=str(Path(args['cuda_output'])/'report.json')
    elif fault=='physical_sha':target=str(Path(args['physical_output'])/'report.json')
    elif fault=='signal_sha':target=str(Path(fixture()[3]['panel']['base']['inputs']['fixed']['output'])/'report.json')
    else:target=None
    if target:monkeypatch.setattr(m,'sha',lambda path:'1'*64 if str(path)==target else original(path))
    elif fault=='future':args['cuda_report_sha256']='0'*64
    elif fault=='consume':monkeypatch.setattr(m,'consume_saved',lambda *a:(_ for _ in ()).throw(ValueError('strict upstream rejected')))
    else:
        target=str(Path(args['cuda_output'])/'report.json')
        def sha(path):
            if str(path)==target:
                counter.append(1)
                if len(counter)>1:return '1'*64
            return original(path)
        monkeypatch.setattr(m,'sha',sha)
    with pytest.raises(ValueError):m.material(args)
    if fault in ('source','test_source','doc_source','cuda_sha','physical_sha','future','consume'):assert calls==[]


def test_counter_units_and_unknown_aggregation():
    value={a:0 for a in m.AXES};assert m.sum_counts([value,value])==value
    with pytest.raises(ValueError):m.counters(dict(value,extra=0))
    value['predict_samples']=None
    with pytest.raises(ValueError):m.counters(value)
    value['predict_samples']=False
    with pytest.raises(ValueError):m.counters(value)
    assert m.mean_known([None,None],2)['full_denominator_mean'] is None
