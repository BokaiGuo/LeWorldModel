"""RAM callbacks only. Fake counters must never become real-model evidence."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import moving_prediction_budget_contract_20261027_v1 as p


def cell(value): return dict(status='actual', value=value, basis='RAM fixture, not scientific evidence')


def fixture():
    methods = []
    for i, arm in enumerate(p.ARMS):
        n = [136, 134, 72, 84, 1802, 88][i]
        def counters(n):
            c = {k: cell(dict(attempted=n, completed=n)) for k in ('predict', 'forward_samples', 'action_encoder')}
            c['predict_API'] = (cell(dict(attempted=n, completed=n)) if i in (0,4) else
                                dict(status='unmeasured',value=None,reason='not separately metered'))
            c['backward'] = cell(dict(attempted=900 if i==4 else 0, completed=900 if i==4 else 0))
            c['pred_projector'] = dict(status='unmeasured',value=None,reason='not separately metered')
            return c
        methods.append(dict(algorithm=arm, source_id=p.digest(arm), contract_fields=dict(horizon=2, sealed_parameter=i),
            root_receipts=[dict(source_slot=s,task_sha256=t,counters=counters(n)) for s,t in enumerate(p.TASKS)],
            two_root_totals=counters(2*n)))
    return dict(schema='p1_moving_baseline_budget_matrix_20261027_v1.matrix',
        status='saved_resource_projection_not_admitted', equal_total_compute=False, P1_admission=False, methods=methods,
        inherited_encoding_shared_once=cell(dict(encoder=dict(attempted=8,completed=8),projector=dict(attempted=8,completed=8))),
        model_invocations=[], physical_cohorts=[], RGB_cohorts=[])


def policy(mode='predict_API_calls_matched', cap=2, sizes=None, scope=None):
    groups=p.calibration_groups(fixture())
    return dict(schema=p.STEM+'.MAIN_declaration',authority='MAIN',declaration_id='RAM_MAIN_mock', issued_at_ns=1,
        matrix_sha256=p.MATRIX_SHA,population='existing_calibration_two_roots',mode=mode,cap=cap,
        allocation='per_root_per_arm',groups=groups,
        scope=dict(prepare=True,search=True,ADforward=True,replay=True,failed_attempts='charged_no_refund',count_on='before_dispatch') if scope is None else scope,
        batch=dict(allowed_sizes=[1,4] if sizes is None else sizes,max_predictor_forwards_per_API=4,
                   API_predictor_batch_relation='same_logical_batch'),
        no_outcome_based_configuration_change=True)


def meter(d=None,**kwargs):
    d=policy() if d is None else d
    return p.PredictionBudgetMeter(d,p.digest(d),'MAIN',p.calibration_groups(fixture()),now_ns=10,**kwargs)


GROUP='arm0:root0'


def api(m,callback=lambda _:None,stage='search',ad=False,batch=1,group=GROUP):
    return m.invoke('predict_API', callback,group_id=group,stage=stage,ad_forward=ad,batch_size=batch)


def counts(m,group=GROUP,scope='included'):
    return m.snapshot()['counts'][group][scope]


def test_API_mode_one_batch_call_not_B_API_or_AE_plus_predict():
    m=meter(policy(cap=1))
    def body(parent):
        m.invoke('action_encoder',lambda _:None,parent=parent,batch_size=4,rows_per_sample=3)
        m.invoke('predictor_forward',lambda _:None,parent=parent,batch_size=4)
        m.invoke('pred_projector',lambda _:None,parent=parent,batch_size=4,rows_per_sample=3)
        m.invoke('predictor_forward',lambda _:None,parent=parent,batch_size=4)
        return 'callback returned'
    assert api(m,body,batch=4)=='callback returned'
    c=counts(m)
    assert c['predict_API_calls']==dict(attempted=1,completed=1)
    assert c['predictor_forward_calls']['attempted']==2
    assert c['predictor_forward_samples']['attempted']==8
    assert c['action_encoder_rows']['attempted']==c['pred_projector_rows']['attempted']==12
    assert c['action_encoder_calls']['attempted']==1
    fired=[]
    with pytest.raises(p.BudgetExceeded): api(m,lambda _:fired.append(True))
    assert fired==[] and counts(m)['predict_API_calls']['attempted']==1


def test_sample_mode_caps_batch_items_not_API_count_or_context_rows():
    m=meter(policy('predictor_forward_samples_matched',cap=8))
    def body(parent): return m.invoke('predictor_forward',lambda _:42,parent=parent,batch_size=4)
    assert api(m,body,batch=4)==api(m,body,batch=4)==42
    assert counts(m)['predict_API_calls']['attempted']==2
    assert counts(m)['predictor_forward_samples']['attempted']==8
    fired=[]
    with pytest.raises(p.BudgetExceeded):
        api(m,lambda parent:m.invoke('predictor_forward',lambda _:fired.append(True),parent=parent,batch_size=4),batch=4)
    assert fired==[]
    assert counts(m)['predict_API_calls']==dict(attempted=3,completed=2)
    assert counts(m)['predictor_forward_samples']==dict(attempted=8,completed=8)


@pytest.mark.parametrize('mode',p.MODES)
def test_failure_is_charged_and_never_refunded(mode):
    m=meter(policy(mode,cap=1,sizes=[1]))
    def fail(_): raise RuntimeError('RAM deliberate failure')
    with pytest.raises(RuntimeError):
        if mode=='predict_API_calls_matched': api(m,fail)
        else: api(m,lambda parent:m.invoke('predictor_forward',fail,parent=parent))
    primary=p.MODES[mode]
    assert counts(m)[primary]==dict(attempted=1,completed=0)
    with pytest.raises(p.BudgetExceeded):
        if mode=='predict_API_calls_matched': api(m)
        else: api(m,lambda parent:m.invoke('predictor_forward',lambda _:None,parent=parent))
    assert counts(m)[primary]==dict(attempted=1,completed=0)


@pytest.mark.parametrize('stage,ad,flag',[('prepare',False,'prepare'),('search',True,'ADforward'),('replay',False,'replay')])
@pytest.mark.parametrize('mode',p.MODES)
def test_MAIN_scope_exclusions_are_explicit_observed_not_erased(stage,ad,flag,mode):
    d=policy(mode,cap=1); d['scope'][flag]=False; m=meter(d)
    def body(parent): m.invoke('predictor_forward',lambda _:None,parent=parent)
    for _ in range(2): api(m,body,stage=stage,ad=ad)
    primary=p.MODES[mode]
    assert counts(m)[primary]['attempted']==0
    assert counts(m,scope='observed')[primary]['attempted']==2
    api(m,body)
    assert counts(m)[primary]['attempted']==1


def test_AD_forward_is_subset_not_extra_prediction_charge():
    m=meter(policy('predictor_forward_samples_matched',cap=4))
    api(m,lambda parent:m.invoke('predictor_forward',lambda _:None,parent=parent,batch_size=4),ad=True,batch=4)
    assert counts(m)['predictor_forward_samples']['attempted']==4
    assert any(e.get('ad_forward') is True and e.get('kind')=='predictor_forward' for e in m.snapshot()['events'])


@pytest.mark.parametrize('mode',p.MODES)
def test_explicit_internal_chunking_counts_actual_child_samples_not_API_batch(mode):
    d=policy(mode,cap=4 if mode=='predictor_forward_samples_matched' else 1)
    d['batch']['API_predictor_batch_relation']='bounded_child_batches'
    m=meter(d)
    def body(parent):
        for _ in range(4):m.invoke('predictor_forward',lambda _:None,parent=parent,batch_size=1)
    api(m,body,batch=4)
    assert counts(m)['predict_API_calls']['attempted']==1
    assert counts(m)['predictor_forward_calls']['attempted']==4
    assert counts(m)['predictor_forward_samples']['attempted']==4


def test_token_parent_batch_fanout_and_terminal_invariants():
    d=policy();d['batch']['max_predictor_forwards_per_API']=1; m=meter(d)
    parent=m.before_dispatch('predict_API',group_id=GROUP,stage='search',ad_forward=False)
    with pytest.raises(p.ContractError,match='inherit'):
        m.before_dispatch('predictor_forward',parent=parent,stage='prepare')
    with pytest.raises(p.ContractError,match='batch mismatch'):
        m.before_dispatch('predictor_forward',parent=parent,batch_size=4)
    child=m.before_dispatch('predictor_forward',parent=parent)
    with pytest.raises(p.ContractError,match='before its children'):m.after_return(parent)
    m.after_failure(child,'RAMFailure')
    with pytest.raises(p.ContractError,match='fanout'):m.before_dispatch('predictor_forward',parent=parent)
    m.after_return(parent)
    with pytest.raises(p.ContractError,match='terminal'):m.after_return(parent)
    with pytest.raises(p.ContractError,match='live API'):m.before_dispatch('action_encoder',parent=parent)
    with pytest.raises(p.ContractError,match='forged'):meter().after_return(parent)
    with pytest.raises(p.ContractError,match='forged'):m.after_return(p.Ticket(parent.owner,parent.serial))


@pytest.mark.parametrize('kind',p.KINDS[1:])
def test_direct_unwrapped_model_module_not_accepted(kind):
    with pytest.raises(p.ContractError):meter().before_dispatch(kind)


@pytest.mark.parametrize('batch',[True,0,-1,2,1.5])
def test_batch_invalid_or_not_predeclared_rejected(batch):
    m=meter()
    with pytest.raises(p.ContractError):api(m,batch=batch)
    assert counts(m)['predict_API_calls']['attempted']==0


def test_predictor_tokens_not_samples_and_false_unknown_tags_rejected():
    m=meter()
    with pytest.raises(p.ContractError):m.before_dispatch('predict_API',group_id=GROUP,stage='search',ad_forward=False,rows_per_sample=3)
    with pytest.raises(p.ContractError):m.before_dispatch('predict_API',group_id=GROUP,stage='ADforward',ad_forward=False)
    with pytest.raises(p.ContractError):m.before_dispatch('predict_API',group_id=GROUP,stage='search',ad_forward=0)


def test_pending_and_snapshot_mutation_not_final_evidence():
    m=meter();t=m.before_dispatch('predict_API',group_id=GROUP,stage='search',ad_forward=False)
    x=m.snapshot();assert x['engineering_report_ready'] is False and x['pending']==1
    x['counts'][GROUP]['included']['predict_API_calls']['attempted']=999
    assert counts(m)['predict_API_calls']['attempted']==1
    m.after_return(t)
    assert m.snapshot()['engineering_report_ready'] is True


def test_full_fake_ledger_cannot_be_real_model_or_science_evidence():
    m=meter(policy(cap=1))
    for g in p.calibration_groups(fixture()):api(m,group=g['group_id'])
    x=m.snapshot()
    assert x['callback_attempt_allocation_matched'] is True
    assert x['actual_model_execution_verified'] is x['scientific_admission'] is False
    assert x['equal_total_compute'] is x['equal_FLOPs'] is False
    assert x['evidence_kind']=='unverified_callback_observations_not_real_model_receipts'
    assert all(c['value'] is None for c in x['other_resources'].values())


@pytest.mark.parametrize('mode',p.MODES)
def test_concurrent_atomic_attempt_cap(mode):
    m=meter(policy(mode,cap=2,sizes=[1]));fired=[]
    def work(_):
        try:
            if mode=='predict_API_calls_matched':api(m,lambda _:fired.append(True))
            else:api(m,lambda parent:m.invoke('predictor_forward',lambda _:fired.append(True),parent=parent))
            return 'ok'
        except p.BudgetExceeded:return 'denied'
    with ThreadPoolExecutor(max_workers=8) as ex: results=list(ex.map(work,range(20)))
    assert results.count('ok')==2 and len(fired)==2
    assert counts(m)[p.MODES[mode]]==dict(attempted=2,completed=2)
    assert m.snapshot()['pending']==0


@pytest.mark.parametrize('when',['before','after'])
def test_sink_failure_poisoned_closed_no_refund(when):
    def sink(e):
        if e['event']==('before_dispatch' if when=='before' else 'after_return'):raise OSError('RAM sink failure')
    m=meter(sink=sink);fired=[]
    with pytest.raises(OSError):api(m,lambda _:fired.append(True))
    assert len(fired)==(0 if when=='before' else 1)
    assert counts(m)['predict_API_calls']['attempted']==1
    assert counts(m)['predict_API_calls']['completed']==(0 if when=='before' else 1)
    assert m.snapshot()['engineering_report_ready'] is False
    with pytest.raises(p.ContractError,match='fault'):api(m)


def test_external_sink_cannot_reenter_meter_to_hide_dispatch():
    m=None
    def sink(_):api(m)
    m=meter(sink=sink)
    with pytest.raises(p.ContractError):api(m)
    assert m.snapshot()['fault'] is not None


@pytest.mark.parametrize('edit',['schema','authority','matrix','mode','cap_bool','cap_zero','groups','time','future','retune',
    'no_scope','bad_scope','search_excluded','refund','count_on','batch_empty','batch_bool','batch_dup','fanout','population','extra','batch_relation'])
def test_bad_MAIN_declaration_rejected(edit):
    d=policy()
    if edit=='schema':d['schema']='legacy_allaxes'
    elif edit=='authority':d['authority']='worker'
    elif edit=='matrix':d['matrix_sha256']='f'*64
    elif edit=='mode':d['mode']='equal_total_compute'
    elif edit=='cap_bool':d['cap']=True
    elif edit=='cap_zero':d['cap']=0
    elif edit=='groups':d['groups'][0]['sealed_configuration_sha256']='f'*64
    elif edit=='time':d['issued_at_ns']=True
    elif edit=='future':d['issued_at_ns']=11
    elif edit=='retune':d['no_outcome_based_configuration_change']=False
    elif edit=='no_scope':d.pop('scope')
    elif edit=='bad_scope':d['scope']['prepare']=1
    elif edit=='search_excluded':d['scope']['search']=False
    elif edit=='refund':d['scope']['failed_attempts']='refund'
    elif edit=='count_on':d['scope']['count_on']='after_return'
    elif edit=='batch_empty':d['batch']['allowed_sizes']=[]
    elif edit=='batch_bool':d['batch']['allowed_sizes']=[True]
    elif edit=='batch_dup':d['batch']['allowed_sizes']=[1,1]
    elif edit=='fanout':d['batch']['max_predictor_forwards_per_API']=0
    elif edit=='population':d['population']='new_roots_self_certified'
    elif edit=='batch_relation':d['batch']['API_predictor_batch_relation']='guess_from_AE'
    else:d['reported_outcome']=0
    with pytest.raises(p.ContractError):meter(d)


def test_SHA_trust_anchor_and_predeclaration_copy():
    d=policy();groups=p.calibration_groups(fixture())
    with pytest.raises(p.ContractError):p.declaration(d,'f'*64,'MAIN',groups,10)
    with pytest.raises(p.ContractError):p.declaration(d,p.digest(d),'worker',groups,10)
    m=meter(d);d['cap']=999
    assert m.snapshot()['cap_per_group']==2


def test_supplied_group_list_cannot_collapse_denominator_or_accept_bool_slot():
    d=policy();d['groups'][1]=deepcopy(d['groups'][0])
    with pytest.raises(p.ContractError):p.declaration(d,p.digest(d),'MAIN',d['groups'],10)
    d=policy();d['groups'][0]['source_slot']=False
    with pytest.raises(p.ContractError):p.declaration(d,p.digest(d),'MAIN',d['groups'],10)


def test_calibration_ready_not_scientific_admission_or_matched_or_allaxes_gate():
    matrix=fixture();old=deepcopy(matrix);x=p.calibration_report(matrix)
    assert matrix==old
    assert x['report_ready'] is True and x['saved_input_verified'] is False
    assert x['primary_mode'] is None and x['prediction_budget_match']=='unestablished'
    assert x['meter_configuration_ready'] is False
    assert x['scientific_admission'] is x['new_root_certificate_issued'] is x['launch_authorized'] is False
    assert x['legacy_all_axes_contract_supports_prediction_budget_claim'] is False
    assert x['other_resource_equality_required'] is x['walltime_equality_required'] is False
    assert x['methods']==matrix['methods'] and x['shared_accounting']['inherited_encoding_shared_once']==matrix['inherited_encoding_shared_once']
    assert x['methods'][1]['two_root_totals']['predict_API']['value'] is None
    for mode,axis in [('predict_API_calls_matched','predict_API'),('predictor_forward_samples_matched','forward_samples')]:
        d=policy(mode);r=p.calibration_report(matrix,d,p.digest(d),'MAIN',10)
        assert r['primary_mode']==mode and r['selected_saved_counter_axis']==axis
        assert r['prediction_budget_match']=='unestablished' and r['report_ready'] is True
        assert r['meter_configuration_ready'] is r['common_cap_allocation_predeclared'] is True


def test_missing_zero_rejected_and_all_other_resources_preserved():
    m=fixture();m['methods'][1]['two_root_totals']['pred_projector']['value']=0
    with pytest.raises(p.ContractError):p.calibration_report(m)
    m=fixture();m['author_reported_values']={'Q':0}
    x=p.calibration_report(m)
    assert 'author_reported_values' not in x
    assert x['methods'][4]['root_receipts'][0]['counters']['backward']['value']['completed']==900


def test_cli_RAM_SHA_gate_is_distinct_from_callback_evidence(monkeypatch):
    m=fixture();calls=[]
    def read(path,h=None):calls.append(h);return m
    monkeypatch.setattr(p,'read_json',read)
    monkeypatch.setattr('sys.argv',['contract'])
    with redirect_stdout(io.StringIO()) as out:p.main()
    x=json.loads(out.getvalue());assert calls==[p.MATRIX_SHA]
    assert x['saved_input_verified'] is True and x['report_ready'] is True
    assert x['scientific_admission'] is False


def test_reader_RAM_rejects_bad_SHA_duplicates_nonfinite_and_symlinks(monkeypatch):
    raw=b'{"x":1}'
    monkeypatch.setattr(Path,'is_file',lambda _:True)
    monkeypatch.setattr(Path,'is_symlink',lambda _:False)
    monkeypatch.setattr(Path,'resolve',lambda x:x)
    monkeypatch.setattr(Path,'stat',lambda _:SimpleNamespace(st_size=len(raw)))
    monkeypatch.setattr(Path,'read_bytes',lambda _:raw)
    assert p.read_json('/ram/file',p.sha(raw))==dict(x=1)
    with pytest.raises(p.ContractError):p.read_json('/ram/file','f'*64)
    for raw in (b'{"x":1,"x":2}',b'{"x":NaN}'):
        with pytest.raises(p.ContractError):p.read_json('/ram/file')
    monkeypatch.setattr(Path,'is_symlink',lambda _:True)
    with pytest.raises(p.ContractError):p.read_json('/ram/file')
