"""RAM saved-JSON fixtures only; no production assembler/consumer or tmp files."""
from copy import deepcopy
import csv
import io
from pathlib import Path
import runpy
from types import SimpleNamespace
import pytest
from experiments.spectral_reachability import p1_moving_baseline_reader_table_20261027_v1 as m


def fixture(extended_rgb=True):
    defs=runpy.run_path(str(Path(__file__).with_name('test_p1_moving_baseline_comparison_20261027_v3.py')))
    f,e,rr,p,events,arrays,pixels,compare=defs['raw_extended_fixture']()
    if extended_rgb:
        receipts=m.comparison.extended_rgb_receipts(rr,p,events,arrays,pixels,compare)
        e['rgb']=dict(state='complete',entry=dict(state='complete',report=dict(path='/ram/extendedRGB/report.json',sha256='a'*64),prepare_sha256='b'*64),report=rr,receipts=receipts)
    r=m.comparison.assemble(*f,extended=e)
    v2defs=runpy.run_path(str(Path(__file__).with_name('test_p1_moving_baseline_comparison_20261027_v2.py')))
    old,_=v2defs['kino_fixture']()
    manifest=m.comparison.template();manifest['existing']['sources']=[deepcopy(b['entry']) for b in old[2]]
    manifest['additional_sources']=[deepcopy(b['entry']) for b in f[2]]
    # RAM physical fixture lacked manifest-only source_sha256; enrich BOTH
    # manifest and already-assembled provenance, updating the digest join.
    for run in r['physical_runs']:
        old_id=run['source_id'];entry=deepcopy(run['provenance'])
        entry['source_sha256']=m.comparison.EXT_PHYSICAL_SHA if run['arms']==list(m.comparison.EXTRA_ARMS) else m.comparison.prior.PHYSICAL_SOURCE_SHA
        run['provenance']=entry;run['source_id']=m.base.digest(entry)
        for row in r['rows']:
            if row.get('physical_source_id')==old_id:row['physical_source_id']=run['source_id']
    manifest['existing']['physical']=deepcopy(r['physical_runs'][0]['provenance'])
    manifest['existing']['rgb']=deepcopy(r['RGB_runs'][0]['provenance'])
    manifest['existing']['additional_cohorts']=[dict(arms=[m.comparison.prior.KINO],physical=deepcopy(r['physical_runs'][1]['provenance']),rgb=deepcopy(r['RGB_runs'][1]['provenance']))]
    manifest['new_downstream']=dict(arms=list(m.comparison.EXTRA_ARMS),physical=deepcopy(r['physical_runs'][2]['provenance']),rgb=deepcopy(e['rgb']['entry']))
    mr=dict(path='/ram/comparison_v3_manifest.json',sha256='1'*64);cr=dict(path='/ram/comparison_v3.json',sha256='2'*64)
    r['manifest']=deepcopy(mr);r['bindings_sha256']={str(Path(m.comparison.__file__).resolve()):m.COMPARISON_SHA}
    return manifest,r,mr,cr


def csv_rows(data):return list(csv.DictReader(io.StringIO(data.decode())))


def test_complete12_reader_rows_preserve_each_root_no_rank_or_average():
    f=fixture();out=m.render(*f);rows=csv_rows(out['reader_rows_csv'])
    assert len(rows)==12 and [(x['algorithm'],x['calibration_slot']) for x in rows]==[(a,str(s)) for a in m.ARMS for s in (0,1)]
    assert all(x['native_success']=='false' and x['full_horizon']=='true' and x['model_Q']=='0.25' for x in rows)
    assert all(x['implementation']=='local_not_official_reproduction' and x['equal_total_compute']=='false' for x in rows)
    assert all(x['scope']=='known_two_root_calibration_only' and x['comparison_sha256']=='2'*64 for x in rows)
    text=out['reader_table_md'].decode();assert '不是独立确认' in text and 'NA是缺失' in text and '不同' in text
    assert '排名或P1/P2通过' in text and '逐根' in text and 'mean' not in rows[0]


def test_each_root_numbers_not_collapsed_to_mean():
    f=list(fixture());r=f[1];r['rows'][0]['model_Q']=m.base.actual(.12345678901234568,'fixture')
    r['rows'][1]['model_Q']=m.base.actual(.9,'fixture')
    rows=csv_rows(m.render(*f)['reader_rows_csv'])
    assert rows[0]['model_Q']=='0.12345678901234568' and rows[1]['model_Q']=='0.9'


def test_unmeasured_not_zero_AD_forward_not_added_and_scalar_J_kept_distinct():
    rows=csv_rows(m.render(*fixture())['reader_rows_csv']);lookup={(r['algorithm'],r['calibration_slot']):r for r in rows}
    grasp=lookup[m.comparison.EXTRA_ARMS[0],'0'];sra=lookup[m.comparison.EXTRA_ARMS[1],'0']
    assert grasp['predict_completed']=='1802' and grasp['backward_completed']=='900' and grasp['AD_forward_in_predict']=='1800'
    assert grasp['grad_API_completed']=='NA' and grasp['jacobian_completed']=='0'
    assert grasp['predict_API_completed']=='1802'
    assert sra['predict_completed']=='88' and sra['AD_forward_in_predict']=='2' and sra['mathematical_reverse_directions']=='192'
    assert sra['backward_completed']=='NA' and sra['jacobian_completed']=='1' and sra['bounded_solve_completed']=='1'


def test_physical_cohort_cost_once_and_RGB_separate_shared_encoding_footer():
    out=m.render(*fixture());cohorts=csv_rows(out['reader_cohorts_csv']);ph=[x for x in cohorts if x['kind']=='physical']
    assert len(ph)==3 and [x['public_step_completed'] for x in ph]==['120','40','80']
    assert [x['physics_step_completed'] for x in ph]==['1224','408','816']
    assert [x['encoder_completed'] for x in cohorts if x['kind']=='RGB']==['12','4','8']
    inherited=[x for x in cohorts if x['kind']=='inherited_encoding_shared_once']
    assert len(inherited)==1 and inherited[0]['encoder_completed']==inherited[0]['projector_completed']=='8'
    assert all(x['scope']=='whole_invocation_not_per_root_or_arm' for x in cohorts)
    assert '共享原encoder8/projector8只记一次' in out['reader_table_md'].decode()


def test_actual_physical_missing_RGB_native_false_but_actual_Q_NA():
    rows=csv_rows(m.render(*fixture(False))['reader_rows_csv'])
    for r in rows:
        if r['algorithm'] in m.comparison.EXTRA_ARMS:
            assert r['native_success']=='false' and r['realQ_terminal10']=='NA' and r['realQ_terminal10_status']=='missing'


@pytest.mark.parametrize('edit',['missing_root','duplicate_root','old_task','bool_slot','wrong_manifest_sha','wrong_manifest_path','official','equal_compute',
    'schema','source_binding','wrong_model_source','wrong_physical_source','wrong_physical_census','wrong_RGB_census','wrong_denominator','wrong_model_census',
    'NA_as_zero','bool_as_zero','bad_counter','nan_Q','reverse_negative','wrong_model_status'])
def test_strict_contract_rejects_mixed_or_invented_evidence(edit):
    f=list(fixture(False));mfest,r,_,_=f;row=r['rows'][-1]
    if edit=='missing_root':r['rows'].pop()
    elif edit=='duplicate_root':r['rows'][-1]=deepcopy(r['rows'][-2])
    elif edit=='old_task':row['task_sha256']='e'*64
    elif edit=='bool_slot':row['source_slot']=True
    elif edit=='wrong_manifest_sha':r['manifest']['sha256']='e'*64
    elif edit=='wrong_manifest_path':r['manifest']['path']='/ram/other.json'
    elif edit=='official':r['official_reproduction']=True
    elif edit=='equal_compute':r['equal_total_compute']=True
    elif edit=='schema':r['schema']='old.report'
    elif edit=='source_binding':r['bindings_sha256']={}
    elif edit=='wrong_model_source':r['runs'][0]['provenance']['report']['sha256']='e'*64
    elif edit=='wrong_physical_source':row['physical_source_id']='e'*64
    elif edit=='wrong_physical_census':r['physical_runs'][-1]['native_successes']=1
    elif edit=='wrong_RGB_census':r['RGB_runs'][0]['actual_images']=11
    elif edit=='wrong_denominator':r['denominator']['expected_model_rows']=10
    elif edit=='wrong_model_census':r['model_layer_complete']=False
    elif edit=='NA_as_zero':row['realQ_terminal10']=m.base.actual(0.,'fabricated')
    elif edit=='bool_as_zero':row['native_success']=m.base.actual(0,'not a boolean')
    elif edit=='bad_counter':row['resources']['value']['predict']['value']['completed']=89
    elif edit=='nan_Q':row['model_Q']['value']=float('nan')
    elif edit=='reverse_negative':row['resources']['value']['mathematical_reverse_directions']['value']=-1
    else:row['model_status']='missing'
    with pytest.raises(ValueError):m.render(*f)


def test_external_SHA_input_boundary_no_consumer_or_neural(monkeypatch):
    f=fixture();refs=[]
    monkeypatch.setattr(m,'sha',lambda p:m.COMPARISON_SHA)
    def load(ref):refs.append(ref);return f[0] if len(refs)==1 else f[1]
    monkeypatch.setattr(m.base,'load_ref',load)
    monkeypatch.setattr(m,'render',lambda *a:dict(reader_rows_csv=b'a',reader_cohorts_csv=b'b',reader_table_md=b'c'))
    monkeypatch.setattr(m.base,'reference',lambda ref:None)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='floor'):m.run(f[2]['path'],f[2]['sha256'],f[3]['path'],f[3]['sha256'],'/ram/new_table')
    assert refs==[f[2],f[3]]


def test_source_freeze_and_cap_and_no_input_mutation():
    f=fixture();saved=deepcopy(f);out=m.render(*f)
    assert f==saved and sum(map(len,out.values()))<m.CAP
    assert m.sha(m.comparison.__file__)==m.COMPARISON_SHA


@pytest.mark.parametrize('state',['missing','partial'])
def test_missing_partial_model_and_physical_are_NA_never_zero(state):
    f=list(fixture(False));mf,r,_,_=f;arm=m.comparison.EXTRA_ARMS[-1]
    entry=mf['additional_sources'][-1];old_id=m.base.digest(entry)
    entry.pop('consume');entry.update(state=state,reason='not released')
    if state=='missing':entry.pop('report')
    new_id=m.base.digest(entry)
    run=next(x for x in r['runs'] if x['source_id']==old_id)
    run.update(source_id=new_id,provenance=deepcopy(entry),state=state)
    r['physical_runs'].pop();mf['new_downstream']=m.comparison.deferred()
    for row in r['rows']:
        if row['arm'] in m.comparison.EXTRA_ARMS:
            row.update(physical_status='missing',rgb_status='missing',native_success=m.base.missing('not run'),
                full_horizon=m.base.missing('not run'),physical_returned_steps=m.base.missing('not run'),
                realQ_root=m.base.missing('no observation'),realQ_prefix5=m.base.missing('no observation'),realQ_terminal10=m.base.missing('no observation'))
            row.pop('physical_source_id',None)
        if row['arm']==arm:
            row.update(source_id=new_id,model_status=state,model_Q=m.base.missing('not released',state),resources=m.base.missing('not released',state))
    r['denominator'].update(actual_model_rows=10,actual_physical_rows=8)
    r.update(status='partial_comparison_not_admitted',model_layer_complete=False)
    rows=csv_rows(m.render(*f)['reader_rows_csv'])
    unknown=[x for x in rows if x['algorithm']==arm]
    assert len(unknown)==2 and all(x['model_Q']==x['native_success']==x['predict_completed']=='NA' for x in unknown)
    assert all(x['model_Q_status']==state for x in unknown)
