"""RAM metadata projection only; no physical consumers or neural import/runner."""
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import agent_CEM8_baseline_table_20261028_v1 as m
import test_agent_CEM8_physical_analysis_20261028_v1 as hegel


def h(v): return hashlib.sha256(str(v).encode()).hexdigest()


def ledger(n,predict='predict',batch=1,ordinal=None):
    result=[]
    for i in range(n):
        for kind in ('action_encoder',predict,'pred_proj'):
            entry=dict(index=len(result),completed=True,resource=kind,
                       inputs=[dict(shape=[batch,3,192] if kind==predict else [batch,3,10])])
            if ordinal is not None: entry['scope']=dict(plan_index=ordinal)
            result.append(entry)
    return result


def fixture():
    a=hegel.m.assemble(*hegel.fixture()); cr=[]; sr=[]; sl=[]; ops=[]
    for i,key in enumerate(m.ORDER):
        plan=a['consumes']['cem8']['rows'][i]['selected_suffix']['plan']; plan['model_Q']=.2+i
        a['consumes']['cem8']['rows'][i]['suffix_attempted']=5
        cr.append(dict(index=i,source_slot=key[0],condition=key[1],prefix_arm=key[2],status='suffix_sealed',
                       plan=deepcopy(plan),module_ledger=ledger(65),resources={k:dict(attempted=65,completed=65) for k in ('predict','action_encoder','pred_proj')}))
        for j in range(3):
            ordinal=3*i+j
            p=a['consumes']['spectral24']['rows'][ordinal]['selected_suffix']['plan']; p.update(winner_index=2,predicted_Q=.5+ordinal)
            a['consumes']['spectral24']['rows'][ordinal]['suffix_attempted']=5
            sr.append(dict(index=ordinal,status='sealed',result=dict(winner=dict(index=2,Q=.5+ordinal))))
            sl.extend(ledger(13,'predictor',ordinal=ordinal))
            ops.extend(dict(case=ordinal,kind=kind,completed=True) for kind in ('jacobian','svd','svd'))
    c=dict(schema=m.CEM+'.report',rows=cr,errors=[],resources={k:dict(attempted=520,completed=520) for k in ('predict','action_encoder','pred_proj')},
           contract=dict(jacobian=0,VJP=0,SVD=0,AD=0,shared_resource_axes=dict(teacher8=dict(predict=104),all24=dict(predict=312))))
    s=dict(schema=m.TEACHER+'.report',rows=sr,ledger=sl,operations=ops,errors=[],predict_API=312,wall_seconds=23.5,
           contract=dict(jacobians=24,mathematical_VJP_directions=4608))
    return a,c,s


def test_fixed32_long_rows_and16_nonadditive_budgets():
    a,c,s=fixture(); before=deepcopy((a,c,s)); out=m.project(a,c,s)
    assert (a,c,s)==before
    assert len(out['rows'])==32 and len(out['resource_rows'])==16
    assert {(r['anchor_index'],r['method'],r['endpoint_kind']) for r in out['rows']}=={
        (i,method,endpoint) for i in range(8) for method in m.METHODS for endpoint in m.ENDPOINTS}
    for r in out['rows']:
        v=a['rows'][r['anchor_index']][r['method']][r['endpoint_kind']]
        assert all(r[k]==v[k] for k in m.METRICS)
        assert r['native_success'] is v['native']['native_success'] and r['actual_suffix_returned_steps']==5
        assert r['model_budget_axes']['predict_API_calls']['completed']==(65 if r['method']=='cem' else 13)
        assert r['model_budget_axes']['predictor_forward_samples']['completed']==(65 if r['method']=='cem' else 13)
    assert out['resource_rows'][0]['axes']['jacobian_calls']==0
    assert out['resource_rows'][1]['axes']['jacobian_calls']==1
    assert out['resource_rows'][1]['axes']['planning_SVD_calls']==2
    assert out['resource_rows'][1]['axes']['mathematical_VJP_directions']==192
    assert out['resource_rows'][1]['axes']['backward_engine_API_calls'] is None
    assert out['resource_groups']['teacher8']['add_to_parent'] is False
    assert out['resource_groups']['inherited_shared_once']['total_historical_cost'] is None
    assert not out['equal_total_compute'] and not out['equal_FLOPs'] and not out['metric_recomputed_here']


def test_budget_API_samples_batch_dimension_and_unknowns():
    entries=ledger(3,batch=4); x=m.ledger_axes(entries,'predict')
    assert x['predict_API_calls']==dict(attempted=3,completed=3)
    assert x['predictor_forward_samples']==dict(attempted=12,completed=12)
    entries[4]['completed']=False; x=m.ledger_axes(entries,'predict')
    assert x['predict_API_calls']==dict(attempted=3,completed=2)
    assert x['predictor_forward_samples']==dict(attempted=12,completed=8)
    a,c,s=fixture(); del c['contract']['SVD']; out=m.project(a,c,s)
    assert out['resource_rows'][0]['axes']['planning_SVD_calls'] is None
    assert out['resource_groups']['cem8']['wall_seconds'] is None
    assert out['resource_groups']['factorial_all24']['wall_seconds']==23.5


@pytest.mark.parametrize('fault',['schema','status','denominator','order','teacher_ordinal','task','selection','flags','observed4','teacher_variant',
                                'CEM_plan','teacher_Q','suffix_count','early_padding','identity_fill','batch','resource_count','partial','bool_count'])
def test_projection_failclosed(fault):
    a,c,s=fixture(); p=a['rows'][0]; saved=a['consumes']['cem8']['rows'][0]
    if fault=='schema':a['schema']='other.analysis'
    elif fault=='status':a['status']='raw_success'
    elif fault=='denominator':a['fixed_denominator']=4
    elif fault=='order':a['rows'].reverse()
    elif fault=='teacher_ordinal':p['spectral_ordinal']=0
    elif fault=='task':p['task_id']=h('wrong')
    elif fault=='selection':p['selection_digest']=h('wrong')
    elif fault=='flags':a['equal_total_compute']=True
    elif fault=='observed4':a['consumes']['cem8']['rows']=a['consumes']['cem8']['rows'][:4]
    elif fault=='teacher_variant':a['consumes']['spectral24']['rows'][1]['variant']='open_full'
    elif fault=='CEM_plan':c['rows'][0]['plan']['model_Q']=99
    elif fault=='teacher_Q':s['rows'][1]['result']['winner']['Q']=99
    elif fault=='suffix_count':p['cem']['returned_suffix_steps']=4
    elif fault=='early_padding':
        saved['responses']=saved['responses'][:3]; p['cem']['returned_suffix_steps']=3
    elif fault=='identity_fill':p['identity_unknown']=['goal']
    elif fault=='batch':c['rows'][0]['module_ledger'][1]['inputs'][0]['shape'][0]=True
    elif fault=='resource_count':c['rows'][0]['resources']['predict']['completed']=64
    elif fault=='partial':s['errors']=['failure']
    else:p['cem']['returned_suffix_steps']=True
    with pytest.raises((ValueError,KeyError)):m.project(a,c,s)


def test_earlystop_and_engineering_failure_unknown_preserved():
    a,c,s=fixture(); pair=a['rows'][0]; v=pair['cem']; saved=a['consumes']['cem8']['rows'][0]
    v.update(status='native_stopped',returned_suffix_steps=3,horizon10=None,missing_checkpoints=[10]); saved.update(status='native_stopped',suffix_attempted=3)
    saved['responses']=saved['responses'][:3]
    out=m.project(a,c,s); endpoint,h10=out['rows'][:2]
    assert endpoint['actual_suffix_returned_steps']==3 and endpoint['metric_available']
    assert h10['native_success'] is h10['native7_distance'] is None and not h10['metric_available']
    v.update(status='engineering_failed',endpoint=None,first_error=dict(message='actual failure'));saved['status']='engineering_failed'
    out=m.project(a,c,s)
    assert out['rows'][0]['native_success'] is None and out['rows'][0]['first_error']['message']=='actual failure'
    assert len(out['rows'])==32


def test_csv_roundtrip_distinguishes_false_null_and_repeated_budget():
    a,c,s=fixture(); a['rows'][0]['cem']['horizon10']=None
    result=m.project(a,c,s); rows=list(csv.DictReader(io.StringIO(m.csv_bytes(result).decode())))
    assert len(rows)==32 and rows[0]['native_success']=='false' and rows[1]['native_success']=='null'
    assert json.loads(rows[0]['model_budget_axes'])['predict_API_calls']['completed']==65
    assert rows[0]['budget_repeated_across_endpoint_kinds']=='true'


def test_load_full_actual_hash_inventory_and_no_analyzer_invocation(monkeypatch):
    a,c,s=fixture(); root=Path(m.__file__).resolve().parents[2]; analyzer=root/'experiments/spectral_reachability'/(m.ANALYZER+'.py')
    a['input_sha256']={str(analyzer):h('analyzer'),'/RAM/cem.json':h('c'),'/RAM/teacher.json':h('s'),'/RAM/all-input.npy':h('input')}
    a['input_directories']=['/RAM/cem_output','/RAM/teacher_output']
    a['consumes']['cem8']['compiler_report_sha256']=h('c');a['consumes']['spectral24']['compiler_report_sha256']=h('s')
    calls=[]
    def read(path,expected):
        calls.append((path,expected)); return {'/RAM/analysis.json':a,'/RAM/cem.json':c,'/RAM/teacher.json':s}[str(path)]
    monkeypatch.setattr(m,'read_bound',read)
    monkeypatch.setattr(m,'sha',lambda p:a['input_sha256'].get(str(p),h('analysis') if str(p)=='/RAM/analysis.json' else h('own')))
    out=m.load('/RAM/analysis.json',h('analysis'),h('analyzer'),'/RAM/cem.json',h('c'),'/RAM/teacher.json',h('s'))
    assert len(calls)==3 and '/RAM/all-input.npy' in out['input_sha256']
    monkeypatch.setattr(m,'sha',lambda p:h('wrong'))
    with pytest.raises(ValueError):m.load('/RAM/analysis.json',h('analysis'),h('analyzer'),'/RAM/cem.json',h('c'),'/RAM/teacher.json',h('s'))


def test_read_bound_duplicate_nonfinite_and_external_sha(tmp_path):
    for body in (b'{"x":1,"x":2}',b'{"x":NaN}'):
        path=tmp_path/'input.json';path.write_bytes(body)
        with pytest.raises(ValueError):m.read_bound(str(path),hashlib.sha256(body).hexdigest())
    path=tmp_path/'input.json';path.write_bytes(b'{"x":1}')
    with pytest.raises(ValueError):m.read_bound(str(path),h('wrong'))
    with pytest.raises(ValueError):m.read_bound(str(path),'0'*64)


def test_exclusive_write_cap_floor_inputs_and_no_overwrite(tmp_path,monkeypatch):
    path=tmp_path/(m.NAME+'_RAM.json'); result=dict(input_sha256={},input_directories=[])
    body=m.raw(dict(rows=[])); assert m.write_new(str(path),body,result)==hashlib.sha256(body).hexdigest()
    with pytest.raises(ValueError):m.write_new(str(path),body,result)
    path=tmp_path/(m.NAME+'_blocked.json'); result['input_directories']=[str(tmp_path)]
    with pytest.raises(ValueError):m.write_new(str(path),body,result)
    result['input_directories']=[];monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR-1))
    with pytest.raises(ValueError):m.write_new(str(path),body,result)
    assert not path.exists()


def test_CLI_help_no_load(capsys):
    with pytest.raises(SystemExit) as exc:m.main(['--help'])
    assert exc.value.code==0 and '--analysis-sha' in capsys.readouterr().out
