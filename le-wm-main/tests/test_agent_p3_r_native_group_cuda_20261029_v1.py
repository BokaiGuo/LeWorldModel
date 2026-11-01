"""Explicit NumPy/native-buffer fixtures only; no production prepare/model IO."""
from copy import deepcopy
import runpy
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p3_r_native_group_cuda_20261029_v1 as m

f=runpy.run_path(str(m.ROOT/'tests/test_agent_p3_r_native_orchestrator_20261029_v1.py'))


@pytest.fixture(scope='module')
def saved():
    t=f['r_task']();u=deepcopy(t)
    u['goal']=[0.]*191+[1.];u['binding']['goal_sha256']=m.digest(u['goal'])
    p=dict(tasks=[t,u],profile=m.profile());s=f['Memory']()
    rows=m.execute(p,lambda t,s:f['NonlinearRAM'](t,s),s,permit_ram=True)
    return p,rows,s


def events_for(s,slot):return [e['payload'] for e in s.events if e['kind']=='group_event' and e['slot']==slot]


@pytest.mark.parametrize('h,predict',[(1,38),(3,60)])
def test_static_full_signal_bank9_exact_caps(h,predict):
    c=m.caps(h)
    assert set(c)==set(m.core.AXES) and c['predict_API']==c['predict_samples']==predict
    assert c['predict_context_tokens']==3*predict and c['pred_proj_rows']==3*predict
    assert c['jacobian_API']==10 and c['signal_compile_calls']==10
    assert c['VJP_directions']==c['grad_API']==c['raw_backward_callbacks']==1920
    assert c['array_SVD_calls']==10 and c['solve_calls']==9 and c['program_compile_calls']==10
    assert c['replay_calls']==18 and c['metric_calls']==18 and c['schedule_decisions']==0
    assert all(c[k]==0 for k in ('encoder_calls','encoder_samples','projector_calls','projector_samples'))
    assert m.tree.parent.default_caps('adaptive')['replay_calls']==2


def test_all_four_full9_and_no_plan_or_beam_claim(saved):
    p,rows,s=saved
    assert [(r['root'],r['horizon']) for r in rows]==[(0,1),(0,3),(1,1),(1,3)]
    for row in rows:
        r=row['result'];assert row['status']=='complete_native_R_group_not_admitted',r['errors']
        assert len(r['native_calls'])==28 and len(r['signal_mappings'])==1
        assert r['resource_receipt']['attempted']==r['resource_receipt']['completed']==m.caps(row['horizon'])
        g=r['R_groups'][0]
        assert len(g['rows'])==8 and g['control']['status']=='complete'
        assert all(x['status']=='complete' and len(x['actions'])==5*row['horizon'] and len(x['suffix_actions'])==5 for x in g['rows'])
        assert g['resources']==m.tree.group_cost(row['horizon'])
        assert g['bank']['resources']==m.tree.spectral.bank_cost()
        assert g['eligible_indices']==[x['index'] for x in g['rows'] if x['immediate']>g['control']['immediate'] and x['reach']<g['control']['reach']]
        assert r['actual_cp5_start'] is False and r['root_context_role']==m.ROLE
        assert r['model_plan_available'] is False and 'model_plan' not in r
    assert rows[0]['result']['R_groups'][0]['eligible_indices']
    assert all(r['result']['R_groups'][0]['status']=='no_witness' for r in rows[2:])
    assert len(s.capsules)==4


def test_complete_four_group_saved_native_algebra(saved):
    p,rows,s=saved
    out=m.validate_rows(p,rows,s.events,permit_fixture=True,capsule_reader=s.reader)
    assert len(out)==4 and all(x['complete'] for x in out)
    assert [x['AD_primal_predict_API'] for x in out]==[10,12,10,12]
    assert [x['Jacobian_elements'] for x in out]==[19200,23040,19200,23040]
    assert sum(x['resources']['attempted']['predict_API'] for x in out)==196


@pytest.mark.parametrize('damage',['J','raw','suffix','eligible','winneronly','zero','cost','normalizer','resource','signal','lateh','contextrole'])
def test_group_mutations_refuse(saved,damage):
    p,rows,s=saved;r=deepcopy(rows[0]['result']);g=r['R_groups'][0]
    if damage=='J':g['rows'][0]['signal']['jacobian'][0][0]+=.125
    if damage=='raw':g['rows'][0]['actions'][0][0]+=.125
    if damage=='suffix':g['rows'][0]['suffix_actions'][0][0]+=.125
    if damage=='eligible':g['rows'][0]['eligible']=not g['rows'][0]['eligible']
    if damage=='winneronly':g['eligible_indices']=[g['diagnostic_reach_winner']]
    if damage=='zero':g['control']['status']='not_requested'
    if damage=='cost':g['rows'][0]['immediate']+=1.
    if damage=='normalizer':r['signal_mappings'][0]['parent_task']['action_std'][0]*=2
    if damage=='resource':r['resource_receipt']['caps']['replay_calls']=2
    if damage=='signal':r['parent_signal']['nominal'][0]+=.25
    if damage=='lateh':r['horizon']=3
    if damage=='contextrole':r['actual_cp5_start']=True
    with pytest.raises(ValueError):m.validate_group(p['tasks'][0],1,r,events_for(s,0),permit_fixture=True,
        capsule_reader=lambda cid,call:s.reader(0,cid,call))


@pytest.mark.parametrize('damage',['unknown','orphan','missing','callid'])
def test_native_event_mutations_refused(saved,damage):
    p,rows,s=saved;es=deepcopy(events_for(s,0))
    if damage=='unknown':es.append(dict(kind='io',call=None,payload=dict(kind='unknown')))
    if damage=='orphan':es[0]['call']=999
    if damage=='missing':es.pop(next(i for i,e in enumerate(es) if e['payload']['kind']=='direction'))
    if damage=='callid':es[0]['call']=True
    with pytest.raises(ValueError):m.validate_group(p['tasks'][0],1,rows[0]['result'],es,permit_fixture=True,
        capsule_reader=lambda cid,call:s.reader(0,cid,call))


@pytest.mark.parametrize('damage',['missing','reorder','orphan','extra','authority'])
def test_four_denominator_dispatch_mutations(saved,damage):
    p,rows,s=saved;rs=deepcopy(rows);es=deepcopy(s.events)
    if damage=='missing':rs.pop()
    if damage=='reorder':rs[0],rs[1]=rs[1],rs[0]
    if damage=='orphan':es.append(dict(kind='group_event',slot=4,payload=dict(kind='io')))
    if damage=='extra':es.append(dict(kind='group_attempt',identity=m.roster()[0]))
    if damage=='authority':rs[0]['result']['root_task_sha256']='0'*64
    with pytest.raises(ValueError):m.validate_rows(p,rs,es,permit_fixture=True,capsule_reader=s.reader)


def test_native_failure_no_refund_or_retry_all_four():
    p=dict(tasks=[f['r_task']()]*2,profile=m.profile());s=f['Memory']()
    rows=m.execute(p,lambda t,s:f['fixtures']['NumpyNative'](t,s,fail=True),s,permit_ram=True)
    assert rows[0]['result']['status']=='partial_runtime_failure'
    assert rows[0]['result']['resource_receipt']['attempted']['predict_API']==1
    assert rows[0]['result']['resource_receipt']['completed']['predict_API']==0
    assert all(x['status']=='blocked_after_runtime_failure' for x in rows[1:])
    out=m.validate_rows(p,rows,s.events,permit_fixture=True,capsule_reader=s.reader)
    assert len(out)==1 and not out[0]['complete']


def test_failure_mid_R_retains_zero_and_all8_partial_slots():
    class Failing(f['NonlinearRAM']):
        def call(self,name,fn,inputs,shape=None):
            if name=='predictor':
                def fail():raise RuntimeError('RAM edge0 predictor failure')
                return f['fixtures']['NumpyNative'].call(self,name,fail,inputs,shape)
            return super().call(name,fn,inputs,shape)
    count=[0]
    def factory(t,s):
        count[0]+=1
        return (Failing if count[0]==5 else f['NonlinearRAM'])(t,s)
    t=f['r_task']();s=f['Memory']()
    r=m.group(t,1,factory,s,permit_ram=True,capsule=lambda cid,call:s.capsule(0,cid,call))
    assert r['status']=='partial_runtime_failure' and len(r['R_groups'])==1
    g=r['R_groups'][0]
    assert g['control']['status']=='complete' and g['rows'][0]['status']=='failed'
    assert all(x['status']=='not_requested' for x in g['rows'][1:])
    assert r['resource_receipt']['attempted']['replay_calls']==3
    assert r['resource_receipt']['completed']['replay_calls']==2
    out=m.validate_group(t,1,r,s.events,permit_fixture=True,capsule_reader=lambda cid,call:s.reader(0,cid,call))
    assert not out['complete']


def test_actual_capsule_exclusive_fs_bytes(tmp_path):
    path=tmp_path/'new';s=m.Store(path,m.profile());call=[dict(literal='only RAM data')]
    result=s.capsule(0,0,call)
    assert m.sha(result['path'])==result['sha256']==m.digest(call)
    assert m.signal.read(result['path'])==call
    with pytest.raises(ValueError,match='exclusive'):s.capsule(0,0,call)
    s.close()


def test_explicit_RAM_permission_only(saved):
    p,rows,s=saved
    with pytest.raises(ValueError,match='RAM production'):
        m.validate_group(p['tasks'][0],1,rows[0]['result'],events_for(s,0),capsule_reader=lambda cid,call:s.reader(0,cid,call))


def test_frozen_sources_protocol_and_separate_engineering_profile():
    assert all(m.sha(p)==h for p,h in m.TREE_PINS.items())
    assert m.sha(m.PROTOCOL)==m.PROTOCOL_SHA
    assert m.profile()['output_bytes']==2*1024**3 and m.profile()['wall_seconds']==1800
    assert m.rules()['final_replay_calls']==0 and m.rules()['plan_release'] is False
    assert m.rules()['actual_cp5_start'] is False


def test_auth_cold_import_cli_no_production():
    with pytest.raises(ValueError,match='MAIN source audit'):m.run('/not/read','bad','/not/written',authorization='worker')
    code='''
import sys
from experiments.spectral_reachability import agent_p3_r_native_group_cuda_20261029_v1 as m
assert 'torch' not in sys.modules and 'gymnasium' not in sys.modules
assert not m.rules()['physical_release']
'''
    subprocess.run([sys.executable,'-c',code],check=True)
    subprocess.run([sys.executable,'-m',m.__name__,'--help'],check=True,capture_output=True)
