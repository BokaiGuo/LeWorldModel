"""Synthetic NumPy callbacks only. No neural, production prepare or fullstrict."""
from copy import deepcopy
import runpy
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_actual_cp5_r_delivery_20261029_v1 as m

f=runpy.run_path(str(m.ROOT/'tests/test_agent_p3_r_native_orchestrator_20261029_v1.py'))


def material():
    tasks=[];pro=[]
    for root in range(2):
        original=f['r_task']()
        if root:
            original['goal']=[0.]*191+[1.];original['binding']['goal_sha256']=m.digest(original['goal'])
            original['binding']['root_sha256']=m.digest('RAM second known root')
        warm=np.full((10,2),np.float32(.0625),np.float32);prefix=np.full((5,2),np.float32(.125),np.float32)
        original['root_context']['action_history']=m.resolver.normalize(original,warm).tolist()
        original['binding']['context_sha256']=m.digest(original['root_context'])
        ctx=deepcopy(original['root_context']);raw=np.concatenate((warm[-5:],prefix))
        ctx['action_history']=m.resolver.normalize(original,raw).tolist()
        task=m.resolver.local_task(original,ctx,dict(source_slot=root,control_boundary=5,fixture=True),m.resolver.NAME)
        p=dict(source_slot=root,original_task=original,original_task_sha256=m.digest(original),
            original_observed_absolute_steps=[0,5,10],original_control_steps=[-10,-5,0],
            actual_cp5_absolute_steps=[5,10,15],actual_cp5_control_steps=[-5,0,5],raw10=raw.tolist(),prefix5=prefix.tolist(),
            branch_witnesses=[dict(episode_id=f's{root}_{arm}') for arm in m.resolver.ARMS],
            local_task_sha256=m.digest(task))
        tasks.append(task);pro.append(p)
    return dict(tasks=tasks,provenance=pro,profile=m.profile(),sources={'/RAM/fixture':'a'*64})


class RAM(f['NonlinearRAM']):
    def replay(self,context,actions,segmented,ledger):
        if segmented:
            # Same bounded schedule is used by the HOST class. The injected
            # native-buffer dynamics remains pure NumPy, never a torch model.
            schedule=m.segmented_packets(actions)
            assert [i for i,_,reset in schedule if reset]==([1] if len(actions)==15 else [])
        return super().replay(context,actions,segmented,ledger)


@pytest.fixture(scope='module')
def saved():
    p=material();s=f['Memory']();rows=m.execute(p,lambda t,s:RAM(t,s),s,permit_ram=True)
    return p,rows,s


def events_for(s,slot):return [e['payload'] for e in s.events if e['kind']=='group_event' and e['slot']==slot]


@pytest.mark.parametrize('h,n',[(1,40),(3,66)])
def test_complete_27_axes_and_no_witness_costs(h,n):
    c=m.caps(h)
    assert set(c)==set(m.core.AXES) and c['predict_API']==c['predict_samples']==n
    assert c['jacobian_API']==10 and c['VJP_directions']==1920 and c['replay_calls']==20
    assert c['pred_proj_rows']==c['predict_context_tokens']==n*3
    assert c['solve_calls']==9 and c['array_SVD_calls']==10 and c['program_compile_calls']==10
    assert c['metric_calls']==18 and c['schedule_decisions']==0
    assert m.base.caps(h)['predict_API']==n-2*h and m.base.caps(h)['replay_calls']==18
    assert m.parent.default_caps('adaptive')['replay_calls']==2
    assert all(c[k]==0 for k in ('encoder_calls','encoder_samples','projector_calls','projector_samples'))


def test_full_four_slots_delivery_and_no_witness(saved):
    p,rows,s=saved
    assert [x['status'] for x in rows]==['model_delivery_verified']*2+['no_witness']*2
    for row in rows:
        r=row['result'];assert not r['errors'];g=r['R_groups'][0]
        assert len(g['rows'])==8 and g['control']['status']=='complete'
        assert r['eligible_order']==sorted(g['eligible_indices'],key=lambda i:(g['rows'][i]['immediate'],i))
        assert r['actual_cp5_start'] is True and r['root_context_role']==m.ROLE
        if row['root']==0:
            d=r['delivery'];assert d['candidate_index']==r['eligible_order'][0]
            assert d['task']==p['tasks'][0] and d['base_task']==p['provenance'][0]['original_task']
            assert d['raw_edge']==g['rows'][d['candidate_index']]['actions']
            assert d['commit_raw5']==d['raw_edge'][:5] and d['lookahead_raw']==d['raw_edge'][5:]
            assert len(d['raw_edge'])==row['horizon']*5 and len(d['commit_raw5'])==5
            assert d['raw_edge_float32_sha256']==m.raw_hash(d['raw_edge'])
            assert d['model_delivery_verified'] and d['physical_release'] is False and d['execution_authorized'] is False
            assert all(v['status']=='returned' and v['path']==d['path'] for v in r['verification'])
            assert r['verification'][1]['segments']==([5] if row['horizon']==1 else [5,10])
            assert len(r['native_calls'])==30 and r['resource_receipt']['attempted']==m.caps(row['horizon'])
        else:
            assert r['delivery'] is None and r['model_delivery_verified'] is False and r['eligible_order']==[]
            assert all(v['status']=='not_requested' for v in r['verification'])
            assert len(r['native_calls'])==28 and r['resource_receipt']['attempted']==m.base.caps(row['horizon'])
    assert len(s.capsules)==4


def test_saved_all4_native_and_delivery_algebra(saved):
    p,rows,s=saved
    out=m.validate_rows(p,rows,s.events,permit_fixture=True,capsule_reader=s.reader)
    assert len(out)==4 and all(x['complete'] for x in out)
    assert [(x['slot'],x['root'],x['horizon']) for x in out]==[(0,0,1),(1,0,3),(2,1,1),(3,1,3)]
    assert [x['model_delivery_verified'] for x in out]==[True,True,False,False]
    assert [x['AD_primal_predict_API'] for x in out]==[10,12,10,12]


def test_selection_is_immediate_then_index_not_reach():
    g=dict(rows=[dict(index=i,eligible=i in (1,3,5),immediate={1:4.,3:2.,5:2.}.get(i,0.),reach=-i) for i in range(8)],eligible_indices=[1,3,5])
    assert m.ranked(g)==[3,5,1]


@pytest.mark.parametrize('damage',['order','bool','cohort'])
def test_material_rejects_rebased_root_relabelling(damage):
    p=material();prior=dict(tasks=[deepcopy(x['original_task']) for x in p['provenance']]);cp=deepcopy(p)
    if damage=='order':cp['tasks'].reverse();cp['provenance'].reverse()
    if damage=='bool':cp['provenance'][0]['source_slot']=False
    if damage=='cohort':prior['tasks'][0]['binding']['root_sha256']=m.digest('different root')
    with pytest.raises(ValueError):m.material(prior,cp)


def test_h3_explicit_5_plus10_schedule_and_native_code_uses_it():
    raw=np.arange(30,dtype=np.float32).reshape(15,2)/32
    schedule=m.segmented_packets(raw.tolist())
    assert [reset for _,_,reset in schedule]==[False,True,False]
    assert np.array_equal(np.concatenate([p for _,p,_ in schedule]),raw)
    assert [r for _,_,r in m.segmented_packets(raw[:5].tolist())]==[False]
    import inspect
    code=inspect.getsource(m.NativeDelivery.replay)
    assert 'segmented_packets(actions)' in code and 'if reset:' in code


@pytest.mark.parametrize('damage',['raw10','older','prefix','time','goal','normalizer','base','local','branches'])
def test_input_provenance_exact_join_refuses(damage):
    p=material();t=deepcopy(p['tasks'][0]);pro=deepcopy(p['provenance'][0])
    if damage=='raw10':pro['raw10'][0][0]+=.25
    if damage=='older':t['root_context']['latent_history'][0][0]=.25;t['binding']['context_sha256']=m.digest(t['root_context']);pro['local_task_sha256']=m.digest(t)
    if damage=='prefix':pro['prefix5'][0][0]+=.25
    if damage=='time':pro['actual_cp5_absolute_steps']=[0,5,10]
    if damage=='goal':t['goal'][0]+=.25;t['binding']['goal_sha256']=m.digest(t['goal']);pro['local_task_sha256']=m.digest(t)
    if damage=='normalizer':t['action_std'][0]*=2;t['binding']['normalizer_sha256']=m.digest([t['action_mean'],t['action_std']]);pro['local_task_sha256']=m.digest(t)
    if damage=='base':pro['original_task_sha256']='0'*64
    if damage=='local':pro['local_task_sha256']='0'*64
    if damage=='branches':pro['branch_witnesses'].pop()
    with pytest.raises(ValueError):m.input_pair(t,pro)


@pytest.mark.parametrize('damage',['rank','candidate','raw','commit','lookahead','suffix','path','segments','J','zero','resource','role','authority'])
def test_saved_delivery_mutations_refused(saved,damage):
    p,rows,s=saved;r=deepcopy(rows[1]['result']);d=r['delivery'];g=r['R_groups'][0]
    if damage=='rank':r['eligible_order']=list(reversed(r['eligible_order']))
    if damage=='candidate':d['candidate_index']=(d['candidate_index']+1)%8
    if damage=='raw':d['raw_edge'][0][0]+=.125
    if damage=='commit':d['commit_raw5'][0][0]+=.125
    if damage=='lookahead':d['lookahead_raw'][0][0]+=.125
    if damage=='suffix':d['raw_edge']+=g['rows'][0]['suffix_actions']
    if damage=='path':r['verification'][1]['path'][0]['latent_history'][0][0]+=.125
    if damage=='segments':r['verification'][1]['segments']=[5,5,5]
    if damage=='J':g['rows'][0]['signal']['jacobian'][0][0]+=.125
    if damage=='zero':g['control']['status']='not_requested'
    if damage=='resource':r['resource_receipt']['caps']['replay_calls']=18
    if damage=='role':r['actual_cp5_start']=False
    if damage=='authority':d['source_closure_sha256']='0'*64
    with pytest.raises(ValueError):m.validate_group(p['tasks'][0],p['provenance'][0],3,r,events_for(s,1),source_sha=m.digest(p['sources']),
        permit_fixture=True,capsule_reader=lambda cid,call:s.reader(1,cid,call))


@pytest.mark.parametrize('damage',['unknown','selection','duplicate','orphan'])
def test_event_census_rejects_without_model(saved,damage):
    p,rows,s=saved;es=deepcopy(events_for(s,0))
    if damage=='unknown':es.append(dict(kind='io',call=None,payload=dict(kind='reward')))
    if damage=='selection':next(e['payload'] for e in es if e['payload']['kind']=='delivery_selection')['eligible_order']=[]
    if damage=='duplicate':es.append(deepcopy(next(e for e in es if e['payload']['kind']=='delivery_selection')))
    if damage=='orphan':es[0]['call']=999
    with pytest.raises(ValueError):m.validate_group(p['tasks'][0],p['provenance'][0],1,rows[0]['result'],es,source_sha=m.digest(p['sources']),
        permit_fixture=True,capsule_reader=lambda cid,call:s.reader(0,cid,call))


def test_replay_failure_keeps_cost_and_blocks_later_slots():
    class Fail(RAM):
        def replay(self,*args):raise RuntimeError('RAM verification failure')
    p=material();s=f['Memory']();counter=[0]
    def factory(t,s):
        counter[0]+=1;return (Fail if counter[0]==30 else RAM)(t,s)
    rows=m.execute(p,factory,s,permit_ram=True);r=rows[0]['result']
    assert r['status']=='partial_runtime_failure' and r['delivery'] is None
    assert [v['status'] for v in r['verification']]==['returned','failed']
    assert r['resource_receipt']['attempted']['replay_calls']==20 and r['resource_receipt']['completed']['replay_calls']==19
    assert all(x['status']=='blocked_after_runtime_failure' for x in rows[1:])
    out=m.validate_rows(p,rows,s.events,permit_fixture=True,capsule_reader=s.reader)
    assert len(out)==1 and not out[0]['complete']


def test_frozen_pins_flags_cli_and_worker_authorization():
    assert all(m.sha(p)==h for p,h in m.PINS.items()) and m.sha(m.PROTOCOL)==m.PROTOCOL_SHA
    assert m.CP5_SHA=='ce6670d84f6f0f712bdfe5e1b7471be65b0441338c18010f79b3967ef28397c7'
    with pytest.raises(ValueError,match='MAIN source audit'):m.run('/not/read','bad','/not/written',authorization='worker')
    code='''
import sys
from experiments.spectral_reachability import agent_p3_actual_cp5_r_delivery_20261029_v1 as m
assert 'torch' not in sys.modules and 'gymnasium' not in sys.modules
assert not m.rules()['physical_release'] and not m.rules()['execution_authorized']
'''
    subprocess.run([sys.executable,'-c',code],check=True)
    subprocess.run([sys.executable,'-m',m.__name__,'--help'],check=True,capture_output=True)
