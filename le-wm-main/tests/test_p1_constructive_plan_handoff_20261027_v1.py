"""RAM-only metadata fixtures; no tmp, neural or physics execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import pytest
from experiments.spectral_reachability import p1_constructive_plan_handoff_20261027_v1 as m
from experiments.spectral_reachability import p1_constructive_plan_physical_20261027_v1 as physical


def fixture():
    exports={};plans=[]
    for i,blind in enumerate(m.planner.encode.APPROVED_INPUTS.values()):
        path=f'/MEMORY/task_{i}.json.gz'
        exports[path]=dict(source_slot=i,blind_source_sha256=blind,blind_sha256=blind,task_id=f't{i}',
            task_contract=dict(blind_source_sha256=blind))
        for arm in m.planner.ARMS:
            plans.append(dict(schema=m.planner.NAME+'.model_plan',order=i,arm=arm,commands=[[.25,-.25]]*10,
                with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,
                whole_run_report_sha256='a'*64,task_export=path,task_sha256='b'*64,task_id=f't{i}',selection_sha256='c'*64))
    consumed=dict(status='model_commands_verified_for_separate_MAIN_executor',new_physics=0,P1_admission=False,plans=plans)
    return exports,consumed


def build(exports,consumed):
    def sha(path):return 'a'*64 if Path(path).name=='report.json' else 'b'*64
    with patch.object(m,'closure',return_value={'bridge':'f'*64}),patch.object(m,'sha',sha),patch.object(m.planner,'consume',return_value=consumed) as consume,patch.object(m.planner,'read',side_effect=lambda p,*a:deepcopy(exports[str(p)])),patch.object(m.planner,'validate_export_envelope',lambda _:None):
        result=m.build('/MEMORY/planner','a'*64)
        consume.assert_called_once_with(Path('/MEMORY/planner'),'a'*64)
        return result


def test_exact_join_keeps_own_commands_and_not_native_metadata():
    exports,c=fixture();h=build(exports,c)
    assert h['schema']==m.NAME+'.handoff' and h['errors']==[] and len(h['rows'])==4
    assert [(x['source_slot'],x['arm']) for x in h['rows']]==[(i,a) for i in range(2) for a in m.planner.ARMS]
    assert all(x['commands']==p['commands'] for x,p in zip(h['rows'],c['plans']))
    assert not h['certificate_read'] and not h['sourcepack_read']


@pytest.mark.parametrize('change',['slot','blind','alias','taskid','missing','order','arm','report','physical','precision','duplicate','taskcrossarm','release'])
def test_bad_joins_and_partial_release_fail_closed(change):
    e,c=fixture();p=c['plans'][0];x=e[p['task_export']]
    if change=='slot':x['source_slot']=False
    elif change=='blind':x['blind_source_sha256']='f'*64
    elif change=='alias':x['blind_sha256']='f'*64
    elif change=='taskid':x['task_id']='other'
    elif change=='missing':c['plans'].pop()
    elif change=='order':p['order']=False
    elif change=='arm':p['arm']='other'
    elif change=='report':p['whole_run_report_sha256']='f'*64
    elif change=='physical':p['selection_uses_physics']=True
    elif change=='precision':p['commands']=[[.1,.1]]*10
    elif change=='duplicate':
        for p2 in c['plans'][2:]:p2.update(task_export=p['task_export'],task_id=p['task_id'])
    elif change=='taskcrossarm':c['plans'][1].update(task_export=c['plans'][2]['task_export'],task_id='t1')
    elif change=='release':c['status']='partial_or_failed'
    with pytest.raises(ValueError):build(e,c)


def evaluator():
    return dict(schema=m.NAME+'.evaluator_map',sourcepack='/NOT_OPENED/sourcepack.json',sourcepack_sha256='d'*64,
        source_prepared='/NOT_OPENED/source_prepare.json',source_prepare_sha256='e'*64,
        roots=[dict(root_slot=i,root_sha256=str(i)*64) for i in range(2)],wall_seconds=300)


def test_selection_matches_frozen_physical_contract_and_digest():
    e,c=fixture();h=build(e,c);ev=evaluator()
    with patch.object(m,'verify_handoff',return_value=h),patch.object(m,'external',side_effect=lambda p,h:Path(p)),patch.object(m,'read',return_value=ev),patch.object(m,'sha',side_effect=lambda p:'b'*64 if str(p).endswith('map.json') else 'a'*64):
        selected=m.selection('/MEMORY/handoff.json','a'*64,'/MEMORY/map.json','b'*64)
    data=dict(references=[dict(root_sha256=r['root_sha256'],blind_sha256=list(m.planner.encode.APPROVED_INPUTS.values())[i]) for i,r in enumerate(ev['roots'])])
    with patch.object(physical,'sha',return_value='a'*64),patch.object(physical,'read',return_value=h):plans,_=physical.selection(selected,data)
    assert len(plans)==4
    assert all(p['selection_digest']==physical.source.digest({k:v for k,v in p.items() if k!='selection_digest'}) for p in plans)


@pytest.mark.parametrize('change',['boolslot','rootorder','extra','wall','relative'])
def test_evaluator_mapping_is_explicit_and_failclosed(change):
    e,c=fixture();h=build(e,c);ev=evaluator()
    if change=='boolslot':ev['roots'][0]['root_slot']=False
    elif change=='rootorder':ev['roots'].reverse()
    elif change=='extra':ev['native_outcome']='not permitted'
    elif change=='wall':ev['wall_seconds']=601
    elif change=='relative':ev['sourcepack']='relative.json'
    with patch.object(m,'verify_handoff',return_value=h),patch.object(m,'external',side_effect=lambda p,h:Path(p)),patch.object(m,'read',return_value=ev),pytest.raises(ValueError):m.selection('/MEMORY/handoff.json','a'*64,'/MEMORY/map.json','b'*64)


def test_bad_handoff_reconstruction_not_accepted():
    with patch.object(m,'external',return_value=Path('/MEMORY/handoff')),patch.object(m,'read',return_value=dict(planner_output='/MEMORY',planner_report_sha256='a'*64)),patch.object(m,'build',return_value={'wrong':True}),pytest.raises(ValueError):m.verify_handoff('/MEMORY/handoff','a'*64)


def test_no_neural_or_simulator_import():
    assert not any(n in sys.modules for n in ('torch','gymnasium','stable_worldmodel','pymunk'))


def test_exclusive_write_disk_gate_before_open():
    with patch.object(m.shutil,'disk_usage',return_value=NS(free=m.FLOOR-1)),pytest.raises(ValueError):m.write_new('/MEMORY_MUST_NOT_EXIST/file.json',{})
