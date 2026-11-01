"""Saved-array/PNG semantics and in-memory adversaries; no temp or result files."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_line_search_gn_physical_saved_audit_20261027_v1 as a


@pytest.fixture(scope='module')
def bundle():return a.load_bundle()


def event(b,kind,index=0):return [e for e in b['events'] if e['kind']==kind][index]


def rechain(b):
    """Avoid accepting hash failure as evidence that semantic checks work."""
    previous=None
    for i,e in enumerate(b['events']):
        e.update(index=i,previous_sha256=previous)
        e['event_sha256']=a.digest({k:v for k,v in e.items() if k!='event_sha256'})
        previous=e['event_sha256']


def test_actual_saved_bindings_census_and_independent_metrics(bundle):
    r=a.audit(bundle)
    assert r['all_checks'] and r['event_count']==282 and r['branches']==2 and r['PNG']==6
    assert r['counts']=={k:dict(attempted=n,completed=n) for k,n in dict(reset=4,step=20,render=22).items()}
    assert [x['random_reset_seed'] for x in r['roots']]==[202781011,202781014]
    assert [x['source_seed'] for x in r['roots']]==[202761010,202761011]
    assert [x['object_position_error_final'] for x in r['roots']]==[9.16707397108197,13.691347852133942]
    assert [x['wrapped_object_angle_error_final'] for x in r['roots']]==[3.017656615692022,1.7825133275760676]
    assert all(x['object_pose_stationary_all11'] and x['recorded_contact_steps']==0 for x in r['roots'])
    assert all(not x['native_final']['success'] and x['native_wrapped_success_disagreements']==0 for x in r['roots'])
    assert r['new_physical_steps']==r['new_neural_calls']==r['new_independent_samples']==0 and not r['P1_admission']


@pytest.mark.parametrize('defect',[
    'dispatch','raw_float32','native','wrapped','reward','goal_state','info_pose','state_return',
    'object_metric','reset_seed','placement_request','saved_poststate_reset','missing_render','duplicate_step',
    'reordered_step','guard_stale','PNG_pixels','PNG_state','resource_summary','partial_branch','winner',
    'Q','observed_history','source_recipe','selection_digest','final_binding','extra_event','close_receipt',
    'reset_authority','parent_raw_binding',
])
def test_semantic_tampering_rejected_even_with_recomputed_chain(bundle,defect):
    b=deepcopy(bundle);row=b['report']['rows'][0];plan=b['prepared']['plans'][0]
    if defect=='dispatch':row['trace'][0]['command'][0]+=.01
    elif defect=='raw_float32':plan['commands'][0][0]+=1e-12
    elif defect=='native':row['trace'][0]['native_official_evaluation']['state_distance']+=1.
    elif defect=='wrapped':row['trace'][0]['pose_diagnostics']['independent']['wrapped_angle_error']+=.1
    elif defect=='reward':row['trace'][0]['reward']+=.5
    elif defect=='goal_state':row['trace'][0]['official_info']['goal_state'][0]+=1.
    elif defect=='info_pose':row['trace'][0]['official_info']['block_pose'][0]+=1.
    elif defect=='state_return':row['trace'][0]['observation']['state'][0]+=1.
    elif defect=='object_metric':row['object_only']['final']['object_position_error']+=1.
    elif defect=='reset_seed':event(b,'reset_observation')['seed']+=1
    elif defect=='placement_request':event(b,'original_placement_request')['state'][0]+=1.
    elif defect=='saved_poststate_reset':event(b,'reset_observation',1)['options']['state']=b['packets'][0]['state'].tolist()
    elif defect=='missing_render':b['events'].remove(event(b,'render_returned'))
    elif defect=='duplicate_step':b['events'].insert(0,deepcopy(event(b,'step_attempted')))
    elif defect=='reordered_step':
        first=b['events'].index(event(b,'step_attempted'));second=b['events'].index(event(b,'step_returned'))
        b['events'][first],b['events'][second]=b['events'][second],b['events'][first]
    elif defect=='guard_stale':event(b,'rgb_guard_observed',1)['receipt']['observations'][-1]['primitive_step']=0
    elif defect=='PNG_pixels':b['images']['0_line_search_GN_5.png'][0,0,0]^=1
    elif defect=='PNG_state':row['checkpoints'][1]['state'][0]+=1.
    elif defect=='resource_summary':b['report']['counts']['render']['completed']=21
    elif defect=='partial_branch':row['full_horizon']=False
    elif defect=='winner':b['winners'][0]['final_prediction']['commands'][0][0]+=.01
    elif defect=='Q':b['tasks'][0]['projection'][0][0]=2.
    elif defect=='observed_history':b['observed'][0]['raw_action_history'][0,0]=.1
    elif defect=='source_recipe':b['source_pack']['plan']['recipe']['position_jitter']=17.
    elif defect=='selection_digest':plan['selection_sha256']='0'*64
    elif defect=='final_binding':b['expected_prepare_sha']='0'*64
    elif defect=='extra_event':b['events'].append(dict(kind='unrecognized',selection_sha256='0'*64))
    elif defect=='reset_authority':b['reset_audit']['all_reproduction_checks']=False
    elif defect=='parent_raw_binding':b['source_pack']['slots'][0]['raw_source_sha256']='0'*64
    else:event(b,'branch_closed')['receipt']['status']='failed'
    rechain(b)
    with pytest.raises(ValueError):a.audit(b,recheck_bindings=False)


def test_hashchain_independently_rejects_torn_or_changed_event(bundle):
    b=deepcopy(bundle);event(b,'step_attempted')['receipt']['command'][0]+=.01
    with pytest.raises(ValueError,match='hashchain event content'):a.audit(b,recheck_bindings=False)


def test_wrong_external_report_SHA_stops_before_other_inputs():
    with pytest.raises(ValueError,match='SHA mismatch'):a.load_bundle(expected='0'*64)


def test_postread_TOCTOU_rejected(bundle,monkeypatch):
    original=a.sha;path=next(iter(bundle['bindings']))
    monkeypatch.setattr(a,'sha',lambda p:'0'*64 if str(p)==path else original(p))
    with pytest.raises(ValueError,match='TOCTOU'):a.audit(bundle)


def test_native_angle_bug_is_preserved_not_silently_wrapped():
    g=np.zeros(7);s=np.zeros(7);s[4]=4*np.pi+1.
    native=a.native(g,s);d=a.diagnostic(g,s)
    assert native['success'] and d['native_formula']['negative_angle_term']
    assert d['native_vs_wrapped_success_disagreement'] and not d['independent']['same_allxy_threshold_wrapped_success']
    assert np.isclose(d['independent']['wrapped_angle_error'],1.)


def test_audit_imports_no_producer_or_simulator_in_source():
    import ast
    from pathlib import Path
    tree=ast.parse(Path(a.__file__).read_text())
    imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    names=[n.module or '' for n in imports if isinstance(n,ast.ImportFrom)]
    names.extend(alias.name for n in imports if isinstance(n,ast.Import) for alias in n.names)
    assert not any(any(x in n for x in ('torch','gymnasium','stable_worldmodel','pymunk','physical_20261027_v1')) for n in names)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Attribute) and n.attr in ('verify','analyze')]
    assert not calls
