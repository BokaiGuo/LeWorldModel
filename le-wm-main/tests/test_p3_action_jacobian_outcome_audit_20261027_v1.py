"""Saved-receipt mutations and bounded arithmetic; no model/environment."""
from copy import deepcopy
from pathlib import Path
import builtins
import subprocess
import sys

import pytest

from experiments.spectral_reachability import p3_action_jacobian_outcome_audit_20261027_v1 as a

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def saved():
    return a.load_bundle(ROOT)[0]


def test_real_all_guards_counts_and_only_complete_paired_h3(saved):
    report=a.audit(saved)
    assert report['status']=='saved_outcomes_verified'
    assert report['root_classifications']=={'full-path':3,'guard-partial':1,'unmatched':0}
    assert report['branch_classifications']=={'full_H3':6,'partial':2,'unmatched':0}
    assert len(report['branches'])==8 and report['comparison']['eligible_root_count']==3
    assert report['accounting']['physical_env_step_attempts']==report['accounting']['physical_env_step_returns']==102
    assert report['accounting']['terminal_encode_samples']==8
    assert report['accounting']['inherited_cost']['predict_samples'] is None
    assert report['new_independent_roots']==0 and not report['P3_admission']
    partial=next(p for p in report['pairs'] if p['classification']=='guard-partial')
    assert partial['actual_steps']==[6,6] and partial['deltas'] is None
    assert all(b['original_branch']['failures'] for b in report['branches'] if not b['full_H3'])
    assert report['comparison']['metrics']['pose_cost']['dynamic_wins']==0
    assert report['comparison']['metrics']['pose_cost']['ties']==2
    assert report['comparison']['metrics']['pose_cost']['dynamic_losses']==1


def test_root_join_independent_of_saved_array_order(saved):
    changed={**saved,'physical':deepcopy(saved['physical'])}
    changed['physical']['rows'].reverse();changed['physical']['encodings'].reverse();changed['physical']['pairs'].reverse()
    assert a.audit(changed)==a.audit(saved)


@pytest.mark.parametrize('mutation',['duplicate','missing','root','repetition','bool_count','count',
    'guard_count','guard_root','full_partial','scope','pose','official','context','model','encode_count',
    'encode_root','unknown_cost','pipeline','schema','partial_pair','planning_cost'])
def test_mutations_fail_closed(saved,mutation):
    b={**saved,'physical':deepcopy(saved['physical'])};r=b['physical'];row=r['rows'][0]
    if mutation=='duplicate':r['rows'][1]=deepcopy(row)
    elif mutation=='missing':r['rows'].pop()
    elif mutation=='root':row['root_sha256']='0'*64
    elif mutation=='repetition':row['repetition']=1
    elif mutation=='bool_count':row['physical_env_step_returns']=True
    elif mutation=='count':row['physical_env_step_returns']=14
    elif mutation=='guard_count':row['guard']['observations'][0]['canonical_agent_pixels']+=1
    elif mutation=='guard_root':row['guard']['root_id']='0'*64
    elif mutation=='full_partial':r['rows'][-1]['full_horizon_completed']=True
    elif mutation=='scope':r['rows'][-1]['endpoint_scope']='full_H3'
    elif mutation=='pose':row['pose']['pose_cost']+=1
    elif mutation=='official':row['official']['official_success']=True
    elif mutation=='context':row['context_identity']['state'][0]+=1
    elif mutation=='model':r['model_unchanged']=False
    elif mutation=='encode_count':r['encoding_stats']['encode_samples']=7
    elif mutation=='encode_root':r['encodings'][0]['root_sha256']='0'*64
    elif mutation=='unknown_cost':r['inherited_cost']['predict_samples']=0
    elif mutation=='pipeline':r['total_pipeline_cost_equivalent']=True
    elif mutation=='schema':r['schema']='v2'
    elif mutation=='partial_pair':r['pairs'][-1]['classification']='full-path'
    else:
        b['planning']=deepcopy(b['planning']);b['planning']['rows'][0]['search']['best_plan']['cost']+=1
    with pytest.raises((ValueError,KeyError)):a.audit(b)


@pytest.mark.parametrize('mutation',['order','command','post_guard_dispatch','final_hash','pixels'])
def test_journal_and_pixel_mutations_rejected_independently(saved,mutation):
    b={**saved,'events':deepcopy(saved['events']),'pixels':deepcopy(saved['pixels'])}
    key=next(iter(b['events']));events=b['events'][key]
    if mutation=='order':events[1],events[2]=events[2],events[1]
    elif mutation=='command':events[1]['command'][0]+=.1
    elif mutation=='post_guard_dispatch':b['pixels'][key][0]['canonical_agent_pixels']=0
    elif mutation=='final_hash':events[-1]['rgb_file_sha256']='0'*64
    else:b['pixels'][key][1]['canonical_agent_pixels']+=1
    with pytest.raises(ValueError):a.audit(b)


def test_no_scalar_singular_value_or_endpoint_completion_shortcut():
    common={'context_identity':{'root':'same'},'endpoint_fresh':True,'physical_env_step_returns':6,'full_H3':False}
    assert a.classification(common,dict(common))=='guard-partial'
    assert a.classification(common,{**common,'physical_env_step_returns':15,'full_H3':True})=='unmatched'
    assert a.classification(common,{**common,'endpoint_fresh':False})=='unmatched'


def test_label_arithmetic_and_bad_numbers():
    state=[0.,0.,3.,4.,0.,0.,0.];goal=[0.]*7
    labels=a.state_labels(state,goal)
    assert labels['pose']['pose_cost']==5 and labels['official']['official_success'] is True
    with pytest.raises(ValueError):a.state_labels([float('nan')]*7,goal)
    with pytest.raises(ValueError):a.state_labels([True]*7,goal)


@pytest.mark.parametrize('mutation',['abstained','root','pose','official','partial_full'])
def test_direct_branch_check_not_only_auxiliary_parity(saved,mutation):
    index=-1 if mutation=='partial_full' else 0
    row=deepcopy(saved['physical']['rows'][index]);key=row['root_sha256'],row['method']
    plan=next(p for p in saved['protocol']['plans'] if (p['root_sha256'],p['method'])==key)
    if mutation=='abstained':row['guard']['abstained']=True
    elif mutation=='root':row['guard']['root_id']='0'*64
    elif mutation=='pose':row['pose']['pose_cost']+=1
    elif mutation=='official':row['official']['official_state_distance']+=1
    else:row['full_horizon_completed']=True
    packet=next(p for p in saved['scheduler']['packets'] if p['root_sha256']==key[0])
    with pytest.raises(ValueError):
        a.inspect_branch(plan,row,saved['events'][key],saved['pixels'][key],saved['protocol']['references'][key[0]],packet)


def test_import_without_neural_simulator_or_svd():
    script=('import sys; from experiments.spectral_reachability import p3_action_jacobian_outcome_audit_20261027_v1; '
            'assert not any(n.split(".")[0] in {"torch","numpy","gymnasium","pymunk","mujoco"} for n in sys.modules)')
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_new_output_source_seal_and_no_runtime_import(tmp_path,monkeypatch):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        assert name.split('.')[0] not in {'torch','gym','gymnasium','mujoco','pymunk','numpy'},name
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    out=tmp_path/'audit';result=a.run(ROOT,out)
    assert result['source_verification']['status']=='verified'
    assert result['new_predict_samples']==result['new_physical_steps']==result['new_encode_samples']==0
    assert (out/'report.json').is_file()
    with pytest.raises(ValueError,match='NEW'):a.run(ROOT,out)
