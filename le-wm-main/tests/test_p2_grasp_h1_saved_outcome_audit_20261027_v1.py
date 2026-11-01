"""Pure saved evidence/mutation tests, no inference or simulator."""
from copy import deepcopy
import importlib
from pathlib import Path
import subprocess
import sys
import pytest

MOD='experiments.spectral_reachability.p2_grasp_h1_saved_outcome_audit_20261027_v1'
def m():return importlib.import_module(MOD)


@pytest.fixture(scope='module')
def bundle():return m().load_bundle(Path(__file__).resolve().parents[1])[0]


def test_import_safe():
    subprocess.run([sys.executable,'-c',f'import {MOD};import sys;assert not set(["torch","gymnasium","pymunk","numpy"]) & set(sys.modules)'],check=True)


def test_actual_census_budget_and_pair_denominators(bundle):
    out=m().audit(bundle)
    assert out['branch_counts']=={'completed':4,'rejected_raw_out_of_box':4}
    assert out['pair_counts']=={'full-path':1,'unmatched':3}
    assert out['budget']['selected']==dict(attempts=60,returns=60)
    assert out['budget']['warmup']==dict(attempts=60,returns=60)
    assert out['reset_budget']['root_collection']==dict(attempts=32,returns=32)
    assert out['comparison']['complete_pairs']==1
    assert len(out['rows'])==8 and len(out['pairs'])==4
    assert out['P2_admission'] is False


@pytest.mark.parametrize('mutation',['missing','duplicate','root','goal','context','path','rejected_step','pixel','guard','step_order','budget','reset','warmup','pose','partial','source','counter','false_neural'])
def test_saved_counterexamples_rejected(bundle,mutation):
    b=deepcopy(bundle);report=b['report'];rows=report['rows'];row=next(r for r in rows if r['status']=='completed');key=(row['root_sha256'],row['method'])
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows[0]=deepcopy(rows[1])
    elif mutation=='root':row['root_sha256']='0'*64
    elif mutation=='goal':row['context_identity']['goal_rgb_sha256']='0'*64
    elif mutation=='context':row['context_identity']['input_sha256']='0'*64
    elif mutation=='path':row['commands'][0][0]+=.01
    elif mutation=='rejected_step':next(r for r in rows if r['status']=='rejected_raw_out_of_box')['physical_env_step_attempts']=1
    elif mutation=='pixel':b['pixels'][key][0]['canonical_agent_pixels']=0
    elif mutation=='guard':row['guard']['primitive_commands_authorized']=14
    elif mutation=='step_order':b['events'][key][1]['primitive_step']=2
    elif mutation=='budget':report['all_environment_step_counts']['selected']['attempts']=59
    elif mutation=='reset':report['all_environment_reset_counts']['root_collection']['returns']=31
    elif mutation=='warmup':next(e for e in b['setup'] if e['kind']=='env_step_attempt' and e['phase']=='warmup')['command']=[.1,0.]
    elif mutation=='pose':row['pose']['pose_cost']+=1
    elif mutation=='partial':row['full_horizon_completed']=False
    elif mutation=='source':report['consumer_prepare_sha256']='0'*64
    elif mutation=='counter':report['total_environment_step_attempts']=60
    else:report['new_neural_encodes']=1
    with pytest.raises((ValueError,KeyError)):m().audit(b)


def test_pairing_not_array_order(bundle):
    b=deepcopy(bundle);b['report']['rows'].reverse()
    assert m().audit(b)['pairs']==m().audit(bundle)['pairs']


def test_saved_synthetic_guard_partial_kept_not_promoted(bundle):
    mod=m();b=deepcopy(bundle);row=next(x for x in b['report']['rows'] if x['status']=='completed');key=row['root_sha256'],row['method']
    plan=next(x for x in b['protocol']['plans'] if (x['root_sha256'],x['method'])==key)
    packet=next(x for x in b['protocol']['packets'] if x['root_sha256']==key[0])
    events=b['events'][key][:19];pixels=b['pixels'][key][:7];pixels[-1]['canonical_agent_pixels']=0
    row.update(status='observation_unavailable',physical_env_step_attempts=6,physical_env_step_returns=6,
        attempted_commands=6,completed_commands=6,remaining_commands_not_executed=9,actual_endpoint_primitive_steps=6,
        full_horizon_completed=False,endpoint_fresh=True,endpoint_scope='guard_partial_or_terminated',
        trace=row['trace'][:6],final_state=events[-1]['state'],terminal_rgb_path=events[-1]['rgb_path'],terminal_rgb_sha256=events[-1]['rgb_file_sha256'])
    row['guard'].update(observations=pixels,primitive_commands_authorized=6,abstained=True)
    row.update(mod.labels(row['final_state'],row['context_identity']['goal']))
    mod.inspect_branch(plan,row,events,pixels,packet)
    row['full_horizon_completed']=True
    with pytest.raises(ValueError,match='classification'):mod.inspect_branch(plan,row,events,pixels,packet)


def test_output_exclusive_and_no_runtime_imports(tmp_path):
    out=tmp_path/'audit'
    subprocess.run([sys.executable,'-c',f'from {MOD} import main;import sys;main();assert not set(["torch","gymnasium","pymunk","numpy"]) & set(sys.modules)',
        '--output',str(out)],check=True,capture_output=True,text=True)
    assert (out/'report.json').is_file() and (out/'source_seal/manifest.json').is_file()
    with pytest.raises(ValueError,match='NEW'):m().run(Path(__file__).resolve().parents[1],out)
