import json
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_p2_stagec_composition_saved_consume_20261028_v1 as m

def base():
    rows=[];plans=[];physical=[];enc=[]
    for i,(root,method) in enumerate((r,a) for r in range(4) for a in m.METHODS):
        key='r'+str(root)+method
        plan=dict(root_index=root,root_sha256=key,method=method,selection_sha256='s'+str(i))
        row=dict(root_sha256=key,method=method,selection_sha256='s'+str(i),planned_commands=10,planned_horizon_tokens=2,
            attempted_commands=10,completed_commands=10,actual_endpoint_primitive_steps=10,full_horizon_completed=True,failures=[],
            journal_path=f'/fixture/{key}.events.jsonl',terminal_rgb_path=f'/fixture/{key}_rgb_10.png')
        plans.append(plan);rows.append(row);physical.append(dict(root_sha256=key,method=method,failures=[]));enc.append(dict(root_sha256=key,method=method,status='completed',attempted_encode_samples=1))
    return dict(schema=m.EXPECTED_SCHEMA,status='calibration_recorded',rows=rows,encodings=enc,P1_admission=False,P2_admission=False,calibration_only=True,no_neural_predict=True,physical_efficacy_verified=False,fair_total_budget_comparison=False,root_selection_changed=False,new_predict_calls=0,selected_path_env_step_attempts=80,selected_path_env_step_returns=80,attempted_terminal_encode_samples=8),dict(schema=m.PROTOCOL_SCHEMA,plans=plans),dict(schema=m.PHYSICAL_SCHEMA,rows=physical)

def test_fixed_census_and_scope():
    r,p,v=base(); assert len(r['rows'])==len(p['plans'])==len(v['rows'])==8
    assert m.strict_rows(r,p,v)==[(x,y) for x in range(4) for y in m.METHODS]
    assert r['physical_efficacy_verified'] is False

@pytest.mark.parametrize('field', ['P2_admission','new_predict_calls','selected_path_env_step_returns'])
def test_scope_or_budget_mutation_rejected(field):
    r,p,v=base(); r[field]=True if field=='P2_admission' else 1
    with pytest.raises(ValueError):m.strict_rows(r,p,v)

def test_missing_cell_rejected():
    r,p,v=base();p['plans'].pop();
    with pytest.raises(ValueError):m.strict_rows(r,p,v)

def test_composition_is_not_a_claim():
    r,p,v=base();assert m.strict_rows(r,p,v);assert m.METHODS==('joint_h2_no_composition','conditional_suffix_composition')
