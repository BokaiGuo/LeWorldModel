import json
from copy import deepcopy
from pathlib import Path
import pytest
from experiments.spectral_reachability.analyze_pusht_geometry_20261027 import generation, summarize


def actual():
    return json.loads((Path(__file__).resolve().parents[1]/'results/pusht_geometry_prefix_20261027_v3_cuda_v1/report.json').read_text())


def test_actual_generation_does_not_mislabel_changed_greedy_as_new_winner():
    rows=generation(actual())
    assert len(rows)==4
    assert all(a['selected_is_common_winner'] and a['objective_improvement_vs_common']==0 for r in rows for a in r['arms'])
    assert sum(a['selected']!=a['greedy'] for r in rows for a in r['arms'])==1
    assert all(0<r['epsilon_halfstep_relative_difference']<.03 for r in rows)


def test_unsealed_or_failed_model_plan_not_analyzable():
    r=actual();r['status']='integrity_fail'
    with pytest.raises(ValueError):generation(r)


def test_zero_gradient_has_no_fabricated_cosine():
    r=actual();r['plans'][0]['gradient_grids'][0]['gradients']['geometry']=[0.]*10
    with pytest.raises(ValueError):generation(r)


def physical():
    return json.loads((Path(__file__).resolve().parents[1]/'results/pusht_geometry_physical_20261027_v2_cuda_v1/report.json').read_text())


def test_actual_guarded_receipts_are_not_rewritten_as_passing():
    r=summarize(actual(),physical())
    assert r['original_physical_status']=='integrity_fail'
    assert r['original_gates']['encodings_complete'] is False
    assert len(r['blocked_endpoints'])==2
    assert r['actual_steps']==952 and r['actual_encoder_samples']==190
    assert r['P1_admission'] is False and r['theory_negative'] is False
    qualified=[g for g in r['groups'] if g['arm']=='geometry' and g['model_non_greedy']]
    assert len(qualified)==1 and qualified[0]['root_index']==1
    roles=qualified[0]['roles']
    assert all(v['fixed_terminal_available'] for v in roles.values())
    assert roles['selected']['terminal_pose']>roles['greedy']['terminal_pose']
    assert roles['selected']['terminal_official_distance']<roles['greedy']['terminal_official_distance']


@pytest.mark.parametrize('mutation', ['drop_row','duplicate_encoding','hide_gate','runtime_error','fake_terminal','unexplained_missing','rewrite_status'])
def test_invalid_saved_receipts_fail_closed(mutation):
    p=physical()
    if mutation=='drop_row':p['rows'].pop()
    elif mutation=='duplicate_encoding':p['encodings'].append(deepcopy(p['encodings'][0]))
    elif mutation=='hide_gate':del p['gates']['frozen_eval']
    elif mutation=='runtime_error':p['failures'].append({'phase':'runtime'})
    elif mutation=='rewrite_status':p['status']='calibration_recorded'
    else:
        e=next(e for e in p['encodings'] if e['status']=='blocked')
        if mutation=='fake_terminal':e.update(status='completed',comparable=True,attempted_encode_samples=1)
        else:
            b=next(b for b in p['rows'] if (b['root_sha256'],b['method'])==(e['root_sha256'],e['method']))
            b['status']='completed'
    with pytest.raises(ValueError):summarize(actual(),p)


def test_empty_model_gate_set_is_not_pass():
    p=actual();p['gates']={}
    with pytest.raises(ValueError):generation(p)
