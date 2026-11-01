from copy import deepcopy
import pytest
from experiments.spectral_reachability import main_nonlinear_suffix_effect_20261028_v1 as m


def fixture():
    commands=[[0.,0.]]*10
    context=dict(latent_history=[[0.]*192]*3,action_history=[[0.]*10]*2)
    slot=dict(index=0,root_slot=0,condition='off',task=dict(goal=[0.]*192,root_context=context),
        original_generation_winner=dict(index=3),candidates=[])
    rows=[]
    for j in range(42):
        q=float(j+2);score=q-1
        slot['candidates'].append(dict(index=j,original_terminal_Q=q,prefix_bits=m.source.bits(commands[:5])))
        rows.append(dict(index=j,mid=context,commands=commands,selected_Q=score,winner_iteration=1,
            gradients=[{}]*8,iterates=[dict(Q=q)]+[dict(Q=score)]*8))
    return slot,rows


def test_envelope_decomposition_not_physical_claim():
    s,r=fixture();out=m.analyze(s,r)
    assert out['original_index']==3 and out['terminal_only_index']==out['optimized_index']==0
    assert out['terminal_reselection_delta']==-3
    assert out['nonlinear_suffix_envelope_delta']==out['fixed_original_prefix_suffix_delta']==-1
    assert out['model_uphill_prefixes']==0 and out['best_uphill'] is None


def test_no_uphill_inference_from_terminal_only():
    s,r=fixture();r[0]['mid']=deepcopy(r[0]['mid']);r[0]['mid']['latent_history'][-1]=[1.]*192
    out=m.analyze(s,r)
    assert out['selected_model_uphill'] and out['model_uphill_prefixes']==1


@pytest.mark.parametrize('damage',['drop','order','prefix','incumbent','worse','iterate'])
def test_fixed_library_mutation_rejected(damage):
    s,r=fixture()
    if damage=='drop':r.pop()
    elif damage=='order':r.reverse()
    elif damage=='prefix':r[0]['commands']=[[.1,.1]]*10
    elif damage=='incumbent':s['candidates'][0]['original_terminal_Q']=0
    elif damage=='worse':r[0]['selected_Q']=99
    else:r[0]['iterates'].pop()
    with pytest.raises(ValueError):m.analyze(s,r)
