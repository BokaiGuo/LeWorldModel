"""No neural calls: synthetic saved-ABI arithmetic and mutation tests."""
import pytest
from experiments.spectral_reachability import agent_p1_fixed_prefix_gradient_diagnostic_20261029_v1 as m
import test_agent_p1_fixed_prefix_saved_handoff_20261029_v1 as fixture

inputs=fixture.inputs
complete=fixture.complete
group=fixture.group
saved=fixture.saved
panel=fixture.panel


def test_full64_wiring_and_unscored_shadow(panel):
    p,r=panel; groups=m.analyze(p,r,'a'*64)
    assert len(groups)==64 and sum(len(x['steps']) for x in groups)==512
    assert all(s['shadow_terminal_Q'] is None and s['shadow_score_status']=='not_evaluated'
               and s['native_gradient_correctness']=='not_established' for x in groups for s in x['steps'])
    assert all(x['selected_iteration'] is None for x in groups)


@pytest.mark.parametrize('field',['raw_gradient','gradient_context','logit','witness'])
def test_corruption_fails_closed(panel,field):
    p,r=panel; result=r['rows'][0]['result']
    if field=='raw_gradient':result['gradients'][0]['raw_gradient'][0][0]+=1
    elif field=='gradient_context':result['gradients'][0]['context']['latent_history'][-1][0]+=1
    elif field=='logit':result['iterates'][1]['logits'][0]+=1
    else:result['iterates'][0]['witness']=True
    with pytest.raises(ValueError):m.analyze(p,r,'a'*64)
