"""Integer-only envelope check; not production output certification."""
from experiments.spectral_reachability import agent_p3_c3_k2_count_bound_20261029_v2 as m


def test_every_conditional_component_bounded():
    x = m.bound()
    assert (x['slots'],x['requests']) == (6,864)
    assert all(x['parts'][k] <= x['old_parts'][k] for k in x['parts'])
    assert x['conditional_bytes'] <= x['old_bytes']
    assert x['output_fit_proved'] is False
