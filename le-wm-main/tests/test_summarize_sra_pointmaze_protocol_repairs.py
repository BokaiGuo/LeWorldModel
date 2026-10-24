import inspect

from experiments.spectral_reachability import summarize_sra_pointmaze_protocol_repairs as summary


def test_protocol_repair_summary_keeps_p1_and_p2_contracts_separate():
    source = inspect.getsource(summary)
    assert '"continuation_score", "discrete_reachable_set"' in source
    assert '"effect_score", "goal_projection"' in source
    assert "a protocol-repair panel group mixes physical contracts" in source
