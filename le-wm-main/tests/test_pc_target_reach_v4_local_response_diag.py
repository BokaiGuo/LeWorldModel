import json

from experiments import pc_target_reach_v4_local_response_diag_20261104_codex as diagnostic


def test_result_is_json_serializable():
    zero = [[0.0, 0.0] for _ in range(5)]
    row = {"root": 0, "task": {"start": [-0.5, 0., 0.], "goal": [0.5, 0.]},
           "candidates": [{"name": "zero", "prefix": zero}, {"name": "selected_final", "prefix": zero}],
           "greedy_index": 0, "lookahead_index": 0, "selected_tail": zero + zero}
    result = diagnostic.build(row)
    json.dumps(result, allow_nan=False)
    assert isinstance(result["nonprogress"], bool)
