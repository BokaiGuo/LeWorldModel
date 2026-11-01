from experiments.spectral_reachability import p1_backtrack_capability_consume_20261030 as consume


def test_fixed_direction_difference_and_missing_pairs():
    anchor = {"index": 0, "root_slot": 0, "condition": "off", "role": "incumbent",
              "root_state": [0, 0, 0, 0, 0, 0, 0], "goal": [1, 0, 0, 0, 0, 0, 0]}
    endpoints = [[0.0] * 7 for _ in range(21)]
    endpoints[1] = [2.0, 0, 0, 0, 0, 0, 0]
    endpoints[2] = [-1.0, 0, 0, 0, 0, 0, 0]
    endpoints[3] = None
    result = consume.summarize(anchor, endpoints)
    assert result["projection"][:3] == [0.0, 2.0, -1.0]
    assert result["central_difference"][0] == [6.0, 0, 0, 0]
    assert result["asymmetry"][0] == [1.0, 0, 0, 0]
    assert result["central_difference"][1] is None
    assert result["complete_endpoints"] == 20


def test_undefined_common_direction_is_not_replaced():
    anchor = {"index": 0, "root_slot": 0, "condition": "off", "role": "incumbent",
              "root_state": [0] * 7, "goal": [0] * 7}
    result = consume.summarize(anchor, [[0] * 7 for _ in range(21)])
    assert result["direction"] is None
    assert result["projection"] == [None] * 21
