import copy

import pytest

from experiments.spectral_reachability import p1_diverse_cuda_replay_consume_20261030_main as consumer


def fixture(monkeypatch):
    monkeypatch.setattr(consumer.replay, "bindings", lambda: {"test": "hash"})
    zero = [0.0] * 192
    token = [0.0] * 10
    task = {"action_mean": [0.0, 0.0], "action_std": [1.0, 1.0],
            "root_context": {"latent_history": [zero, zero], "action_history": [token, token]}}
    candidate = {"mid": {"latent_history": [zero, zero], "action_history": [token, token]}}
    report = {"model_before": "m", "rows": [{"result": {"candidates": [candidate]}} for _ in range(8)]}
    backtrack = {"rows": [{"final": {"endpoint": zero}} for _ in range(160)]}
    rows = [{"slot": i // 4, "arm": str(i // 2), "role": str(i % 2),
             "candidate_index": 0, "task_binding": {}, "raw15_le_float32_hex": "00" * 120}
            for i in range(16)]
    exact = dict(whole_segmented=True, source_prefix=True, source_endpoint=True,
                 source_prefix_action_history=True)
    contexts = [{"latent_history": [zero, zero], "action_history": [token, token]} for _ in range(3)]
    saved = [dict(**row, exact=exact, whole_endpoints=[zero] * 3,
                  segmented_endpoints=[zero] * 3, action_tokens=[token] * 3, contexts=contexts)
             for row in rows]
    value = dict(schema=consumer.replay.NAME, calls=dict(predict=96, action_encoder=96),
                 model_before="m", model_after="m", training=False, physical_release=False,
                 P1_admission=False, rows=saved, complete=True)
    return value, {"rows": rows, "files": {"test": "hash"}}, rows, backtrack, report, {"requests": [{"task": task}] * 4}


def test_full_roster_verifies(monkeypatch):
    assert consumer.verify(*fixture(monkeypatch))["all_exact"] is True


@pytest.mark.parametrize("mutation", ["whole", "tokens", "identity", "context", "completion"])
def test_rejects_tampering(monkeypatch, mutation):
    args = copy.deepcopy(fixture(monkeypatch))
    value = args[0]
    if mutation == "whole":
        value["rows"][0]["whole_endpoints"] = copy.deepcopy(value["rows"][0]["whole_endpoints"])
        value["rows"][0]["whole_endpoints"][0][0] = 1
    elif mutation == "tokens":
        value["rows"][0]["action_tokens"][0][0] = 1
    elif mutation == "identity":
        value["rows"][0]["candidate_index"] = 1
    elif mutation == "context":
        value["rows"][0]["contexts"][0]["action_history"][0][0] = 1
    else:
        value["complete"] = False
    with pytest.raises(ValueError):
        consumer.verify(*args)
