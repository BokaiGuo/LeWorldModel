"""Reference admission only; no CPU neural experiments or efficacy fixtures."""
import copy

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_original_margin_checkpoint import validate_reference, summarize
from experiments.spectral_reachability.reacher_d2p_context import WARMUP_POLICY


def fixture():
    method = {"first_receipt": {"non_greedy_witness_found": True},
        "first_action_witness_raw": [[.1, -.1]] * 5, "zero_prefix_action_witness_raw": [[0., 0.]] * 5}
    report = {"schema": "d2p_reacher_p1_reconfiguration.v1", "protocol": {"action_block": 5, "reconfiguration_horizon": 1,
        "warmup_policy": WARMUP_POLICY}, "rows": [{"seed": s, "methods": {"reconfiguration": copy.deepcopy(method)}} for s in range(6)]}
    prior = {"status": "completed", "gates": {"cuda_only": True}, "rows": [{"seed": s} for s in reversed(range(6))]}
    return report, prior


def test_all_roots_joined_by_identity():
    assert validate_reference(*fixture()) == list(range(6))


@pytest.mark.parametrize("kind", ["partial", "duplicate", "extra", "truthy", "skip", "unbounded", "wrong_shape", "warmup", "qpos"])
def test_reference_fail_closed(kind):
    report, prior = fixture()
    if kind == "partial":
        prior["rows"].pop()
    elif kind == "duplicate":
        report["rows"][0]["seed"] = 1
    elif kind == "extra":
        prior["rows"].append({"seed": 99})
    elif kind == "truthy":
        prior["gates"]["cuda_only"] = 1
    elif kind == "skip":
        report["rows"][0]["methods"]["reconfiguration"]["first_receipt"]["non_greedy_witness_found"] = False
    elif kind == "unbounded":
        report["rows"][0]["methods"]["reconfiguration"]["first_action_witness_raw"][0] = [2., 0.]
    elif kind == "wrong_shape":
        report["rows"][0]["methods"]["reconfiguration"]["first_action_witness_raw"].pop()
    elif kind == "warmup":
        report["protocol"]["warmup_policy"] = "different"
    else:
        report["protocol"]["qpos_readout_binding"] = {"enabled": True}
    with pytest.raises(ValueError):
        validate_reference(report, prior)


def test_summary_keeps_all_roots_and_reversals():
    rows = [{"seed": s, "checkpoints": {name: {precision: {"decomposition": {"predicted_margin": -1 if s == 2 else 1,
        "observed_margin": 1}, "candidate_latent_l2": 1., "zero_latent_l2": 2.} for precision in ("float32", "float64")}
        for name in ("receipt_checkpoint", "official_checkpoint")}} for s in range(6)]
    result = summarize(rows)
    assert len(result) == 4
    assert all(r["order_agreements"] == 5 and r["reversed_roots"] == [2] for r in result.values())
    with pytest.raises(ValueError):
        summarize(rows[:-1])
