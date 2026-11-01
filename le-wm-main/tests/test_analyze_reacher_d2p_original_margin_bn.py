"""Stored-vector integrity fixtures only; no neural or efficacy experiments."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_original_margin_bn import VARIANTS, summarize


def fixture():
    cell = {"vectors": {"prediction": [1.] * 192, "prediction_zero": [0.] * 192,
        "observed": [-1.] * 192, "observed_zero": [0.] * 192, "goal": [2.] * 192},
        "costs": {"prediction": 1., "prediction_zero": 4., "observed": 9., "observed_zero": 4.},
        "candidate_latent_l2": float(np.sqrt(768)), "zero_latent_l2": 0.,
        "decomposition": {"predicted_margin": -3., "observed_margin": 5., "margin_error": 8.,
            "zero_successor_bias_contribution": 0., "response_linear_error_contribution": 8.,
            "response_quadratic_error_contribution": 0.}}
    seeds = list(range(6))
    gates = {k: True for k in ("cuda_only", "all6_fixed_roots", "all4_variants_each_root", "both_native_baselines_bit_exact",
        "only_declared_bn_buffers_changed", "forward_sample_accounting", "inputs_unchanged", "crossover_unchanged",
        "implementation_unchanged")}
    common = {"order_agreements": 0, "mean_candidate_latent_l2": float(np.mean([cell["candidate_latent_l2"]] * 6)),
        "mean_zero_latent_l2": 0.}
    protocol = {"root_seeds": seeds, "physical_labels_used_for_planning": False, "scope": "Fixture only"}
    report = {"schema": "reacher_d2p_original_margin_bn.v1", "status": "completed", "gates": gates,
        "sealed_protocol": copy.deepcopy(protocol), "actual_model_predict_samples": 48,
        "rows": [{"seed": s, "variants": {v: copy.deepcopy(cell) for v in VARIANTS}} for s in seeds],
        "summary": {v: copy.deepcopy(common) for v in VARIANTS}}
    cross_gates = {k: True for k in ("cuda_only", "all6_fixed_roots", "all12_physical_packets_exact_replay",
        "all_original_four_costs_replay", "all_checkpoints_frozen_eval", "forward_sample_accounting", "inputs_unchanged",
        "implementation_unchanged", "action_embedding_reference_fp32_bit_exact")}
    crossover = {"schema": "reacher_d2p_original_margin_checkpoint.v1", "status": "completed", "gates": cross_gates,
        "sealed_protocol": protocol, "actual_model_predict_samples": 48,
        "rows": [{"seed": s, "checkpoints": {name: {p: copy.deepcopy(cell) for p in ("float32", "float64")}
            for name in ("receipt_checkpoint", "official_checkpoint")}} for s in seeds],
        "summary": {name + "/" + p: {**common, "reversed_roots": seeds}
            for name in ("receipt_checkpoint", "official_checkpoint") for p in ("float32", "float64")}}
    return report, crossover


def test_complete_six_by_four_vector_contract():
    report, crossover = fixture()
    assert summarize(report, crossover)["root_seeds"] == list(range(6))


def test_pairing_is_by_root_identity_not_collection_order():
    report, crossover = fixture()
    expected = summarize(report, crossover)
    report["rows"].reverse()
    crossover["rows"] = crossover["rows"][2:] + crossover["rows"][:2]
    assert summarize(report, crossover) == expected


@pytest.mark.parametrize("kind", ["partial_gates", "false_gate", "integer_gate", "missing_root", "duplicate_root",
    "crossover_duplicate", "crossover_false_gate", "crossover_summary", "missing_variant", "corrupt_vector",
    "corrupt_summary", "labels_opened", "compute_count", "protocol_roots"])
def test_incomplete_or_tampered_evidence_rejected(kind):
    report, crossover = fixture()
    if kind == "partial_gates":
        report["gates"] = {"cuda_only": True}
    elif kind == "false_gate":
        report["gates"]["only_declared_bn_buffers_changed"] = False
    elif kind == "integer_gate":
        report["gates"]["cuda_only"] = 1
    elif kind == "missing_root":
        report["rows"].pop()
    elif kind == "duplicate_root":
        report["rows"][-1]["seed"] = 0
    elif kind == "crossover_duplicate":
        crossover["rows"][-1]["seed"] = 0
    elif kind == "crossover_false_gate":
        crossover["gates"]["all12_physical_packets_exact_replay"] = False
    elif kind == "crossover_summary":
        crossover["summary"]["official_checkpoint/float32"]["order_agreements"] = 6
    elif kind == "missing_variant":
        del report["rows"][0]["variants"][VARIANTS[0]]
    elif kind == "corrupt_vector":
        report["rows"][0]["variants"][VARIANTS[1]]["vectors"]["prediction"][0] = float("nan")
    elif kind == "corrupt_summary":
        report["summary"][VARIANTS[0]]["order_agreements"] = 6
    elif kind == "labels_opened":
        report["sealed_protocol"]["physical_labels_used_for_planning"] = True
    elif kind == "compute_count":
        report["actual_model_predict_samples"] = 24
    else:
        report["sealed_protocol"]["root_seeds"][-1] = 9
    with pytest.raises(ValueError):
        summarize(report, crossover)


def test_changed_visual_chart_rejected_even_if_cell_algebra_is_consistent():
    report, crossover = fixture()
    altered = report["rows"][0]["variants"][VARIANTS[1]]
    for name in altered["vectors"]:
        altered["vectors"][name] = [v + 1. for v in altered["vectors"][name]]
    with pytest.raises(ValueError, match="chart changed"):
        summarize(report, crossover)


def test_native_baseline_requires_bit_exact_full_vectors():
    report, crossover = fixture()
    baseline = report["rows"][0]["variants"][VARIANTS[0]]
    # Below the cost/decomposition tolerances, but not an exact native replay.
    baseline["vectors"]["prediction"][0] += 1e-12
    with pytest.raises(ValueError, match="baseline not bit exact"):
        summarize(report, crossover)
