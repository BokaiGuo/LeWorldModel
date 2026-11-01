"""Independent numeric-contract fixtures, not neural/physical evidence."""
import copy

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import reacher_d2p_recentered_reach as runtime
from experiments.spectral_reachability import analyze_reacher_d2p_recentered_reach as a


def fixture():
    generator = torch.Generator().manual_seed(104)
    j = torch.randn(192, 3, generator=generator)
    offset = torch.randn(192, generator=generator)
    goal = offset+j@torch.tensor([.7, -.6, .8])
    initial = torch.tensor([-.8, .2, -.4])
    _, _, receipt = runtime.refine(lambda x: offset+j@x, initial, goal)
    for step in receipt["trace"]:
        step["qp_certificate"]["device"] = "cuda:synthetic-fixture"
    return receipt, goal.numpy().astype(float), initial.numpy().astype(float)


def test_complete_independent_affine_qp_and_monotone_chain():
    receipt, goal, command = fixture()
    value = np.asarray(receipt["initial_vector"])
    for step in receipt["trace"]:
        command, value, cost, errors = a.verify_step(step, goal, command, value)
        assert len(errors) == 6
    assert cost == pytest.approx(receipt["final_objective"])


@pytest.mark.parametrize("bad", ["anchor", "intercept", "box", "local", "certificate", "trial", "cost", "accept", "missing", "nan"])
def test_independent_verifier_rejects_changed_local_problem_and_action(bad):
    receipt, goal, command = fixture()
    s = copy.deepcopy(receipt["trace"][0])
    value = np.asarray(receipt["initial_vector"])
    if bad == "anchor": s["anchor_command"][0] += .1
    elif bad == "intercept": s["affine_residual"][0] += .1
    elif bad == "box": s["qp_action"][0] = 1.1
    elif bad == "local": s["local_affine_objective"] += 1.
    elif bad == "certificate": s["qp_certificate"]["objective_duality_gap_upper_bound"] = 1.
    elif bad == "trial": s["trials"][0]["command"][0] += .1
    elif bad == "cost": s["trials"][0]["objective"] += 1.
    elif bad == "accept": s["accepted_scale"] = .123
    elif bad == "missing": s["trials"].pop()
    else: s["response"][0][0] = float("nan")
    with pytest.raises(ValueError): a.verify_step(s, goal, command, value)


def test_root_identity_join_rejects_duplicate_and_missing():
    with pytest.raises(ValueError): a.index([{"seed": 0}, {"seed": 0}], [0, 1])
    with pytest.raises(ValueError): a.index([{"seed": 0}], [0, 1])
    assert sorted(a.index([{"seed": 1}, {"seed": 0}], [0, 1])) == [0, 1]


def test_confirmation_pairing_has_explicit_eight_root_contract():
    result = a.cohort_paired([-1.] * 8)
    assert result["mean_difference"] == -1.
    assert result["improved"] == result["n"] == 8
    assert result["ci95_descriptive"] == [-1., -1.]
    for bad in ([0.] * 7, [0.] * 16, [float("nan")] * 8):
        with pytest.raises(ValueError):
            a.cohort_paired(bad)


def test_confirmation_admits_identity_only_and_rejects_source_overlap():
    from pathlib import Path
    from experiments.spectral_reachability.audit_reacher_d2p_recentered_reach import confirmation_cohorts, digest
    frozen = {"status": "completed", "gates": {"integrity": True},
        "sealed_protocol": {"contract": runtime.CONTRACT,
            "implementation_sha256": {"recenter": digest(Path(runtime.__file__))}},
        "rows": [{"seed": i, "source_episode": i+100} for i in range(16)]}
    entries = [(None, {"seed": i, "source_episode": i+100}) for i in list(range(8))+list(range(16,24))]
    assert confirmation_cohorts(frozen, entries) == {"alignment": list(range(8)), "confirmation": list(range(16,24))}
    broken = copy.deepcopy(entries)
    broken[-1][1]["source_episode"] = 115
    with pytest.raises(ValueError): confirmation_cohorts(frozen, broken)
    broken = copy.deepcopy(entries)
    broken[0][1]["source_start_row"] = 33
    with pytest.raises(ValueError): confirmation_cohorts(frozen, broken)
    changed = copy.deepcopy(frozen)
    changed["sealed_protocol"]["implementation_sha256"]["recenter"] = "wrong"
    with pytest.raises(ValueError): confirmation_cohorts(changed, entries)
    with pytest.raises(ValueError): confirmation_cohorts(frozen, entries[:-1])
