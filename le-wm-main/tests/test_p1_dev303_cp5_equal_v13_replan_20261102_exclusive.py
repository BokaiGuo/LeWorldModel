from copy import deepcopy
import hashlib
import io

import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_dev303_cp5_equal_v13_replan_20261102_exclusive as subject,
)


@pytest.fixture(scope="module")
def protocol():
    return subject.protocol()


def test_protocol_binds_development_root_both_actual_histories(protocol):
    assert [row["role"] for row in protocol["roles"]] == list(subject.ARMS)
    _, arrays = subject.input_arrays()
    for row in protocol["roles"]:
        actual = arrays[row["role"] + "_raw_action_history"][5:]
        assert np.array_equal(actual, np.asarray(row["actual_prefix"], np.float32))
        assert row["actual_prefix_sha256"] == hashlib.sha256(actual.tobytes()).hexdigest()
    assert protocol["P1_admission"] is False
    assert protocol["independent_confirmation"] is False
    assert protocol["physical_actions_authorized"] is False


def test_both_arms_same_initial_bank_rng_and_budget(protocol):
    zero = np.asarray(protocol["initial_mean"], np.float32)
    assert zero.shape == (2, 5, 2) and not zero.any()
    assert all(np.array_equal(row["contact_seed_tail"], zero) for row in protocol["roles"])
    seeds = np.asarray(protocol["seed_tails"], np.float32)
    assert seeds.shape == (1, 2, 5, 2) and not seeds.any()
    banks = [subject.engine.base.common.make_bank(
        np.random.default_rng(protocol["random_seed"]), zero,
        np.full_like(zero, protocol["algorithm"]["initial_std"]), seeds, 0)
        for _ in protocol["roles"]]
    assert np.array_equal(banks[0], banks[1])
    assert protocol["per_arm_budget"] == {
        "rgb_encode_calls": 1, "action_encoder_calls": 10, "predict_calls": 10,
        "predict_samples": 3200,
    }


def test_kernel_binding_restores_global_state():
    original = subject.engine.NAME, subject.engine.protocol
    with subject._kernel_protocol():
        assert subject.engine.NAME == subject.NAME
        assert subject.engine.protocol is subject.protocol
    assert (subject.engine.NAME, subject.engine.protocol) == original


def test_reject_export_sha_drift(monkeypatch):
    monkeypatch.setattr(subject, "ARCHIVE_SHA", "0" * 64)
    with pytest.raises(ValueError, match="fixed input SHA"):
        subject.input_arrays()


def test_reject_task_sha_drift(monkeypatch):
    monkeypatch.setattr(subject, "TASK_PREP_SHA", "0" * 64)
    with pytest.raises(ValueError, match="fixed input SHA"):
        subject.protocol()


def test_no_silent_archive_action_tamper(monkeypatch):
    original = subject.export.read_regular
    path = subject.EXPORT_DIR / "rgb_action_history.npz"
    raw = original(path)
    with np.load(io.BytesIO(raw), allow_pickle=False) as content:
        values = {key: content[key].copy() for key in content.files}
    values["zero_prefix_raw_action_history"][9, 0] = 0.5
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    changed = buffer.getvalue()
    monkeypatch.setattr(subject.export, "read_regular",
                        lambda p: changed if subject.Path(p) == path else original(p))
    with pytest.raises(ValueError, match="fixed input SHA"):
        subject.input_arrays()


def test_postrun_audit_rejects_unequal_counts(tmp_path, monkeypatch, protocol):
    prepared = tmp_path / "prep.json"
    output = tmp_path / "out"
    output.mkdir()
    report = {
        "schema": subject.NAME + ".report.v1", "protocol_sha256": "prep",
        "status": "model_only_complete_not_admitted", "P1_admission": False,
        "physical_execution": False, "post_cp5_physical_outcomes_used": False,
        "model_before": {"original_state_dict": protocol["model_sha256"]},
        "model_after": {"original_state_dict": protocol["model_sha256"]},
        "attempts": {"rgb_encode": 2, "action_encoder": 19, "predict": 20},
        "rows": [{"role": role} for role in subject.ARMS],
    }
    raw = subject.export.canonical(report)
    (output / "report.json").write_bytes(raw)
    monkeypatch.setattr(subject.engine, "run", lambda *_: subject.export.sha(raw))
    with pytest.raises(ValueError, match="equal-budget"):
        subject.run_cuda(prepared, "prep", output)
    assert not (output / "audit.json").exists()
