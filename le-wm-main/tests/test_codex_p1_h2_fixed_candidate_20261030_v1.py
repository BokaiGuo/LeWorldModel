"""Real saved blind inputs plus an analytic array-only, non-neural backend."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_h2_fixed_candidate_20261030_v1 as kernel

BUNDLE = Path(__file__).resolve().parents[1] / "results/codex_p1_blind_bundle_export_20261030_v1_MAIN01.json"
SHA = "b5748d81de35df501cb6b7d2d325642080994701a9fa7a634e50ac1c22462856"


@pytest.fixture(scope="module")
def bundle():
    return kernel.blind.read_verified(BUNDLE, SHA)[0]


class LinearBackend:
    """Synthetic deterministic arithmetic, never a model."""
    def __init__(self, ledger):
        self.ledger = ledger
        self.j = np.zeros((192, 10))
        self.j[0, 0], self.j[1, 1] = 1, 2

    def step(self, context, actions):
        for resource in ("action_encoder", "predict"):
            e = self.ledger.begin(resource)
            self.ledger.end(e)
        c = deepcopy(context)
        z = np.asarray(c["latent_history"][-1]) + self.j @ np.asarray(actions).reshape(10)
        c["latent_history"] = c["latent_history"][1:] + [z.tolist()]
        token = kernel.observed.base.d2p.normalized_history(actions)[0].astype(float).tolist()
        c["action_history"] = c["action_history"][1:] + [token]
        return c

    def nominal(self, context):
        return np.asarray(self.step(context, np.zeros((5, 2), np.float32))["latent_history"][-1])

    def jacobian(self, context, operation):
        operation("jacobian", lambda: self.step(context, np.zeros((5, 2), np.float32)))
        return self.j.copy()

    def replay(self, context, actions):
        return self.step(context, actions)


def test_real_compiler_two_packets_equal_receipts_and_predicted_context(bundle):
    events = []
    result = kernel.compile_row(bundle, 0, np.zeros(192, np.float32),
                                lambda packet, ledger: LinearBackend(ledger), events.append)
    assert result["physical_authorized"] is False and len(result["raw10"]) == 10
    assert result["packets"][1]["context"] == result["packets"][0]["winner"]["context"]
    for n, packet in enumerate(result["packets"]):
        assert [c["kind"] for c in packet["candidates"]] == ["zero_nominal", kernel.LABEL, "fresh_observed_ridge"]
        assert np.array_equal(packet["candidates"][1]["commands"], kernel.fixed_candidates()[n])
        assert packet["independent_replay"] == packet["winner"]["context"]
        assert result["counts"][n] == {k: {"attempted": v, "completed": v}
                                      for k, v in kernel.observed.budget().items()}
    assert "original_imagined_suffix" not in str(events)
    assert "original_suffix" not in str(events)


def test_fixed_rule_is_same_for_all_conditions_and_feasible(bundle):
    spec = kernel.specification()
    assert len(bundle["formal_roster"]) == 24 and spec["no_dedup"] and spec["no_early_exit"]
    fixed = np.asarray(spec["raw_packets"], np.float32)
    assert np.array_equal(fixed, kernel.fixed_candidates())
    assert np.all(fixed[0, :, 0] == .125) and np.all(fixed[1, :, 0] == -.125)
    assert np.all(fixed[:, :, 1] == 0)
    assert np.max(np.linalg.norm(np.arctanh(fixed.astype(float)).reshape(2, 10), axis=1)) < .75*np.sqrt(5)


def test_all_sixteen_rows_use_same_fixed_rules_with_synthetic_backend(bundle):
    # cp5 latent is synthetic: this is a wiring test, not evidence about RGB encoding.
    for row_index in range(16):
        result = kernel.compile_row(bundle, row_index, np.zeros(192, np.float32),
                                    lambda packet, ledger: LinearBackend(ledger), lambda e: None)
        assert result["specification_sha256"] == kernel.blind.digest(kernel.specification())
        for packet in range(2):
            assert np.array_equal(result["packets"][packet]["candidates"][1]["commands"],
                                  kernel.fixed_candidates()[packet])


def test_bad_bundle_or_row_fails_before_backend(bundle):
    bad = deepcopy(bundle)
    bad["rows"][0]["outcome"] = 1
    made = []
    with pytest.raises(ValueError):
        kernel.compile_row(bad, 0, np.zeros(192, np.float32), lambda *a: made.append(a), lambda e: None)
    with pytest.raises(ValueError):
        kernel.compile_row(bundle, 16, np.zeros(192, np.float32), lambda *a: made.append(a), lambda e: None)
    assert not made


def test_failed_first_packet_never_runs_second(bundle):
    made = []
    class Broken(LinearBackend):
        def jacobian(self, context, operation):
            return operation("jacobian", lambda: (_ for _ in ()).throw(RuntimeError("AD failed")))
    def factory(packet, ledger):
        made.append(packet)
        return Broken(ledger)
    with pytest.raises(RuntimeError, match="AD failed"):
        kernel.compile_row(bundle, 0, np.zeros(192, np.float32), factory, lambda e: None)
    assert made == [0]


def test_incomplete_receipts_fail_closed(bundle):
    class Unmetered(LinearBackend):
        def step(self, context, actions):
            self.ledger = kernel.observed.Ledger(lambda e: None)
            return super().step(context, actions)
    with pytest.raises(ValueError):
        kernel.compile_row(bundle, 0, np.zeros(192, np.float32),
                           lambda packet, ledger: Unmetered(ledger), lambda e: None)
