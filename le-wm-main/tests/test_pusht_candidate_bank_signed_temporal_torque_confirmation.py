from pathlib import Path

import numpy as np

from experiments import run_pusht_candidate_bank_signed_temporal_torque_confirmation as candidate_signed
from experiments.run_pusht_balanced_temporal_torque_confirmation import RootExample


def _row(root_id: int, preference: int, plus: float, minus: float) -> RootExample:
    return RootExample(
        root_id=root_id,
        root_digest=f"root-{root_id}",
        rgb_frames=np.zeros((3, 4, 4, 3), dtype=np.uint8),
        history_actions=np.zeros((2, 2), dtype=np.float32),
        preference=preference,
        preference_margin=plus - minus,
        zero_corrected_plus_response=plus,
        zero_corrected_minus_response=minus,
    )


def test_candidate_bank_pair_relabels_actions_by_geometric_moment(monkeypatch) -> None:
    # The candidate bank enumerates negative tangent first.  The protocol must
    # not mistake enumeration order for the physical moment label.
    monkeypatch.setattr(candidate_signed, "boundary_inward_normal", lambda *_args: np.asarray([1.0, 0.0]))
    monkeypatch.setattr(
        candidate_signed,
        "candidate_actions",
        lambda *_args, **_kwargs: np.asarray([[0.18, -0.55], [0.18, 0.55], [-0.18, 0.0]]),
    )
    monkeypatch.setattr(
        candidate_signed,
        "action_moment_sign",
        lambda _state, action, **_kwargs: float(np.sign(np.asarray(action)[1])),
    )
    actions, audit = candidate_signed.candidate_bank_action_pair(
        object(), np.zeros(7), inward_magnitude=0.18, tangent_magnitude=0.55,
        deployed_tangent_magnitudes=(0.55,), deployed_contact_normal_magnitudes=(0.18,),
    )
    assert audit["pair_valid"]
    assert audit["matching_candidate_indices"] == [0, 1]
    np.testing.assert_allclose(actions["plus"], [0.18, 0.55])
    np.testing.assert_allclose(actions["minus"], [0.18, -0.55])


def test_candidate_bank_pair_requests_the_full_declared_deployment_bank(monkeypatch) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(candidate_signed, "boundary_inward_normal", lambda *_args: np.asarray([1.0, 0.0]))

    def builder(_env, _state, tangent_magnitudes, inward_magnitude, **kwargs):
        seen["tangent_magnitudes"] = tuple(tangent_magnitudes)
        seen["inward_magnitude"] = inward_magnitude
        seen["normal_magnitudes"] = tuple(kwargs["contact_normal_magnitudes"])
        return np.asarray([
            [0.02, -0.06], [0.02, 0.06], [0.02, -0.10], [0.02, 0.10],
            [0.02, -0.16], [0.02, 0.16], [0.04, -0.06], [0.04, 0.06],
            [0.04, -0.10], [0.04, 0.10], [0.04, -0.16], [0.04, 0.16],
            [0.08, -0.06], [0.08, 0.06], [0.08, -0.10], [0.08, 0.10],
            [0.08, -0.16], [0.08, 0.16], [-0.08, 0.0],
        ])

    monkeypatch.setattr(candidate_signed, "candidate_actions", builder)
    monkeypatch.setattr(
        candidate_signed,
        "action_moment_sign",
        lambda _state, action, **_kwargs: float(np.sign(np.asarray(action)[1])),
    )
    actions, audit = candidate_signed.candidate_bank_action_pair(
        object(), np.zeros(7), inward_magnitude=0.08, tangent_magnitude=0.16,
    )
    assert audit["pair_valid"]
    assert audit["candidate_bank_size"] == 19
    assert seen == {
        "tangent_magnitudes": (0.06, 0.10, 0.16),
        "inward_magnitude": 0.08,
        "normal_magnitudes": (0.02, 0.04, 0.08),
    }
    np.testing.assert_allclose(actions["plus"], [0.08, 0.16])


def test_candidate_bank_pair_refuses_a_selected_action_outside_deployed_support(monkeypatch) -> None:
    monkeypatch.setattr(candidate_signed, "boundary_inward_normal", lambda *_args: np.asarray([1.0, 0.0]))
    with np.testing.assert_raises_regex(ValueError, "member of the deployed candidate bank"):
        candidate_signed.candidate_bank_action_pair(
            object(), np.zeros(7), inward_magnitude=0.18, tangent_magnitude=0.16,
        )


def test_signed_pair_task_ranking_transduces_each_turn_direction_without_retraining() -> None:
    rows = [_row(1, 1, 0.3, -0.1), _row(2, 0, -0.2, 0.4)]
    ranking = candidate_signed.signed_pair_task_ranking(np.asarray([0.9, 0.1]), rows)
    assert ranking["roots"] == 2
    assert ranking["contexts"] == 4
    assert ranking["candidate_pair_ranking_accuracy"] == 1.0
    assert ranking["candidate_pair_ranking_accuracy_by_desired_turn_sign"] == {"1": 1.0, "-1": 1.0}
    assert ranking["mean_signed_effect_regret"] == 0.0


def test_signed_pair_task_ranking_charges_the_observed_effect_gap_when_wrong() -> None:
    rows = [_row(1, 1, 0.3, -0.1)]
    ranking = candidate_signed.signed_pair_task_ranking(np.asarray([0.1]), rows)
    assert ranking["candidate_pair_ranking_accuracy"] == 0.0
    assert ranking["mean_signed_effect_regret"] == 0.4


def test_signed_pair_release_gate_is_calibration_only_and_can_abstain() -> None:
    probability = np.asarray([0.98, 0.02, 0.51, 0.49, 0.90, 0.10])
    target = np.asarray([1, 0, 1, 0, 1, 0])
    gate = candidate_signed.select_signed_pair_release_gate(
        probability, target, minimum_balanced_accuracy=0.90, minimum_coverage=0.5, minimum_per_class=1,
    )
    assert gate["authorized"]
    released = candidate_signed.apply_signed_pair_release_gate(np.asarray([0.99, 0.51, 0.01]), gate)
    assert released.tolist() == [True, False, True]


def test_signed_pair_task_ranking_reports_abstention_without_scoring_unreleased_roots() -> None:
    rows = [_row(1, 1, 0.3, -0.1), _row(2, 0, -0.2, 0.4)]
    ranking = candidate_signed.signed_pair_task_ranking(
        np.asarray([0.9, 0.1]), rows, release_mask=np.asarray([True, False]),
    )
    assert ranking["roots"] == 2
    assert ranking["released_roots"] == 1
    assert ranking["abstained_roots"] == 1
    assert ranking["contexts"] == 2
    assert ranking["candidate_pair_ranking_accuracy"] == 1.0


def test_candidate_bank_pair_rejects_missing_mirror(monkeypatch) -> None:
    monkeypatch.setattr(candidate_signed, "boundary_inward_normal", lambda *_args: np.asarray([1.0, 0.0]))
    monkeypatch.setattr(
        candidate_signed, "candidate_actions", lambda *_args, **_kwargs: np.asarray([[0.18, 0.55], [-0.18, 0.0]]),
    )
    actions, audit = candidate_signed.candidate_bank_action_pair(
        object(), np.zeros(7), inward_magnitude=0.18, tangent_magnitude=0.55,
        deployed_tangent_magnitudes=(0.55,), deployed_contact_normal_magnitudes=(0.18,),
    )
    assert actions == {}
    assert not audit["pair_valid"]


def test_candidate_bank_launcher_is_cuda_only_and_immutable() -> None:
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "run_pusht_candidate_bank_signed_temporal_torque_confirmation_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "run_with_torch_cudnn.sh .venv/bin/python -m" in text
    assert "run_pusht_candidate_bank_signed_temporal_torque_confirmation" in text
    assert "Refusing to reuse immutable protocol output" in text
    assert "GPU_PREFLIGHT_ATTEMPTS" in text
    assert "experiments.gpu_preflight --device cuda" in text
    assert "no CPU fallback will be used" in text
    assert "--contact-inward-magnitude 0.08 --contact-tangent-magnitude 0.16" in text
    assert "--deployed-tangent-magnitudes 0.06 0.10 0.16" in text
    assert "--deployed-contact-normal-magnitudes 0.02 0.04 0.08" in text
