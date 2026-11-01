"""Protocol fixtures only; no CPU neural experiment."""
import copy

import pytest
import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_suffix_action_chart import validate_reference, terminal_rgb_batch


def test_terminal_rgb_negative_strides_preserve_values_and_orientation():
    image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)[::-1]
    goal = image[:, ::-1]
    batch = terminal_rgb_batch(image, goal, 10)
    for name, expected in (("pixels", image), ("goal", goal)):
        assert np.array_equal(batch[name][0, 0], expected)
        assert batch[name].flags.c_contiguous
        assert all(stride >= 0 for stride in batch[name].strides)
    assert batch["action"].shape == (1, 1, 10)


def reference():
    seeds = list(range(8))
    report = {"schema": "reacher_d2p_suffix_continuation_confirmation.v1", "status": "completed",
        "gates": dict.fromkeys(("all_declared_roots", "all_execution_gates", "cuda_only", "implementation_unchanged",
            "selection_unchanged", "source_episodes_disjoint", "weights_unchanged"), True),
        "sealed_protocol": {"root_seeds": seeds},
        "rows": [{"seed": seed, "source_episode": seed + 100, "source_start_row": 1, "decision": "progress"} for seed in seeds]}
    selection = {"rows": [{**row, "receipt": {"dispatch": row["decision"]}} for row in reversed(report["rows"])]}
    return report, selection


def test_reference_pairs_by_identity_not_order():
    joined = validate_reference(*reference())
    assert [old["seed"] for old, _ in joined] == list(range(8))
    assert all(old["seed"] == choice["seed"] for old, choice in joined)


@pytest.mark.parametrize("value", [False, 1, "true", None])
def test_reference_rejects_non_strict_gate(value):
    report, selection = reference()
    report["gates"]["cuda_only"] = value
    with pytest.raises(ValueError, match="strict"):
        validate_reference(report, selection)


def test_reference_rejects_missing_required_gate():
    report, selection = reference()
    del report["gates"]["source_episodes_disjoint"]
    with pytest.raises(ValueError, match="strict"):
        validate_reference(report, selection)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "source", "decision", "subset"])
def test_reference_rejects_incomplete_or_mismatched_identity(kind):
    report, selection = reference()
    if kind == "missing":
        selection["rows"].pop()
    elif kind == "duplicate":
        selection["rows"][-1] = copy.deepcopy(selection["rows"][0])
    elif kind == "source":
        selection["rows"][0]["source_start_row"] += 1
    elif kind == "decision":
        selection["rows"][0]["receipt"]["dispatch"] = "other"
    else:
        report["sealed_protocol"]["root_seeds"].pop()
        report["rows"].pop()
        selection["rows"].pop(0)
    with pytest.raises(ValueError):
        validate_reference(report, selection)
