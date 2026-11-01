"""Full-coordinate fixed-rule stored-data tests, not neural experiments."""
import copy

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_packet_causal_residual import summarize


def fixture():
    predictions, observations = {"rows": []}, {"rows": []}
    for seed in range(8):
        branches = {f: {"raw": [[1.] * 192 for _ in range(4)], "terminal_unit": [[1.25] * 192 for _ in range(4)],
            "recurrent_unit": [[1.25] * 192 for _ in range(4)]} for f in ("native_dispatch", "zero")}
        observed = {f: {"packets": [{"vectors": {"free": [1.] * 192, "observed": [2.] * 192, "goal": [0.] * 192}} for _ in range(4)]} for f in branches}
        predictions["rows"].append({"seed": seed, "goal_vector": [0.] * 192, "innovation_vector": [.25] * 192, "branches": branches})
        observations["rows"].append({"seed": seed, "branches": observed})
    return predictions, observations


def test_fixed_rule_complete_arithmetic_and_identity_join():
    predicted, observed = fixture()
    result = summarize(predicted, observed)
    assert result["mean_latent_l2_by_packet"]["raw"] == pytest.approx([192 ** .5] * 4)
    assert result["mean_latent_l2_by_packet"]["terminal_unit"] == pytest.approx([.75 * 192 ** .5] * 4)
    predicted["rows"].reverse()
    observed["rows"].reverse()
    assert summarize(predicted, observed) == result


@pytest.mark.parametrize("kind", ["partial", "duplicate", "wrong_seed", "branch", "variant", "shape", "nan", "delta_nan", "goal", "baseline", "coefficient", "first_step"])
def test_invalid_fixed_rule_receipt_rejected(kind):
    predicted, observed = copy.deepcopy(fixture())
    row = predicted["rows"][0]
    branch = row["branches"]["native_dispatch"]
    if kind == "partial":
        predicted["rows"].pop()
    elif kind == "duplicate":
        predicted["rows"][-1]["seed"] = 0
    elif kind == "wrong_seed":
        observed["rows"][-1]["seed"] = 9
    elif kind == "branch":
        row["branches"].pop("zero")
    elif kind == "variant":
        branch.pop("terminal_unit")
    elif kind == "shape":
        branch["raw"][0].pop()
    elif kind == "nan":
        branch["recurrent_unit"][1][0] = float("nan")
    elif kind == "delta_nan":
        row["innovation_vector"][0] = float("nan")
    elif kind == "goal":
        row["goal_vector"][0] += 1e-12
    elif kind == "baseline":
        branch["raw"][0][0] += 1e-12
    elif kind == "coefficient":
        branch["terminal_unit"][1][0] += .01
    else:
        branch["recurrent_unit"][0][0] += 1e-12
    with pytest.raises(ValueError):
        summarize(predicted, observed)
