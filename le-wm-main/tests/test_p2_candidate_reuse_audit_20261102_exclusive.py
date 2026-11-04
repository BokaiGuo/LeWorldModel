from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import p2_candidate_reuse_audit_20261102_exclusive as audit


@pytest.fixture(scope="module")
def data():
    pair = json.loads((audit.NEW / "pair.json").read_bytes())
    case = json.loads(audit.PANEL.read_bytes())["cases"][0]
    old = [json.loads((audit.prior.reuse.OLD_RUN / f"branch_{i}.json").read_bytes())
           for i in range(2)]
    return pair, case, old


def test_real_candidate_identity(data):
    result = audit.compare(*data)
    assert [x["candidate_action_and_cost_matches"] for x in result["rows"]] == [16, 16]
    assert result["new_context_sha256"] != result["prior_context_sha256"]
    assert set(result["prior_context_keys"]) - set(result["new_context_keys"]) == {
        "action_dim", "latent_dim", "context_length"}


def test_changed_input_candidate_or_score_rejected(data):
    pair, case, old = data
    changed = deepcopy(pair)
    changed["context"]["latent_history"][0][0] += 1
    with pytest.raises(ValueError, match="latent_history"):
        audit.compare(changed, case, old)
    changed = deepcopy(pair)
    changed["branches"][0]["candidates"][0]["raw_float32_hex"] = "00" * 80
    with pytest.raises(ValueError, match="candidate mismatch"):
        audit.compare(changed, case, old)
    changed = deepcopy(pair)
    changed["branches"][1]["candidates"][0]["cost"] += 1
    with pytest.raises(ValueError, match="candidate mismatch"):
        audit.compare(changed, case, old)
