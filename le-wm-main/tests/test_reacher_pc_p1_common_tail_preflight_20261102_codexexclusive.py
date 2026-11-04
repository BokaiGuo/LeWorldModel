import copy
import json

import pytest

from experiments.spectral_reachability import reacher_pc_p1_common_tail_preflight_20261102_codexexclusive as preflight


@pytest.fixture(scope="module")
def metadata():
    return tuple(json.loads(p.read_text()) for p in (preflight.Q_PATH, preflight.SOURCE, preflight.BANK))


def test_real_source_metadata_and_checkpoint():
    result = preflight.material()
    assert [r["seed"] for r in result["roots"]] == [20263160, 20263161]
    assert not result["checkpoint_loaded"] and not result["physical_outcomes_read"]
    assert not result["runtime_authorized"] and not result["P1_admission"]


@pytest.mark.parametrize("fault", ["root", "metric", "fd", "source", "parameters"])
def test_source_mismatch_fails_closed(metadata, fault):
    q, source, bank = copy.deepcopy(metadata)
    if fault == "root":
        source["rows"][1]["root_rgb_sha256"] = "wrong"
    elif fault == "metric":
        q["rows"][0]["goal_metric"]["gram"][0][1] += 100
    elif fault == "fd":
        q["rows"][0]["goal_metric"]["gradient_checks"][0]["relative_error"] = 2
    elif fault == "source":
        q["fixed_bank_sha256"] = "wrong"
    else:
        q["parameter_hash_after"] = "wrong"
    with pytest.raises(ValueError):
        preflight.validate_metadata(q, source, bank)
