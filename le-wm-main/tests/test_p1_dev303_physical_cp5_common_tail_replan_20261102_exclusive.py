import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_dev303_physical_cp5_common_tail_replan_20261102_exclusive as subject,
)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fixture():
    source = {
        "seed": subject.bridge.SEED,
        "queue_sha256": subject.bridge.QUEUE_SHA,
        "source_receipt_sha256": subject.bridge.RECEIPT_SHA,
    }
    roles = []
    for index, role in enumerate(subject.ROLES):
        actions = np.zeros((5, 2), dtype="<f4")
        actions[0, 0] = index / 20
        roles.append({
            "role": role,
            "candidate_index": index,
            "candidate_label": role,
            "actions": actions.tolist(),
            "actions_sha256": sha(actions.tobytes()),
        })
    contract = {
        "schema": subject.bridge.NAME + ".contract.v1",
        "source": source,
        "selection_input": "rgb_only",
        "feedback_boundary": 5,
        "physical_execution_authorized": False,
        "independent_confirmation": False,
        "P1_admission": False,
        "roles": roles,
    }
    return source, contract


def test_cpu_contract_seals_common_tail_replan_boundary():
    source, selector = fixture()
    value = subject.build_contract(source, selector)
    assert value["schema"] == subject.NAME + ".contract.v1"
    assert value["status"] == "cpu_contract_sealed_waiting_for_cuda"
    assert [row["role"] for row in value["roles"]] == list(subject.ROLES)
    assert all(row["tail_policy"] == "cp5_common_tail_replan_from_actual_rgb"
               for row in value["roles"])
    assert value["common_tail"]["selection_after_cp5"] == "forbidden"
    assert value["same_replanning_budget_required"] is True
    assert value["P1_admission"] is False


def test_contract_rejects_selector_or_physical_boundary_drift():
    source, selector = fixture()
    selector["physical_execution_authorized"] = True
    with pytest.raises(ValueError, match="selector/bridge boundary"):
        subject.build_contract(source, selector)


def test_contract_rejects_action_tampering():
    source, selector = fixture()
    selector["roles"][1]["actions"][0][0] = 2.0
    with pytest.raises(ValueError, match="bounded finite prefix"):
        subject.build_contract(source, selector)


def test_write_new_is_exclusive(tmp_path):
    path = tmp_path / "contract.json"
    digest = subject.write_new(path, {"P1_admission": False})
    assert digest == sha(path.read_bytes())
    with pytest.raises(FileExistsError):
        subject.write_new(path, {"P1_admission": False})
