"""RAM mutations: no model, simulator, or research-result fixture fabrication."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import agent_p2_control_evidence_resolver_20261028_v1 as m


def fixture():
    task = {"source_slot": 0, "task_id": "task0", "blind_sha256": "blind0",
            "blind_source_sha256": "source0", "history_semantics": "actual moving",
            "with_target": False, "task": {"model_sha256": "weights", "model_config_sha256": "config",
            "packet_logit_cap": 1.6, "control_dim": 2, "packet_steps": 5, "representation": "native192",
            "action_mean": [0.1, 0.2], "action_std": [1.0, 1.0],
            "root_context": {"states": [[1.0] * 192] * 3, "actions": [[0.0] * 10] * 2},
            "root_raw_action_history": [[0.1, 0.0]] * 5 + [[0.0, 0.1]] * 5,
            "latent_dim": 192, "projection": {"type": "identity"}, "horizon": 2,
            "goal": [1.0] * 192, "goal_source_sha256": "goalsha"},
            "task_contract": {"model_bindings": {"/weights": "weights"},
            "normalization": {"mean": [0.1, 0.2], "std": [1.0, 1.0], "training_provenance_verified": False},
            "observed_steps": [0, 5, 10], "pixel_sha256": ["a", "b", "c", "d"], "metric": "identity_native192_no_fit"}}
    tc = hashlib.sha256(json.dumps(task["task_contract"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    task["task"]["task_contract_sha256"] = tc
    ref = {key: task[key] for key in ("source_slot", "task_id", "blind_sha256")}
    ref.update(arm="beam", task_contract_sha256=tc, task_export="/task.gz", task_sha256="tasksha")
    return task, ref


def test_complete_task_components():
    task, ref = fixture()
    evidence, missing = m.task_evidence(ref, task, Path("/task.gz"), "tasksha", [])
    assert missing == []
    assert set(evidence) == set(m.TASK_FIELDS)
    assert evidence["observation_spec"]["/with_target"]["value"] is False


@pytest.mark.parametrize("field", ["source_slot", "task_id", "blind_sha256"])
def test_cross_identity_rejected(field):
    task, ref = fixture()
    ref[field] = "wrong"
    with pytest.raises(ValueError, match="join mismatch"):
        m.task_evidence(ref, task, Path("/task.gz"), "s", [])


def test_missing_ref_identity_refused():
    task, ref = fixture()
    del ref["blind_sha256"]
    with pytest.raises(ValueError, match="incomplete"):
        m.task_evidence(ref, task, Path("/task.gz"), "s", [])


def test_contract_content_mutation_rejected():
    task, ref = fixture()
    task["task_contract"]["observed_steps"] = [0, 1, 2]
    with pytest.raises(ValueError, match="content digest"):
        m.task_evidence(ref, task, Path("/task.gz"), "s", [])


def test_missing_goal_not_zero():
    task, ref = fixture()
    del task["task"]["goal"]
    evidence, missing = m.task_evidence(ref, task, Path("/task.gz"), "s", [])
    assert "/task/goal" not in evidence["goal_identity"]
    assert missing[0]["field"] == "goal_identity"


def test_missing_path_does_not_search_by_sha(monkeypatch):
    monkeypatch.setattr(m, "load", lambda *args: pytest.fail("must not search"))
    with pytest.raises(ValueError, match="missing explicit link"):
        m.pair({"prepare_sha256": "x"}, Path("/report"), "r", "", "prepared", "prepare_sha256", [])


def test_link_preserves_hash_and_path_pointers(monkeypatch):
    monkeypatch.setattr(m, "load", lambda p, s: {"ok": True})
    doc, path, sha, chain = m.pair({"prepared": "/p", "prepare_sha256": "s"}, "/r", "r", "", "prepared", "prepare_sha256", [])
    assert path == Path("/p") and sha == "s"
    assert chain[0]["hash_from"]["pointer"] == "/prepare_sha256"


def test_load_rejects_sha_and_relative_path():
    with pytest.raises(ValueError, match="absolute"):
        m.load("relative", "sha")
    with pytest.raises(ValueError, match="SHA drift"):
        m.load(Path(m.__file__).absolute(), "0" * 64)


def test_partial_rows_never_shrink_denominator(monkeypatch):
    task, ref = fixture()
    second = copy.deepcopy(ref)
    second["source_slot"] = 1
    del second["task_export"]
    report = {"rows": [ref, second], "P1_admission": False}
    def load(path, expected):
        return report if str(path) == "/report" else task
    monkeypatch.setattr(m, "load", load)
    result = m.resolve_asset("no_composition", Path("/report"), "r")
    assert len(result["rows"]) == 2 and len(result["fixed_rows"]) == 2
    assert result["controls"]["goal_identity"] is None
    assert result["controls"]["resource_axes"] is None
    assert any(i.get("row") == 1 for i in result["missing"])


def test_duplicate_root_arm_refused(monkeypatch):
    _, ref = fixture()
    monkeypatch.setattr(m, "load", lambda *a: {"P1_admission": False, "rows": [ref, ref]})
    with pytest.raises(ValueError, match="duplicate"):
        m.resolve_asset("no_composition", Path("/report"), "r")


def test_v3_catalog_exact_binding_compatibility():
    value = {"assets": {"beam": {"controls": {"goal_identity": [1], "resource_axes": None}}}}
    binding = m.catalog(Path("/output"), "sha", value)["beam"]["goal_identity"]
    assert set(binding) == {"path", "sha256", "pointer"}
    assert m.contract.pointer(value, binding["pointer"]) == [1]
    assert "resource_axes" not in m.catalog(Path("/output"), "sha", value)["beam"]


def test_v3_real_evidence_reader_compatibility(monkeypatch):
    value = {"assets": {"beam": {"controls": {"goal_identity": [1.0] * 192}}}}
    binding = m.catalog(Path("/output"), "sha", value)["beam"]["goal_identity"]
    monkeypatch.setattr(m.contract, "sha", lambda p: "sha")
    monkeypatch.setattr(m.contract, "read", lambda p: value)
    assert m.contract.evidence_value(binding) == [1.0] * 192
    binding["sha256"] = "drift"
    with pytest.raises(ValueError, match="drift"):
        m.contract.evidence_value(binding)


def test_saved_three_assets_and_all_rows():
    value = m.build()
    assert {n: len(a["rows"]) for n, a in value["assets"].items()} == {
        "kinodynamic": 2, "no_composition": 2, "sampling": 4}
    for name in ("kinodynamic", "no_composition"):
        assert all(value["assets"][name]["controls"][f] is not None for f in m.TASK_FIELDS)
    assert value["assets"]["sampling"]["controls"]["task_identity"] is None
    assert not value["matched"] and value["composition_benefit"] is None
    assert not value["P1_admission"] and not value["P2_admission"]
