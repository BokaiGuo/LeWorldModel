"""Offline real-source preflight and synthetic class-declaration tests."""
import copy

import pytest

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_h15_preproduction_manifest_20261030_codex as pre
from test_p2_h15_neural_ledger_adapter_20261030_codex import contract


def inputs(root=0):
    info = pre.preflight()
    task = pre.host.bridge.build_bridge()["roots"][root]["task"]
    c = contract(task)
    files = pre.host.historical_inventory()
    path = next(p for p in files if p.endswith("checkpoint_compat.py"))
    # Deliberately synthetic: pinning this claim cannot prove an actual class graph.
    evidence = {"model_hashes": info["model_hashes"],
                "classes": [{"name": "", "class_name": "synthetic.Model",
                             "path": path, "sha256": files[path]}]}
    return c, core.digest(c), core.digest(task), evidence, core.digest(evidence)


def test_real_preflight_rebuild():
    value = pre.preflight()
    assert value["inventory_file_count"] == 398
    assert value["inventory_sha256"] == "6f5c971d17aeedf7aeb8f47a6232169ffa0dd2c0abfac2dc028073b49d920079"
    assert len({x["task_sha256"] for x in value["tasks"]}) == 2
    assert value["tasks"][0]["checkpoint_file_sha256"] != value["model_hashes"]["original_state_dict"]
    assert not value["class_graph_available"] and not value["ready_to_execute"]


@pytest.mark.parametrize("root", [0, 1])
def test_candidate_is_only_a_pinned_declaration(root):
    args = inputs(root)
    value = pre.candidate(root, *args)
    assert pre.verify_candidate(value, core.digest(value), *args) == value
    assert not value["class_graph_completeness_verified"]
    assert not value["class_evidence_authenticity_verified"]
    assert not value["ready_to_execute"] and not value["P2_admission"]
    assert value["joint_manifest"]["base"]["model_hashes"] == pre.preflight()["model_hashes"]


@pytest.mark.parametrize("field", ["contract", "task", "evidence", "buffers", "source", "graph", "root"])
def test_reject_bad_bindings(field):
    c, cs, ts, evidence, es = inputs()
    root = 0
    if field == "contract":
        cs = "0" * 64
    elif field == "task":
        ts = pre.preflight()["tasks"][1]["task_sha256"]
    elif field == "evidence":
        es = "0" * 64
    elif field == "buffers":
        evidence = copy.deepcopy(evidence)
        evidence["model_hashes"]["all_buffers"] = "0" * 64
        es = core.digest(evidence)
    elif field == "source":
        evidence = copy.deepcopy(evidence)
        evidence["classes"][0]["sha256"] = "0" * 64
        es = core.digest(evidence)
    elif field == "graph":
        evidence = copy.deepcopy(evidence)
        evidence["classes"] = []
        es = core.digest(evidence)
    else:
        root = 1
    with pytest.raises(ValueError):
        pre.candidate(root, c, cs, ts, evidence, es)


def test_inventory_drift_missing_and_symlink(tmp_path):
    file = tmp_path / "source.py"
    file.write_text("first")
    digest = pre.host.bridge.sha(file)
    files = {str(file): digest}
    assert pre.checked_inventory(files) == core.digest(files)
    file.write_text("changed")
    with pytest.raises(ValueError, match="inventory file"):
        pre.checked_inventory(files)
    link = tmp_path / "link.py"
    link.symlink_to(file)
    with pytest.raises(ValueError, match="inventory file"):
        pre.checked_inventory({str(link): pre.host.bridge.sha(file)})
    with pytest.raises(ValueError, match="inventory file"):
        pre.checked_inventory({str(tmp_path / "missing"): digest})


def test_report_anchor_tamper_rejected(monkeypatch):
    monkeypatch.setattr(pre, "REPORT_SHA", "0" * 64)
    with pytest.raises(ValueError):
        pre.preflight()
