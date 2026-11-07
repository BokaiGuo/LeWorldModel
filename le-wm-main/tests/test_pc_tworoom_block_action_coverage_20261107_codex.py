import ast
import inspect
import json

import numpy as np

from experiments import pc_tworoom_block_action_coverage_20261107_codex as exp
from experiments.pc_input_permissions import file_hash


def test_block_actions_are_legal_reproducible_and_blockwise_constant():
    actions = exp.block_actions(7, 6)
    assert actions.shape == (6, 25, 2)
    assert np.max(np.abs(actions)) <= 1
    np.testing.assert_array_equal(actions, exp.block_actions(7, 6))
    for block in range(5):
        np.testing.assert_array_equal(actions[:, 5 * block:5 * block + 5],
                                      np.repeat(actions[:, 5 * block:5 * block + 1], 5, axis=1))


def test_planning_does_not_access_physical_truth():
    tree = ast.parse(inspect.getsource(exp.plan))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "ORACLE_ROWS" not in names
    assert not ({"proprio", "PHYSICAL", "RESULT"} & attrs)


def test_completed_panel_binds_sealed_evidence():
    plan = json.loads(exp.PLAN.read_text())
    physics = json.loads(exp.PHYSICAL.read_text())
    result = json.loads(exp.RESULT.read_text())
    audit_path = exp.PREFIX.with_name(exp.PREFIX.name + "_audit.json")
    audit = json.loads(audit_path.read_text())
    assert plan["track"] == "primary_rgb"
    assert plan["protocol_sha256"] == file_hash(exp.PROTOCOL)
    assert plan["runner_sha256"] == file_hash(exp.__file__)
    assert physics["plan_sha256"] == file_hash(exp.PLAN)
    assert result["physical_sha256"] == file_hash(exp.PHYSICAL)
    assert audit["plan_sha256"] == file_hash(exp.PLAN)
    assert len(plan["rows"]) == len(physics["rows"]) == len(audit["rows"]) == 10
    for p, q in zip(plan["rows"], physics["rows"], strict=True):
        assert p["row"] == q["row"]
        assert len(p["scores"]) == 300
        assert p["choice"] == int(np.argmin(p["scores"]))
        assert len(q["all_candidates"]) == (300 if p["row"] in exp.ORACLE_ROWS else 0)
