import ast
import inspect
import json

from experiments import pc_pointmaze_archive_render_domain_20261107_codex as experiment
from experiments.pc_input_permissions import file_hash


def test_prediction_does_not_read_future_frames_or_real_successors():
    tree = ast.parse(inspect.getsource(experiment.predict))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "archive_path" not in names
    assert "ACTUAL" not in attrs
    assert "RESULT" not in attrs


def test_sealed_plan_and_outputs_bind_current_inputs():
    plan = json.loads(experiment.PLAN.read_text())
    result = json.loads(experiment.RESULT.read_text())
    assert plan["track"] == result["track"] == "official_state_assisted"
    assert plan["protocol_sha256"] == file_hash(experiment.PROTOCOL)
    assert plan["script_sha256"] == file_hash(experiment.__file__)
    assert plan["archive_input_sha256"] == file_hash(experiment.INPUT)
    assert plan["prediction_sha256"] == file_hash(experiment.PRED)
    assert result["plan_sha256"] == file_hash(experiment.PLAN)
    assert [r["root"] for r in result["rows"]] == list(range(1880, 1884))
