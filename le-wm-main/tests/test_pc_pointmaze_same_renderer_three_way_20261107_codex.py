import ast
import inspect

from experiments import pc_pointmaze_same_renderer_three_way_20261107_codex as run


def test_prediction_stage_does_not_open_physical_or_successor_artifacts():
    tree = ast.parse(inspect.getsource(run.predict))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "physical" not in names
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "FRAMES" not in attributes
    assert "PHYSICAL" not in attributes
    assert "PREFIX" not in attributes
    assert "MODEL_METADATA" in names
    assert run.ACTIONS == (-.1, -.03, .03, .1)
