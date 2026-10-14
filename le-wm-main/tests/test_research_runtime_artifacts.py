import json

from experiments.research_runtime.artifacts import artifact_is_valid, atomic_write_json


def test_atomic_json_write_and_schema_validation(tmp_path):
    path = tmp_path / "artifact.json"
    atomic_write_json(path, {"schema_version": 1, "value": 3})
    assert artifact_is_valid(path, required_keys=("schema_version", "value")) == (True, "valid")
    assert json.loads(path.read_text())["value"] == 3


def test_corrupt_or_schema_incomplete_artifact_is_not_reusable(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("not json")
    assert artifact_is_valid(path) == (False, "invalid_json")
    path.write_text("{}")
    valid, reason = artifact_is_valid(path, required_keys=("schema_version",))
    assert not valid
    assert reason == "missing_keys:schema_version"
