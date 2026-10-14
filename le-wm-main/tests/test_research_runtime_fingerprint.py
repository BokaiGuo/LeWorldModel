from experiments.research_runtime.fingerprint import (
    build_run_identity,
    fingerprint_file,
    fingerprint_inputs,
)


def test_file_fingerprint_changes_when_input_changes(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text('{"x": 1}\n')
    first = fingerprint_file(path)
    path.write_text('{"x": 2}\n')
    second = fingerprint_file(path)
    assert first["sha256"] != second["sha256"]
    assert fingerprint_inputs((path,))["sha256"] != ""


def test_missing_input_is_explicit_and_identity_is_content_based(tmp_path):
    missing = tmp_path / "missing.ckpt"
    lock = fingerprint_inputs((missing,))
    assert lock["files"][0]["exists"] is False
    left = build_run_identity(
        protocol_version="v1", config={"seed": 1}, input_lock=lock, code={"commit": "a"}
    )
    right = build_run_identity(
        protocol_version="v1", config={"seed": 2}, input_lock=lock, code={"commit": "a"}
    )
    assert left["sha256"] != right["sha256"]
