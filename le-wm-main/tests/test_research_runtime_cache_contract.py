import json

from experiments.research_runtime.cache_contract import build_contract


def test_shell_contract_changes_with_declared_input(tmp_path):
    source = tmp_path / "runner.sh"
    source.write_text("echo one\n")
    first = build_contract(
        repo=tmp_path,
        protocol_version="v1",
        config={"env": "pusht", "policy": "demo", "device": "cuda"},
        inputs=[source],
    )
    source.write_text("echo two\n")
    second = build_contract(
        repo=tmp_path,
        protocol_version="v1",
        config={"env": "pusht", "policy": "demo", "device": "cuda"},
        inputs=[source],
    )
    assert first["run_identity"]["sha256"] != second["run_identity"]["sha256"]
