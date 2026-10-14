import json
import os
import subprocess
import sys

import pytest

from experiments.run_unified_research_protocol import (
    PROTOCOL_VERSION,
    build_manifest,
    write_manifest,
)


def test_unified_manifest_contains_one_ordered_study_surface(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = repo / "results" / "unified"
    manifest = build_manifest(
        repo=repo,
        output_root=output,
        source=repo / "source.json",
        split_seeds=(11, 12),
        adapter_seeds=(17, 19),
    )
    assert manifest["protocol_version"] == PROTOCOL_VERSION
    assert manifest["study_scope"] == "covariance_predictor_memory_input_output"
    assert [stage["name"] for stage in manifest["stages"]] == [
        "geometry",
        "controlled_alignment",
        "learned_transfer",
        "contact_control",
        "claim_matrix",
    ]
    transfer = manifest["stages"][2]
    assert len(transfer["commands"]) == 5 * 2 + 5
    assert any("--legacy-memory-input" in command for command in transfer["commands"])
    assert any("--memory-action-mode" in command for command in transfer["commands"])
    assert any("--memory-state-mode" in command for command in transfer["commands"])
    assert any("run_adaptive_time_frequency_memory_audit.py" in " ".join(command) for command in transfer["commands"])
    assert all("theory" not in stage["name"] for stage in manifest["stages"])


def test_unified_manifest_is_immutable(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = repo / "results" / "unified"
    manifest = build_manifest(repo=repo, output_root=output, source=repo / "source.json")
    path = output / "manifest.json"
    digest = write_manifest(path, manifest)
    payload = json.loads(path.read_text())
    assert payload["manifest_sha256"] == digest
    with pytest.raises(FileExistsError, match="refusing to overwrite frozen unified manifest"):
        write_manifest(path, manifest)


def test_unified_runner_script_help_is_executable_from_repo_root():
    repo = os.path.dirname(os.path.dirname(__file__))
    completed = subprocess.run(
        [sys.executable, "experiments/run_unified_research_protocol.py", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--output-root" in completed.stdout
