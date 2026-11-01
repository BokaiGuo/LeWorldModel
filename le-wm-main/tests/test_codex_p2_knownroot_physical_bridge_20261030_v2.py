import copy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import (
    codex_p2_knownroot_physical_bridge_20261030_v2 as bridge,
)


def bindings(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"sealed source")
    return {str(source): bridge.file_sha(source)}


def plan(root="r" * 64):
    return {
        "root_sha256": root,
        "commands": [[0.1, 0.2] for _ in range(10)],
        "selection_uses_physical_outcome": False,
    }


def case(root="r" * 64):
    return {
        "root_sha256": root, "task_id": "task-0", "task_sha256": "t" * 64,
        "goal_sha256": "g" * 64, "warmup_sha256": "w" * 64,
        "renderer": "native-rgb", "action_domain": {"shape": [2], "dtype": "float32"},
    }


def test_existing_cuda_plan_is_rejected_without_physical_binding(tmp_path):
    with pytest.raises(ValueError, match="missing physical binding"):
        bridge.compile_packet(plans=[plan()], physical_cases=[{"root_sha256": "r" * 64}],
                              source_bindings=bindings(tmp_path))


def test_packet_binds_root_and_actions_and_is_exclusive(tmp_path):
    packet = bridge.compile_packet(plans=[plan()], physical_cases=[case()],
                                   source_bindings=bindings(tmp_path))
    path = tmp_path / (bridge.NAME + "_packet.json")
    assert len(packet["plans"][0]["commands"]) == 10
    assert bridge.save_packet(path, packet)
    with pytest.raises(FileExistsError):
        bridge.save_packet(path, packet)


class Env:
    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.step_index = 0
        self.closed = False

    def reset(self):
        return {"state": "initial"}

    def step(self, action):
        if self.fail_at == self.step_index:
            raise RuntimeError("synthetic physical failure")
        self.step_index += 1
        return {"step": self.step_index, "action": action}

    def render(self):
        return b"rgb-" + bytes([self.step_index])

    def close(self):
        self.closed = True


def test_synthetic_physical_run_and_strict_consume(tmp_path):
    packet = bridge.compile_packet(plans=[plan()], physical_cases=[case()],
                                   source_bindings=bindings(tmp_path))
    output = tmp_path / (bridge.NAME + "_run")
    report = bridge._fixture_only_run_physical(packet=packet, output=output, factory=lambda spec: Env(),
                                               authorization=bridge.AUTH)
    assert report["status"] == "physical_recorded_not_admitted"
    consumed = bridge.consume(output, bridge.file_sha(output / "report.json"))
    assert consumed == {"status": "physical_recorded_not_admitted", "rows": 1,
                        "errors": [], "P2_admission": False}


def test_failure_preserves_partial_report_and_journal(tmp_path):
    packet = bridge.compile_packet(plans=[plan()], physical_cases=[case()],
                                   source_bindings=bindings(tmp_path))
    output = tmp_path / (bridge.NAME + "_failed")
    report = bridge._fixture_only_run_physical(packet=packet, output=output,
                                               factory=lambda spec: Env(fail_at=2), authorization=bridge.AUTH)
    assert report["status"] == "partial_or_failed"
    assert report["rows"][0]["status"] == "failed_partial"
    assert (output / "journal.jsonl").is_file()
    tampered = json.loads((output / "report.json").read_text())
    tampered["rows"][0]["actions"].append(["tampered"])
    (output / "report.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="saved row/journal"):
        bridge.consume(output, bridge.file_sha(output / "report.json"))


def test_tampered_packet_selection_is_not_silently_accepted(tmp_path):
    packet = bridge.compile_packet(plans=[plan()], physical_cases=[case()],
                                   source_bindings=bindings(tmp_path))
    altered = copy.deepcopy(packet)
    altered["plans"][0]["commands"][0][0] = 9.9
    assert bridge.digest({k: v for k, v in altered.items()
                          if k != "packet_sha256"}) != altered["packet_sha256"]


def test_prototype_native_entry_is_blocked_before_factory(tmp_path):
    calls = []
    with pytest.raises(RuntimeError, match="BLOCKED"):
        bridge.run_physical(packet={}, output=tmp_path / "output",
                            factory=lambda _: calls.append(1), authorization=bridge.AUTH)
    assert not calls
