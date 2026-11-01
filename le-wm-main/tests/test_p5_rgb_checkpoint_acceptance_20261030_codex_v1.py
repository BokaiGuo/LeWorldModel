import copy
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p5_rgb_checkpoint_acceptance_20261030_codex_v1 as acc

BASE_SHA = "853f22a07ff01693422e242bf032619cfe7d96548a1740b70062d92f69ccf126"
GPU = "GPU-12345678-1234-1234-1234-123456789abc"


class FakeBackend:
    def metadata(self):
        return {"device": "cuda:0", "device_name": "synthetic only", "torch_version": "fake",
                "cuda_version": "fake", "strict_load": True, "frozen": True, "eval": True,
                "checkpoint_sha256": acc.abi.bridge.FILES["checkpoint"][1]}

    def encode(self, values):
        return np.zeros((1, 6), dtype=np.float32)

    def step(self, values, index):
        return np.zeros((1, 6), dtype=np.float32)


@pytest.fixture(scope="module")
def base():
    p = acc.abi.bridge.ROOT / ("results/" + acc.NAME + "_BASELINE02.json")
    raw = p.read_bytes()
    assert acc.abi.bridge.digest(raw) == BASE_SHA
    return acc.abi.bridge.read_json(raw)


@pytest.fixture
def evidence(tmp_path, monkeypatch, base):
    prepared = acc.runner.load_prepared()
    monkeypatch.setattr(acc.abi, "materialize", lambda row: {})
    journal = acc.runner.Journal(tmp_path / "journal")
    try:
        assert acc.runner.run_protocol(prepared, FakeBackend(), journal, 3600) == "complete"
    finally:
        journal.close()
    jb = (tmp_path / "journal" / "events.jsonl").read_bytes()
    run_name = acc.runner.NAME + "_TEST"
    command = [str(acc.abi.bridge.ROOT / ".venv/bin/python"), "-m",
               "experiments.spectral_reachability.p5_rgb_cuda_runner_20261030_codex_v1",
               "--authorization", acc.runner.AUTH, "--device-index", "0",
               "--max-seconds", "60", "--output-name", run_name]
    snapshot = {"uuid": GPU, "free_mib": 1000, "total_mib": 2000,
                "mig_mode": "Disabled", "compute_mode": "Default"}
    events = [{"event": "created", "run_name": run_name, "gpu_uuid": GPU,
               "prepared_sha256": acc.runner.PREP_SHA, "gate_sha256": acc.sup.GATE_SHA,
               "retry": False, "min_free_mib": 100, "min_total_mib": 2000,
               "hard_seconds": 120, "runner_seconds": 60},
              {"event": "admitted", "snapshot": snapshot},
              {"event": "launch_attempt", "command": command},
              {"event": "launched", "pid": 123},
              {"event": "finished", "status": "complete_self_report_only", "retry": False,
               "execution": {"status": "exited", "returncode": 0, "pid": 123},
               "journal": {"sha256": acc.abi.bridge.digest(jb)}}]
    sb = b"".join(acc.abi.bridge.canonical(e) for e in events)
    state = copy.deepcopy(base["state"])
    for record in state["parameters"].values():
        record["device"] = "cuda:0"
    for record in state["normalization"].values():
        record["device"] = "cuda:0"
    bundle = {"schema": acc.NAME + ".runtime_bundle",
              "baseline_sha256": acc.abi.bridge.digest(acc.abi.bridge.canonical(base)),
              "supervisor_sha256": acc.abi.bridge.digest(sb), "journal_sha256": acc.abi.bridge.digest(jb),
              "capture_producer_sha256": "a" * 64, "before": state, "after": copy.deepcopy(state)}
    return bundle, sb, jb


def test_real_checkpoint_baseline(base):
    code = ("from experiments.spectral_reachability import "
            "p5_rgb_checkpoint_acceptance_20261030_codex_v1 as a; "
            "b=a.baseline(); assert a.abi.bridge.digest(a.abi.bridge.canonical(b)) == "
            + repr(BASE_SHA))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-c", code], cwd=acc.abi.bridge.ROOT, env=env, check=True)
    assert base["state_dict_entries"] == 26 and base["parameter_count"] == 93842
    assert base["actual_neural_forward_calls"] == base["actual_cuda_calls"] == 0
    assert base["runtime_evidence_available"] is False


def test_synthetic_complete_is_only_self_report(base, evidence):
    bundle, sb, jb = evidence
    report = acc.accept(base, bundle, sb, jb)
    assert report["status"] == "self_report_consistency_only_not_scientific"
    assert report["condition_counts"]["off"]["complete"] == 82
    assert report["condition_counts"]["on"]["complete"] == 82
    assert not report["P5_admission"]


@pytest.mark.parametrize("mutation", ["changed_parameter", "unchanged_wrong_parameter",
                                      "module_training", "precision", "autocast", "missing_snapshot"])
def test_reject_mutated_states(base, evidence, mutation):
    bundle, sb, jb = evidence
    name = next(iter(bundle["before"]["parameters"]))
    if mutation == "changed_parameter":
        bundle["after"]["parameters"][name]["sha256"] = "0" * 64
    elif mutation == "unchanged_wrong_parameter":
        for side in ("before", "after"):
            bundle[side]["parameters"][name]["sha256"] = "0" * 64
    elif mutation == "module_training":
        for side in ("before", "after"):
            bundle[side]["modules"][0]["training"] = True
    elif mutation == "precision":
        bundle["after"]["precision"]["matmul_precision"] = "medium"
        if bundle["before"]["precision"]["matmul_precision"] == "medium":
            bundle["after"]["precision"]["matmul_precision"] = "highest"
    elif mutation == "autocast":
        for side in ("before", "after"):
            bundle[side]["precision"]["cuda_autocast_enabled"] = True
    else:
        del bundle["after"]
    with pytest.raises(ValueError):
        acc.accept(base, bundle, sb, jb)


def test_reject_partial_supervision_and_wrong_journal(base, evidence):
    bundle, sb, jb = evidence
    with pytest.raises(ValueError, match="binding"):
        acc.accept(base, bundle, sb, jb + b"\n")
    changed = sb.replace(b"complete_self_report_only", b"partial")
    bundle["supervisor_sha256"] = acc.abi.bridge.digest(changed)
    with pytest.raises(ValueError, match="completion"):
        acc.accept(base, bundle, changed, jb)
