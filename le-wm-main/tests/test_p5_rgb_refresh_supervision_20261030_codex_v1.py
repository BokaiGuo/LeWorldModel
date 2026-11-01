import copy
import io
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p5_rgb_refresh_supervision_20261030_codex_v1 as m


def prepared():
    rows = []
    for i in range(164):
        steps = []
        for h in range(1, 6):
            steps.append({"horizon": h, "previous_plan_step": h + 3, "current_plan_step": h + 4,
                          "target_plan_step": h + 5, "previous_rgb": {"model_input_chw_float32_sha256": "prev"},
                          "current_rgb": {"model_input_chw_float32_sha256": "current"}, "action_sha256": "action"})
        rows.append({"index": i, "root_slot": i // 82, "condition": ("off", "on")[(i // 41) % 2],
                     "task_id": str(i), "selection_digest": str(i), "steps": steps})
    return {"rows": rows, "pairs": [{}] * 80}


def state(device):
    record = {"shape": [1], "dtype": "float32", "sha256": "hash", "device": device}
    return {"modules": [{"name": "", "class": "Model", "training": False}],
            "parameters": {"x": {**record, "requires_grad": False}},
            "normalization": {"_state_mean": record, "_state_std": record},
            "precision": {"default_dtype": "torch.float32", "matmul_precision": "highest",
                          "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": True,
                          "cuda_autocast_enabled": False}}


class Backend:
    model = torch = None
    def metadata(self):
        return {"device": "cuda:0", "device_name": "synthetic", "torch_version": "synthetic",
                "cuda_version": "synthetic", "strict_load": True, "frozen": True, "eval": True,
                "checkpoint_sha256": m.bridge.FILES["checkpoint"][1]}
    def encode(self, values):
        return np.zeros((1, 6), dtype="<f4")
    def step(self, values, index):
        return np.ones((1, 6), dtype="<f4")


def make_run(tmp_path):
    p, base = prepared(), {"state": state("cpu")}
    directory = tmp_path / "run"
    journal = m.runner.Journal(directory)
    try:
        backend = m.InstrumentedBackend(Backend(), journal, directory, base, capture=lambda *x: state("cuda:0"))
        assert m.refresh.execute(p, m.PREP_SHA, backend, journal, 10, clock=lambda: 0,
                                 load=lambda step: None) == "complete"
    finally:
        journal.close()
    return p, base, directory


def test_complete_strict_review_and_snapshots(tmp_path):
    p, base, directory = make_run(tmp_path)
    result = m.review_run(directory, base, p, m.source_sha())
    assert result["encode_returned"] == result["predict_returned"] == 820
    assert len(result["rows"]) == 164
    assert result["condition_counts"]["off"] == {"denominator": 82, "complete": 82}


def rechain(events):
    prev = "0" * 64
    for i, event in enumerate(events):
        event["seq"], event["prev"] = i, prev
        event.pop("hash", None)
        event["hash"] = m.bridge.digest(m.bridge.canonical(event))
        prev = event["hash"]
    return b"".join(m.bridge.canonical(e) for e in events)


@pytest.mark.parametrize("mutation", ["missing", "order", "identity", "output", "extra", "start"])
def test_journal_fail_closed_even_if_rehashed(tmp_path, mutation):
    p, _, directory = make_run(tmp_path)
    raw = (directory / "events.jsonl").read_bytes()
    events = [m.bridge.read_json(line) for line in raw.splitlines()]
    if mutation == "missing":
        events.pop(4)
    elif mutation == "order":
        events[1], events[3] = events[3], events[1]
    elif mutation == "identity":
        events[2]["current_plan_step"] = 9
    elif mutation == "output":
        events[2]["output"]["values"][0] = 2
    elif mutation == "extra":
        events[2]["future_label"] = 1
    else:
        events[0]["prepared_sha256"] = "wrong"
    changed = rechain(events)
    with pytest.raises(ValueError):
        m.review_journal(p, changed, m.bridge.digest(changed))


@pytest.mark.parametrize("kind", ["snapshot", "precision", "anchor", "external_sha"])
def test_snapshot_and_external_sha_rejected(tmp_path, kind):
    p, base, directory = make_run(tmp_path)
    if kind != "external_sha":
        path = directory / "state_after.json"
        obj = m.bridge.read_json(path.read_bytes())
        if kind == "snapshot":
            obj["state"]["parameters"]["x"]["sha256"] = "wrong"
        elif kind == "precision":
            obj["state"]["precision"]["cuda_autocast_enabled"] = True
        else:
            obj["anchor_seq"] -= 1
        path.write_bytes(m.bridge.canonical(obj))
    with pytest.raises(ValueError):
        m.review_run(directory, base, p, m.source_sha(), "bad" if kind == "external_sha" else None)


def supervisor_events():
    uuid = "GPU-12345678-1234-1234-1234-123456789abc"
    created = {"event": "created", "producer_sha256": "source", "baseline_sha256": m.wiring.BASE_SHA,
               "prepared_sha256": m.PREP_SHA, "gpu_uuid": uuid, "run_suffix": "X",
               "min_free_mib": 100, "min_total_mib": 200, "hard_seconds": 20, "runner_seconds": 10, "retry": False}
    snap = {"uuid": uuid, "total_mib": 200, "free_mib": 100, "compute_mode": "Default", "mig_mode": "Disabled"}
    return [created, {"event": "admitted", "snapshot": snap},
            {"event": "launch_attempt", "command": m.command("X", 10, "source")},
            {"event": "launched", "pid": 123}, {"event": "finished", "retry": False,
            "status": "complete_self_report_only", "review": {"x": 1},
            "execution": {"status": "exited", "returncode": 0, "elapsed_seconds": 5, "pid": 123}}]


@pytest.mark.parametrize("change", ["memory", "timeout", "retry", "command", "uuid", "pid", "mig"])
def test_supervisor_denies_invalid(change):
    events = supervisor_events()
    if change == "memory":
        events[1]["snapshot"]["free_mib"] = 99
    elif change == "timeout":
        events[-1]["execution"]["status"] = "timeout"
    elif change == "retry":
        events[0]["retry"] = True
    elif change == "command":
        events[2]["command"].append("--extra")
    elif change == "uuid":
        events[1]["snapshot"]["uuid"] = "GPU-other"
    elif change == "pid":
        events[-1]["execution"]["pid"] = 456
    else:
        events[1]["snapshot"]["mig_mode"] = "Enabled"
    with pytest.raises(ValueError):
        m.validate_supervisor(events, "X", "source")


def test_supervisor_valid():
    assert m.validate_supervisor(supervisor_events(), "X", "source") == {"x": 1}


def test_actual_external_timeout_cpu_only(tmp_path):
    with (tmp_path / "out").open("wb") as out, (tmp_path / "err").open("wb") as err:
        result = m.sup.supervise([sys.executable, "-c", "import time; time.sleep(30)"], {},
                                 str(tmp_path), out, err, 0.2)
    assert result["status"] == "timeout" and result["kill"] == "sigkill_sent"
    assert result["returncode"] is not None and result["elapsed_seconds"] < 5


def test_failed_capture_before_any_forward(tmp_path):
    p, base = prepared(), {"state": state("cpu")}
    journal = m.runner.Journal(tmp_path / "run")
    try:
        backend = m.InstrumentedBackend(Backend(), journal, tmp_path / "run", base,
                                        capture=lambda *x: state("cpu"))
        assert m.refresh.execute(p, m.PREP_SHA, backend, journal, 10, clock=lambda: 0,
                                 load=lambda step: None) == "partial"
    finally:
        journal.close()
    events = [m.bridge.read_json(line) for line in (tmp_path / "run" / "events.jsonl").read_bytes().splitlines()]
    assert [e["kind"] for e in events] == ["start", "attempt", "failed", "end"]
