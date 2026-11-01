import json
import sys
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p3_feedback_supervised_launch_20261030_codex_v1 as m

GPU = "GPU-12345678-1234-1234-1234-123456789abc"


def args():
    return SimpleNamespace(authorization=m.AUTH, gpu_uuid=GPU, hard_seconds=10, min_free_mib=100,
                           min_total_mib=200, suffix="test", reference="reviewed-record",
                           source_sha=m.binding.budget.sha(m.__file__))


def test_gpu_query_and_uuid_identity():
    def query(cmd, **kw):
        assert cmd[2] == GPU
        text = GPU + ", 200, 100, Default\n" if "memory.total" in cmd[3] else "Disabled\n"
        return SimpleNamespace(stdout=text)
    value = m.gpu_snapshot(GPU, query)
    m.admit(value, 100, 200)
    with pytest.raises(ValueError):
        m.admit(value, 101, 200)
    with pytest.raises(ValueError):
        m.gpu_snapshot("0", query)
    with pytest.raises(ValueError):
        m.gpu_snapshot(GPU, lambda *a, **k: SimpleNamespace(stdout="wrong, 200, 100, Default\n"))


def test_cpu_child_timeout(tmp_path):
    with (tmp_path / "out").open("wb") as out, (tmp_path / "err").open("wb") as err:
        result = m.supervise([sys.executable, "-c", "import time; time.sleep(30)"], {}, out, err, 0.2)
    assert result["status"] == "timeout" and result["returncode"] is not None


@pytest.mark.parametrize("mode", ["memory", "drift", "timeout"])
def test_launch_refuses_or_preserves_partial(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(m.binding, "RESULTS", tmp_path)
    checks = []
    def preflight():
        checks.append(True)
        if mode == "drift" and len(checks) == 2:
            raise ValueError("drift")
    monkeypatch.setattr(m, "preflight", preflight)
    monkeypatch.setattr(m, "gpu_snapshot", lambda uuid: dict(uuid=uuid, total_mib=200,
                        free_mib=1 if mode == "memory" else 100, compute_mode="Default", mig_mode="Disabled"))
    dispatched = []
    def supervise(cmd, env, out, err, seconds):
        dispatched.append((cmd, env))
        return dict(status="timeout", returncode=-9, pid=42, elapsed_seconds=seconds)
    monkeypatch.setattr(m, "supervise", supervise)
    assert m.launch(args()) == 2
    journal = tmp_path / (m.NAME + "_test") / "supervision.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert events[-1]["retry"] is False
    assert len(dispatched) == (1 if mode == "timeout" else 0)
    if dispatched:
        assert dispatched[0][1]["CUDA_VISIBLE_DEVICES"] == GPU
        assert events[-1]["execution"]["status"] == "timeout"
    with pytest.raises(ValueError, match="output exists"):
        m.launch(args())


def test_authorization_rejected_before_preflight(monkeypatch):
    monkeypatch.setattr(m, "preflight", lambda: pytest.fail("must not dispatch"))
    a = args()
    a.authorization = "wrong"
    with pytest.raises(ValueError):
        m.launch(a)


def test_reviewed_source_sha_required(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "preflight", lambda: None)
    monkeypatch.setattr(m.binding, "RESULTS", tmp_path)
    a = args()
    a.source_sha = "0" * 64
    with pytest.raises(ValueError, match="source SHA"):
        m.launch(a)
    assert list(tmp_path.iterdir()) == []
