from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from experiments.spectral_reachability import p1_backtrack_capability_prepare_20261030 as prep
from experiments.spectral_reachability import p1_backtrack_capability_runner_20261030 as runner


class Store:
    def __init__(self, path):
        path.mkdir()
        self.path, self.previous = path, "a" * 64
        (path / "journal.jsonl").write_text("fixture")
        self.events = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))

    def put(self, name, value, final=False):
        self.report = value

    def check(self, *args):
        return None


def anchor():
    return dict(index=0, root_slot=0, condition="off", arm="a", role="b", task_id="c" * 64,
                prefix5_float32_hex=np.zeros((5, 2), "<f4").tobytes().hex())


def test_failure_retains_21_rows_and_stops_without_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "Store", Store)
    monkeypatch.setattr(runner.prior.base, "validate_env", lambda *_: None)
    monkeypatch.setattr(runner.prior.base, "native_hooks", lambda *_: nullcontext())
    monkeypatch.setattr(runner.prior, "episode", lambda *_: (_ for _ in ()).throw(ValueError("drift")))
    body = SimpleNamespace(render_action=False, render_size=224, render_mode="rgb_array")
    factory = lambda: SimpleNamespace(unwrapped=body, close=lambda: None)
    report = runner.execute_shard(anchor(), prep.probes(), {"roots": [{}], "source_plan": {}},
                                  tmp_path / "shard", factory, {})
    assert report["status"] == "partial_or_failed"
    assert len(report["rows"]) == 21
    assert report["rows"][0]["status"] == "engineering_failed"
    assert all(row["status"] == "not_started" for row in report["rows"][1:])
    assert report["counts"]["constructor"]["attempted"] == report["counts"]["close"]["completed"] == 1


def test_shard_caps_match_full_planned_work():
    c = runner.CAPS
    assert c["constructor"] == c["close"] == 21
    assert c["public_step"] == 21 * 20
    assert c["physics_step"] == 2 * c["reset"] + 10 * c["public_step"]
    assert c["explicit_render"] == 21 * (11 + 11) * 3
