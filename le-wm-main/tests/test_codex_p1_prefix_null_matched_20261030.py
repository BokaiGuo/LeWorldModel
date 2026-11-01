import copy
import hashlib
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_prefix_null_matched_20261030 as run
from experiments.spectral_reachability import codex_p1_prefix_null_matched_saved_20261030 as saved


@pytest.fixture(scope="module")
def prepared():
    return run.build()[0]


def test_real_source_fixed_all_pairs_and_identical_probes(prepared):
    assert prepared["fixed_pairs"] == 8
    assert prepared["fixed_branches"] == 336
    assert len(prepared["anchors"]) == 16
    assert len(prepared["probes"]) == 21
    assert prepared["P1_admission"] is prepared["independent_confirmation"] is False
    for i in range(8):
        a, b = prepared["anchors"][2*i:2*i+2]
        assert (a["role"], b["role"]) == ("selected_prefix", "zero_prefix")
        assert bytes.fromhex(b["prefix5_float32_hex"]) == bytes(40)
        assert a["task_id"] == b["task_id"]
        assert len(a["trace"]) == len(b["trace"]) == 51
    for item in prepared["probes"]:
        assert hashlib.sha256(bytes.fromhex(item["raw5_float32_hex"])).hexdigest() == item["raw5_sha256"]


def test_source_drift_is_rejected(monkeypatch):
    monkeypatch.setattr(run, "SOURCE_SHA", "0"*64)
    with pytest.raises((ValueError, AssertionError)):
        run.build()


def test_consumer_rejects_missing_or_partial(monkeypatch, tmp_path, prepared):
    monkeypatch.setattr(run, "verify", lambda *_: (prepared, None))
    monkeypatch.setattr(run, "ROOT", tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    directory = results / (run.NAME+"_run_fixture")
    directory.mkdir()
    path = directory / "run_report.json"
    report = dict(schema=run.NAME+".run.v1", status="partial_or_failed", fixed_branches=336,
                  fixed_anchors=16, P1_admission=False, independent_confirmation=False, shards=[])
    path.write_bytes(run.native.raw(report))
    with pytest.raises(ValueError, match="complete fixed run"):
        saved.consume(directory, run.sha(path), "/unused", "0"*64)


def test_prepare_verification_rejects_drift(monkeypatch, tmp_path, prepared):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    path = results / (run.NAME+"_prepare_fixture.json")
    path.write_bytes(run.native.raw(prepared))
    changed = copy.deepcopy(prepared)
    changed["probes"][0]["raw5_float32_hex"] = "01"*40
    monkeypatch.setattr(run, "build", lambda: (changed, None))
    with pytest.raises(ValueError, match="drift"):
        run.verify(path, run.sha(path))


def test_run_consumer_wiring_all_shards_and_fixed_bank(monkeypatch, tmp_path, prepared):
    p = copy.deepcopy(prepared)
    p["bindings"] = {}
    monkeypatch.setattr(run, "ROOT", tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    prepare_path = results / (run.NAME + "_prepare_fixture.json")
    prepare_path.write_bytes(run.native.raw(p))
    digest = run.sha(prepare_path)
    monkeypatch.setattr(run, "verify", lambda path, sha: (p, {}) if
                        Path(path) == prepare_path and sha == digest else pytest.fail("seal mismatch"))
    monkeypatch.setattr(run.native, "constructor_contract", lambda: None)
    calls = []
    def shard(anchor, bank, data, path, factory, bindings):
        calls.append((anchor["index"], copy.deepcopy(bank)))
        path.mkdir()
        report = {"status": "recorded_not_consumed"}
        (path / "report.json").write_bytes(run.native.raw(report))
        return report
    monkeypatch.setattr(run.core, "execute_shard", shard)
    directory = results / (run.NAME + "_run_fixture")
    result = run.run(str(prepare_path), digest, str(directory))
    assert result["status"] == "recorded_not_consumed"
    assert [c[0] for c in calls] == list(range(16))
    assert all(c[1] == p["probes"] for c in calls)
    def check(path, anchor, bank, sha):
        assert run.sha(path / "report.json") == sha
        assert bank == p["probes"]
        return {"anchor": anchor["index"], "statuses": ["probe_completed"]*20 +
                ["censored_out_of_domain"], "endpoints": [None]*21}
    monkeypatch.setattr(saved.checks, "check_shard", check)
    verdict = saved.consume(str(directory), result["sha256"], str(prepare_path), digest)
    assert verdict["fixed_branches"] == 336
    assert verdict["counts"]["censored_out_of_domain"] == 16
    assert verdict["P1_admission"] is False
