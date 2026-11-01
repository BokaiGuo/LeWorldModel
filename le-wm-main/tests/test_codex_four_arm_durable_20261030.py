import hashlib
import json
import subprocess
import sys

import pytest

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as driver
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer


def contract():
    identity = {k: "a" * 64 for k in spec.IDENTITY}
    arms = {a: {"initializer_sha256": str(i + 1) * 64,
                "optimizer_sha256": identity["optimizer"],
                "optimizer_config_sha256": identity["optimizer_config"]}
            for i, a in enumerate(spec.ARMS)}
    return {"schema": spec.NAME + ".contract.v1", "identity": identity, "arms": arms,
            "scope": "standalone_per_root_per_arm_including_attributable_setup",
            "phase_policy_sha256": "b" * 64,
            "levels": [{"id": "a", "caps": {k: 2 for k in spec.AXES}},
                       {"id": "b", "caps": {k: 4 for k in spec.AXES}}]}


def charge(**v):
    return {k: v.get(k, 0) for k in spec.AXES}


def setup(tmp_path):
    c = contract()
    sha = spec.digest(c)
    return c, sha, driver.Driver(tmp_path / "run", c, sha, "structured", "a", "run-1")


def verify(result, sha):
    return consumer.consume(result["journal"], result["sha256"], sha,
                            "structured", "a", "run-1")


def test_completed_and_nonoverwrite(tmp_path):
    c, sha, d = setup(tmp_path)
    d.dispatch("setup", "encode", charge(encode_calls=1, encode_samples=2), lambda: 17)
    d.dispatch("common_suffix", "backward", charge(backward_calls=1), lambda: None)
    r = d.finish()
    result = verify(r, sha)
    assert result["status"] == "closed_unverified"
    assert result["charged_upper_bounds"]["encode_samples"] == 2
    assert not result["coverage_verified"] and not result["fairness_proven"]
    with pytest.raises((FileExistsError, ValueError)):
        driver.Driver(tmp_path / "run", c, sha, "structured", "a", "run-1")


def test_failure_persists_attempt_and_cannot_continue(tmp_path):
    _, sha, d = setup(tmp_path)
    def boom():
        raise RuntimeError("backend")
    with pytest.raises(RuntimeError):
        d.dispatch("jacobian", "backward", charge(backward_calls=1, vjp_rows=2), boom)
    with pytest.raises(ValueError):
        d.dispatch("trial", "solve", charge(solve_calls=1), lambda: None)
    r = verify(d.finish(), sha)
    assert r["status"] == "failed_or_blocked"
    assert r["charged_upper_bounds"]["backward_calls"] == 1
    assert r["attempted_events"] == 1 and r["completed_events"] == 0


def test_over_cap_rejected_before_call_and_preserves_prefix(tmp_path):
    _, sha, d = setup(tmp_path)
    touched = []
    d.dispatch("trial", "predict", charge(predict_calls=1, predict_samples=2), lambda: None)
    with pytest.raises(ValueError, match="budget"):
        d.dispatch("trial", "predict", charge(predict_calls=1, predict_samples=1), lambda: touched.append(1))
    r = verify(d.finish(), sha)
    assert not touched and r["status"] == "failed_or_blocked"
    assert r["charged_upper_bounds"]["predict_samples"] == 2
    assert r["blocked_events"] == 1


@pytest.mark.parametrize("alteration", ["truncate", "reorder", "delete_terminal"])
def test_corrupt_even_with_reanchored_file_hash(tmp_path, alteration):
    _, sha, d = setup(tmp_path)
    d.dispatch("trial", "solve", charge(solve_calls=1), lambda: None)
    r = d.finish()
    path = tmp_path / "run" / "events.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    if alteration == "truncate":
        raw = b"".join(lines)[:-4]
    elif alteration == "reorder":
        raw = b"".join([lines[0], lines[2], lines[1], *lines[3:]])
    else:
        raw = b"".join(lines[:-1])
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        consumer.consume(path, hashlib.sha256(raw).hexdigest(), sha, "structured", "a", "run-1")


def test_wrong_external_binding_and_digest(tmp_path):
    _, sha, d = setup(tmp_path)
    r = d.finish()
    with pytest.raises(ValueError):
        consumer.consume(r["journal"], r["sha256"], sha, "native", "a", "run-1")
    with pytest.raises(ValueError):
        consumer.consume(r["journal"], "0" * 64, sha, "structured", "a", "run-1")


def test_persist_intent_before_callback(tmp_path):
    _, sha, d = setup(tmp_path)
    def inspect():
        rows = (tmp_path / "run" / "events.jsonl").read_text().splitlines()
        assert json.loads(rows[-1])["kind"] == "intent"
    d.dispatch("screening", "selection", charge(selection_evaluations=1), inspect)
    assert verify(d.finish(), sha)["completed_events"] == 1


def rechain(rows):
    previous = None
    output = []
    for i, row in enumerate(rows):
        body = {"index": i, "previous": previous, "kind": row["kind"], "payload": row["payload"]}
        previous = spec.digest(body)
        output.append(spec.canonical({**body, "sha256": previous}) + b"\n")
    return b"".join(output)


@pytest.mark.parametrize("tamper", ["overcap_intent", "wrong_total", "return_before_intent", "false_failure"])
def test_semantic_forgery_rejected_even_when_rechained(tmp_path, tamper):
    _, sha, d = setup(tmp_path)
    d.dispatch("trial", "predict", charge(predict_calls=1, predict_samples=1), lambda: None)
    result = d.finish()
    path = tmp_path / "run" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if tamper == "overcap_intent":
        rows[1]["payload"]["reservation"]["predict_samples"] = 3
    elif tamper == "wrong_total":
        rows[-1]["payload"]["charged_upper_bounds"]["predict_samples"] = 0
    elif tamper == "return_before_intent":
        rows[1], rows[2] = rows[2], rows[1]
    else:
        rows[-1]["payload"]["status"] = "failed_or_blocked"
    raw = rechain(rows)
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        consumer.consume(path, hashlib.sha256(raw).hexdigest(), sha, "structured", "a", "run-1")


def test_intent_sync_failure_never_dispatches_and_cannot_verify(tmp_path, monkeypatch):
    _, sha, d = setup(tmp_path)
    touched = []
    def fail_sync(_fd):
        raise OSError("sync failed")
    with monkeypatch.context() as patch:
        patch.setattr(driver.os, "fsync", fail_sync)
        with pytest.raises(OSError, match="sync failed"):
            d.dispatch("trial", "solve", charge(solve_calls=1), lambda: touched.append(1))
    assert not touched
    r = d.finish()
    with pytest.raises(ValueError):
        verify(r, sha)


def test_partial_os_writes_are_completed(tmp_path, monkeypatch):
    _, sha, d = setup(tmp_path)
    original = driver.os.write
    def short_write(fd, data):
        return original(fd, data[:min(len(data), 7)])
    with monkeypatch.context() as patch:
        patch.setattr(driver.os, "write", short_write)
        d.dispatch("trial", "solve", charge(solve_calls=1), lambda: None)
    assert verify(d.finish(), sha)["charged_upper_bounds"]["solve_calls"] == 1


def test_cli_success_and_incomplete_journal_nonzero(tmp_path):
    _, sha, d = setup(tmp_path)
    d.dispatch("trial", "solve", charge(solve_calls=1), lambda: None)
    result = d.finish()
    path = tmp_path / "run" / "events.jsonl"
    def call(expected):
        return subprocess.run([sys.executable, "-B", "-m",
                               "experiments.spectral_reachability.codex_four_arm_consumer_20261030",
                               "--journal", str(path), "--sha256", expected, "--contract-sha256", sha,
                               "--arm", "structured", "--level", "a", "--run-id", "run-1"],
                              capture_output=True, text=True, check=False)
    assert call(result["sha256"]).returncode == 0
    raw = b"".join(path.read_bytes().splitlines(keepends=True)[:-2])
    path.write_bytes(raw)
    assert call(hashlib.sha256(raw).hexdigest()).returncode != 0
