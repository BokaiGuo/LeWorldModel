"""RAM parser tests and bounded actual original log extraction; no execution."""
from copy import deepcopy
import io
import os
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p4_original_evidence_extract_20261028_v1 as m


def record(payload):
    return m.canonical({"type": "response_item", "payload": payload}) + b"\n"


def fixture():
    call = "call_fixture"
    invocation = record({"type": "custom_tool_call", "call_id": call, "input": "command --flag"})
    text = 'Traceback (most recent call last):\n  File "/unopened/code.py", line 17, in reset\nRuntimeError: failed\n'
    result = record({"type": "custom_tool_call_output", "call_id": call, "output": [{"type": "input_text", "text": text}]})
    raw = invocation + result
    binding = {"path": "/unopened/origin.jsonl", "bytes": len(raw), "sha256": m.digest(raw)}
    dispatch = {"call_id": call, "log": binding["path"], "line": 1, "command": "command --flag"}
    return raw, binding, [dispatch]


def test_exact_frame_pointer_and_literal_span():
    raw, binding, dispatches = fixture()
    selected = m.scan_prefix(io.BytesIO(raw), binding, dispatches)
    frame = selected[1]["frames"][0]
    assert frame["observed_path"] == "/unopened/code.py" and frame["source_line"] == 17
    assert frame["pointer"] == "/payload/output/0/text"
    payload = m.parse(raw.splitlines()[1])["payload"]
    start, stop = frame["character_span"]
    assert payload["output"][0]["text"][start:stop] == frame["literal_frame"]
    assert frame["loaded_bytes_sha256"] is None


def test_never_read_appended_suffix():
    raw, binding, dispatches = fixture()
    stream = io.BytesIO(raw + b'FORBIDDEN_RESERVED_SUFFIX')
    m.scan_prefix(stream, binding, dispatches)
    assert stream.tell() == len(raw)


@pytest.mark.parametrize("damage", ["hash", "short", "line", "log", "command"])
def test_prefix_and_dispatch_mutations_refused(damage):
    raw, binding, dispatches = fixture()
    if damage == "hash":
        binding["sha256"] = "0" * 64
    elif damage == "short":
        raw = raw[:-10]
    elif damage == "line":
        dispatches[0]["line"] = 99
    elif damage == "log":
        dispatches[0]["log"] = "/another/log.jsonl"
    else:
        dispatches[0]["command"] = "not executed command"
    with pytest.raises(ValueError):
        m.scan_prefix(io.BytesIO(raw), binding, dispatches)


@pytest.mark.parametrize("kind,text", [
    ("custom_tool_call", 'Traceback (most recent call last):\n  File "/fake.py", line 1, in main'),
    ("custom_tool_call_output", 'File "/fake.py", line 1, in main'),
    ("custom_tool_call_output", 'print("Traceback (most recent call last):")\n  File "/fake.py", line 1, in main'),
    ("custom_tool_call_output", 'Traceback (most recent call last):\\n  File "/fake.py", line 1, in main'),
])
def test_commands_source_text_and_escaped_trace_are_not_runtime_evidence(kind, text):
    assert m.frames({"type": kind, "output": [{"type": "input_text", "text": text}]}) == []


def test_mention_of_known_call_in_other_message_is_not_a_join():
    raw, binding, dispatches = fixture()
    extra = record({"type": "custom_tool_call_output", "call_id": "other", "output": [{"type": "input_text", "text": "call_fixture"}]})
    binding["bytes"] += len(extra)
    binding["sha256"] = m.digest(raw + extra)
    assert len(m.scan_prefix(io.BytesIO(raw + extra), binding, dispatches)) == 2


def test_duplicate_keys_nonfinite_and_overlarge_line_refused(monkeypatch):
    with pytest.raises(ValueError):
        m.parse(b'{"a":0,"a":1}')
    with pytest.raises(ValueError):
        m.parse(b'{"a":NaN}')
    raw, binding, dispatches = fixture()
    monkeypatch.setattr(m, "LINE_CAP", 5)
    with pytest.raises(ValueError):
        m.scan_prefix(io.BytesIO(raw), binding, dispatches)


@pytest.fixture(scope="module")
def actual():
    return m.extract()


def test_actual_saved_evidence_is_partial_not_reset_accounting(actual):
    assert len(actual["rows"]) == 10
    populated = [row for row in actual["rows"] if row["runtime_path_observations"]]
    assert len(populated) == 2 and all(row["source_id"] == "B" for row in populated)
    assert populated[0]["runtime_path_observations"][0]["origin"]["jsonl_line"] == 22794
    assert populated[1]["runtime_path_observations"][0]["origin"]["jsonl_line"] == 22817
    assert all(row["reset_attempts"] is None and row["exhaustive"] is False for row in actual["rows"])
    assert actual["reset_blockers_cleared"] == 0 and actual["source_run_authorized"] is False


def test_actual_origin_span_record_sha_pointer_and_literal_match(actual):
    for row in actual["rows"]:
        if not row["runtime_path_observations"]:
            continue
        origin = row["runtime_path_observations"][0]["origin"]
        start, stop = origin["record_byte_span"]
        assert stop <= origin["prefix_bytes"]
        with m.regular(origin["path"]).open("rb") as stream:
            stream.seek(start)
            raw = stream.read(stop - start)
        assert m.digest(raw) == origin["record_sha256"]
        record = m.parse(raw)
        assert record["payload"]["call_id"] == row["call_id"]
        for frame in row["runtime_path_observations"]:
            value = record
            for token in frame["pointer"].strip("/").split("/"):
                value = value[int(token)] if isinstance(value, list) else value[token]
            a, b = frame["character_span"]
            assert value[a:b] == frame["literal_frame"]


def test_actual_saved_full_consume_roundtrip(tmp_path, actual):
    path = tmp_path / (m.NAME + "_full_roundtrip.json")
    m.write(path, actual)
    result = m.consume(path, m.digest(m.canonical(actual)))
    assert result["status"] == "partial_evidence_verified_not_authority"
    assert result["reset_blockers_cleared"] == 0


def test_actual_read_scope_and_no_path_follow(monkeypatch):
    allowed = {str(item["path"]) for item in m.catalog()["log_bindings"]} | {
        str(m.CATALOG), str(m.ROOT / "experiments/spectral_reachability" / (m.NAME + ".py")),
        str(m.ROOT / "tests" / ("test_" + m.NAME + ".py")), str(m.ROOT / "innerdocs" / (m.NAME + ".md"))}
    original = m.regular
    def guard(path):
        assert str(path) in allowed
        return original(path)
    monkeypatch.setattr(m, "regular", guard)
    assert m.extract()["new_neural_calls"] == 0


def test_saved_consume_and_tamper(tmp_path, monkeypatch, actual):
    monkeypatch.setattr(m, "extract", lambda: deepcopy(actual))
    path = tmp_path / (m.NAME + "_test.json")
    m.write(path, actual)
    assert m.consume(path, m.digest(m.canonical(actual)))["source_run_authorized"] is False
    with pytest.raises(FileExistsError):
        m.write(path, actual)
    value = deepcopy(actual)
    value["rows"][0]["reset_attempts"] = 0
    changed = tmp_path / (m.NAME + "_changed.json")
    m.write(changed, value)
    with pytest.raises(ValueError, match="reconstruction"):
        m.consume(changed, m.digest(m.canonical(value)))


def test_cold_import_does_not_import_neural_or_physics():
    code = """import sys
class Guard:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','numpy','gym','gymnasium','pymunk','stable_worldmodel'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from experiments.spectral_reachability import agent_p4_original_evidence_extract_20261028_v1
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=m.ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
