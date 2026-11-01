"""RAM joins, negative attribution cases, and the same saved log prefixes."""
from copy import deepcopy
import io
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p4_original_evidence_extract_20261028_v2 as m


def engine(**extra):
    return {"chunk_id": "fixture", "wall_time_seconds": 0.1, "output": "", **extra}


def output(call, results):
    return {"type": "custom_tool_call_output", "call_id": call,
            "output": [{"type": "input_text", "text": m.b.canonical(result).decode()} for result in results]}


def where(call, line, path="/bound/log.jsonl"):
    return {"path": path, "call_id": call, "jsonl_line": line, "prefix_bytes": 10000,
            "prefix_sha256": "a" * 64, "record_sha256": "b" * 64, "record_byte_span": [0, 99]}


def feed(tracker, payload, line, path="/bound/log.jsonl"):
    tracker.receive({"type": "response_item", "payload": payload}, where(payload["call_id"], line, path))


def tracker():
    obj = m.Tracker([{"call_id": "parent", "source_id": "A", "log": "/bound/log.jsonl", "line": 1, "command": "launch"}])
    feed(obj, {"type": "custom_tool_call", "call_id": "parent", "name": "exec", "input": "launch"}, 1)
    feed(obj, output("parent", [engine(session_id=11), engine(session_id=22)]), 2)
    return obj


def poll(source, call="poll"):
    return {"type": "custom_tool_call", "call_id": call, "name": "exec", "input": source}


def test_parser_literal_actual_calls_not_comments_or_strings():
    source = '// tools.write_stdin({session_id:99});\ntext(await tools.write_stdin({session_id:11,chars:""}));'
    result = m.poll_inputs(poll(source))
    assert result["pure_single"] is True
    assert [row["session_id"] for row in result["polls"]] == [11]
    a, b = result["polls"][0]["input_span"]
    assert source[a:b] == 'text(await tools.write_stdin({session_id:11,chars:""}))'


@pytest.mark.parametrize("source", [
    'text("tools.write_stdin({session_id:11})");',
    'if(false) text(await tools.write_stdin({session_id:11}));',
    'for(const s of [11]) {text(await tools.write_stdin({session_id:s}));}',
    'text(await tools.write_stdin({session_id:some_variable}));',
    'text(await tools.write_stdin({session_id:true}));',
    'text(await tools.write_stdin({session_id:11,session_id:22}));',
    'function f(){text(await tools.write_stdin({session_id:11}));}',
    'text(await tools.exec_command({cmd:"echo tools.write_stdin({session_id:11})"}));',
])
def test_mentions_dynamic_conditional_duplicate_inputs_not_joins(source):
    assert m.poll_inputs(poll(source))["polls"] == []


def test_direct_function_argument_join():
    args = m.b.canonical({"session_id": 11, "chars": ""}).decode()
    value = m.poll_inputs({"type": "function_call", "name": "functions.write_stdin", "arguments": args})
    assert value["polls"][0]["session_id"] == 11 and value["pure_single"]


def test_multiple_poll_returns_never_positionally_assigned():
    obj = tracker()
    feed(obj, poll('text(await tools.write_stdin({session_id:11}));text(await tools.write_stdin({session_id:22}));'), 3)
    feed(obj, output("poll", [engine(exit_code=0), engine(exit_code=1)]), 4)
    assert all(not row["terminal_observations"] for row in obj.sessions)
    assert all(row["poll_calls"] == ["poll"] for row in obj.sessions)
    assert [row["exit_code"] for row in obj.batches[0]["result_observations"]] == [0, 1]


def test_return_explicit_id_is_used_not_output_order():
    obj = tracker()
    feed(obj, poll('text(await tools.write_stdin({session_id:11}));text(await tools.write_stdin({session_id:22}));'), 3)
    feed(obj, output("poll", [engine(session_id=22, exit_code=7), engine(session_id=11, exit_code=0)]), 4)
    assert [row["terminal_observations"][0]["exit_code"] for row in obj.sessions] == [0, 7]
    assert all(row["subprocess_source_assignment"] is None for row in obj.sessions)


def test_unique_single_literal_poll_and_exact_call_return():
    obj = tracker()
    feed(obj, poll('text(await tools.write_stdin({session_id:11}));'), 3)
    feed(obj, output("different_call", [engine(exit_code=9)]), 4)
    assert not obj.sessions[0]["terminal_observations"]
    feed(obj, output("poll", [engine(exit_code=0)]), 5)
    assert obj.sessions[0]["terminal_observations"][0]["join"] == "single_literal_poll_and_same_call_id"
    assert not obj.sessions[1]["terminal_observations"]


def test_single_poll_plus_other_exec_is_still_ambiguous():
    obj = tracker()
    feed(obj, poll('text(await tools.write_stdin({session_id:11}));text(await tools.exec_command({cmd:"another"}));'), 3)
    feed(obj, output("poll", [engine(exit_code=0)]), 4)
    assert not obj.sessions[0]["terminal_observations"]


def test_explicit_wrong_session_not_joined_to_poll_target():
    obj = tracker()
    feed(obj, poll('text(await tools.write_stdin({session_id:11}));'), 3)
    feed(obj, output("poll", [engine(session_id=22, exit_code=0)]), 4)
    assert all(not row["terminal_observations"] for row in obj.sessions)


def test_return_before_poll_refused_and_duplicate_return_refused():
    obj = tracker()
    feed(obj, poll('text(await tools.write_stdin({session_id:11}));'), 3)
    with pytest.raises(ValueError, match="order"):
        feed(obj, output("poll", [engine(exit_code=0)]), 2)
    feed(obj, output("poll", [engine(exit_code=0)]), 4)
    with pytest.raises(ValueError, match="join"):
        feed(obj, output("poll", [engine(exit_code=0)]), 5)


def test_parent_return_requires_same_log_namespace():
    obj = m.Tracker([{"call_id": "parent", "source_id": "A", "log": "/bound/log.jsonl", "line": 1, "command": "launch"}])
    feed(obj, {"type": "custom_tool_call", "call_id": "parent", "name": "exec", "input": "launch"}, 1)
    with pytest.raises(ValueError, match="parent return join"):
        feed(obj, output("parent", [engine(session_id=11)]), 2, "/different/log.jsonl")


def test_cross_log_session_namespace_is_not_assumed():
    obj = tracker()
    feed(obj, poll('text(await tools.write_stdin({session_id:11}));'), 3, "/other/log.jsonl")
    feed(obj, output("poll", [engine(session_id=11, exit_code=0)]), 4, "/other/log.jsonl")
    assert all(not row["poll_calls"] and not row["terminal_observations"] for row in obj.sessions)


def test_stdout_numbers_and_fake_session_json_not_births():
    fake = {"type": "custom_tool_call_output", "output": [{"type": "input_text", "text": '{"session_id":11,"score":[1,2,3]}' }]}
    assert m.engine_results(fake) == []


def test_sanitized_trace_and_ledger_not_score_arrays():
    ledger = {"call_id": "printed", "attempts": [{"index": 0, "reset_kind": "random", "disposition": "failed", "identity_sha256": None}]}
    text = 'Traceback (most recent call last):\n  File "/never/open.py", line 8, in reset\nRuntimeError: failed\n' + m.b.canonical(ledger).decode() + '\n{"score":["DO_NOT_COPY_OUTCOMES_869"]}\n'
    result = {"value": engine(exit_code=1, output=text)}
    evidence = m.sanitized_evidence(result)
    assert evidence["traceback_frames"][0]["observed_path"] == "/never/open.py"
    assert evidence["reset_ledger_candidates"][0]["record"] == ledger
    assert evidence["reset_ledger_candidates"][0]["exhaustive"] is False
    assert "DO_NOT_COPY_OUTCOMES_869" not in m.b.canonical(evidence).decode()


def test_prefix_hash_and_no_suffix_read():
    obj = tracker()
    raw = m.b.canonical({"type": "response_item", "payload": poll('text(await tools.write_stdin({session_id:11}));')}) + b"\n"
    binding = {"path": "/bound/log.jsonl", "bytes": len(raw), "sha256": m.b.digest(raw)}
    stream = io.BytesIO(raw + b"FORBIDDEN_SUFFIX")
    m.scan(stream, binding, obj)
    assert stream.tell() == len(raw)
    binding["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA"):
        m.scan(io.BytesIO(raw), binding, tracker())


@pytest.fixture(scope="module")
def actual():
    return m.extract()


def test_actual_saved_chain_and_terminal_ambiguity(actual):
    assert actual["summary"] == {"parents": 3, "sessions": 8, "poll_batches": 3,
                                 "sessions_with_literal_poll": 8, "sessions_with_joined_terminal": 0,
                                 "batch_terminal_observations": 10, "unassigned_terminal_observations": 10,
                                 "traceback_frames": 0, "reset_ledger_candidates": 0}
    assert {s["session_id"] for s in actual["sessions"]} == {71462, 37004, 43033, 29686, 39278, 18657, 42594, 59270}
    assert all(s["subprocess_source_assignment"] is None and s["reset_attempts"] is None for s in actual["sessions"])
    assert all(s["unclosed_poll_reasons"][0]["reason"] == "no_unique_terminal_result_session_join" for s in actual["sessions"])
    assert actual["source_run_authorized"] is False and actual["reset_blockers_cleared"] == 0


def test_actual_origin_byte_sha_and_decoded_exit(actual):
    for batch in actual["poll_batches"]:
        for observation in batch["result_observations"]:
            ref = observation["origin"]
            a, b = ref["record_byte_span"]
            assert b <= ref["prefix_bytes"]
            with Path(ref["path"]).open("rb") as stream:
                stream.seek(a)
                raw = stream.read(b - a)
            assert m.b.digest(raw) == ref["record_sha256"]
            value = m.b.parse(raw)
            for token in ref["pointer"].strip("/").split("/"):
                value = value[int(token)] if isinstance(value, list) else value[token]
            assert m.b.digest(value.encode()) == ref["json_text_sha256"]
            assert m.b.parse(value.encode())["exit_code"] == observation["exit_code"]


def test_actual_file_reads_do_not_follow_stdout_paths(monkeypatch):
    allowed = {str(Path(m.b.__file__).resolve()), str(m.b.CATALOG),
               str(Path(m.__file__).resolve()), str(m.ROOT / "tests" / ("test_" + m.NAME + ".py")),
               str(m.ROOT / "innerdocs" / (m.NAME + ".md"))} | {row["path"] for row in m.b.catalog()["log_bindings"]}
    regular = m.b.regular
    reads = []
    def guarded(path):
        assert str(path) in allowed
        reads.append(str(path))
        return regular(path)
    monkeypatch.setattr(m.b, "regular", guarded)
    m.extract()
    assert set(reads) == allowed


def test_frozen_v1_hash_is_required(monkeypatch):
    monkeypatch.setattr(m, "BASE_SHA", "0" * 64)
    with pytest.raises(ValueError, match="v1 dependency SHA"):
        m.extract()


def test_actual_saved_consume_and_mutation(tmp_path, actual, monkeypatch):
    path = tmp_path / (m.NAME + "_roundtrip.json")
    m.write(path, actual)
    assert m.consume(path, m.b.digest(m.b.canonical(actual)))["source_run_authorized"] is False
    with pytest.raises(FileExistsError):
        m.write(path, actual)
    monkeypatch.setattr(m, "extract", lambda: deepcopy(actual))
    value = deepcopy(actual)
    value["sessions"][0]["terminal_observations"] = [{"exit_code": 0}]
    changed = tmp_path / (m.NAME + "_changed.json")
    m.write(changed, value)
    with pytest.raises(ValueError, match="reconstruction"):
        m.consume(changed, m.b.digest(m.b.canonical(value)))


def test_cold_import_no_model_physics_or_numpy():
    code = """import sys
class Guard:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','numpy','gym','gymnasium','pymunk','stable_worldmodel'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from experiments.spectral_reachability import agent_p4_original_evidence_extract_20261028_v2 as m
assert m.extract()['new_physics_calls'] == 0
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=m.ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
