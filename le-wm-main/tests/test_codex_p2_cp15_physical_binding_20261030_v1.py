import pytest

from experiments.spectral_reachability import codex_p2_cp15_physical_binding_20261030_v1 as b


def test_prefix_guard_precedes_dispatch_and_rejects_stop_or_corruption():
    binding = {"base": 0, "prefix_reference": [{"kind": "real"}],
               "plans": [{"arm": "joint", "feedback_context_sha256": "context",
                          "suffix_raw_sha256": "hash", "suffix_raw_float32_hex": "0000"}]}
    seen = []
    def emit(event):
        seen.append(event["kind"])
    def dispatch(raw):
        seen.append("dispatch")
        return raw
    assert b.guarded_suffix_dispatch([{"kind": "real"}], binding, "joint", False, emit, dispatch) == b"\0\0"
    assert seen == ["feedback_prefix_guard", "dispatch"]
    for prefix, stopped in (([], False), ([{"kind": "real"}], True)):
        seen.clear()
        with pytest.raises(ValueError):
            b.guarded_suffix_dispatch(prefix, binding, "joint", stopped, emit, dispatch)
        assert seen == []


def test_durable_guard_failure_prevents_any_dispatch():
    binding = {"base": 0, "prefix_reference": [],
               "plans": [{"arm": "joint", "feedback_context_sha256": "context",
                          "suffix_raw_sha256": "hash", "suffix_raw_float32_hex": "0000"}]}
    dispatched = []
    def fail(_):
        raise OSError("journal")
    with pytest.raises(OSError):
        b.guarded_suffix_dispatch([], binding, "joint", False, fail, dispatched.append)
    assert not dispatched
