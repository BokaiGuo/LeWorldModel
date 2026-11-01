from copy import deepcopy
import time

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_fixed_prefix_suffix_physical_20261030_main as m


def fixture():
    report, binding, traces = {"status": "complete", "rows": []}, {"rows": []}, []
    joined = {"task_rows": []}
    for slot in range(4):
        joined["task_rows"].append(dict(task_id=f"t{slot}", root_slot=slot // 2,
                                       condition=slot % 2, native_root_sha256=f"h{slot}"))
    for i in range(8):
        slot = i // 2
        prefixes = np.full((4, 5, 2), i / 10, dtype=np.float32)
        actions = np.zeros((4, 3, 5, 2), dtype=np.float32)
        actions[:, 0] = prefixes
        row = dict(slot=slot, arm=f"a{i % 2}", indices=[0, 1, 2, 3])
        report["rows"].append(deepcopy(row))
        binding["rows"].append({**row, "prefixes": prefixes.tolist(),
                                "task": {"binding": {"task_id": f"t{slot}"}}})
        traces.append([{"kind": "inputs"}] + [{"kind": "optimized"}] * 4 +
                      [{"kind": "selection", "payload": {"actions": actions.tolist()}}])
    return report, binding, joined, traces


def test_fixed_roster_and_exact_prefix():
    report, binding, joined, traces = fixture()
    plans = m.plans_from_traces(report, binding, joined, traces)
    assert len(plans) == 32
    assert [p["role"] for p in plans[:4]] == [f"candidate_{i}" for i in range(4)]
    assert [p["index"] for p in plans] == list(range(32))
    action = np.frombuffer(bytes.fromhex(plans[4]["raw15_float32_hex"]), dtype="<f4").reshape(3, 5, 2)
    np.testing.assert_array_equal(action[0], np.asarray(binding["rows"][1]["prefixes"][0], dtype=np.float32))


@pytest.mark.parametrize("corruption", ["task", "prefix", "domain", "trace"])
def test_rejects_tampered_inputs(corruption):
    report, binding, joined, traces = fixture()
    if corruption == "task":
        joined["task_rows"][0]["task_id"] = "wrong"
    elif corruption == "prefix":
        traces[0][-1]["payload"]["actions"][0][0][0][0] = 0.5
    elif corruption == "domain":
        traces[0][-1]["payload"]["actions"][0][1][0][0] = 2.0
    else:
        traces[0][1]["kind"] = "selection"
    with pytest.raises(Exception):
        m.plans_from_traces(report, binding, joined, traces)


def test_cleanup_still_runs_if_journal_fails():
    class BrokenStore:
        def check(self):
            pass

        def emit(self, *args, **kwargs):
            raise OSError("journal unavailable")

    meter = m.Meter(BrokenStore(), time.monotonic())
    called = []
    meter.call("close", lambda: called.append(True), cleanup=True)
    assert called == [True]
    assert meter.fault and len(meter.secondary) == 2
    with pytest.raises(Exception):
        meter.call("constructor", lambda: None)
