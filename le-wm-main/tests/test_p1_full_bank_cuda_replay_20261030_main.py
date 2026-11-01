import copy

import pytest

from experiments.spectral_reachability import p1_full_bank_cuda_replay_20261030_main as wrapper


def install_fixture(monkeypatch):
    prior = {"rows": []}
    choices = []
    old_rows = []
    for i in range(8):
        item = {"slot": i, "arm": f"arm{i}", "task_binding": {"id": i},
                "selected_index": i + 1, "raw15_float32_hex": "ab" * 120}
        choices.append(item)
        old_rows.extend([{"slot": i, "arm": f"arm{i}", "task_binding": {"id": i},
                          "role": "incumbent", "candidate_index": 0,
                          "raw15_le_float32_hex": "00" * 120}, {}])
    full = {"rows": choices, "P1_admission": False}
    monkeypatch.setattr(wrapper, "original_sources", lambda: (old_rows, {}, {}, {}))
    monkeypatch.setattr(wrapper.selector, "build", lambda *args: copy.deepcopy(full))
    monkeypatch.setattr(wrapper.replay.prior.one.saved, "read_verified",
                        lambda path, sha: copy.deepcopy(full if path == wrapper.SELECTED else prior))
    return full, old_rows


def test_roster_uses_full_bank_witnesses(monkeypatch):
    full, _ = install_fixture(monkeypatch)
    rows, *_ = wrapper.sources()
    assert len(rows) == 16
    assert [x["candidate_index"] for x in rows[1::2]] == list(range(1, 9))
    assert rows[1]["raw15_le_float32_hex"] == full["rows"][0]["raw15_float32_hex"]
    assert all(x["role"] == "incumbent" for x in rows[::2])


def test_task_mismatch_fails(monkeypatch):
    _, old_rows = install_fixture(monkeypatch)
    old_rows[0]["task_binding"] = {"id": "wrong"}
    with pytest.raises(ValueError, match="task identity"):
        wrapper.sources()


def test_selector_drift_fails(monkeypatch):
    install_fixture(monkeypatch)
    monkeypatch.setattr(wrapper.selector, "build", lambda *args: {"rows": []})
    with pytest.raises(ValueError, match="reconstruction"):
        wrapper.sources()
