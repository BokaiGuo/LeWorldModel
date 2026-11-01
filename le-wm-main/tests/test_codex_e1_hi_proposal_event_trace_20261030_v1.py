import copy
import pytest

from experiments.spectral_reachability import codex_e1_hi_proposal_event_trace_20261030_v1 as m


@pytest.fixture(scope="module")
def sources():
    return m.load()


def test_real_events_conserve_whole_receipt(sources):
    out = m.analyze(*sources)
    assert out["status"] == "incomplete" and not out["equal_budget_certified"]
    assert len(out["row_execution_projections"]) == 4
    assert out["unallocated_aggregate_modules"]["high_predictor"]["input_rows"] == 392
    assert all(x["allocation"] == "row_execution_only_not_target_causal_cost"
               for x in out["row_execution_projections"])
    assert len(out["row_execution_projections"][0]["modules"]["encoder"]["event_set_sha256"]) == 64


@pytest.mark.parametrize("change", [
    lambda d: d[0]["rows"][0]["events"][0].update(returned=False),
    lambda d: d[0]["rows"][0]["events"][0].update(index=2),
    lambda d: d[0]["rows"][0]["events"][1].update(parent=99),
    lambda d: d[0]["rows"][0]["events"][0]["layouts"][0]["shape"].__setitem__(0, 2),
    lambda d: d[0]["rows"][0]["events"].pop(),
    lambda d: d[1].update(report_sha256="0" * 64),
    lambda d: d[1]["rows"][0].update(high_target_sha256="0" * 64),
    lambda d: d[2]["rows"][1].update(blind_sha256="0" * 64),
    lambda d: d[0]["rows"][0].update(condition="on"),
])
def test_tampering_fails_closed(sources, change):
    data = copy.deepcopy(sources)
    change(data)
    with pytest.raises((ValueError, KeyError)):
        m.analyze(*data)


def test_pinned_hash_mismatch():
    with pytest.raises(ValueError):
        m.util.read_bound(m.REPORT, "0" * 64, 4 * 1024 * 1024)


def test_exclusive_output_never_overwrites(monkeypatch, tmp_path):
    path = tmp_path / "analysis.json"
    monkeypatch.setattr(m, "OUTPUT", path)
    monkeypatch.setattr(m.sys, "argv", ["audit"])
    assert m.main() == 0
    first = path.read_bytes()
    assert m.main() == 2
    assert path.read_bytes() == first
