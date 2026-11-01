from copy import deepcopy
import pytest

from experiments.spectral_reachability import codex_p2_feedback_shared_candidate_20261030_v1 as m


def inputs():
    prep = m.source.read(str(m.PATHS["prepare"]), m.source.PINS["prepare"])
    report = m.source.read(str(m.PATHS["report"]), m.source.PINS["report"])
    row = m.source.read(str(m.PATHS["report"].parent / "slot1.json.gz"),
                        report["artifact_sha256"]["slot1.json.gz"], True)
    summary = m.source.read(str(m.SUMMARY), m.SUMMARY_SHA)
    return prep["jobs"][1], row, summary["jobs"][1]


def test_real_saved_context_and_raw_bits():
    result = m.build()
    assert len(result["pairs"]) == 4
    assert all(p["arms"][1]["candidate_index"] == 1 for p in result["pairs"])
    assert not result["P2_admission"] and result["new_neural_calls"] == 0
    assert all(a["physical_outcome"] is None for p in result["pairs"] for a in p["arms"])


def test_context_and_candidate_tampering_rejected():
    job, row, summary = inputs()
    bad = deepcopy(job)
    bad["input_context"]["latent_history"][0][0] += 1
    with pytest.raises(ValueError):
        m.make_pair(bad, row, summary)
    bad = deepcopy(row)
    bad["result"]["evaluations"][1]["Q"] += 1
    with pytest.raises(ValueError):
        m.make_pair(job, bad, summary)
