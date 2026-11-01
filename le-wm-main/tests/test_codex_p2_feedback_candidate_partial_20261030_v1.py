from copy import deepcopy
import pytest
from experiments.spectral_reachability import codex_p2_feedback_candidate_partial_20261030_v1 as m


def test_real_partial_and_event_tamper():
    path=m.physical.ROOT/"results/codex_p2_feedback_candidate_physical_20261030_v1_RUN_JOB0"
    result=m.audit(path,"5a1a34f3eaa461c07be44cb312600bb59369ecc6210a87d75a69bfdcac210330",
                   "2fbc4e6490b0d123e53e3e5fef7726a6fe4973c302e79b97e8de4950a4f2f0d7")
    assert result["status"]=="partial_journal_checked_not_full_strict"
    assert result["rows"][1]["returned"]==29 and not result["full_native_saved_validation"]
    report=m.physical.native.decode((path/"report.json").read_bytes())
    events=[m.physical.native.decode(x) for x in (path/"journal.jsonl").read_bytes().splitlines()]
    prep,_=m.physical.verify(report["execution_binding"]["prepared"],report["execution_binding"]["prepare_sha256"])
    altered=deepcopy(events)
    altered[1]["kind"]="unknown"
    with pytest.raises(ValueError):
        m.audit_events(report,altered,prep)
