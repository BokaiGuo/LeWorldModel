"""Synthetic admission-audit contracts only; not simulator/GPU evidence."""
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_soft_limit_source import summarize


def fixture():
    seeds = list(range(202650000,202650008))+list(range(202651000,202651008))
    return [{"seed":s,"v1_admitted":s!=202651005,"v2_source_qpos_exact":True,
        "full_offset_target_inside_nominal_limits":True,"v1_v2_context_exact":True if s!=202651005 else None,
        "source_limit_violation_radians":[0.,.0369 if s==202651005 else 0.]} for s in seeds]


def test_complete_unchanged_source_set_and_explicit_rejection():
    result = summarize(fixture())
    assert result["v1_accepted_and_exactly_unchanged"]==15
    assert result["neural_predict_samples"]==0 and result["P1_admission"] is False


@pytest.mark.parametrize("mutation",["partial","duplicate","root","target","rgb","rejection","penetration"])
def test_no_skipping_clipping_or_extra_failure(mutation):
    rows = fixture()
    if mutation=="partial": rows.pop()
    elif mutation=="duplicate": rows[1]["seed"]=rows[0]["seed"]
    elif mutation=="root": rows[0]["v2_source_qpos_exact"]=False
    elif mutation=="target": rows[0]["full_offset_target_inside_nominal_limits"]=False
    elif mutation=="rgb": rows[0]["v1_v2_context_exact"]=False
    elif mutation=="rejection": rows[0]["v1_admitted"]=False
    else: rows[13]["source_limit_violation_radians"][1]=0.
    with pytest.raises(ValueError): summarize(rows)
