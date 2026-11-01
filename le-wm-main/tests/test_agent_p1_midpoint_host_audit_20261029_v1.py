import pytest
from experiments.spectral_reachability import agent_p1_midpoint_host_audit_20261029_v1 as audit


def test_rejects_missing_result_fields():
    with pytest.raises(ValueError):
        audit.audit({'request': {}}, {}, {}, [])


def test_contract_does_not_claim_production():
    assert audit.method.contract()['production_ready'] is False
    assert audit.method.contract()['fixed_denominator'] == 4
