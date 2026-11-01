"""Regression against the actual saved exploratory receipt; no neural inference."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability import p1_long_suffix_gradient_fd_consume_20261029 as audit


@pytest.fixture(scope='module')
def artifacts():
    producer=audit.producer
    source=producer.consumer.producer.one.source
    saved=producer.consumer.producer.one.saved
    value=saved.read_verified(source.ROOT/'results/p1_long_suffix_gradient_fd_20261029_MAIN01.json',
        'a8a602ec73453ccb7de139dbb9ef0708ce7400f0c9bc7143c90c640e0ec5b12b')
    report=saved.read_verified(source.REPORT,source.REPORT_SHA)
    prepared=saved.read_verified(report['prepared'],report['prepare_sha256'])
    long_result=saved.read_verified(source.ROOT/'results/p1_long_suffix_host_20261029_MAIN01.json',producer.RESULT_SHA)
    return value,report,prepared,long_result


def test_actual_saved_receipt(artifacts):
    result=audit.verify(*artifacts)
    assert result['directions']==8 and result['max_relative_error']<.002


@pytest.mark.parametrize('field,new_value',[
    ('source_report_sha256','0'*64),('epsilon',0.01),('P1_admission',True),('training',True)])
def test_reject_contract_tampering(artifacts,field,new_value):
    value,report,prepared,long_result=artifacts
    bad=deepcopy(value);bad[field]=new_value
    with pytest.raises(ValueError,match='source/call'):
        audit.verify(bad,report,prepared,long_result)


def test_reject_tampered_endpoint(artifacts):
    value,report,prepared,long_result=artifacts
    bad=deepcopy(value);bad['rows'][0]['directions'][0]['plus_endpoint'][0]+=1
    with pytest.raises(ValueError,match='scalar mismatch'):
        audit.verify(bad,report,prepared,long_result)
