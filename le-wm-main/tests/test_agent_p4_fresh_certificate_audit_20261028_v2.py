"""RAM mutation tests plus bounded saved metadata audit; no root/model/physics."""
from copy import deepcopy
import json
import pytest
from experiments.spectral_reachability import agent_p4_fresh_certificate_audit_20261028_v2 as m

def test_real_audit_returns_blocked_not_internal_sha_failure():
    a=m.audit()
    assert a['status']=='blocked_no_legal_fresh_certificate' and a['consume_status']=='blocked'
    assert a['certificate_count']==2 and a['certificate_created'] is False
    assert a['freshness']['certified_fresh_roots']==[] and a['history']['history_certificate'] is None
    assert a['reserved_roots_opened'] is False
    assert all(x['disposition']=='construction_only_not_fresh_or_history_certificate' for x in a['certificate_candidates'])

@pytest.mark.parametrize('key,field',[(
    'root_manifest','freshness_certified'),('consumer_manifest','certified_fresh_roots'),
    ('coverage','coverage_complete'),('exclusion','approval'),('sourcepack','global_freshness')])
def test_pinned_metadata_mutation_fails_closed(monkeypatch,key,field):
    original=m.json_input
    def fake(k):
        value=deepcopy(original(k))
        if k==key:
            value[field]=True if field in ('freshness_certified','coverage_complete') else ['fake'] if field=='certified_fresh_roots' else {'approved':True} if field=='approval' else 'fresh_same_task'
        return value
    monkeypatch.setattr(m,'json_input',fake)
    with pytest.raises(ValueError):m.audit()

def test_construction_headers_cannot_be_upgraded():
    h=m.certificate_header('certificate_s0')
    assert h['arrays']['root_history.npy']['shape']==[3,224,224,3]
    assert h['arrays']['future_states.npy']['shape']==[10,7]
    assert h['disposition']=='construction_only_not_fresh_or_history_certificate'

def test_consume_rejects_tampered_report(tmp_path):
    original=m.audit();p=tmp_path/'report.json';p.write_text(json.dumps(original,sort_keys=True,separators=(',',':')));h=m.digest(p)
    changed=deepcopy(original);changed['certificate_created']=True;p.write_text(json.dumps(changed,sort_keys=True,separators=(',',':')))
    with pytest.raises(ValueError):m.consume(p,h)

def test_report_never_creates_authority():
    a=m.audit();assert a['approval_created'] is False and a['certificate_created'] is False
    assert a['new_neural_calls']==a['new_physics_calls']==0
