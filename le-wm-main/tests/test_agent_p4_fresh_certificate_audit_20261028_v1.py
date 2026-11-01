"""Saved metadata/NPZ-header tests only; no roots, model, environment, or physics."""
from copy import deepcopy
import json
import pytest
from experiments.spectral_reachability import agent_p4_fresh_certificate_audit_20261028_v1 as m


def test_real_pinned_inventory_is_blocked():
    a=m.audit()
    assert a['status']=='blocked_no_legal_fresh_certificate'
    assert a['consume_status']=='blocked'
    assert a['certificate_count']==2
    assert all(x['disposition']=='construction_only_not_fresh_or_history_certificate' for x in a['certificate_candidates'])
    assert a['freshness']==dict(root_manifest=False,consumer_manifest=False,exclusion_manifest=False,certified_fresh_roots=[],source_run_authorized=False)
    assert a['history']['history_certificate'] is None and a['history']['coverage_complete'] is False
    assert a['reserved_roots_opened'] is False and a['certificate_created'] is False


@pytest.mark.parametrize('key,field',[
    ('root_manifest','freshness_certified'),('consumer_manifest','certified_fresh_roots'),
    ('coverage','coverage_complete'),('exclusion','approval'),('sourcepack','global_freshness')])
def test_metadata_mutation_fail_closed(monkeypatch,key,field):
    original=m.read_json
    def fake(k):
        v=deepcopy(original(k))
        if k==key:
            if field=='freshness_certified':v[field]=True
            elif field=='certified_fresh_roots':v[field]=['fake']
            elif field=='coverage_complete':v[field]=True
            elif field=='approval':v[field]={'approved':True,'authority':'MAIN','coverage_scope':['fake']}
            else:v[field]='fresh_same_task'
        return v
    monkeypatch.setattr(m,'read_json',fake)
    with pytest.raises(ValueError):m.audit()


def test_certificate_header_is_not_fresh_upgrade():
    c=m.npz_header('certificate_s0')
    assert c['arrays']['root_history.npy']['shape']==[3,224,224,3]
    assert c['arrays']['future_states.npy']['shape']==[10,7]
    assert c['disposition']=='construction_only_not_fresh_or_history_certificate'


def test_consume_rejects_report_mutation(tmp_path):
    a=m.audit();p=tmp_path/'report.json';p.write_text(json.dumps(a,sort_keys=True,separators=(',',':')))
    h=m.sha(p)
    a['certificate_created']=True;p.write_text(json.dumps(a,sort_keys=True,separators=(',',':')))
    with pytest.raises(ValueError):m.consume(p,h)


def test_output_contract_never_certifies():
    a=m.audit()
    assert a['approval_created'] is False and a['certificate_created'] is False
    assert not a['freshness']['certified_fresh_roots']
    assert 'freshness' in ' '.join(a['blockers'])
