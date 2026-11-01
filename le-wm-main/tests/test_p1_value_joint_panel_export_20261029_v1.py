from copy import deepcopy
import pytest
from experiments.spectral_reachability import p1_value_joint_panel_export_20261029_v1 as m
from test_agent_p1_joint_vreach_saved_audit_20261029_v1 import paid, requests, inputs, complete


def test_one_real_ram_audited_arm_has_exact_identity_and_missing_physics(paid):
    q,result,receipt,events=paid
    rows=m.export_arm(q, result['arm'], result, receipt, events, 'a'*64)
    assert len(rows)==20 and all(r['physical']['status']=='missing' for r in rows)
    assert rows[0]['identity']['root_sha256']==q['task']['binding']['root_sha256']
    assert rows[0]['prefix_hex']+rows[0]['suffix_hex']==result['candidates'][0]['raw_float32_bits']
    assert rows[0]['parent_context_digest']==m.core.digest(result['candidates'][0]['mid'])
    broken=deepcopy(result); broken['candidates'][0]['commands'][0][0] += .1
    with pytest.raises(ValueError): m.export_arm(q,result['arm'],broken,receipt,events,'a'*64)
    wrong_root=deepcopy(q); wrong_root['task']['binding']['root_sha256']='f'*64
    with pytest.raises(ValueError): m.export_arm(wrong_root,result['arm'],result,receipt,events,'a'*64)
    wrong_context=deepcopy(result); wrong_context['candidates'][0]['mid']['action_history'][0][0] += .1
    with pytest.raises(ValueError): m.export_arm(q,result['arm'],wrong_context,receipt,events,'a'*64)


def test_receipt_must_match_actual_full_consumer(monkeypatch, tmp_path):
    report=tmp_path/'report.json';report.write_text('{}')
    receipt=tmp_path/'receipt.json';receipt.write_text('{}')
    monkeypatch.setattr(m.host,'consume',lambda *args: {'complete': True})
    with pytest.raises(ValueError, match='receipt'):
        m.build(str(tmp_path),m.sha(report),str(receipt),m.sha(receipt))


def test_exclusive_write(tmp_path):
    path=tmp_path/'x.json'
    m.write(path,{'a':1})
    with pytest.raises(FileExistsError): m.write(path,{'a':2})
