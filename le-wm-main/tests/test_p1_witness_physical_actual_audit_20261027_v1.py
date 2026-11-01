"""Pure arithmetic, PNG and hash-chain fixtures; no physics or neural calls."""
from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import p1_witness_physical_actual_audit_20261027_v1 as m


@pytest.mark.parametrize('turns',[-2,-1,0,1,2])
def test_wrapped_pose_invariance_but_native_not_assumed_invariant(turns):
    goal=np.zeros(7);state=np.zeros(7);state[4]=.7
    original=m.metrics(goal,state);goal[4]+=turns*2*np.pi
    shifted=m.metrics(goal,state)
    assert shifted['object_angle']==pytest.approx(original['object_angle'],abs=2e-15)
    assert shifted['wrapped_success']==original['wrapped_success']


def test_negative_native_bug_and_threshold_boundary_preserved():
    goal=np.zeros(7);state=np.zeros(7);goal[4]=-4.;state[4]=3.
    value=m.metrics(goal,state)
    assert value['negative_native_angle'] and value['native_success'] and not value['wrapped_success']
    state[:2]=[20.,0.]
    assert not m.metrics(goal,state)['native_success']


def test_saved_metrics_tamper_rejected_without_model():
    goal=np.zeros(7);state=np.zeros(7);state[0]=30
    d=m.physical.pose_diagnostics(goal,state)
    official=dict(success=False,state_distance=30.)
    m.check_metric(goal,state,d,official)
    d=deepcopy(d);d['independent']['object_position_error']=1.
    with pytest.raises(ValueError):m.check_metric(goal,state,d,official)


def test_png_file_and_pixel_hashes_recomputed(tmp_path):
    a=np.zeros((224,224,3),np.uint8);a[0,0]=[65,105,225]
    path=tmp_path/'frame.png';Image.fromarray(a).save(path)
    record=dict(path=str(path),file_sha256=m.physical.sha(path),rgb_sha256=m.physical.core.sha(a.tobytes()))
    assert np.array_equal(m.png_array(record,tmp_path),a)
    record['rgb_sha256']='a'*64
    with pytest.raises(ValueError,match='pixel SHA'):m.png_array(record,tmp_path)


def test_png_wrong_resolution_rejected(tmp_path):
    path=tmp_path/'frame.png';a=np.zeros((32,32,3),np.uint8);Image.fromarray(a).save(path)
    record=dict(path=str(path),file_sha256=m.physical.sha(path),rgb_sha256=m.physical.core.sha(a.tobytes()))
    with pytest.raises(ValueError,match='PNG224'):m.png_array(record,tmp_path)


def test_source_chain_byte_corruption_rejected(tmp_path):
    store=m.physical.Store(tmp_path/'fixture');store.emit('first',value=1);store.emit('second',value=2)
    path=store.path/'journal.jsonl'
    assert len(m.physical.source.read_journal(path))==2
    raw=path.read_bytes();path.write_bytes(raw.replace(b'"value": 1',b'"value": 3'))
    with pytest.raises(ValueError):m.physical.source.read_journal(path)


def test_actual_saved_full_audit_including_branch_closed_nested_identity(monkeypatch):
    report=m.physical.ROOT/'results/p1_witness_physical_20261027_v1_main01/report.json'
    if not report.is_file():pytest.skip('actual physical artifacts unavailable')
    old=m.physical.read
    def reorder(path):
        value=old(path)
        if str(path)==str(report):value['rows'].reverse()
        return value
    monkeypatch.setattr(m.physical,'read',reorder)
    result,manifest=m.audit(m.physical.ROOT/'results/p1_witness_physical_20261027_v1_prepared01',
        'e4fc7b7adeac3a23036196adaadfbbe69d8873f1e8649ebbf5fe9e8fb1eca0ff',report,
        '48df27e54eca33159c44a834ce0be1a13723553dc9d64dde996b7b4ca9eaf747')
    assert result['status']=='saved_audit_passed' and result['checked_steps']==80
    assert result['checked_frames']==88 and result['checked_png']==24
    assert result['negative_native_angle_steps']==result['native_wrapped_success_disagreements']==0
    assert len(manifest['images'])==len({(x['root_sha256'],x['arm'],x['step']) for x in manifest['images']})==24
    assert result['official_successes']==0 and not result['P1_admission']
    assert m.physical.sha(report)=='48df27e54eca33159c44a834ce0be1a13723553dc9d64dde996b7b4ca9eaf747'
