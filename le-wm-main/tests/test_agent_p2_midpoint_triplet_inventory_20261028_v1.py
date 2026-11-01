"""RAM and saved-only inventory regressions; no model/simulator."""
from copy import deepcopy
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_midpoint_triplet_inventory_20261028_v1 as m


def history():
    rgb = [np.full((224,224,3),i,np.uint8) for i in (1,2,3)]
    goal = np.full((224,224,3),4,np.uint8)
    frames = [dict(path='actual-'+str(i),warmup_step=5*(i+1),observation_id='event-'+str(i),
                   root_sha256='RAM-root',actual_observation=True,pixel_sha256=m.pixel_sha(r)) for i,r in enumerate(rgb)]
    ref = dict(root_sha256='RAM-root',history_rgb_sha256=hashlib.sha256(np.stack(rgb).tobytes()).hexdigest(),
               goal_rgb_sha256=m.pixel_sha(goal))
    return ref,frames,rgb,goal


def test_exact_history_identity_not_filename():
    args=history()
    result=m.validate_history(*args)
    assert result['state_at_warmup10'] is None
    assert result['history_rgb_sha256']==args[0]['history_rgb_sha256']


@pytest.mark.parametrize('damage',['stack','goal','role','event','alias','root','not_actual','pixels','missing'])
def test_wrong_or_copied_history_rejected(damage):
    ref,frames,rgb,goal=history()
    if damage=='stack': ref['history_rgb_sha256']='bad'
    if damage=='goal': goal[0,0,0]=0
    if damage=='role': frames[1]['warmup_step']=15
    if damage=='event': frames[1]['observation_id']=frames[0]['observation_id']
    if damage=='alias': frames[1]['path']=frames[0]['path']
    if damage=='root': frames[1]['root_sha256']='other'
    if damage=='not_actual': frames[1]['actual_observation']=False
    if damage=='pixels': rgb[1][0,0,0]=0
    if damage=='missing': frames.pop()
    with pytest.raises(ValueError): m.validate_history(ref,frames,rgb,goal)


def test_equal_pixels_at_distinct_actual_times_allowed():
    ref,frames,rgb,goal=history()
    rgb=[rgb[0].copy() for _ in rgb]
    for f,r in zip(frames,rgb): f['pixel_sha256']=m.pixel_sha(r)
    ref['history_rgb_sha256']=hashlib.sha256(np.stack(rgb).tobytes()).hexdigest()
    assert m.validate_history(ref,frames,rgb,goal)['frames']==frames


@pytest.fixture(scope='module')
def saved(): return m.inventory(scan=False)


def test_actual_all8_partial_root_evidence_retained(saved):
    assert len(saved['rows'])==8 and len(saved['accepted_history_stacks'])==2
    assert saved['exact_triplet_branches']==2
    assert saved['stageB_execution_ready'] is False and saved['stageB_adapter_delivered'] is False
    assert saved['reconstruction']['missing_root_slots']==[1,2,3]
    assert saved['new_neural_calls']==saved['new_physics_steps']==0
    for r in saved['rows']:
        assert r['frames'][1] is not None and r['frames'][2] is not None
        assert (r['frames'][0] is not None)==(r['root_slot']==0)


def test_fixed_reconstruction_meter_and_no_subset(saved):
    r=m.reconstruction(saved['rows'])
    assert len(r['missing_branch_slots'])==6
    assert r['planned_public_calls']['original_collector_reset_attempts']==32
    assert r['planned_public_calls']['missing_root_reset_attempts']==3
    assert r['planned_public_calls']['zero_warmup_step_attempts']==45
    assert r['planned_public_calls']['selected_action_step_attempts']==0
    assert r['planned_public_calls']['returned_counts'] is None
    assert r['encoder']==r['predict']==r['AD']==0
    assert r['downstream_only_after_all8']['public_predict']==r['downstream_only_after_all8']['manual_predict']==8
    changed=deepcopy(saved['rows'])
    for row in changed: row['native_success']=True; row['score']=-1e30
    assert m.reconstruction(changed)==r
    with pytest.raises(ValueError): m.reconstruction(changed[:-1])


def test_discovery_ignores_self_and_reserved_paths(tmp_path):
    ref=history()[0]
    p=tmp_path/'arbitrary.json'
    p.write_text(m.gap.encode(dict(root=ref['root_sha256'],png='not-an-authority.png')).decode())
    fresh=tmp_path/'reserved4fresh'; fresh.mkdir()
    (fresh/'report.json').write_bytes(p.read_bytes())
    own=tmp_path/(m.NAME+'_MAIN01.json'); own.write_bytes(p.read_bytes())
    r=m.discover({'RAM-root':ref},tmp_path)
    assert list(r['matched_metadata_sha256'])==[str(p)]
    assert r['candidate_image_documents'][0]['admission'].startswith('discovery_only')
    assert r['reserved_and_fresh_paths_excluded'] is True


def test_precise_import_guard_in_cold_saved_process():
    code='''
import sys
class Reject:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','gym','gymnasium','pymunk','stable_worldmodel','stable_pretraining'}:
   raise AssertionError('forbidden runtime import '+fullname)
sys.meta_path.insert(0,Reject())
from experiments.spectral_reachability import agent_p2_midpoint_triplet_inventory_20261028_v1 as m
r=m.inventory(scan=False)
assert len(r['rows'])==8 and r['stageB_execution_ready'] is False
'''
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True,timeout=60,
                     env={**os.environ,'OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
    assert p.returncode==0,p.stdout+p.stderr


def test_input_drift_failclosed(tmp_path):
    path=tmp_path/'saved.json'; path.write_text('{}')
    x=m.Inputs(); x.bind(path,m.sha(path)); path.write_text('{"changed":true}')
    with pytest.raises(ValueError): x.finish()


def test_verify_exclusive_output_and_snapshot(saved,tmp_path):
    path=tmp_path/'inventory.json'; m.mid.write(path,saved)
    r=m.verify(path,m.sha(path))
    assert r['stageB_execution_ready'] is False
    with pytest.raises(FileExistsError): m.mid.write(path,saved)
    with pytest.raises(ValueError): m.verify(path,'0'*64)


def test_verify_reconstructs_rows_not_just_counter_claim(saved,tmp_path):
    changed=deepcopy(saved)
    changed['rows'][2]['frames'][0]=deepcopy(changed['rows'][2]['frames'][1])
    path=tmp_path/'mutated.json'; m.mid.write(path,changed)
    with pytest.raises(ValueError): m.verify(path,m.sha(path))
