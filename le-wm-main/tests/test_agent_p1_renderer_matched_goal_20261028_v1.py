"""Fixed factual25 endpoints, paired real goal renderings, saved arithmetic only."""
from copy import deepcopy
from functools import lru_cache
from fractions import Fraction
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_renderer_matched_goal_20261028_v1 as m


@lru_cache(maxsize=1)
def actual():
    return m.acquire()


def test_real_four_goals_already_saved_no_cuda_gap():
    report,parent,goals,evidence=actual()
    assert set(goals)=={(s,c) for s in (0,1) for c in ('off','on')}
    for root in (0,1):
        assert goals[root,'off']['physical_goal']==goals[root,'on']['physical_goal']
        assert goals[root,'off']['pixel_sha256']!=goals[root,'on']['pixel_sha256']
    assert evidence['new_goal_RGB_needed']==[] and evidence['new_neural_calls']==0


def toy():
    z0=np.zeros((5,192),np.float32);z1=z0.copy();g0=np.zeros(192,np.float32);g1=g0.copy()
    z0[:,0]=[0,1,2,3,4];z1[:,0]=[1,2,3,4,5];g1[0]=3
    return z0,z1,g0,g1


def test_two_by_two_not_alternate_endpoints_with_original_goal():
    off,on,g0,g1=toy();p=m.factorial_pair(off,on,g0,g1,0,1,'off')
    assert p['margins']['off_off']['sign']==-1
    assert p['margins']['on_off']['sign']==-1
    assert p['margins']['on_on']['sign']==1
    assert p['original_vs_alternate_same_goal_strict_reversal'] is False
    assert p['matched_renderer_strict_reversal'] is True
    assert p['goal_shift_alternate']['exact']==['1','32']
    assert p['goal_shift_identity_residual']==['0','1']
    assert p['interaction_identity_residual']==['0','1']


@pytest.mark.parametrize('condition',['off','on'])
def test_exact_factorial_identities_and_physical_preference_not_reselected(condition):
    off,on,g0,g1=toy();on[2,1]=np.float32(.3)
    p=m.factorial_pair(off,on,g0,g1,1,2,condition)
    assert p['goal_shift_identity_residual']==p['interaction_identity_residual']==['0','1']
    assert p['original_condition']==condition


def test_full_saved_panel_all80_and_original_selection_unchanged():
    r,parent,goals,evidence=actual();out=m.analyze(r,parent,goals)
    assert len(out['pairs'])==80 and out['fixed_pairs']==out['available_pairs']==80 and out['unknown_pairs']==0
    assert [(p['anchor_index'],p['i'],p['j']) for p in out['pairs']]==[(a,i,j) for a in range(8) for i in range(5) for j in range(i+1,5)]
    assert [a['selected_word_unchanged'] for a in out['anchors']]==[a['seal']['selected_word'] for a in r['anchors']]
    assert out['P1_admission'] is False
    assert set(out['summary']['versus_physical'])=={'original_matched','alternate_original_goal','alternate_matched'}
    assert all(set(v)=={'native_distance','all_xy','wrapped_angle'} for v in out['summary']['versus_physical'].values())
    for anchor in out['anchors']:
        w=anchor['selected_word_unchanged'];zero=anchor['words'][0];selected=anchor['words'][w]
        assert anchor['selected_vs_zero_unchanged']['all_xy_delta']==selected['all_xy']-zero['all_xy']


@pytest.mark.parametrize('mutation',['physical_goal','goal_pose','prepost','swap_condition','missing_on','pixel','order'])
def test_real_goal_triplet_mutations_rejected(mutation):
    overlay=m.v2.teacher.overlay;source=m.v2.teacher.read(overlay.SOURCE,overlay.SOURCE_SHA)
    events=m.source_events(overlay.SOURCE.parent/'journal.jsonl')
    frames=deepcopy([e for e in events if e['kind']=='fresh_overlay_frame' and e['source_slot']==0 and e['key']=='s0_construction' and e['receipt']['reference_key']=='20'])
    r,parent,goals,evidence=actual();goal=deepcopy(goals[0,'off']['physical_goal'])
    if mutation=='physical_goal':goal[5]+=1
    elif mutation=='goal_pose':frames[1]['receipt']['pre']['goal_pose'][0]+=1;frames[1]['receipt']['post']['goal_pose'][0]+=1
    elif mutation=='prepost':frames[1]['receipt']['post']['snapshot']['state'][0]+=1
    elif mutation=='swap_condition':frames[1]['receipt']['with_target']=False
    elif mutation=='missing_on':frames.pop(1)
    elif mutation=='pixel':frames[1]['receipt']['pixel_sha256']='0'*64
    elif mutation=='order':frames[0],frames[1]=frames[1],frames[0]
    with pytest.raises(ValueError):m.validate_goal_triplet(source,frames,0,goal)


@pytest.mark.parametrize('mutation',['goal','condition','word','time','selected'])
def test_analysis_identity_mutations_rejected(mutation):
    original,parent,goals,evidence=actual();r=deepcopy(original)
    if mutation=='goal':r['rows'][0]['goal'][5]+=1
    elif mutation=='condition':r['rows'][0]['condition']='on'
    elif mutation=='word':r['rows'][0]['word']=1
    elif mutation=='time':r['rows'][0]['terminal_return']['step']=9
    elif mutation=='selected':r['anchors'][0]['seal']['selected_word']=(r['anchors'][0]['seal']['selected_word']+1)%5
    with pytest.raises(ValueError):m.analyze(r,parent,goals)


def test_exact_tie_not_called_reversal():
    z=np.zeros((5,192),np.float32);g=np.zeros(192,np.float32)
    p=m.factorial_pair(z,z,g,g,0,1,'on')
    assert all(v['sign']==0 for v in p['margins'].values())
    assert p['matched_renderer_strict_reversal'] is False


@pytest.mark.parametrize('mutation',['pixels','coordinated_preprocess','coordinated_latent'])
def test_goal_chain_mutations_rejected_independently(mutation,monkeypatch):
    original=m.legacy_array
    def changed(base,receipt):
        a=original(base,receipt);name=receipt['file']
        if mutation=='pixels' and name=='image_03_pixels.npy.gz':
            a=a.copy();a.flat[0]^=np.uint8(1)
        elif mutation=='coordinated_preprocess' and name in ('image_03_preprocessed.npy.gz','module_0006_input0.npy.gz'):
            a=a.copy();a.flat[0]+=np.float32(1)
        elif mutation=='coordinated_latent' and name in ('image_03_latent.npy.gz','module_0007_output.npy.gz'):
            a=a.copy();a.flat[0]+=np.float32(1)
        return a
    monkeypatch.setattr(m,'legacy_array',changed)
    with pytest.raises(ValueError):m.acquire()


def test_original_source_journal_hash_not_skipped(tmp_path):
    import json
    overlay=m.v2.teacher.overlay
    firstline=(overlay.SOURCE.parent/'journal.jsonl').read_bytes().splitlines()[0]
    event=json.loads(firstline);event['kind']='tampered_start'
    path=tmp_path/'journal.jsonl';path.write_bytes(m.v2.phys.adapter.paired.raw(event))
    with pytest.raises(ValueError,match='original paired source journal'):m.source_events(path)
