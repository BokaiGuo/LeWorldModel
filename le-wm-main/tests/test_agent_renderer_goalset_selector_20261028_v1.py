"""RAM arithmetic/identity tests; no neural or physical execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_renderer_goalset_selector_20261028_v1 as m


def fixture():
    spec=importlib.util.spec_from_file_location('goalset_old_RAM',Path(__file__).with_name('test_agent_p1_terminal_witness_selector_20261028_v1.py'))
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    p,q,r=h.fixture()
    # Native float32 literal coordinates, not lossy arbitrary Python doubles.
    for c in r['candidates']:
        c['predicted_terminal']=np.asarray(c['predicted_terminal'],np.float32).astype(float).tolist()
        c['context']['latent_history'][-1]=deepcopy(c['predicted_terminal'])
        c['nonlinear_terminal_cost']=m.old.base.cost(c['predicted_terminal'],p['goal'],q)
    r['winner']=deepcopy(r['candidates'][0])
    slot=dict(index=0,root_slot=0,condition='off',task_id='t',task_sha256='s',blind_sha256='b',packet=deepcopy(p),q=deepcopy(q))
    a=dict(**deepcopy(slot),physical_goal=[1.]*7,physical_goal_sha256=m.digest([1.]*7),
        goal_vectors=np.stack([np.zeros(192,np.float32),np.full(192,2.,np.float32)]).tolist(),goalset_sha256='g')
    previous=m.old.select_panel(p,q,r,3)
    return slot,previous,r,a


def test_goalset_changes_terminal_selection_without_changing_old_scores():
    s,p,r,a=fixture();saved=deepcopy((s,p,r,a));out=m.select_panel(s,p,r,a,3)
    assert out['goalset_terminal']['index']==0
    assert out['original_stage_plus_ridge']==p['original_stage_plus_ridge']
    assert out['original_terminal_Q']==p['nonlinear_terminal_Q']
    assert out['candidate_denominator']==out['terminal_available']==3
    assert out['candidates'][0]['goalset_Q']==0.
    assert (s,p,r,a)==saved


def test_goal_branch_tie_off_and_candidate_tie_first_index():
    z=np.ones(192,np.float32);gs=np.stack([np.zeros(192,np.float32),np.full(192,2.,np.float32)])
    x=m.score(z,gs);assert x['active_branch']=='off' and x['branch_Q']==[1.,1.]
    s,p,r,a=fixture();a['goal_vectors'][1]=deepcopy(a['goal_vectors'][0])
    out=m.select_panel(s,p,r,a,3)
    assert out['goalset_terminal']['index']==p['nonlinear_terminal_Q']['index']


def test_full_dimension_mean_once_and_duplicates():
    z=np.zeros(192,np.float32);z[0]=2;g=np.zeros((2,192),np.float32)
    assert m.score(z,g)['value']==4/192
    assert m.score(z,g)['active_branch']=='off'


@pytest.mark.parametrize('kind',['dtype','shape','nan','bool'])
def test_score_rejects_invalid_native_arrays(kind):
    z=np.zeros(192,np.float32);g=np.zeros((2,192),np.float32)
    if kind=='dtype':z=z.astype(np.float64)
    elif kind=='shape':g=g[:1]
    elif kind=='bool':z=z.astype(bool)
    else:z[0]=np.nan
    with pytest.raises(ValueError):m.score(z,g)


@pytest.mark.parametrize('kind',['goal','goal_zero_sign','context','history_zero_sign','root','condition','task','blind','physical','q'])
def test_expected_slot_independently_copied_binding_mutations_rejected(kind):
    s,p,r,a=fixture()
    if kind=='goal':s['packet']['goal'][0]=1.
    elif kind=='goal_zero_sign':s['packet']['goal'][0]=-0.
    elif kind=='context':s['packet']['root_context']['latent_history'][0][0]=2.
    elif kind=='history_zero_sign':s['packet']['root_context']['action_history'][0][0]=-0.
    elif kind=='root':s['root_slot']=1
    elif kind=='condition':s['condition']='on'
    elif kind=='task':s['task_sha256']='wrong'
    elif kind=='blind':s['blind_sha256']='wrong'
    elif kind=='physical':a['physical_goal'][5]+=1
    else:s['q'][0][0]=2.
    with pytest.raises(ValueError):m.select_panel(s,p,r,a,3)


@pytest.mark.parametrize('kind',['missing','count','raw','terminal','prefix','order','previous'])
def test_candidate_witness_mutations_rejected_no_subset_argmin(kind):
    s,p,r,a=fixture()
    if kind=='missing':r['candidates'][0].pop('predicted_terminal')
    elif kind=='count':r['candidates'].pop()
    elif kind=='raw':r['candidates'][0]['commands'][0][0]=-0.
    elif kind=='terminal':r['candidates'][0]['predicted_terminal'][0]+=1.
    elif kind=='prefix':r['candidates'][0]['predicted_prefix'][0]+=1.
    elif kind=='order':r['candidates'].reverse()
    else:p['nonlinear_terminal_Q']['index']=0
    with pytest.raises((ValueError,KeyError)):m.select_panel(s,p,r,a,3)


def test_outcomes_do_not_enter_model_selection():
    s,p,r,a=fixture();expected=m.select_panel(s,p,r,a,3)
    r['native_success']=True;r['candidates'][0]['reward']=-1e30
    s['physical_outcome']={'winner':2}
    assert m.select_panel(s,p,r,a,3)==expected


def test_literal_rejects_unrepresentable_not_silently_cast():
    with pytest.raises(ValueError):m.native_literal([.1]*192,(192,))


def test_image_bridge_rejects_coordinated_preprocessing_and_bad_hidden():
    px=np.zeros((224,224,3),np.uint8)
    pre=m.goals.strict.v3.pixels.preprocess_pixels(px)
    hidden=np.zeros((1,257,192),np.float32);pin=hidden[:,0].copy();out=pin.copy();z=out[0].copy()
    m.check_image(px,pre,pre.reshape(1,3,224,224).copy(),hidden,pin,out,z)
    with pytest.raises(ValueError):m.check_image(px,pre+1,(pre+1).reshape(1,3,224,224),hidden,pin,out,z)
    with pytest.raises(ValueError):m.check_image(px,pre,pre.reshape(1,3,224,224),hidden[:,:197],pin,out,z)


def test_write_is_exclusive_and_result_tampering_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    p=tmp_path/'results'/f'{m.NAME}_RAM.json';v={'schema':m.NAME+'.analysis','x':1}
    receipt=m.write(p,v)
    assert receipt['sha256']==m.sha(p)
    with pytest.raises(ValueError):m.write(p,v)
    m.verify_saved(p,receipt['sha256'],v)
    with pytest.raises(ValueError):m.verify_saved(p,receipt['sha256'],dict(v,x=2))


def test_full1056_no_cross_library_union_or_pruning():
    total=0
    for family,n in [('allmode',42),('mixedmode',222)]:
        for index in range(4):
            s,p,r,a=fixture();template=deepcopy(r['candidates'])
            r['candidates']=[deepcopy(template[j%3]) for j in range(n)]
            for j,c in enumerate(r['candidates']):c['index']=j;c['candidate_id']=f'prefix_{j}'
            r['winner']=deepcopy(r['candidates'][0]);p=m.old.select_panel(s['packet'],s['q'],r,n)
            out=m.select_panel(s,p,r,a,n);total+=out['candidate_denominator']
            assert len(out['candidates'])==n and out['goalset_terminal']['index']==0
    assert total==1056


def test_real_goal_authority_and_current_prepares_without_outcomes(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('outcome-bearing acquire must not be used')
    monkeypatch.setattr(m.goals,'acquire',forbidden)
    a=m.acquire_targets();bindings={};m.validate_current_prepares(a,bindings)
    assert len(a['slots'])==4 and len(a['images'])==16 and len(bindings)==2
    assert [s['condition'] for s in a['slots']]==['off','on','off','on']
    for i in (0,2):
        assert a['slots'][i]['physical_goal']==a['slots'][i+1]['physical_goal']
        assert a['slots'][i]['goal_vectors']==a['slots'][i+1]['goal_vectors']


@pytest.mark.parametrize('kind',['pixels','preprocess','latent','history'])
def test_actual_saved_image_chain_mutations_rejected(kind,monkeypatch):
    original=m.goals.legacy_array
    def changed(folder,rec):
        a=original(folder,rec)
        if kind=='pixels' and rec['file']=='image_03_pixels.npy.gz':a=a.copy();a.flat[0]^=np.uint8(1)
        elif kind=='preprocess' and rec['file']=='image_03_preprocessed.npy.gz':a=a.copy();a.flat[0]+=np.float32(1)
        elif kind=='latent' and rec['file']=='image_03_latent.npy.gz':a=a.copy();a.flat[0]+=np.float32(1)
        elif kind=='history' and rec['file']=='image_00_latent.npy.gz':a=a.copy();a.flat[0]+=np.float32(1)
        return a
    monkeypatch.setattr(m.goals,'legacy_array',changed)
    with pytest.raises(ValueError):m.acquire_targets()


@pytest.mark.parametrize('kind',['goal','renderer','outline'])
def test_actual_native_goal_authority_not_only_matching_latent(kind,monkeypatch):
    original=m.source.root_metadata
    def changed(data):
        a=deepcopy(original(data))
        if kind=='goal':a[0]['goal'][5]+=1.
        elif kind=='outline':a[0]['goal_pose'][0]+=1.
        else:a[0]['renderer']['unexpected']=True
        return a
    monkeypatch.setattr(m.source,'root_metadata',changed)
    with pytest.raises(ValueError):m.acquire_targets()
