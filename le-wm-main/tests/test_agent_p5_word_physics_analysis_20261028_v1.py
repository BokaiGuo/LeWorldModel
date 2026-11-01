"""RAM saved-schema arithmetic; no simulator/model and no scientific evidence."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_word_physics_analysis_20261028_v1 as m


def frame(frame, word, move_block=False, twist=0.):
    token = dict(xy=1.,yx=2.,x_minus_x=-1.,zero_zero=0.)[word] if frame>10 else 0.
    public = dict(bodies=dict(
        agent=dict(position=[token*(frame-10),3.],velocity=[token,0.]),
        block=dict(position=[9.+(token if move_block else 0.),7.],angle=twist,
                   velocity=[0.,0.],angular_velocity=0.)),
        env=dict(coverage_arr=[.2]*frame,n_contact_points=0,
                 goal_pose=dict(values=[256.,256.,float(np.pi/4)],dtype='<f8',shape=[3]),
                 success_threshold=.95))
    transition = dict(kind='step',done=False,reward=float(np.clip(.2/.95,0,1)),
                      info=dict(n_contacts=0,final_coverage=.2,max_coverage=.2))
    image = np.zeros((224,224,3),np.uint8)
    image[0,0,:] = int(abs(token)*20)
    return public,transition,image


def fixture():
    branches,rows,models = [],[],[]
    for seed in m.SEEDS:
        for word in m.WORDS:
            branch = dict(seed=seed,word=word,case=f'seed_{seed}_{word}',root_sha256=f'root_{seed}',
                          native_goal=frame(10,word)[0]['env']['goal_pose'])
            branches.append(branch)
            rows.append(dict(**branch,status='completed',error=None))
        models.append(dict(seed=seed,root_sha256=f'root_{seed}',status='completed',meter=dict(saved_fixture=True),
            contrasts=dict(xy_vs_yx_patch_mse=1.,x_minus_x_vs_zero_patch_mse=2.,interpretation='model-only')))
    return branches,rows,models


def calculate(branches,rows,models,loader=None):
    return m.assemble(branches,rows,models,loader or (lambda row:lambda f:frame(f,row['word'])))


def test_all_frames_separated_and_model_parallel():
    data = calculate(*fixture())
    assert len(data['baselines'])==12
    assert len(data['suffix_frame_rows'])==120
    assert len(data['pair_frame_rows'])==60
    assert len(data['model_only_parallel'])==3
    assert [(x['seed'],x['word'],x['frame']) for x in data['suffix_frame_rows']]==[
        (s,w,f) for s in m.SEEDS for w in m.WORDS for f in range(11,21)]
    for pair in data['pair_frame_rows']:
        assert pair['delta']['agent_xy']['l2']>0
        assert pair['delta']['agent_velocity']['l2']>0
        assert pair['delta']['block_xy']['l2']==0
        assert pair['delta']['block_angle']['principal_left_minus_right_rad']==0
    assert any(x['rgb']['changed_pixels']>0 for x in data['pair_frame_rows'])
    assert not any(x['any_exact_block_pose_difference'] for x in data['pair_descriptive_summaries'])
    assert all(x['task_success'] is None and x['object_noncommutativity_claim'] is False for x in data['pair_descriptive_summaries'])
    assert data['model_only_parallel'][0]['model_only_contrasts']['xy_vs_yx_patch_mse']==1.


def test_intermediate_block_motion_not_hidden_by_same_endpoint():
    def loader(row):
        def load(f):
            p,t,rgb = frame(f,row['word'],move_block=f==13)
            return p,t,rgb
        return load
    result = calculate(*fixture(),loader=loader)
    pair = [x for x in result['pair_frame_rows'] if x['left_word']=='xy' and x['seed']==m.SEEDS[0]]
    assert pair[2]['delta']['block_xy']['l2']>0
    assert pair[-1]['delta']['block_xy']['l2']==0
    summary = result['pair_descriptive_summaries'][0]
    assert summary['any_exact_block_pose_difference']
    assert not summary['object_noncommutativity_claim']


def test_raw_pixels_signed_before_subtraction():
    left = np.zeros((224,224,3),np.uint8)
    right = left.copy()
    right[0,0,0]=255
    d = m.rgb_difference(left,right)
    assert d['max_abs_intensity']==255
    assert d['changed_pixels']==d['changed_channels']==1
    assert d['mse_intensity_squared']==255**2/(224*224*3)
    assert d['mae_intensity']==255/(224*224*3)
    assert d['channel_mse_intensity_squared'][0]==255**2/(224*224)
    assert m.rgb_difference(left,left)['exact_equal']


@pytest.mark.parametrize('bad',[np.zeros((224,224,3),np.float32),np.zeros((224,224,4),np.uint8)])
def test_bad_rgb_rejected(bad):
    with pytest.raises(ValueError,match='RGB'):
        m.rgb_difference(bad,bad)


def test_angle_units_kept_separate():
    a = m.quantities(*frame(11,'xy')[:2],11)
    b = copy.deepcopy(a)
    a['block_angle_native_rad']=2*np.pi+.01
    d = m.difference(a,b)
    assert d['block_angle']['left_minus_right_native_rad']>6
    assert d['block_angle']['principal_left_minus_right_rad']==pytest.approx(.01)
    assert d['block_xy']['l2']==0
    assert 'total_cost' not in m.policy()


def test_contact_points_are_not_unique_contacts():
    p,t,_ = frame(11,'xy')
    p['env']['n_contact_points']=11
    t['info']['n_contacts']=2
    q = m.quantities(p,t,11)
    assert q['callback_contact_points']==11 and q['returned_n_contacts']==2
    t['info']['n_contacts']=11
    with pytest.raises(ValueError,match='contact formula'):
        m.quantities(p,t,11)


def test_high_coverage_still_not_done_or_success():
    p,t,_ = frame(11,'xy')
    p['env']['coverage_arr'][-1]=1.
    t['info']['final_coverage']=t['info']['max_coverage']=t['reward']=1.
    q = m.quantities(p,t,11)
    assert not q['native_done'] and q['task_success'] is None
    t['done']=True
    with pytest.raises(ValueError,match='done=False'):
        m.quantities(p,t,11)


@pytest.mark.parametrize('mutation',['missing','reorder','root','failed','model_seed','model_root','negative_mse','goal'])
def test_full_denominator_and_binding_mutations(mutation):
    branches,rows,models = fixture()
    if mutation=='missing':
        rows.pop()
    elif mutation=='reorder':
        rows.reverse()
    elif mutation=='root':
        rows[0]['root_sha256']='other'
    elif mutation=='failed':
        rows[0]['status']='not_run'
    elif mutation=='model_seed':
        models[0]['seed']+=1
    elif mutation=='model_root':
        models[0]['root_sha256']='other'
    elif mutation=='negative_mse':
        models[0]['contrasts']['xy_vs_yx_patch_mse']=-1.
    else:
        branches[0]['native_goal']['values'][0]+=1
    with pytest.raises(ValueError):
        calculate(branches,rows,models)


def test_baseline_identity_required():
    def loader(row):
        def load(f):
            p,t,img = frame(f,row['word'])
            if f==10 and row['word']=='yx':
                p['bodies']['agent']['position'][0]+=1
            return p,t,img
        return load
    with pytest.raises(ValueError,match='baseline'):
        calculate(*fixture(),loader=loader)


@pytest.mark.parametrize('key',['position','velocity','angle','angular_velocity'])
def test_missing_native_quantity_not_zero(key):
    p,t,_ = frame(11,'xy')
    del p['bodies']['block'][key]
    with pytest.raises(KeyError):
        m.quantities(p,t,11)


def test_pinned_real_receipt_only_read_not_analysis():
    for relative,h in m.PINS.items():
        assert m.sha(m.ROOT/relative)==h
    path = m.ROOT/'results/consume_p5_physical_MAIN_20261028_v1_MAIN01.json'
    receipt = m.read_json(path)
    assert receipt['status']=='all12_saved_native_replays_verified'
    assert len(receipt['verified_cases'])==receipt['denominator']==12


def test_missing_authority_fails_before_consumer_import(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    def forbidden(name):
        pytest.fail('must not import before actual receipt exists')
    monkeypatch.setattr(m,'load_source',forbidden)
    with pytest.raises(ValueError,match='regular nonsymlink'):
        m.analyze()


def test_saved_loader_layout_and_mutation(tmp_path):
    case = 'seed_1_xy'
    frame_dir = tmp_path/case/'frame_11'
    frame_dir.mkdir(parents=True)
    p,t,img = frame(11,'xy')
    for name,value in [('full_public_state.json',p),('transition.json',t)]:
        (frame_dir/name).write_bytes(m.canonical(value))
    with (frame_dir/'rgb.npy').open('wb') as f:
        np.save(f,img,allow_pickle=False)
    records = [dict(frame=f,files={}) for f in range(21)]
    records[11]['files'] = {path.name:dict(sha256=m.sha(path),bytes=path.stat().st_size) for path in frame_dir.iterdir()}
    loader = m.record_loader(tmp_path,dict(case=case,records=records))
    assert np.array_equal(loader(11)[2],img)
    (frame_dir/'transition.json').write_text('{}')
    with pytest.raises(ValueError,match='SHA/size'):
        loader(11)


def test_write_verify_no_overwrite_and_output_hash(monkeypatch,tmp_path):
    result = dict(schema=m.TAG,policy=m.policy(),data=calculate(*fixture()))
    monkeypatch.setattr(m,'analyze',lambda:result)
    out = tmp_path/'analysis.json'
    assert m.main(['analyze','--output',str(out)])==0
    assert m.main(['verify','--output',str(out),'--output-sha256',m.sha(out)])==0
    with pytest.raises(ValueError,match='exclusive'):
        m.main(['analyze','--output',str(out)])
    with pytest.raises(ValueError,match='actual input SHA'):
        m.main(['verify','--output',str(out),'--output-sha256','0'*64])
    changed = copy.deepcopy(result)
    changed['data']['suffix_frame_rows'].pop()
    out.write_bytes(m.canonical(changed))
    with pytest.raises(ValueError,match='deterministic'):
        m.main(['verify','--output',str(out),'--output-sha256',m.sha(out)])


def test_strict_json_and_symlink(tmp_path):
    p = tmp_path/'bad.json'
    p.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError,match='duplicate'):
        m.read_json(p)
    p.write_text('{"x":NaN}')
    with pytest.raises(ValueError,match='nonfinite'):
        m.read_json(p)
    link = tmp_path/'link'
    link.symlink_to(p)
    with pytest.raises(ValueError,match='nonsymlink'):
        m.sha(link)
