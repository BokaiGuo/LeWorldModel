"""CPU-only receipt/image arithmetic tests, not model or physical evidence."""
import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p1_checkpoint_action_camera_provenance_20261027_v1 as p


@pytest.mark.parametrize('lfs',[False,True])
def test_hf_object_binding(tmp_path,lfs):
    content=tmp_path/'content';content.write_bytes(b'fixed source')
    blob=content.read_bytes()
    digest=hashlib.sha256(blob).hexdigest() if lfs else hashlib.sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest()
    metadata=tmp_path/'metadata';metadata.write_text('a'*40+'\n'+digest+'\n123456\n')
    assert p.metadata_record(metadata,content,lfs=lfs)['digest_verified']
    content.write_bytes(b'changed')
    with pytest.raises(ValueError):p.metadata_record(metadata,content,lfs=lfs)


def test_metadata_shape_not_silently_accepted(tmp_path):
    content=tmp_path/'content';content.write_bytes(b'fixed')
    metadata=tmp_path/'metadata';metadata.write_text('missing revision')
    with pytest.raises(ValueError):p.metadata_record(metadata,content,lfs=True)


def test_only_selected_source_function_executes(tmp_path):
    source=tmp_path/'source.py'
    source.write_text("raise RuntimeError('module import forbidden')\ndef selected(x):\n return x+2\n")
    assert p.function(source,'selected',{})(3)==5
    with pytest.raises(ValueError):p.function(source,'missing',{})


def test_action_units_are_displacement_not_velocity():
    u=np.array([[.1,-.2]]*5)
    r=p.packet_units(100*u,u,[-.0087,.0068],[.2019,.2002])
    assert r['native_pixel_delta_to_live_token_exact']
    assert r['extra_div100_token_difference_max']>0


@pytest.mark.parametrize('damage',['double_division','wrong_shape','nonfinite','std'])
def test_reject_wrong_unit_contract(damage):
    u=np.ones((5,2))*.1;d=100*u;std=np.ones(2)
    if damage=='double_division':u=u/100
    if damage=='wrong_shape':d=d.reshape(10)
    if damage=='nonfinite':u[0,0]=np.nan
    if damage=='std':std[0]=0
    with pytest.raises(ValueError):p.packet_units(d,u,np.zeros(2),std)


def test_real_transform_implementations_on_asymmetric_fixture():
    rgb=np.zeros((224,224,3),np.uint8)
    rgb[20:80,90:190]=[239,83,17]
    rgb[100:160,10:30]=[17,61,219]
    r=p.image_checks([('fixture_only_not_physical',rgb)])[0]
    assert r['train_tensor_vs_live_max_abs']==0
    assert r['numpy_branch_is_xy_transpose'] and r['numpy_branch_vs_chw_max_abs']>0
    assert r['jpeg95_changed_channel_values']>0 and r['jpeg95_pixel_MAE']>0


def replay_fixture():
    return dict(status='passed',simulator='official',gates={'complete':True},
        episodes=[dict(episode=i,dense=dict(max_state_errors=[.00001],boundary_rgb_archive_mae=[.4]*5))
                  for i in range(16)])


def test_saved_replay_not_mistaken_for_exact_camera():
    r=p.saved_replay(replay_fixture())
    assert r['saved_episodes']==16 and r['saved_boundary_images']==80
    assert not r['exact_camera_identity_established']


@pytest.mark.parametrize('damage',['missing','duplicate','gate','nan'])
def test_saved_replay_fail_closed(damage):
    r=replay_fixture()
    if damage=='missing':r['episodes'].pop()
    if damage=='duplicate':r['episodes'][0]['episode']=1
    if damage=='gate':r['gates']['complete']=False
    if damage=='nan':r['episodes'][0]['dense']['boundary_rgb_archive_mae'][0]=np.nan
    with pytest.raises(ValueError):p.saved_replay(r)


def test_no_model_or_physics_call_in_source():
    import ast
    tree=ast.parse(p.Path(p.__file__).read_text())
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any(n.endswith(('.predict','.step','.reset','.render','.cuda','.load_pretrained')) for n in calls)
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='encode':
            # str.encode for Git object headers is unrelated to model.encode.
            assert isinstance(n.func.value,(ast.Constant,ast.JoinedStr))
    assert 'torch.load' not in calls
