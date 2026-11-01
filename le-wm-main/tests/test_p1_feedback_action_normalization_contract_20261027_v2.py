"""Pure NumPy/receipt fixtures, not checkpoint-training or efficacy evidence."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p1_feedback_action_normalization_contract_20261027_v2 as c


def export_fixture():
    return dict(schema=c.EXPORT_SCHEMA,checkpoint=dict(weights_sha256='a'*64,config_sha256='b'*64),
        provenance=dict(kind='fitted_training_export',training_run_id='fixture only',
            dataset_action_bytes_sha256='c'*64,training_config_sha256='d'*64,normalizer_recipe_sha256='e'*64),
        normalizer=dict(mean=[1.,2.],std=[2.,4.],raw_action_divisor=1.,command_dim=2,frameskip=5,
                        packet_order='time_major_xy',formula='(raw/divisor-mean)/std',std_ddof=1))


def bound_fixture(tmp_path,divisor=1.):
    w=tmp_path/'weights'; w.write_bytes(b'fixture, not weights')
    config=tmp_path/'config'; config.write_text('{}')
    value=export_fixture(); value['checkpoint']=dict(weights_sha256=c.audit.sha(w),config_sha256=c.audit.sha(config))
    value['normalizer']['raw_action_divisor']=divisor
    path=tmp_path/'metadata.json'; path.write_text(json.dumps(value))
    return c.BoundActionNormalizer(path,trusted_export_sha256=c.audit.sha(path),weights=w,config=config),path,w,config


def test_valid_export_is_content_binding_not_authority():
    s=c.validate_export(export_fixture(),'a'*64,'b'*64)
    assert s==dict(mean=[1.,2.],std=[2.,4.],divisor=1.)


@pytest.mark.parametrize('damage',['schema','weight','config','archive','missing_run','no_recipe',
                                  'order','frameskip','negative_std','nan_mean','scale','formula','ddof'])
def test_reject_bad_export(damage):
    v=export_fixture()
    if damage=='schema': v['schema']='descriptive_archive_statistics'
    if damage=='weight': v['checkpoint']['weights_sha256']='f'*64
    if damage=='config': v['checkpoint']['config_sha256']='f'*64
    if damage=='archive': v['provenance']['kind']='archive_statistics'
    if damage=='missing_run': v['provenance']['training_run_id']=' '
    if damage=='no_recipe': v['provenance']['normalizer_recipe_sha256']=''
    if damage=='order': v['normalizer']['packet_order']='coordinate_major'
    if damage=='frameskip': v['normalizer']['frameskip']=1
    if damage=='negative_std': v['normalizer']['std'][0]=-1
    if damage=='nan_mean': v['normalizer']['mean'][0]=float('nan')
    if damage=='scale': v['normalizer']['raw_action_divisor']=0
    if damage=='formula': v['normalizer']['formula']='infer_from_outcomes'
    if damage=='ddof': v['normalizer']['std_ddof']=2
    with pytest.raises(ValueError): c.validate_export(v,'a'*64,'b'*64)


def test_bound_packet_order_float32(tmp_path):
    n,_,_,_=bound_fixture(tmp_path)
    raw=np.array([[-1.,1.],[0.,.1],[.2,.3],[.4,.5],[.6,.7]],np.float32)
    actual=n.normalize(raw)
    expected=((raw-np.array([1.,2.],np.float32))/np.array([2.,4.],np.float32)).reshape(10)
    assert actual.dtype==np.float32
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(n.normalize(np.stack([raw,raw])),np.stack([expected,expected]))


def test_explicit_divisor_not_guessed(tmp_path):
    n,_,_,_=bound_fixture(tmp_path,100.)
    np.testing.assert_array_equal(n.normalize(np.ones((5,2),np.float32)),
        np.tile((np.array([.01,.01],np.float32)-[1.,2.])/[2.,4.],5).astype(np.float32))


def test_tensor_boundary_preserves_gradient_dtype_device(tmp_path):
    import torch  # CPU arithmetic only, no model/encoder/predictor instantiated.
    n,_,_,_=bound_fixture(tmp_path)
    raw=torch.zeros((2,5,2),dtype=torch.float32,requires_grad=True)
    token=n.normalize_tensor(raw)
    assert token.shape==(2,10) and token.device==raw.device and token.dtype==raw.dtype
    np.testing.assert_array_equal(token.detach().numpy(),n.normalize(np.zeros((2,5,2),np.float32)))
    token.sum().backward()
    np.testing.assert_array_equal(raw.grad.numpy(),np.tile(np.array([.5,.25],np.float32),(2,5,1)))
    with pytest.raises(ValueError): n.normalize_tensor(torch.zeros(10))
    with pytest.raises(ValueError): n.normalize_tensor(torch.zeros((5,2),dtype=torch.int64))
    with pytest.raises(ValueError): n.normalize_tensor(torch.ones((5,2))*2)


@pytest.mark.parametrize('raw',[np.zeros((2,5)),np.zeros(10),np.full((5,2),np.nan),np.ones((5,2))*1.01])
def test_reject_wrong_raw_packet(tmp_path,raw):
    n,_,_,_=bound_fixture(tmp_path)
    with pytest.raises(ValueError): n.normalize(raw)


def test_no_default_or_self_certification(tmp_path):
    _,path,w,config=bound_fixture(tmp_path)
    with pytest.raises(TypeError): c.BoundActionNormalizer(path,weights=w,config=config)
    with pytest.raises(ValueError): c.BoundActionNormalizer(path,trusted_export_sha256='0'*64,weights=w,config=config)
    trusted=c.audit.sha(path); w.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError): c.BoundActionNormalizer(path,trusted_export_sha256=trusted,weights=w,config=config)


def prior_fixture():
    return dict(status='completed',gates={'complete':True},actual_predict_samples=3968,physical_future_branches=128,
        protocol=dict(modes=['identity_gn4','nuisance_gn4'],iterations=4,damping=.01,
                      root_ids=['0'*64,c.audit.ROOT,'2'*64,'3'*64]),
        rows=[dict(root_id=r,proposal_index=i,branches={m:dict(forecast={'metric_same_for_spectrum_and_GN':True})
            for m in ['identity_gn4','nuisance_gn4']}) for r in range(4) for i in range(16)])


def test_actual_prior_census_contract():
    assert c.check_prior_identity(prior_fixture())['not_exact_current_six_context_protocol']


@pytest.mark.parametrize('damage',['gate','missing','duplicate','mixed_objective','mode','budget'])
def test_do_not_treat_partial_or_mixed_prior_as_completed(damage):
    r=prior_fixture()
    if damage=='gate': r['gates']['complete']=False
    if damage=='missing': r['rows'].pop()
    if damage=='duplicate': r['rows'][0]=deepcopy(r['rows'][1])
    if damage=='mixed_objective': r['rows'][0]['branches']['identity_gn4']['forecast']['metric_same_for_spectrum_and_GN']=False
    if damage=='mode': r['protocol']['modes']=['identity_gn4']
    if damage=='budget': r['actual_predict_samples']=0
    with pytest.raises(ValueError): c.check_prior_identity(r)


def witness_fixture():
    feedback={'cases':[]}; plans=[]
    for role in ['selected','greedy','zero']:
        prefix=np.ones((5,2),np.float32)*.1
        feedback['cases'].append(dict(role=role,prefix=prefix.tolist(),between_raw=np.zeros((2,5,2)).tolist()))
        mean,std=c.audit.public_action_stats(c.REPO/'experiments/run_pusht_counterfactual.py')
        history=(np.stack([np.zeros((5,2)),prefix]).astype(np.float32).reshape(2,10)-mean)/std
        for arm in ['imagined','actual']:
            candidates=[dict(candidate_id=f'suffix_{i}',suffix_params=[0.]*10,
                commands=prefix.tolist()+np.zeros((5,2)).tolist()) for i in range(10)]
            plans.append(dict(role=role,arm=arm,root_sha256=c.audit.ROOT,root_index=1,prefix=prefix.tolist(),
                              context=dict(action_history=history.tolist()),candidates=candidates))
    return feedback,plans


def test_all_six_sixty_witnesses_no_guessed_replacement():
    rows=c.witnesses(*witness_fixture())
    assert len(rows)==6 and sum(len(r['candidates']) for r in rows)==60
    assert all(r['bound_incoming_tokens'] is None for r in rows)
    assert all(x['bound_model_token'] is None for r in rows for x in r['candidates'])


@pytest.mark.parametrize('damage',['history','order','prefix','candidate','suffix','root'])
def test_witness_binding_errors(damage):
    f,p=witness_fixture()
    if damage=='history': p[0]['context']['action_history'][0][0]+=.1
    if damage=='order': p[0],p[1]=p[1],p[0]
    if damage=='prefix': p[0]['prefix'][0][0]=.2
    if damage=='candidate': p[0]['candidates'].pop()
    if damage=='suffix': p[0]['candidates'][0]['commands'][5][0]=.1
    if damage=='root': p[0]['root_index']=2
    with pytest.raises(ValueError): c.witnesses(f,p)


def test_bound_witnesses(tmp_path):
    n,_,_,_=bound_fixture(tmp_path)
    rows=c.witnesses(*witness_fixture(),n)
    assert all(np.asarray(r['bound_incoming_tokens']).shape==(2,10) for r in rows)


def test_no_neural_import_or_run_entrypoint():
    import ast
    tree=ast.parse(c.Path(c.__file__).read_text())
    names=[v.name for node in ast.walk(tree) if isinstance(node,ast.Import) for v in node.names]
    names += [node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert not any(n and n.split('.')[0] in {'torch','gym','gymnasium','stable_worldmodel'} for n in names)
    assert not hasattr(c,'run')
