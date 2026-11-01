"""Small full-coordinate fixtures; no neural/physics execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_terminal_witness_selector_20261028_v1 as s


def fixture():
    ctx=dict(layout='states_with_between_state_actions_v1',context_length=3,
             latent_dim=192,action_dim=10,latent_history=[[1.]*192]*3,
             action_history=[[0.]*10]*2)
    packet=dict(root_context=deepcopy(ctx),goal=[0.]*192)
    q=np.eye(192).tolist();cs=[]
    # Surrogate prefers0; nonlinear terminal prefers1 despite its uphill prefix.
    for i,(prefix,reach,terminal) in enumerate([(0.5,0.1,2.),(2.,0.,0.1),(1.,5.,1.)]):
        params=np.full(10,i/10,dtype=np.float32)
        raw=np.tanh(params).reshape(5,2).tolist()
        suffix=np.zeros((5,2),np.float32).tolist()
        after=deepcopy(ctx);after['latent_history']=deepcopy(ctx['latent_history'][1:])+[[prefix]*192]
        after['action_history']=np.concatenate([ctx['action_history'],s.base.kernel.normalized_history(raw)]).tolist()[-2:]
        end=deepcopy(after);end['latent_history']=deepcopy(after['latent_history'][1:])+[[terminal]*192]
        end['action_history']=s.base.kernel.normalized_history(raw+suffix).tolist()
        cs.append(dict(index=i,completed=True,candidate_id=f'prefix_{i}',proposal_kind=f'toy_{i}',
            q_sha256=s.base.digest(q),prefix_params=params.tolist(),suffix_params=[0.]*10,
            prefix_commands=raw,suffix_commands=suffix,commands=raw+suffix,
            predicted_prefix=[prefix]*192,predicted_terminal=[terminal]*192,
            stage_cost=prefix**2,own_reach_value=reach,objective=prefix**2+reach,
            nonlinear_terminal_cost=s.base.cost([terminal]*192,packet['goal'],q),
            context=end,independent_model_replay_exact=True))
    result=dict(candidates=cs,winner=deepcopy(cs[0]),q_sha256=s.base.digest(q))
    return packet,q,result


def test_separate_scores_uphill_winner_full_denominator():
    p,q,r=fixture();out=s.select_panel(p,q,r,3)
    assert out['candidate_denominator']==out['terminal_available']==3
    assert out['original_stage_plus_ridge']['index']==0
    assert out['nonlinear_terminal_Q']['index']==1
    assert out['candidates'][1]['model_prefix_uphill_from_anchor'] is True
    assert out['P1_admission'] is False


def test_stable_tie_index_not_input_order_selection():
    p,q,r=fixture();r['candidates'][2]['predicted_terminal']=[.1]*192
    r['candidates'][2]['context']['latent_history'][-1]=[.1]*192
    r['candidates'][2]['nonlinear_terminal_cost']=r['candidates'][1]['nonlinear_terminal_cost']
    assert s.select_panel(p,q,r,3)['nonlinear_terminal_Q']['index']==1


def test_missing_terminal_blocks_argmin_and_emits_literal_request():
    p,q,r=fixture();c=r['candidates'][1]
    for k in ('predicted_terminal','nonlinear_terminal_cost','context'):del c[k]
    out=s.select_panel(p,q,r,3)
    assert out['candidate_denominator']==3 and out['terminal_available']==2
    assert out['nonlinear_terminal_Q'] is None
    req=out['literal_forward_requests'][0]
    assert req['suffix_H1']['raw']==c['suffix_commands']
    assert req['full_H2']['commands']==c['commands']
    assert req['suffix_H1']['context']['latent_history'][-1]==c['predicted_prefix']
    assert out['original_stage_plus_ridge']['index']==0


@pytest.mark.parametrize('mutation', ['stage','reach','terminal','goal','raw','negative_zero','context','q','order','winner','count','nonfinite'])
def test_tamper_rejected(mutation):
    p,q,r=fixture();c=r['candidates'][0]
    if mutation=='stage':c['stage_cost']+=1
    elif mutation=='reach':c['objective']+=1
    elif mutation=='terminal':c['nonlinear_terminal_cost']+=1
    elif mutation=='goal':p['goal'][0]=1.
    elif mutation=='raw':c['commands'][0][0]=.5
    elif mutation=='negative_zero':c['prefix_commands'][0][0]=-0.
    elif mutation=='context':c['context']['action_history'][0][0]+=1
    elif mutation=='q':q[0][0]=2.
    elif mutation=='order':r['candidates'].reverse()
    elif mutation=='winner':r['winner']=deepcopy(r['candidates'][1])
    elif mutation=='count':r['candidates'].pop()
    else:c['own_reach_value']=float('nan')
    with pytest.raises((ValueError,AssertionError,RuntimeError)):
        s.select_panel(p,q,r,3)


def test_no_native_or_outcome_influence():
    p,q,r=fixture();expected=s.select_panel(p,q,r,3)
    r['native_reward']=1e10;r['candidates'][0]['physical_success']=True
    r['candidates'][1]['native_distance']=-1e20
    assert s.select_panel(p,q,r,3)==expected


def test_partial_terminal_cannot_be_disguised_as_missing():
    p,q,r=fixture();del r['candidates'][1]['predicted_terminal']
    with pytest.raises((ValueError,AssertionError,RuntimeError)):
        s.select_panel(p,q,r,3)


@pytest.mark.parametrize('family,count', [('allmode',42),('mixedmode',222)])
def test_original_real_shape_saved_execution_and_selector_interface(family,count):
    # Reuse upstream purely algebraic RAM producers, not production saved runs.
    filename=('test_agent_allmode_prefix_20261028_v2.py' if family=='allmode' else
              'test_agent_mixedmode_prefix_20261028_v1.py')
    spec=importlib.util.spec_from_file_location('terminal_selector_RAM_'+family,
        Path(__file__).with_name(filename))
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    slot,row,store=h.execute()
    module=s.base if family=='allmode' else s.mixed
    module.validate_execution(slot,row,lambda a:h.load(store,a))
    s.validate_raw_bits(slot,row,lambda a:h.load(store,a))
    out=s.select_panel(slot['packet'],slot['q'],row['result'],count)
    cs=row['result']['candidates']
    assert out['original_stage_plus_ridge']['index']==row['result']['winner']['index']
    assert out['nonlinear_terminal_Q']['index']==min(cs,key=lambda c:(c['nonlinear_terminal_cost'],c['index']))['index']
    assert len(out['candidates'])==out['terminal_available']==count
    assert out['literal_forward_requests']==[]


def test_array_loader_actual_descriptor_and_byte_mutation(tmp_path):
    from io import BytesIO
    import hashlib
    stream=BytesIO();np.save(stream,np.zeros((5,2),np.float32),allow_pickle=False)
    raw=stream.getvalue();(tmp_path/'raw.npy').write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest()
    desc=dict(file='raw.npy',sha256=digest,dtype='float32',shape=[5,2])
    meter=dict(array_reads=0,array_bytes_read=0)
    load=s.array_loader(tmp_path,dict(artifact_sha256={'raw.npy':digest}),meter)
    assert load(desc).shape==(5,2) and meter['array_bytes_read']==len(raw)
    (tmp_path/'raw.npy').write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    with pytest.raises((ValueError,AssertionError,RuntimeError)):load(desc)
