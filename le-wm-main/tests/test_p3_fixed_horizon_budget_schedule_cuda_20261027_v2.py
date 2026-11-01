"""Metadata callbacks exercise the real runner, never a neural implementation."""
from copy import deepcopy
import builtins
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p3_fixed_horizon_budget_schedule_cuda_20261027_v2 as runner
from experiments.spectral_reachability.d2p_planning_compute_ledger import Binding

ROOT=Path(__file__).resolve().parents[1]


class Metadata:
    def __init__(self,width):
        self.shape=(1,3,width)
        self.device=SimpleNamespace(type='cuda',index=0)


class HookModule:
    def __init__(self): self.hooks=[]
    def register_forward_pre_hook(self,hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda:self.hooks.remove(hook))
    def __call__(self,arg):
        for hook in self.hooks: hook(self,(arg,))
        return arg


class Model:
    def __init__(self,fail_at=None):
        self.action_encoder=HookModule();self.encoder=HookModule();self.calls=0;self.fail_at=fail_at
    def predict(self,embedding,action):
        self.calls+=1
        if self.calls==self.fail_at: raise RuntimeError('injected predict failure')


def packet():
    return dict(root_sha256='a'*64,input_sha256='b'*64,
        root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,
            latent_dim=2,action_dim=10,latent_history=[[0.,0.]]*3,action_history=[[0.]*10]*2),
        primitive_bank=[np.tile([i/16,0.],(5,1)).tolist() for i in range(16)],
        goal=[10.,10.],projection=np.eye(2).tolist())


def callback(model,mismatch=False):
    def replay(context,actions):
        c=deepcopy(context)
        for a in np.asarray(actions).reshape(-1,5,2):
            model.predict(Metadata(2),model.action_encoder(Metadata(10)))
            x=np.asarray(c['latent_history'][-1])+[float(a[0,0])+1,0.]
            c['latent_history']=c['latent_history'][1:]+[x.tolist()]
            c['action_history']=c['action_history'][1:]+[[float(a[0,0]),0.]*5]
        if mismatch and len(actions)==15: c['latent_history'][-1][0]+=.01
        return c
    return replay


def arm(mode='uniform',fail_at=None,mismatch=False):
    model=Model(fail_at)
    r=runner.run_arm(model=model,replay=callback(model,mismatch),p=packet(),
        binding=Binding('seal','task','a'*64,'model'),mode=mode)
    assert model.action_encoder.hooks==model.encoder.hooks==[]
    assert 'predict' not in vars(model)
    return r,model


@pytest.mark.parametrize('mode',['uniform','dynamic'])
def test_production_arm_counts_metadata_only(mode):
    r,m=arm(mode)
    assert r['status']=='completed' and m.calls==147
    assert [p['predict_samples'] for p in r['prediction_receipt']['phases']]==[144,3]
    assert r['action_encoder_samples']==147 and r['processed_action_context_tokens']==441
    assert r['visual_encoder_samples']==0 and r['whole_h3_replay_exact']


@pytest.mark.parametrize('fail_at',[1,16,17,144,145,147])
def test_failure_calls_not_refunded(fail_at):
    r,m=arm(fail_at=fail_at)
    assert r['status']=='incomplete'
    assert r['prediction_receipt']['attempted_predict_samples']==m.calls
    assert r['action_encoder_samples']==m.calls
    assert sum(p['failed_predict_calls'] for p in r['prediction_receipt']['phases'])==1
    assert not all(p['completed'] for p in r['prediction_receipt']['phases'])
    assert not r['P3_admission']


def test_whole_path_mismatch_rejected_with_costs_preserved():
    r,m=arm(mismatch=True)
    assert r['status']=='incomplete' and m.calls==147 and not r['whole_h3_replay_exact']


def test_pair_matching_requires_same_chi_and_contract():
    u,_=arm();d,_=arm('dynamic')
    assert runner.compare([d,u],['a'*64])[0]['classification']=='matched_full_H3'
    d['search']['signal']['chi']+=.1
    assert runner.compare([d,u],['a'*64])[0]['classification']=='unmatched'
    with pytest.raises(ValueError): runner.compare([u,u],['a'*64])


def test_unknown_blocked_bindings_not_treated_as_same_feedback():
    rows=[dict(root_sha256='a'*64,mode=m,status='blocked',binding=None,
               input_sha256='b'*64,search=None) for m in runner.METHODS]
    pair=runner.compare(rows,['a'*64])[0]
    assert pair['classification']=='unmatched' and not pair['same_input_and_feedback']


def test_real_packet_is_16_prefix_commands_not_compiler_suffix_or_old_prediction():
    source=runner.io.decode((ROOT/runner.COMPILATION).read_bytes())
    row=source['rows'][0]
    p=runner.packet(row)
    assert len(p['primitive_bank'])==16
    for bank,receipt in zip(p['primitive_bank'],row['compiled']['receipts']):
        assert bank==[receipt['program_live_commands'][0]]*5
    changed=deepcopy(row)
    changed['compiled']['receipts'][0]['predicted_terminal_vector']=['outcomes_are_not_inputs']
    changed['compiled']['receipts'][0]['jacobian']=['old_jacobians_not_used']
    assert runner.packet(changed)==p


def test_prepare_does_not_import_neural_or_simulator_and_validates(tmp_path,monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        assert name.split('.')[0] not in {'torch','mujoco','pymunk','gymnasium'},name
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    output=tmp_path/'prepared'
    receipt=runner.prepare(ROOT,output)
    digest=hashlib.sha256((output/'prepare.json').read_bytes()).hexdigest()
    body,identity=runner.validate_prepared(output,digest)
    assert identity==receipt['protocol_id']
    assert body['contract']['dynamic']=='K2=2+floor(4*chi+0.5); K3=8-K2'
    assert body['expected_predict_samples']==1176 and body['new_independent_roots']==0
    assert receipt['new_predict_samples']==0 and receipt['model_loaded'] is False
    assert len(body['packets'])==4
    with pytest.raises(ValueError,match='exclusive NEW'): runner.prepare(ROOT,output)
    with pytest.raises(ValueError,match='SHA256'): runner.validate_prepared(output,'0'*64)
    protocol=output/'protocol.json'
    with protocol.open('a') as stream: stream.write(' ')
    with pytest.raises(ValueError,match='byte binding'): runner.validate_prepared(output,digest)
