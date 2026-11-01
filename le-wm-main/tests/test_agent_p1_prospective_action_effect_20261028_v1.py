"""RAM-only prospective contract/causal scheduler tests, never native adapters."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import struct

import pytest

from experiments.spectral_reachability import agent_p1_prospective_action_effect_20261028_v1 as m


def raw(x):
    return struct.pack('<10f',*([x,0.0]*5)).hex()


def selector():
    candidates=[]; selections=[]
    for a in range(8):
        for w in range(5):
            commands=raw([0,.25,-.25,.5,-.5][w])
            candidates.append(dict(suffix_raw_bits=commands,full10_raw_bits=raw(0)+commands,
                original_plan=dict(index=a*5+w,root_slot=a//4,condition=('off','on')[(a//2)%2],
                    role=('Z','S')[a%2],library_index=w,library_label=m.WORDS[w],
                    anchor_id=f'anchor{a}',task_id=f'task{a}',blind_sha256=f'blind{a}',
                    plan_digest=f'plan{a}_{w}')))
        selections.append(dict(anchor_index=a,selected_original_index=a*5+1,
            selection_digest=f'selection{a}',commands_raw_bits=raw(.25)))
    return dict(candidates=candidates,selections=selections)


def job():
    return m.jobs(selector())[0]


class Fake:
    def __init__(self): self.events=[];self.sealed=False;self.requests=[]
    def replay20(self,j,word):
        self.events.append(('replay',word))
        return dict(fingerprint=dict(raw20_bits=raw(0)*2+j['prefix5_bits']+j['first_suffix5_bits'],
                state7=[0.0]*7,snapshot_sha256='a'*64,png_sha256='b'*64,pixel_sha256='c'*64),
            forward=dict(latents=[[0.0]*192]*3,history=[[0.0]*10]*2),handle=word)
    def predict(self,request):
        assert not self.sealed
        assert set(request)=={'schema','anchor_index','word','latents','history','raw5_bits'}
        self.requests.append(deepcopy(request));self.events.append(('predict',request['word']))
        return [float(request['word'])]*192
    def seal(self,value):
        assert len(self.requests)==5
        self.sealed=True;self.events.append(('seal',value['selected_word']))
        self.seal_value=deepcopy(value)
        return hashlib.sha256(m.canonical(value)).hexdigest()
    def advance5(self,start,bits):
        assert self.sealed
        self.events.append(('advance',start['handle']))
        return dict(complete=True,actual_raw5_bits=bits,terminal_state7=[0.0]*7,
                    terminated=False,truncated=False,terminal_absolute_step=25,endpoint=[float(start['handle'])+1]*192,
                    alternate_endpoint=[float(start['handle'])+1]*192,
                    renderer_audit=dict(before_sha256='d'*64,after_sha256='d'*64,restored=True))


def test_fixed8_40_jobs_and_model_only_prefix():
    jobs=m.jobs(selector())
    assert len(jobs)==8 and sum(len(j['words']) for j in jobs)==40
    assert [j['source_slot'] for j in jobs]==[0,0,0,0,1,1,1,1]
    assert all(j['first_suffix5_bits']==raw(.25) for j in jobs)


@pytest.mark.parametrize('change',['drop','order','root','chosen_bits','prefix','zero'])
def test_roster_and_action_mutation_rejected(change):
    s=selector()
    if change=='drop':s['candidates'].pop()
    if change=='order':s['candidates'][0],s['candidates'][1]=s['candidates'][1],s['candidates'][0]
    if change=='root':s['candidates'][0]['original_plan']['root_slot']=2
    if change=='chosen_bits':s['selections'][0]['commands_raw_bits']=raw(.5)
    if change=='prefix':s['candidates'][1]['full10_raw_bits']=raw(.25)+raw(.25)
    if change=='zero':s['candidates'][0]['suffix_raw_bits']=raw(.5)
    with pytest.raises(ValueError):m.jobs(s)


def test_predictions_and_selection_sealed_before_any_label():
    f=Fake();out=m.drive_anchor(job(),[0.0]*192,f)
    assert out['selected_word']==0 and len(out['branches'])==5
    assert [e[0] for e in f.events][:7]==['replay']+['predict']*5+['seal']
    assert sum(e[0]=='replay' for e in f.events)==5
    assert out['complete_pairs']==10 and out['unknown_pairs']==0
    assert out['policy_endpoint']==out['branches'][0]['endpoint']


def test_exact_tie_smallest_word_no_hidden_label_selection():
    f=Fake();out=m.drive_anchor(job(),[1.5]*192,f)
    assert out['selected_word']==1
    assert f.seal_value['selected_word']==1
    assert out['policy_endpoint']==[2.0]*192


@pytest.mark.parametrize('field',['state7','snapshot_sha256','png_sha256','raw20_bits'])
def test_replay_drift_is_unknown_not_pooled(field):
    f=Fake();original=f.replay20
    def replay(j,w):
        start=original(j,w)
        if w==2:start['fingerprint'][field]='changed'
        return start
    f.replay20=replay
    out=m.drive_anchor(job(),[0.0]*192,f)
    assert out['branches'][2]['endpoint'] is None
    assert out['complete_pairs']==6 and out['unknown_pairs']==4
    assert sum(e==('advance',2) for e in f.events)==0


def test_native_early_stop_does_not_make_five_step_label_or_fallback():
    f=Fake();advance=f.advance5
    def early(start,bits):
        out=advance(start,bits)
        if start['handle']==0:out.update(complete=False,terminal_absolute_step=23,terminated=True)
        return out
    f.advance5=early
    out=m.drive_anchor(job(),[0.0]*192,f)
    assert out['selected_word']==0 and out['policy_endpoint'] is None
    assert out['unknown_pairs']==4 and len(out['branches'])==5


def test_predict_failure_retains_all_slots_and_dispatches_nothing():
    f=Fake();predict=f.predict
    def fail(r):
        if r['word']==2:raise RuntimeError('synthetic failure')
        return predict(r)
    f.predict=fail
    out=m.drive_anchor(job(),[0.0]*192,f)
    assert out['status']=='prediction_or_seal_failed'
    assert len(out['branches'])==5 and out['unknown_pairs']==10
    assert not f.sealed and not any(e[0]=='advance' for e in f.events)


def test_non_float32_endpoint_rejected_before_selection():
    f=Fake();f.predict=lambda r:[.1]*192
    out=m.drive_anchor(job(),[0.0]*192,f)
    assert out['selected_word'] is None and out['unknown_pairs']==10


def test_response_contrast_cancels_shared_offset_but_not_goal_ranking():
    p=[[float(w)]*192 for w in range(5)]
    a=[[float(w)-3]*192 for w in range(5)]
    out=m.effects(p,a,[0.0]*192)
    assert out['zero_residual_mse']==m.pack(Fraction(9))
    assert all(x['response_error_mse']==m.pack(Fraction(0)) for x in out['words'])
    assert any(x['strict_reversal'] for x in out['pairs'])
    assert all(x['identity_residual']==m.pack(Fraction(0)) for x in out['pairs'])


def test_action_conditioned_response_error_nonzero_with_zero_reference_error():
    p=[[float(w)]*192 for w in range(5)];a=deepcopy(p);a[2]=[10.0]*192
    out=m.effects(p,a,[0.0]*192)
    assert out['zero_residual_mse']==m.pack(Fraction(0))
    assert out['words'][2]['response_error_mse']==m.pack(Fraction(64))


def test_unknown_zero_blocks_reference_effect_not_other_pair_labels():
    p=[[float(w)]*192 for w in range(5)];a=deepcopy(p);a[0]=None
    out=m.effects(p,a,[0.0]*192)
    assert out['zero_residual_mse'] is None
    assert all(x['response_error_mse'] is None for x in out['words'])
    assert sum(x['available'] for x in out['pairs'])==6


def test_forward_whitelist_rejects_goal_or_state_and_replay_has_full_state():
    r=dict(schema=m.NAME+'.forward',anchor_index=0,word=0,latents=[[0.0]*192]*3,
           history=[[0.0]*10]*2,raw5_bits=raw(0),state=[0.0]*7)
    with pytest.raises(ValueError):m.validate_forward(r)


def test_contract_no_dispatch_cli_or_fresh_authority():
    c=m.contract()
    assert c['new_forward_slots']==40 and c['new_image_encoding_cap']==88
    assert c['replay_public_step_cap']==1000 and c['fresh_roots']==0
    assert not c['run_cli_implemented'] and not c['P1_admission']


def test_actual20_forward_shift_and_real_history_normalized_once():
    old=[[1.0]*192,[2.0]*192,[3.0]*192]
    history=raw(.25)+raw(.5)
    f=m.forward20(old,[4.0]*192,history,[0.0,0.0],[.25,.5])
    assert [v[0] for v in f['latents']]==[2.0,3.0,4.0]
    assert f['history'][0]==[1.0,0.0]*5
    assert f['history'][1]==[2.0,0.0]*5
    assert old[0][0]==1.0


def test_forward20_no_hidden_imagined_context_or_zero_std():
    with pytest.raises(ValueError):m.forward20([[0.0]*192]*3,[0.0]*192,raw(0)*2,[0.0]*2,[0.0]*2)


def test_exclusive_prepare_and_hash_reader(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    p=tmp_path/'results'/f'{m.NAME}_fixture.json'
    receipt=m.write(p,{'run_authorized':False})
    assert m.read(p,receipt['sha256'])=={'run_authorized':False}
    with pytest.raises(FileExistsError):m.write(p,{})
    with pytest.raises(ValueError):m.read(p,'0'*64)
    with pytest.raises(ValueError):m.write(tmp_path/'results'/'foreign.json',{})


def test_forecast_seal_exception_does_not_dispatch_or_fallback():
    f=Fake()
    def fail(_):raise RuntimeError('durable seal failure')
    f.seal=fail
    out=m.drive_anchor(job(),[0.0]*192,f)
    assert out['selected_word'] is None and out['unknown_pairs']==10
    assert not any(e[0]=='advance' for e in f.events)


def test_prepare_full_synthetic_file_layout_and_tamper(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'context_proof',lambda value,jobs:dict(all_three_actual=True,full_saved_history_rebuilt=True))
    (tmp_path/'results').mkdir()
    for folder in ('tests','innerdocs','experiments/spectral_reachability'):
        (tmp_path/folder).mkdir(parents=True,exist_ok=True)
    for path in (tmp_path/'tests'/f'test_{m.NAME}.py',tmp_path/'innerdocs'/f'{m.NAME}.md'):
        path.write_text('synthetic closure file')
    s=selector()
    teacher=dict(anchors=[dict(anchor_id=f'anchor{a}',task_id=f'task{a}',condition=('off','on')[(a//2)%2],
        role=('Z','S')[a%2],root_slot=a//4,task_file=f'known{a}',task_sha256='a'*64) for a in range(8)],
        policy='synthetic_frozen_policy',expected_model_hash='a'*64)
    objects=dict(selector=s,margin=dict(fixed_pairs=80,summary=dict(model_to_observed=dict(available=76,unknown=4))),
                 teacher_prepare=teacher,context_history={},T3={},already_H1H2={},already_API={},already_rolling={})
    inputs={}
    for name,value in objects.items():
        p=tmp_path/'results'/f'{name}.json';p.write_text(json.dumps(value))
        inputs[name]=(str(p.relative_to(tmp_path)),m.sha(p))
    monkeypatch.setattr(m,'INPUTS',inputs)
    seam=tmp_path/'experiments/spectral_reachability/seam.py';seam.write_text('frozen seam')
    monkeypatch.setattr(m,'SEAMS',{'seam.py':m.sha(seam)})
    out=m.build()
    assert len(out['jobs'])==8 and out['run_authorized'] is False
    assert out['contract']['full_upstream_closure_rebuilt_by_preparer'] is False
    assert len(out['source_files'])==3 and len(out['pending'])==4
    seam.write_text('tampered seam')
    with pytest.raises(ValueError,match='seam SHA'):m.build()


def test_same_state_renderer_action_effect_difference():
    p=[[float(w)]*192 for w in range(5)];a=deepcopy(p);other=deepcopy(a);other[2]=[10.0]*192
    out=m.renderer_effects(a,other)
    assert out['words'][2]['effect_change_mse']==m.pack(Fraction(64))
    assert out['words'][1]['effect_change_mse']==m.pack(Fraction(0))


def test_renderer_physical_drift_keeps_original_labels_but_not_alt():
    f=Fake();advance=f.advance5
    def drift(start,bits):
        r=advance(start,bits)
        if start['handle']==2:r['renderer_audit']['after_sha256']='e'*64
        return r
    f.advance5=drift
    out=m.drive_anchor(job(),[0.0]*192,f)
    assert out['complete_pairs']==10
    assert out['branches'][2]['endpoint'] is not None
    assert out['branches'][2]['alternate_endpoint'] is None
    assert out['renderer_effects']['words'][2]['effect_change_mse'] is None


def context_snapshot():
    s=selector();js=m.jobs(s);rows=[]
    for j in js:
        a=j['anchor_index']
        for w in range(5):
            prefix=j['prefix5_bits'];suffix=j['words'][w]['raw5_bits']
            rows.append(dict(index=5*a+w,anchor_id=j['anchor_id'],plan_digest=f'plan{a}_{w}',
                root_slot=j['source_slot'],condition=j['condition'],role=j['role'],input_binding_status='saved_exact',
                history=dict(teacher_latent_absolute_steps=[5,10,15],
                    predictor_action_absolute_steps=[list(range(6,11)),list(range(11,16)),list(range(16,21))],
                    prefix_raw_bits=prefix,suffix_raw_bits=suffix,teacher_raw_bits=raw(0)+prefix,
                    full_public20_raw_bits=raw(0)*2+prefix+suffix),
                RGB_history_frames=[dict(phase=phase,step=step,condition=j['condition'])
                    for phase,step in [('warmup',5),('warmup',10),('prefix',5)]]))
    return dict(fixed_slots=40,rows=rows),js


def test_teacher15_all_three_actual_and_action_time_attestation():
    saved,js=context_snapshot()
    out=m.attest_context(saved,deepcopy(saved),js)
    assert out['all_three_actual'] and out['actual_latent_absolute_steps']==[5,10,15]
    assert out['checked_slots']==40 and out['full_saved_history_rebuilt']


@pytest.mark.parametrize('field',['imagined','time','condition','raw','rebuild'])
def test_context_attestation_fails_before_new_loop(field):
    saved,js=context_snapshot();rebuilt=deepcopy(saved)
    if field=='imagined':saved['rows'][0]['input_binding_status']='predicted_context'
    if field=='time':saved['rows'][0]['history']['teacher_latent_absolute_steps']=[0,5,15]
    if field=='condition':saved['rows'][0]['RGB_history_frames'][0]['condition']='on'
    if field=='raw':saved['rows'][0]['history']['teacher_raw_bits']=raw(.5)*2
    if field=='rebuild':rebuilt['rows'][0]['plan_digest']='different'
    else:rebuilt=deepcopy(saved)
    with pytest.raises(ValueError):m.attest_context(saved,rebuilt,js)
