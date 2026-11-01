"""Small saved-file mocks and arithmetic adapter tests; no CUDA/physics."""
from copy import deepcopy
import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import p4_feedback_root_runtime_20261027_v1 as m
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal


def dump(path,value): path.write_text(json.dumps(value,sort_keys=True,allow_nan=False))


def ctx(latents,raw):
    return dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.asarray(latents).tolist(),action_history=m.normalize(raw).tolist())


def vector(value): return np.full(192,value).tolist()


def frame(value): return np.full((224,224,3),value,np.uint8)


def pack(folder,root):
    """EXPLICIT MOCK source pack; fixture bytes are NOT real scientific evidence."""
    folder.mkdir(); cap_dir=folder/'cap'; cap_dir.mkdir(); res_dir=folder/'response'; res_dir.mkdir()
    policy=folder/'policy'; policy.mkdir(); (policy/'weights.pt').write_bytes(b'mock-not-a-checkpoint')
    dump(policy/'config.json',{'mock':True})
    producer=folder/'mock_source.py'; producer.write_text('MOCK_ONLY = True\n')
    seal=create_seal(files=[producer],args_json='{"mock":true}',output_dir=folder/'producer_seal')
    raw=np.zeros((2,5,2)).tolist(); prefix=np.zeros((5,2)).tolist()
    frames=np.stack([frame(16),frame(32),frame(48)])
    root_rgb=folder/'root.npy'; np.save(root_rgb,frames,allow_pickle=False)
    prefix_rgb=folder/'prefix.png'; Image.fromarray(frame(64)).save(prefix_rgb)
    identity=dict(history=[[0.,0.]]*3,history_rgb_sha256=m.hashlib.sha256(frames.tobytes()).hexdigest())
    plans=[]; rows=[]
    for role in ('selected','greedy','zero'):
        for i in range(41):
            method=role+('_zero' if i==0 else '_probe_'+str(i))
            p=dict(root_sha256=root,role=role,method=method,commands=prefix+prefix,
                saved_candidate=dict(predicted_prefix=vector(3.)))
            p['selection_sha256']=m.content_sha(p); plans.append(p)
            rows.append(dict(root_sha256=root,method=method,selection_sha256=p['selection_sha256'],
                status='completed',completed_commands=10,full_horizon_completed=True,
                trace=[dict(command=[0.,0.]) for _ in range(10)],context_identity=identity))
    dump(cap_dir/'protocol.json',dict(plans=plans))
    slot=dict(root_sha256=root,method='selected_zero',kind='prefix',status='observed',primitive_step=5,
        rgb_path=str(prefix_rgb),rgb_sha256=m.digest(prefix_rgb))
    enc=dict(slot,status='completed',comparable=True,vector=vector(4.))
    cap=dict(schema='pusht_geometry_capability_20261027_v1',status='diagnostic_recorded',gates={'mock':True},
        protocol_sha256=m.digest(cap_dir/'protocol.json'),census=dict(qualified_roots=[root],complete=True),
        source_seals=[seal],rows=rows,endpoint_receipts=[slot],encodings=[enc])
    dump(cap_dir/'report.json',cap)
    bindings={str(p):m.digest(p) for p in (cap_dir/'report.json',root_rgb,prefix_rgb,policy/'weights.pt',policy/'config.json')}
    response=dict(schema='pusht_geometry_feedback_response_20261027_v1',root_sha256=root,
        capability=str(cap_dir/'report.json'),bindings=bindings,training=False,fit=False,heldout_selection=False,
        goal=vector(9.),q=np.eye(192).tolist(),initial_context=ctx([vector(0.),vector(1.),vector(2.)],raw),
        cases=[dict(role='selected',prefix=prefix,between_raw=raw,saved_predicted_prefix=vector(3.),
                    saved_actual_prefix=vector(4.),prefix_rgb=str(prefix_rgb))],root_rgb=str(root_rgb),policy=str(policy))
    dump(res_dir/'protocol.json',response)
    dump(res_dir/'prepare_receipt.json',dict(status='prepared_not_run',protocol_sha256=m.digest(res_dir/'protocol.json'),source_seals=[seal]))
    prior_path=folder/'mock_prior_batch.json'
    dump(prior_path,dict(schema='p4_multiround_feedback_runtime_20261027_v1',status='calibration_recorded',failures=[],
        rows=[dict(root_sha256='f'*64,arm=arm,status='completed',completed_packets=4,selected_returns=20) for arm in m.ARMS]))
    args=dict(root_sha256=root,prior_root_sha256s=['f'*64],capability_path=cap_dir/'report.json',
        prior_batch_refs=[dict(path=str(prior_path),sha256=m.digest(prior_path))],
        capability_sha256=m.digest(cap_dir/'report.json'),response_path=res_dir/'protocol.json',
        response_sha256=m.digest(res_dir/'protocol.json'),response_prepare_path=res_dir/'prepare_receipt.json',
        response_prepare_sha256=m.digest(res_dir/'prepare_receipt.json'))
    binding=m.bind_sources(**args); sha=m.content_sha(binding)
    runtime_seal=create_seal(files=[Path(m.__file__)],args_json=json.dumps(dict(binding_sha256=sha)),output_dir=folder/'runtime_seal')
    return dict(args=args,binding=binding,sha=sha,seal=runtime_seal,root=root)


@pytest.fixture(scope='module')
def packs(tmp_path_factory):
    base=tmp_path_factory.mktemp('p4_root_adapter_mock')
    return [pack(base/'one','1'*64),pack(base/'two','2'*64)]


def adapter(packs,arm='actual',index=0):
    p=packs[index]
    return m.FeedbackAdapter(p['binding'],p['sha'],arm,p['seal'])


def execution(a,start,value,packet=None):
    image=frame(value); packet=np.zeros((5,2)).tolist() if packet is None else packet
    return dict(root_sha256=a.root,arm=a.arm,boundary=start,status='completed',completed_commands=5,
        remaining_commands_not_executed=0,guard_authorized=True,terminal_pixel_sha256=m.rgb_sha(image),
        reset_identity_sha256=a.binding['reset_identity_sha256'],
        trace=[dict(primitive_step=start+i+1,command=packet[i],pixel_sha256=m.rgb_sha(image)) for i in range(5)]),image


def prefix(a):
    receipt,image=execution(a,0,64)
    predicted=ctx([vector(1.),vector(2.),vector(3.)],a.between)
    a.accept_prefix(receipt,image,predicted)


def observe(a):
    previous=m.context(a.source['initial_context'])[0] if a.last_observed is None else m.context(a.last_observed)[0]
    observed=ctx([previous[1],previous[2],vector(4. if a.boundary==5 else a.boundary)],a.between)
    hashes=[m.rgb_sha(f) for f in a.frames]
    return dict(root_sha256=a.root,arm=a.arm,boundary=a.boundary,purpose='real_feedback',context=observed,
        frame_pixel_sha256=hashes,between_raw=a.between.tolist(),encoding_receipt=encoding(a,observed,hashes,'real_feedback'))


def encoding(a,value,frames,purpose):
    return dict(root_sha256=a.root,arm=a.arm,boundary=a.boundary,purpose=purpose,
        binding_sha256=a.binding_sha,device='cuda:0',model_bindings=a.binding['model_bindings'],
        frame_pixel_sha256=frames,value_sha256=m.content_sha(value),errors=[],
        between_raw_sha256=m.content_sha(a.between.tolist()),encoded_frames=len(frames),
        encoder_attempted=1,encoder_completed=1,projector_attempted=1,projector_completed=1)


def plan(a,request):
    candidates=[]
    for i in range(10):
        terminal=vector(float(i)); candidates.append(dict(candidate_id=str(i),predicted_terminal=terminal,
            commands=np.zeros((5,2)).tolist(),Q_cost=float((9-i)**2)))
    selected=candidates[-1]; before=m.context(request['chosen_context'])[0]
    successor=ctx([before[1],before[2],selected['predicted_terminal']],a.between)
    return dict(root_sha256=a.root,arm=a.arm,boundary=a.boundary,binding_sha256=a.binding_sha,
        context=request['chosen_context'],goal_q_sha256=a.binding['goal_q_sha256'],candidates=candidates,
        selected_candidate_id=selected['candidate_id'],commands=selected['commands'],
        successor=successor,verification_successor=deepcopy(successor))


@pytest.mark.parametrize('index',[0,1])
@pytest.mark.parametrize('arm',['actual','imagined'])
def test_parameterized_three_round_mock(packs,index,arm):
    a=adapter(packs,arm,index); prefix(a)
    for boundary in (5,10,15):
        request=a.planning_request(observe(a)); assert request['feedback_used_for_selection']==(arm=='actual')
        if arm=='imagined': assert request['chosen_context']==a.predicted
        else: assert request['chosen_context']==request['observed_context']
        auth=a.accept_selection(plan(a,request),m.content_sha)
        receipt,image=execution(a,boundary,64+boundary)
        a.record_execution(auth['selection_sha256'],receipt,image)
    assert a.phase=='completed20' and len(a.decisions)==3
    image_sha=m.rgb_sha(a.frames[-1]); native=vector(8.)
    result=a.record_terminal_evaluation(dict(root_sha256=a.root,arm=arm,boundary=20,
        purpose='terminal_evaluation_only',pixel_sha256=image_sha,native192=native,
        encoding_receipt=encoding(a,native,[image_sha],'terminal_evaluation_only')))
    assert result['Q_cost']==1.
    report=a.receipt()
    assert report['executed_packets']==4 and not report['P4_admission'] and not report['actual_neural_execution_verified']


def test_two_roots_not_one_hardbound_constant(packs):
    assert packs[0]['binding']['root_sha256']!=packs[1]['binding']['root_sha256']
    assert packs[0]['sha']!=packs[1]['sha']
    assert not packs[0]['binding']['fresh_holdout_verified']


@pytest.mark.parametrize('field',['prior_root_sha256s','excluded_root_sha256s'])
def test_prior_and_excluded_root_rejected(packs,field):
    args=deepcopy(packs[0]['args']); args[field]=[args['root_sha256']]
    with pytest.raises(ValueError,match='same/prior/excluded'): m.bind_sources(**args)


def test_missing_pack_no_placeholder_or_write(tmp_path):
    args=dict(root_sha256='1'*64,prior_root_sha256s=['2'*64],prior_batch_refs=[])
    for name in ('capability','response','response_prepare'):
        args[name+'_path']=tmp_path/'missing.json'; args[name+'_sha256']='0'*64
    with pytest.raises(ValueError): m.bind_sources(**args)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('key',['capability_sha256','response_sha256','response_prepare_sha256'])
def test_wrong_pinned_hash(packs,key):
    args=deepcopy(packs[0]['args']); args[key]='0'*64
    with pytest.raises(ValueError): m.bind_sources(**args)


def test_runtime_seal_wrong_root_binding(packs):
    p=packs[0]
    with pytest.raises(ValueError,match='runtime seal'): m.FeedbackAdapter(p['binding'],p['sha'],'actual',packs[1]['seal'])


def test_invented_prior_root_roster_rejected(packs):
    args=deepcopy(packs[0]['args']); args['prior_root_sha256s']=['a'*64]
    with pytest.raises(ValueError,match='prior-root roster'): m.bind_sources(**args)


def test_missing_prior_batch_evidence_rejected(packs):
    args=deepcopy(packs[0]['args']); args['prior_batch_refs']=[]
    with pytest.raises(ValueError,match='actual prior-batch'): m.bind_sources(**args)


def test_tampered_runtime_binding(packs):
    p=packs[0]; binding=deepcopy(p['binding']); binding['root_sha256']='2'*64
    with pytest.raises(ValueError): m.validate_binding(binding,p['sha'])


@pytest.mark.parametrize('kind',['missing_candidate','root','raw_history','prefix_latent','cap_prefix_rgb'])
def test_source_pack_corruption_mock(packs,monkeypatch,kind):
    original=m.read; args=packs[0]['args']
    def altered(path):
        value=original(path)
        if Path(path)==args['capability_path']:
            if kind=='missing_candidate': value['rows'].pop()
            if kind=='cap_prefix_rgb': value['endpoint_receipts'][0]['rgb_sha256']='bad'
        if Path(path)==args['response_path']:
            if kind=='root': value['root_sha256']='2'*64
            if kind=='raw_history': value['cases'][0]['between_raw'][0][0][0]=1.
            if kind=='prefix_latent': value['cases'][0]['saved_predicted_prefix'][0]=99.
        return value
    monkeypatch.setattr(m,'read',altered)
    with pytest.raises(ValueError): m.bind_sources(**args)


@pytest.mark.parametrize('field,value',[
    ('root_sha256','2'*64),('arm','imagined'),('boundary',10),('purpose','terminal_evaluation_only'),
    ('frame_pixel_sha256',['wrong']*3),('between_raw',np.ones((2,5,2)).tolist()),
])
def test_observation_identity_fail_closed(packs,field,value):
    a=adapter(packs); prefix(a); obs=observe(a); obs[field]=value
    with pytest.raises(ValueError): a.planning_request(obs)
    assert a.phase=='failed'
    with pytest.raises(ValueError): a.planning_request(observe(a))


@pytest.mark.parametrize('change',['context','metric','root','candidate_score','winner','replay','successor_shift','missing_candidate'])
def test_selection_rejected_before_dispatch(packs,change):
    a=adapter(packs); prefix(a); request=a.planning_request(observe(a)); p=plan(a,request); writes=[]
    if change=='context': p['context']=deepcopy(a.predicted)
    if change=='metric': p['goal_q_sha256']='bad'
    if change=='root': p['root_sha256']='2'*64
    if change=='candidate_score': p['candidates'][0]['Q_cost']=0.
    if change=='winner': p['selected_candidate_id']='0'
    if change=='replay': p['verification_successor']['latent_history'][-1][0]=1.
    if change=='successor_shift': p['successor']['latent_history'][0][0]=22.
    if change=='missing_candidate': p['candidates'].pop()
    with pytest.raises(ValueError): a.accept_selection(p,lambda x:writes.append(x))
    assert not writes and a.phase=='failed'


@pytest.mark.parametrize('failure',['raise','wrong_ack'])
def test_failed_persistence_no_authorization(packs,failure):
    a=adapter(packs); prefix(a); request=a.planning_request(observe(a))
    def persist(value):
        if failure=='raise': raise OSError('mock ENOSPC')
        return 'not-durable-sha'
    with pytest.raises((ValueError,OSError)): a.accept_selection(plan(a,request),persist)
    assert a.phase=='failed' and not a.decisions


@pytest.mark.parametrize('change',['partial','wrong_action','wrong_rgb','wrong_step','unguarded','wrong_authorization'])
def test_execution_never_advances_invalid_packet(packs,change):
    a=adapter(packs); prefix(a); request=a.planning_request(observe(a)); auth=a.accept_selection(plan(a,request),m.content_sha)
    receipt,image=execution(a,5,69); sha=auth['selection_sha256']
    if change=='partial': receipt['completed_commands']=4
    if change=='wrong_action': receipt['trace'][0]['command']=[1.,1.]
    if change=='wrong_rgb': image=frame(99)
    if change=='wrong_step': receipt['trace'][1]['primitive_step']=22
    if change=='unguarded': receipt['guard_authorized']=False
    if change=='wrong_authorization': sha='bad'
    with pytest.raises(ValueError): a.record_execution(sha,receipt,image)
    assert a.boundary==5 and len(a.executions)==1 and a.phase=='failed'


def test_terminal_latent_cannot_enter_selection_early(packs):
    a=adapter(packs); prefix(a)
    with pytest.raises(ValueError): a.record_terminal_evaluation(dict(root_sha256=a.root,arm=a.arm,boundary=20,
        purpose='terminal_evaluation_only',pixel_sha256=m.rgb_sha(a.frames[-1]),native192=vector(1.)))
    assert a.phase=='failed'


def test_bad_prefix_blocks_all_feedback(packs):
    a=adapter(packs); receipt,image=execution(a,0,63)
    with pytest.raises(ValueError): a.accept_prefix(receipt,image,ctx([vector(1.),vector(2.),vector(3.)],a.between))
    assert a.phase=='failed'


@pytest.mark.parametrize('key,value',[
    ('device','cpu'),('model_bindings',{}),('value_sha256','bad'),('encoder_completed',0),
    ('encoded_frames',1),('between_raw_sha256','bad'),('purpose','terminal_evaluation_only'),
])
def test_encoding_receipt_rejected(packs,key,value):
    a=adapter(packs); prefix(a); obs=observe(a); obs['encoding_receipt'][key]=value
    with pytest.raises(ValueError): a.planning_request(obs)
    assert a.phase=='failed'


def test_duplicate_execution_authorization_rejected(packs):
    a=adapter(packs); prefix(a); request=a.planning_request(observe(a))
    auth=a.accept_selection(plan(a,request),m.content_sha); receipt,image=execution(a,5,69)
    a.record_execution(auth['selection_sha256'],receipt,image)
    with pytest.raises(ValueError): a.record_execution(auth['selection_sha256'],receipt,image)
    assert a.boundary==10 and a.phase=='failed'


def test_reentrant_persist_cannot_clear_failure(packs):
    a=adapter(packs); prefix(a); request=a.planning_request(observe(a))
    def persist(value):
        try: a.planning_request(observe(a))
        except ValueError: pass
        return m.content_sha(value)
    with pytest.raises(ValueError): a.accept_selection(plan(a,request),persist)
    assert a.phase=='failed' and not a.decisions


def test_no_model_or_physics_imports_or_output_writer():
    tree=ast.parse(Path(m.__file__).read_text()); names=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Import): names.extend(v.name for v in n.names)
        elif isinstance(n,ast.ImportFrom): names.append(n.module or '')
    assert not any(any(k in name for k in ('torch','gym','pymunk','checkpoint','multiround_feedback')) for name in names)
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('mkdir','write_text','write_bytes') for n in ast.walk(tree))
