"""RAM-only bridge fixtures; no reserved assets, model or native producer."""
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p4_blind4_task_bridge_20261028_v2 as m


def h(label):return m.sha(str(label).encode())


def recipe_plan():
    """Native recipe schema and codec; explicit source-only RAM history.

    No collect_current_exclusions, real history census, production prepare,
    reserved path, environment factory or model is invoked.
    """
    codec=m.recipe_codec
    path=str(Path(codec.source.__file__).resolve())
    history=dict(schema=codec.source.NAME+'.exclusions',environment='swm/PushT-v1',
        files={path:codec.source.file_sha(path)},root_hashes=[],seed_families=[],reset_spans=[],unknown_sources=[],
        coverage=dict(status='unknown',authority_sha256=None,scope='RAM codec test only'),physical_future_labels_read=False)
    return codec.build([202798101,202798102],[202798201,202798202],60,history=history)


def packet(slot):
    return dict(rgb_history=np.stack([np.full((224,224,3),slot*10+j,np.uint8) for j in range(3)]),
                goal_rgb=np.full((224,224,3),100+slot,np.uint8),raw_action_history=m.moving.fixed_raw())


def deployment():
    rows=[]
    for i in range(4):
        p=packet(i);pixels=[h0 for h0 in [m.sha(a.tobytes()) for a in [*p['rgb_history'],p['goal_rgb']]]]
        raw=m.sha(p['raw_action_history'].tobytes())
        rows.append(dict(order=i,source_slot=i,ready=True,source_status='reachable_native_replayed',
            blind_npz=f'/RAM/s{i}_blind_inputs.npz',blind_sha256=h(('blind',i)),pixel_sha256=pixels,
            raw_action_sha256=raw,identity=dict(root_state_sha256=h(('state',i)),physical_pose_key=h(('pose',i)),
            root_snapshot_sha256=h(('snapshot',i)),history_pixels_sha256=pixels[:3],raw_history_sha256=raw),
            root_goal_sha256=h(('rootgoal',i)),source_certificate_sha256=h(('cert',i))))
    return dict(schema=m.NAME+'.blind4',fixed_denominator=4,ready=True,slots=rows,sourcepack_sha256=h('source'),
        source_prepare_sha256=h('prepared'),history_certificate_sha256=h('history'),history_approval_sha256=h('approval'),
        recipe_sha256=m.RECIPE_SHA,normalization=m.moving.normalization(),model_bindings=deepcopy(m.moving.MODELS),
        expected_model_tensor_sha256=m.moving.low.MODEL_TENSOR_SHA,image_order=['observed0','observed5','observed10','goal'],
        future_encoding_caps=dict(encoder=16,projector=16,predict=0,AD=0,physics=0),actual_new_encoding=None,
        execution_authorized=False,freshness_certified=False,P1_admission=False,P4_admission=False,scientific_admission=False)


def exports():
    d=deployment()
    return [m.task_export(d,h('bridgeprepare'),i,packet(i),[np.full(192,(4*i+j)/32,np.float32) for j in range(4)]) for i in range(4)]


def test_all_four_real_task_build_and_eight_planner_inputs():
    es=exports();requests=m.planner_inputs(es)
    assert [(r['source_slot'],r['arm']) for r in requests]==[(s,a) for s in range(4) for a in m.ARMS]
    assert len({r['task_id'] for r in requests})==4
    for i,e in enumerate(es):
        t=e['task'];m.core.validate_task(t)
        assert t['root_context']['latent_history'][1]==[np.float32((4*i+1)/32).item()]*192
        assert np.array_equal(t['root_context']['action_history'],m.core.normalized_tokens(t,m.moving.fixed_raw()))
        assert t['root_raw_action_history']==m.moving.fixed_raw().astype(float).tolist()
        assert e['encoding_provenance_verified'] is e['execution_authorized'] is e['P1_admission'] is False
        assert requests[2*i]['gn_task']==requests[2*i+1]['gn_task']==t
        assert requests[2*i]['d2p_packet']==dict(root_context=t['root_context'],goal=t['goal'])
        assert requests[2*i]['q']==np.eye(192).tolist()


@pytest.mark.parametrize('kind',['zero_raw','reorder_raw','dtype','frames','goal','latents64','missing_latent','three_roots','not_ready','slot_bool','norm'])
def test_task_mutations_failclosed(kind):
    d=deployment();p=packet(0);z=[np.full(192,i/32,np.float32) for i in range(4)];slot=0
    if kind=='zero_raw':p['raw_action_history'][:]=0
    elif kind=='reorder_raw':p['raw_action_history']=p['raw_action_history'][::-1].copy()
    elif kind=='dtype':p['raw_action_history']=p['raw_action_history'].astype(np.float64)
    elif kind=='frames':p['rgb_history']=p['rgb_history'][::-1].copy()
    elif kind=='goal':p['goal_rgb'][:]=0
    elif kind=='latents64':z[0]=z[0].astype(np.float64)
    elif kind=='missing_latent':z.pop()
    elif kind=='three_roots':d['slots'].pop()
    elif kind=='not_ready':d['ready']=False
    elif kind=='slot_bool':slot=True
    else:d['normalization']['mean']=[0,0]
    with pytest.raises(ValueError):m.task_export(d,h('prepare'),slot,p,z)


@pytest.mark.parametrize('kind',['count','reorder','witness','cohort','identity','Q','authority','raw','extra_precision','goal_source'])
def test_planner_seals_and_no_leakage(kind):
    es=exports()
    if kind=='count':es.pop()
    elif kind=='reorder':es.reverse()
    elif kind=='witness':es[0]['task_contract']['witness_commands']=[[0,0]]*10
    elif kind=='cohort':es[0]['task_contract']['history_approval_sha256']=h('another cohort')
    elif kind=='identity':es[0]['task_contract']['identity']['root_state']=[0]*7
    elif kind=='Q':es[0]['task']['projection']=np.zeros((192,192)).tolist()
    elif kind=='authority':es[0]['P1_admission']=True
    elif kind=='raw':es[0]['task']['root_raw_action_history'][0][0]+=1e-12
    elif kind=='extra_precision':es[0]['task']['goal'][0]=0.10000000001
    else:es[0]['task']['goal_source_sha256']=h('wrong goal')
    if kind not in ('count','reorder'):
        es[0]['task']['task_contract_sha256']=m.core.digest(es[0]['task_contract'])
        try:es[0]['task_id']=m.core.task_id(es[0]['task'])
        except ValueError:pass
    with pytest.raises(ValueError):m.planner_inputs(es)


@pytest.fixture
def ram(monkeypatch):
    files={};reads=[]
    def put(path,value):
        payload=m.raw(value);files[str(path)]=payload;return m.sha(payload)
    folder=Path('/RAM/native');prepared=Path('/RAM/prepare.json');certpath='/RAM/history.json';approvalpath='/RAM/approval.json'
    original=recipe_plan()
    payload=m.recipe_codec.prepared_bytes(original)
    files[str(m.RECIPE)]=payload
    monkeypatch.setattr(m,'RECIPE_SHA',m.sha(payload))
    recipe={k:deepcopy(original[k]) for k in ('recipe','witness','packet_logit_cap','environment')}
    for path,h0 in original['bindings'].items():
        files[path]=Path(path).read_bytes()
        assert m.sha(files[path])==h0
    identity=m.exclusion.state_identity([1.,2.,3.,4.,0.5,0.,0.])
    identity['root_state']=[1.,2.,3.,4.,0.5,0.,0.]
    identity.update(root_snapshot_sha256=h('snapshot'),raw_history_sha256=h('raw'),history_pixels_sha256=[h(i) for i in range(3)])
    cert=dict(schema='moving_fresh_source_exclusions.v1',protocol={'sha256':m.PROTOCOL_SHA},cutoff_utc='RAM',
        project_root=str(m.ROOT),observed_roots=[identity],evidence_files=[{'family':f} for f in m.exclusion.REQUIRED_FAMILIES],
        source_sha256={str(Path(m.exclusion.__file__).resolve()):m.EXCLUSION_SHA})
    files[str(Path(m.exclusion.__file__).resolve())]=Path(m.exclusion.__file__).read_bytes()
    cert['coverage_content_sha256']=m.exclusion.digest(cert)
    approval=dict(schema=m.exclusion.NAME+'.MAIN_coverage_approval',authority='MAIN',approved=True,
        manifest_content_sha256=cert['coverage_content_sha256'],cutoff_utc='RAM',coverage_scope=list(m.exclusion.REQUIRED_FAMILIES))
    approvalsha=put(approvalpath,approval)
    cert.update(source_run_preconditions_met=True,unresolved_sources=[],approval_errors=[],
        approval=dict(approved=True,coverage_scope=list(m.exclusion.REQUIRED_FAMILIES),external_file=m.ref(approvalpath,approvalsha)))
    # unresolved_sources/approval_errors are real content/extras according to the original certificate schema.
    content={k:v for k,v in cert.items() if k not in {'coverage_content_sha256','approval','approval_errors','source_run_preconditions_met'}}
    cert['coverage_content_sha256']=m.exclusion.digest(content)
    approval['manifest_content_sha256']=cert['coverage_content_sha256'];approvalsha=put(approvalpath,approval)
    cert['approval']['external_file']['sha256']=approvalsha
    certsha=put(certpath,cert)
    d=deployment();rows=[]
    for i in range(4):
        p=packet(i);stream=io.BytesIO();np.savez_compressed(stream,**p);name=f's{i}_blind_inputs.npz';payload=stream.getvalue()
        files[str(folder/name)]=payload
        identity=dict(d['slots'][i]['identity'])
        rows.append(dict(slot=m.slots()[i],reachable=True,status='reachable_native_replayed',actual_root_identity=identity,
            blind_payload=name,blind_payload_sha256=m.sha(payload),root_sha256=h(('rootgoal',i)),certificate_sha256=h(('cert',i))))
    p=dict(schema=m.saved.PRODUCER+'.prepare',mode='fresh_candidate',history_certificate=m.ref(certpath,certsha),
        historical_roots=deepcopy(cert['observed_roots']),bindings={certpath:certsha,approvalpath:approvalsha,
            **cert['source_sha256'],**original['bindings'],str(m.RECIPE):m.RECIPE_SHA},
        slots=m.slots(),fixed_denominator=4,with_target=False,warmup=10,horizon=10,replay_episodes=2,
        saved_telemetry='native_latest_action_float32_bits_and_goal_pose_float64_bits.v2',
        original_recipe=m.ref(m.RECIPE,m.RECIPE_SHA),raw_warmup=m.moving.fixed_raw().tolist(),**recipe)
    pack=dict(sourcepack_complete=True,mode='fresh_candidate',errors=[],rows=rows)
    def provider(path):reads.append(path);return files[path]
    monkeypatch.setattr(m,'source_pins',lambda reader:None)
    monkeypatch.setattr(m,'own_files',lambda:[])
    calls=[]
    def consume(output,psha,prep,h,reader):
        calls.append((output,psha,prep,h))
        return dict(schema=m.saved.NAME+'.receipt',strict_complete=True,fixed_denominator=4,phase_denominator=12)
    monkeypatch.setattr(m.saved,'consume',consume)
    def run():
        prepare_sha=put(prepared,p);source_sha=put(folder/'sourcepack.json',pack)
        return m.prepare_material(str(folder),source_sha,str(prepared),prepare_sha,reader=m.saved.Reader(provider))
    return dict(files=files,reads=reads,p=p,pack=pack,cert=cert,put=put,run=run,calls=calls,certpath=certpath,original=original,
                set_recipe_sha=lambda h0:monkeypatch.setattr(m,'RECIPE_SHA',h0))


def test_prepare_v2_legal_future_source_and_privilege_split(ram):
    result=ram['run']();d=result['deployment']
    assert d['ready'] is True and len(ram['calls'])==1 and len(d['slots'])==4
    assert d['future_encoding_caps']==dict(encoder=16,projector=16,predict=0,AD=0,physics=0)
    encoded=m.raw(d).decode()
    for denied in ('witness_commands','root_state"','history.json','sourcepack.json','construction_certificate.npz'):
        assert denied not in encoded
    assert d['execution_authorized'] is d['freshness_certified'] is d['P1_admission'] is False
    assert result['admin']['original_recipe_plan_sha256']==m.recipe_codec.source.digest(ram['original'])
    for k in ('recipe','witness','packet_logit_cap','environment'):
        assert ram['p'][k]==ram['original'][k]


def test_native_envelope_schema_and_full_helper_verification():
    p=recipe_plan();payload=m.recipe_codec.prepared_bytes(p);e=m.saved.decode(payload)
    assert set(e)=={'schema','gzip_base64','plan_sha256'}
    assert e['schema']=='p1_constructive_reachable_source_20261027_v1.envelope'
    assert 'recipe' not in e and 'witness' not in e
    assert m.recipe_codec.read_prepared(payload)==p


def test_actual_sealed_recipe_top_level_readonly_not_production_decode():
    # Only this explicitly authorized original recipe envelope is read. Do not
    # call its full helper here: that would traverse real historical inputs.
    reader=m.saved.Reader()
    envelope=reader.json(m.RECIPE,m.RECIPE_SHA)
    assert set(envelope)=={'schema','gzip_base64','plan_sha256'}
    assert envelope['schema']==m.recipe_codec.NAME+'.envelope'
    m.core.hash_value(envelope['plan_sha256'])
    assert 'recipe' not in envelope and 'witness' not in envelope
    assert reader.inputs=={str(m.RECIPE):m.RECIPE_SHA}


@pytest.mark.parametrize('case',['recipe','witness','packet_logit_cap','environment'])
def test_each_decoded_field_must_be_inherited(ram,case):
    ram['p'][case]=None
    with pytest.raises(ValueError,match='original recipe/witness/action/environment: '+case):ram['run']()


@pytest.mark.parametrize('case',['flat','semantic','schema','base64','expanded','byte_sha','decoder_binding','inherited_binding','inner_recipe'])
def test_recipe_envelope_and_source_closure_failclosed(ram,case):
    import base64,gzip
    p=ram['p'];payload=ram['files'][str(m.RECIPE)];e=m.saved.decode(payload)
    if case=='decoder_binding':
        p['bindings'][str(Path(m.recipe_codec.__file__).resolve())]=h('drift')
    elif case=='inherited_binding':
        path=str(m.recipe_codec.ENV);p['bindings'][path]=h('drift')
    elif case=='byte_sha':ram['files'][str(m.RECIPE)]+=b' '
    else:
        if case=='flat':payload=m.recipe_codec.raw(ram['original'])
        else:
            if case=='semantic':e['plan_sha256']=h('wrong semantic')
            elif case=='schema':e['schema']='wrong.envelope'
            elif case=='base64':e['gzip_base64']='!!!'
            elif case=='expanded':
                e['gzip_base64']=base64.b64encode(gzip.compress(b' '*(2*1024**2+1))).decode()
            else:
                inner=deepcopy(ram['original']);inner['recipe']={'substituted':True}
                e=m.saved.decode(m.recipe_codec.prepared_bytes(inner))
            payload=m.raw(e)
        # External byte binding is resealed: inner semantic/source checks must still refuse.
        ram['files'][str(m.RECIPE)]=payload
        # Scope local fixture's synthetic recipe SHA only; production pin unchanged.
        newsha=m.sha(payload);p['original_recipe']['sha256']=newsha;p['bindings'][str(m.RECIPE)]=newsha
        # read_recipe uses the fixed recipe SHA: this test seam changes only the fixture pin.
        ram['set_recipe_sha'](newsha)
    with pytest.raises((ValueError,KeyError)):ram['run']()
    assert ram['calls']==[]
    assert not any(path.endswith(('.npz','sourcepack.json')) for path in ram['reads'])


@pytest.mark.parametrize('case',['calibration','v1','missingcert','unknown_coverage','unresolved','approval_false'])
def test_authority_preflight_before_any_source_or_blind_access(ram,case):
    p,c=ram['p'],ram['cert']
    if case=='calibration':p['mode']='calibration_only'
    elif case=='v1':p['schema']='agent_p4_fresh_source_20261028_v1.prepare'
    elif case=='missingcert':p['history_certificate']=None
    else:
        if case=='unknown_coverage':c['source_run_preconditions_met']=False
        elif case=='unresolved':c['unresolved_sources']=['unknown A-H']
        else:c['approval']['approved']=False
        p['history_certificate']['sha256']=ram['put'](ram['certpath'],c)
    with pytest.raises(ValueError):ram['run']()
    assert not ram['calls']
    assert not any(path.endswith(('.npz','sourcepack.json')) for path in ram['reads'])
    assert str(m.RECIPE) not in ram['reads']


def test_k_of_four_never_compresses_denominator(ram):
    row=ram['pack']['rows'][2];row.update(reachable=False,status='rejected_fixed_candidate',blind_payload=None)
    result=ram['run']()
    assert result['status']=='source_k_of_4_downstream_BLOCKED'
    assert len(result['deployment']['slots'])==4 and result['deployment']['slots'][2]['blind_npz'] is None
    with pytest.raises(ValueError):m.task_export(result['deployment'],h('prepare'),0,packet(0),[np.zeros(192,np.float32)]*4)


@pytest.mark.parametrize('case',['slots','warmup','recipe','raw','telemetry','history_binding','history_projector','consumer'])
def test_prepare_contract_closure_rejected(ram,monkeypatch,case):
    p=ram['p']
    if case=='slots':p['slots'][3]['index']=2
    elif case=='warmup':p['warmup']=9
    elif case=='recipe':p['witness']={'substituted':True}
    elif case=='raw':p['raw_warmup'][0][0]=0
    elif case=='telemetry':p['saved_telemetry']='v1'
    elif case=='history_binding':p['bindings'][ram['certpath']]=h('wrong')
    elif case=='history_projector':
        # Reseal content/approval so the missing original source binding is the actual rejection.
        c=ram['cert'];c['source_sha256']={}
        extra={'coverage_content_sha256','approval','approval_errors','source_run_preconditions_met'}
        c['coverage_content_sha256']=m.exclusion.digest({k:v for k,v in c.items() if k not in extra})
        a=c['approval']['external_file'];approval=json.loads(ram['files'][a['path']])
        approval['manifest_content_sha256']=c['coverage_content_sha256'];a['sha256']=ram['put'](a['path'],approval)
        p['bindings'][a['path']]=a['sha256']
        p['history_certificate']['sha256']=ram['put'](ram['certpath'],c)
        p['bindings'][ram['certpath']]=p['history_certificate']['sha256']
    else:
        monkeypatch.setattr(m.saved,'consume',lambda *a,**k:dict(schema=m.saved.NAME+'.receipt',
            strict_complete=False,fixed_denominator=4,phase_denominator=12))
    with pytest.raises(ValueError):ram['run']()


def test_verify_exact_rebuild_and_source_drift(ram,monkeypatch):
    value=ram['run']();path='/RAM/bridge.json';expected=ram['put'](path,value)
    reader=m.saved.Reader
    monkeypatch.setattr(m.saved,'Reader',lambda:reader(lambda p:ram['files'][p]))
    assert m.verify(path,expected)==value
    source=value['admin']['sourcepack']['path']
    ram['files'][source]+=b' '
    with pytest.raises(ValueError,match='input SHA'):m.verify(path,expected)


def test_cold_import_no_native_producer_no_torch():
    code='''
import builtins,sys
old=builtins.__import__
def checked(name,*a,**k):
    if name.split(".")[0] in {"torch","gymnasium","pymunk","stable_worldmodel"}:
        raise AssertionError(name)
    return old(name,*a,**k)
builtins.__import__=checked
from experiments.spectral_reachability import agent_p4_blind4_task_bridge_20261028_v2 as m
assert not any(n.endswith(("agent_p4_fresh_source_20261028_v1","agent_p4_fresh_source_20261028_v2")) for n in sys.modules)
assert m.slots()[3]["index"]==3
'''
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr


def test_new_output_only(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    path=tmp_path/'results'/(m.NAME+'_RAM.json')
    receipt=m.write(path,dict(status='RAM'))
    assert receipt['sha256']==m.sha(path.read_bytes())
    with pytest.raises(FileExistsError):m.write(path,dict(status='RAM'))
    with pytest.raises(ValueError):m.write(Path('/tmp/foreign.json'),dict(status='RAM'))
