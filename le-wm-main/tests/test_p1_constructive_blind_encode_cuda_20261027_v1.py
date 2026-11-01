"""Blind-only fixtures and real array-level GN/D2P handoff; no model/physics."""
from contextlib import nullcontext
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace as NS
import json
import zipfile

import numpy as np
import pytest

from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as m


def packet():
    return dict(rgb_history=np.stack([np.full((224,224,3),i,np.uint8) for i in (0,1,2)]),
                raw_action_history=np.zeros((10,2),np.float32),goal_rgb=np.full((224,224,3),3,np.uint8))


def npz(path,values):
    f=BytesIO(); np.savez_compressed(f,**values); path.write_bytes(f.getvalue()); return m.sha(path)


def protocol(tmp_path,n=2):
    slots=[]
    for i in range(n):
        path=tmp_path/f's{i}_blind_inputs.npz'; values=packet()
        # Keep history image duplicates deliberately, while goals make source file identities distinct.
        values['goal_rgb'][0,0,0]=3+i
        h=npz(path,values)
        slots.append(dict(order=i,source_slot=i,blind_npz=str(path),blind_sha256=h,
            pixel_sha256=[m.digest(rgb.tobytes()) for rgb in [*values['rgb_history'],values['goal_rgb']]],
            raw_action_sha256=m.digest(values['raw_action_history'].tobytes())))
    return dict(slots=slots,normalization=dict(mean=np.asarray([-.0087,.0068],np.float32).tolist(),
                std=np.asarray([.2019,.2002],np.float32).tolist(),source_sha256=m.NORMALIZER_SHA,
                training_provenance_verified=False,raw_semantics='relative raw already action_scale/100; no second division'),
        model_bindings=m.MODELS,arguments=dict(whitelist_sha='a'*64),contract=m.contract(n))


def exported(tmp_path):
    p=protocol(tmp_path,1); slot=p['slots'][0]; values=[np.full(192,i,np.float32) for i in range(4)]
    return m.task_export(p,'b'*64,slot,m.load_blind(slot['blind_npz'],slot['blind_sha256']),values)


def approved_manifest(p):
    return dict(schema=m.NAME+'.approved_blind_whitelist',approved_by='MAIN',purpose='calibration',
        source_admission='MAIN_independently_admitted',admission_sha256='c'*64,with_target=False,
        history_steps=[0,5,10],raw_action_steps=10,
        slots=[{k:s[k] for k in ('order','source_slot','blind_npz','blind_sha256')} for s in p['slots']])


def test_full_two_slot_RGB_only_encoding_task_seals_and_NPYs(tmp_path):
    p=protocol(tmp_path); store=m.Store(tmp_path/'out'); events=[]; ledger=m.Ledger(2,events.append); rows=[]; exports=[]; calls=[]
    def tensor(kind): return NS(device=NS(type='cuda'),shape=(1,3,224,224) if kind=='encoder' else (1,192))
    def encode(rgb):
        assert type(rgb) is np.ndarray and rgb.dtype==np.uint8 and rgb.shape==(224,224,3)
        calls.append(rgb.copy())
        for kind in ('encoder','projector'): ledger.dispatch(kind,tensor(kind),lambda:None)
        return np.full(192,len(calls),np.float32)
    m.encode_all(p,'b'*64,encode,ledger,store,rows,exports)
    assert len(calls)==8 and len(exports)==2 and np.array_equal(calls[0],calls[4])
    assert ledger.counts()=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')}
    assert len(list(store.output.glob('*.npy')))==8
    for row,e in zip(rows,exports):
        assert row['status']=='task_sealed' and e['task_id']==m.core.task_id(e['task'])
        handoff=m.load_task(row['task_path'],row['task_file_sha256'])
        assert handoff['gn_task']==e['task'] and set(handoff['d2p_packet'])=={'root_context','goal'}
        assert e['native_root_sha256'] is None and e['task_contract']['with_target'] is False
        assert e['task']['root_context']['latent_history']==[np.full(192,v).tolist() for v in ([1,2,3] if row['order']==0 else [5,6,7])]
    assert np.any(exports[0]['task']['root_context']['action_history'])
    assert sum(f.stat().st_size for f in store.output.iterdir())<128*1024


def test_raw_zero_is_nonzero_normalized_between2(tmp_path):
    e=exported(tmp_path); t=e['task']; expected=m.core.normalized_tokens(t,np.zeros((10,2),np.float32))
    assert expected.shape==(2,10) and np.array_equal(expected,t['root_context']['action_history']) and np.any(expected)
    assert m.core.task_id(t)==e['task_id']
    tampered=deepcopy(e); tampered['task']['root_context']['action_history']=np.zeros((2,10)).tolist()
    with pytest.raises(ValueError): m.task_interfaces(tampered)
    tampered=deepcopy(e); tampered['task_contract']['with_target']=True
    with pytest.raises(ValueError): m.task_interfaces(tampered)


@pytest.mark.parametrize('bad',['state','witness_commands','certificate','candidate_actions','future_rgb'])
def test_privileged_NPZ_fields_rejected(tmp_path,bad):
    p=packet(); p[bad]=np.zeros(2); path=tmp_path/'x_blind_inputs.npz'; h=npz(path,p)
    with pytest.raises(ValueError,match='blind-only'): m.load_blind(path,h)


@pytest.mark.parametrize('bad',['nonzero','float64','shape','RGBfloat','RGBcount','goalshape','NaN'])
def test_exact_blind_dimensions_and_dtype(tmp_path,bad):
    p=packet()
    if bad=='nonzero': p['raw_action_history'][0,0]=.01
    elif bad=='float64': p['raw_action_history']=np.zeros((10,2))
    elif bad=='shape': p['raw_action_history']=np.zeros((2,10),np.float32)
    elif bad=='RGBfloat': p['rgb_history']=p['rgb_history'].astype(float)
    elif bad=='RGBcount': p['rgb_history']=p['rgb_history'][:2]
    elif bad=='goalshape': p['goal_rgb']=p['goal_rgb'][:112]
    elif bad=='NaN': p['raw_action_history'][0,0]=np.nan
    path=tmp_path/'x_blind_inputs.npz'; h=npz(path,p)
    with pytest.raises(ValueError): m.load_blind(path,h)


def test_externalSHA_and_wrongfilename_and_symlink(tmp_path):
    path=tmp_path/'x_blind_inputs.npz'; h=npz(path,packet())
    with pytest.raises(ValueError): m.load_blind(path,'0'*64)
    other=tmp_path/'certificate.npz'; other.write_bytes(path.read_bytes())
    with pytest.raises(ValueError): m.load_blind(other,h)
    link=tmp_path/'link_blind_inputs.npz'; link.symlink_to(path)
    with pytest.raises(ValueError): m.load_blind(link,h)


def test_duplicate_zip_members_and_object_payload_rejected(tmp_path):
    path=tmp_path/'x_blind_inputs.npz'; f=BytesIO()
    with pytest.warns(UserWarning,match='Duplicate name'):
        with zipfile.ZipFile(f,'w') as archive:
            for name in ('rgb_history.npy','raw_action_history.npy','raw_action_history.npy'): archive.writestr(name,b'no')
    path.write_bytes(f.getvalue())
    with pytest.raises(ValueError): m.load_blind(path,m.sha(path))
    p=packet(); p['goal_rgb']=np.array([object()],dtype=object); h=npz(path,p)
    with pytest.raises(ValueError): m.load_blind(path,h)


@pytest.mark.parametrize('bad',['certificate_path','sourcepack_path','state','unadmitted','with_target','order','duplicate'])
def test_whitelist_exact_fields_no_privileged_paths(tmp_path,bad):
    p=protocol(tmp_path); manifest=approved_manifest(p)
    if bad in ('certificate_path','sourcepack_path','state'): manifest[bad]='DO_NOT_OPEN'
    elif bad=='unadmitted': manifest['source_admission']='pending_independent_MAIN_audit'
    elif bad=='with_target': manifest['with_target']=True
    elif bad=='order': manifest['slots'][0]['order']=1
    elif bad=='duplicate': manifest['slots'][1]['blind_npz']=manifest['slots'][0]['blind_npz']; manifest['slots'][1]['blind_sha256']=manifest['slots'][0]['blind_sha256']
    path=tmp_path/'approved.json'; path.write_bytes(m.canonical(manifest))
    with pytest.raises(ValueError): m.whitelist(path,m.sha(path))


def test_admitted_manifest_no_sourcepack_read(tmp_path):
    p=protocol(tmp_path); manifest=approved_manifest(p); path=tmp_path/'approved.json'; path.write_bytes(m.canonical(manifest))
    got,slots=m.whitelist(path,m.sha(path))
    assert len(slots)==2 and got['admission_sha256']=='c'*64
    assert not any('certificate' in key or 'sourcepack' in key for key in manifest)


def test_failure_projector_or_sync_preserves_nonrefunded_attempt():
    events=[]; ledger=m.Ledger(2,events.append); ledger.image=0
    encoder=NS(device=NS(type='cuda'),shape=(1,3,224,224)); projector=NS(device=NS(type='cuda'),shape=(1,192))
    ledger.dispatch('encoder',encoder,lambda:None)
    def fail(): raise RuntimeError('projector')
    with pytest.raises(RuntimeError): ledger.dispatch('projector',projector,fail)
    assert ledger.counts()==dict(encoder=dict(attempted=1,completed=1),projector=dict(attempted=1,completed=0))
    with pytest.raises(ValueError): ledger.dispatch('encoder',encoder,lambda:None)
    ledger=m.Ledger(1,events.append,sync=fail); ledger.image=0
    with pytest.raises(RuntimeError): ledger.dispatch('encoder',encoder,lambda:None)
    assert ledger.counts()['encoder']==dict(attempted=1,completed=0)


@pytest.mark.parametrize('bad',['cpu','shape','grad','order','cap'])
def test_encoder_predispatch_rejects(bad):
    ledger=m.Ledger(1,lambda e:None); ledger.image=0; t=NS(device=NS(type='cuda'),shape=(1,3,224,224)); kind='encoder'
    if bad=='cpu': t.device.type='cpu'
    elif bad=='shape': t.shape=(3,3,224,224)
    elif bad=='grad': ledger.no_grad=lambda:False
    elif bad=='order': kind='projector'
    elif bad=='cap': ledger.entries=[dict(resource='encoder',completed=True)]*8
    old=len(ledger.entries)
    with pytest.raises(ValueError): ledger.dispatch(kind,t,lambda:None)
    assert len(ledger.entries)==old


def test_NPY_fsync_fault_retains_file_without_task(tmp_path,monkeypatch):
    store=m.Store(tmp_path/'out'); old=m.os.fsync
    def fail(fd): raise OSError('fsync')
    monkeypatch.setattr(m.os,'fsync',fail)
    with pytest.raises(OSError): store.latent(0,np.zeros(192,np.float32))
    assert (store.output/'image_00.native192.npy').exists() and not list(store.output.glob('task*'))
    monkeypatch.setattr(m.os,'fsync',old)
    with pytest.raises(FileExistsError): store.latent(0,np.zeros(192,np.float32))


def test_actual_two_authorized_blind_payloads_only():
    for path,h in m.APPROVED_INPUTS.items():
        p=m.load_blind(path,h)
        assert set(p)=={'rgb_history','raw_action_history','goal_rgb'}
        assert p['rgb_history'].shape==(3,224,224,3) and not p['raw_action_history'].any()
    assert len(m.APPROVED_INPUTS)==2


def test_original_checkpoint_normalizer_saved_only():
    n=m.native_metadata()
    assert n['mean']==np.asarray([-.0087,.0068],np.float32).tolist()
    assert n['std']==np.asarray([.2019,.2002],np.float32).tolist()
    assert n['training_provenance_verified'] is False


def test_real_build_cannot_swap_approved_source_slots(monkeypatch):
    slots=[dict(order=i,source_slot=1-i,blind_npz=path,blind_sha256=h) for i,(path,h) in enumerate(m.APPROVED_INPUTS.items())]
    monkeypatch.setattr(m,'whitelist',lambda *args:({},slots))
    with pytest.raises(ValueError,match='source-slot order'): m.build('mock.json','a'*64)


class LinearBackend:
    """Array-only stand-in for actual compiler APIs; not a learned model."""
    def __init__(self,task):
        self.task=task; self.calls=0; self.j=np.zeros((192,10)); self.j[0,0]=1.; self.j[1,1]=2.
    def step(self,context,actions):
        self.calls+=1; c=deepcopy(context)
        z=np.asarray(c['latent_history'][-1])+self.j@np.asarray(actions).reshape(10)
        c['latent_history']=c['latent_history'][1:]+[z.tolist()]
        token=m.core.normalized_tokens(self.task,np.asarray(actions).reshape(5,2))[0]
        c['action_history']=c['action_history'][1:]+[token.astype(float).tolist()]; return c
    def replay(self,context,actions):
        for raw in np.asarray(actions).reshape(-1,5,2): context=self.step(context,raw)
        return context
    def nominal_jacobian(self,context,operation):
        nominal=self.step(context,np.zeros((5,2))); operation('jacobian',lambda:self.step(context,np.zeros((5,2))))
        return np.asarray(nominal['latent_history'][-1]),self.j.copy()


def test_export_really_plugs_into_existing_full10_conditional_compiler(tmp_path):
    from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v1 as compiler
    e=exported(tmp_path); handoff=m.task_interfaces(e); backend=LinearBackend(handoff['gn_task'])
    result=compiler.compile_arm(handoff['d2p_packet'],np.asarray(handoff['q']),backend,lambda e:None)
    assert result['status']=='completed' and result['whole_and_segmented_exact'] and len(result['candidates'])==10
    assert backend.calls==66 and len(result['root_jacobian'])==192
    assert 'root_sha256' not in handoff['d2p_packet']  # array API does not need hidden/native root identity


def test_export_really_plugs_into_existing_GN_core(tmp_path):
    from experiments.spectral_reachability import p1_conditional_direct_screen_20261027_v1 as screen
    task=m.task_interfaces(exported(tmp_path))['gn_task']
    class GN(LinearBackend):
        def guard(self,meter): return nullcontext()
        def rollout(self,t,raw,meter,mode):
            c=deepcopy(t['root_context']); result=[]
            for actions in np.asarray(raw).reshape(2,5,2):
                meter.call('action_encoder',lambda:None); meter.call('predict',lambda:None)
                c=self.step(c,actions); result.append(c)
            return result
        def jacobian(self,t,inc,meter):
            path=self.rollout(t,inc['commands'],meter,'AD_forward')
            for i in range(192): meter.call('vjp_directions',lambda:None)
            j=np.concatenate((self.j,self.j),axis=1)*(1-np.asarray(inc['commands']).reshape(20)**2)[None]
            return np.asarray(path[-1]['latent_history'][-1]),j
    cfg=dict(method='full_H2_GN',B1=1,B2=2,initial_logits=[np.zeros((2,10)).tolist()],step_scale=1.,damping=.002,extra_line_search_trials=0)
    result=screen.run_direct(task,cfg,GN(task),lambda e:None)
    assert result['status']=='direct_component_completed_not_admitted' and result['error'] is None
