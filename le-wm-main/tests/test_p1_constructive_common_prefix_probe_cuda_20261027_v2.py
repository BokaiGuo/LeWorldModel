"""RAM/mock only: numerical protocol phases, strict original gate, no CUDA."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_common_prefix_probe_cuda_20261027_v2 as m
import test_p1_constructive_common_prefix_probe_cuda_20261027_v1 as fixture


def setup():
    p=fixture.protocol(); p['contract']=m.contract(); store=fixture.Store(); ledger=m.Ledger(store.event)
    anchors=[dict(index=i,status='not_started') for i in range(4)]
    images=[dict(index=i,status='not_started') for i in range(92)]; rows=[dict(index=i,status='not_started') for i in range(84)]
    return p,store,ledger,anchors,images,rows,[]


def callbacks(p,ledger,*,drift=1e-6,bad_anchor=False,terminal=100.,fail_phase=None,fail_resource='encoder'):
    calls=[]; encodings=[]
    def encode(item,phase):
        encodings.append((phase,item['index']))
        def fail():
            if phase==fail_phase and fail_resource=='encoder': raise RuntimeError('encoder primary')
        ledger.call('encoder',fail)
        def project():
            if phase==fail_phase and fail_resource=='projector': raise RuntimeError('projector primary')
        ledger.call('projector',project)
        if phase=='anchor_default':
            value=np.nextafter(np.float32(0.),np.float32(1.)) if bad_anchor=='ulp' else np.float32(-0.) if bad_anchor else np.float32(0.)
        else: value=drift if item['index']<4 else item['index']-3 if item['index']<8 else terminal+item['index']
        return np.full(192,value,np.float32)
    def predict(c,u):
        calls.append((deepcopy(c),u.copy()))
        ledger.call('action_encoder',lambda:None); ledger.call('predict',lambda:None)
        z=np.asarray(c['latent_history'][-1],np.float32)+u.sum(dtype=np.float32)
        return dict(c,latent_history=c['latent_history'][1:]+[z.astype(float).tolist()],
            action_history=[c['action_history'][-1],m.prior.core.normalized_tokens(p['tasks'][0],u)[0].tolist()])
    return encode,predict,calls,encodings


def run_kernel(**kwargs):
    p,store,ledger,anchors,images,rows,diffs=setup(); encode,predict,calls,encodings=callbacks(p,ledger,**kwargs)
    analysis=m.execute(p,encode,predict,ledger,store,anchors,images,rows,diffs)
    return dict(p=p,store=store,ledger=ledger,anchors=anchors,images=images,rows=rows,diffs=diffs,analysis=analysis,calls=calls,encodings=encodings)


def test_full96_encoding84_prediction_phase_census_and_math_chronology():
    r=run_kernel(); caps=m.contract()['caps']
    assert r['ledger'].counts()=={k:dict(attempted=v,completed=v) for k,v in caps.items()}
    assert r['ledger'].phase_counts()=={phase:{k:dict(attempted=v,completed=v) for k,v in c.items()} for phase,c in m.contract()['phase_caps'].items()}
    assert r['encodings']==[('anchor_default',i) for i in range(4)]+[('image_math',i) for i in range(92)]
    assert len(r['calls'])==84 and len(r['diffs'])==4 and all(x['exact_original_task'] for x in r['anchors'])
    for i,(c,u) in enumerate(r['calls']):
        pid=i//21
        assert np.array_equal(c['latent_history'][0],np.full(192,1e-6,np.float32))
        assert c['latent_history'][1]==c['latent_history'][0] and c['latent_history'][-1]==[float(pid+1)]*192
        assert np.array_equal(u,m.physical.probes()[i%21]['commands'])
        raw=np.concatenate((np.zeros((5,2)),r['p']['positions'][pid]['prefix_commands']))
        assert np.array_equal(c['action_history'],m.prior.core.normalized_tokens(r['p']['tasks'][pid//2],raw))
    assert all(not d['math_vs_original_exact_bits'] and d['acceptance_tolerance'] is None and d['max_abs']>0 for d in r['diffs'])


@pytest.mark.parametrize('damage',(True,'ulp'))
def test_default_anchor_bit_mismatch_is_not_tolerated_saved_before_rejection(damage):
    p,store,ledger,anchors,images,rows,diffs=setup(); encode,predict,calls,_=callbacks(p,ledger,bad_anchor=damage)
    with pytest.raises(ValueError,match='EXACT bits'): m.execute(p,encode,predict,ledger,store,anchors,images,rows,diffs)
    assert np.array_equal(np.zeros(192,np.float32),np.full(192,-0.,np.float32))
    assert anchors[0]['status']=='encoded' and anchors[0]['exact_original_task'] is False
    value=-0. if damage is True else np.nextafter(np.float32(0.),np.float32(1.))
    assert anchors[0]['float32_bits']==m.physical.bits(np.full(192,value,np.float32),'<f4')
    assert [e['kind'] for e in store.events][-2:]==['image_saved','original_anchor_authenticated']
    assert ledger.counts()['encoder']==dict(attempted=1,completed=1) and ledger.counts()['predict']['attempted']==0 and calls==[]
    assert all(x['status']=='not_started' for x in images+rows)


def test_formal_difference_is_description_not_new_threshold_or_gate():
    a=run_kernel(drift=0.); b=run_kernel(drift=1e-6)
    assert all(d['math_vs_original_exact_bits'] for d in a['diffs'])
    assert len(a['calls'])==len(b['calls'])==84
    assert a['calls'][0][0]['latent_history']!=b['calls'][0][0]['latent_history']
    assert all(d['default_anchor_authenticated'] for d in b['diffs'])


def test_terminal_labels_never_affect_prediction_inputs_or_directions():
    a=run_kernel(terminal=100.); b=run_kernel(terminal=-700.)
    for (c,u),(d,v) in zip(a['calls'],b['calls'],strict=True): assert c==d and np.array_equal(u,v)
    for x,y in zip(a['analysis']['positions'],b['analysis']['positions'],strict=True): assert x['common_direction']==y['common_direction']
    assert a['analysis']['positions'][0]['rows'][0]['actual_Q']!=b['analysis']['positions'][0]['rows'][0]['actual_Q']


@pytest.mark.parametrize('phase',('anchor_default','image_math'))
@pytest.mark.parametrize('resource',('encoder','projector'))
def test_phase_fault_nonrefund_no_math_predictions_and_durable_prior_outputs(phase,resource):
    p,store,ledger,anchors,images,rows,diffs=setup(); encode,predict,calls,_=callbacks(p,ledger,fail_phase=phase,fail_resource=resource)
    with pytest.raises(RuntimeError,match='primary'): m.execute(p,encode,predict,ledger,store,anchors,images,rows,diffs)
    assert ledger.failed and ledger.counts()['predict']['attempted']==0 and not calls
    assert ledger.phase_counts()[phase][resource]['attempted']==1 and ledger.phase_counts()[phase][resource]['completed']==0
    if phase=='image_math': assert all(x['exact_original_task'] for x in anchors)
    with pytest.raises(ValueError): ledger.call(resource,lambda:None)


def test_missing_terminal_partial_84_denominator_and96_slot_budget_without_padding():
    p,store,ledger,anchors,images,rows,diffs=setup(); p['images'][8]['available']=False
    encode,predict,calls,_=callbacks(p,ledger); a=m.execute(p,encode,predict,ledger,store,anchors,images,rows,diffs)
    assert len(rows)==84 and len(images)==92 and len(anchors)==4 and len(calls)==84
    assert images[8]['status']=='missing' and ledger.counts()['encoder']==dict(attempted=95,completed=95)
    assert all(x['actual_delta'] is None for x in a['positions'][0]['rows'])


def test_phase_caps_wrong_scope_and_sync_failure():
    l=m.Ledger(lambda e:None); l.scope=('anchor_default',0)
    with pytest.raises(ValueError): l.call('predict',lambda:None)
    for i in range(4): l.scope=('anchor_default',i); l.call('encoder',lambda:None); l.call('projector',lambda:None)
    l.scope=('anchor_default',4)
    with pytest.raises(ValueError): l.call('encoder',lambda:None)
    l=m.Ledger(lambda e:None,sync=lambda:(_ for _ in ()).throw(RuntimeError('sync'))); l.scope=('anchor_default',0)
    with pytest.raises(RuntimeError): l.call('encoder',lambda:None)
    assert l.phase_counts()['anchor_default']['encoder']==dict(attempted=1,completed=0)


def fake_runtime():
    flags=dict(math=True,flash=False,mem_efficient=True,cudnn=False)
    cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))
    for key in flags: setattr(cuda,key+'_sdp_enabled',lambda k=key:flags[k])
    torch=SimpleNamespace(backends=SimpleNamespace(cuda=cuda,cudnn=SimpleNamespace(allow_tf32=False)))
    enum=SimpleNamespace(MATH='math',FLASH_ATTENTION='flash',EFFICIENT_ATTENTION='mem_efficient',CUDNN_ATTENTION='cudnn')
    @contextmanager
    def sdpa(allowed):
        old=flags.copy(); allowed=allowed if isinstance(allowed,list) else [allowed]
        flags.update({k:k in allowed for k in flags})
        try: yield
        finally: flags.update(old)
    return torch,SimpleNamespace(SDPBackend=enum,sdpa_kernel=sdpa),flags


@pytest.mark.parametrize('phase',m.PHASES)
def test_actual_phase_switch_flags_and_restore_on_primary_exception(phase):
    torch,attention,flags=fake_runtime(); old=flags.copy(); events=[]
    with pytest.raises(RuntimeError,match='primary'):
        with m.numerical_phase(torch,attention,phase,events.append):
            assert not torch.backends.cuda.matmul.allow_tf32
            assert torch.backends.cudnn.allow_tf32==(phase=='anchor_default')
            assert flags==dict(math=True,flash=phase=='anchor_default',mem_efficient=phase=='anchor_default',cudnn=phase=='anchor_default')
            raise RuntimeError('primary')
    assert flags==old and torch.backends.cuda.matmul.allow_tf32 is True and torch.backends.cudnn.allow_tf32 is False
    assert events[0]['phase']==phase


def test_actual_saved_material_with_original_proved_pixel_identities_RAM_only():
    p=m.material(physical_report=str(m.ROOT/'results/p1_constructive_common_prefix_probe_physical_20261027_v1_main01/report.json'),physical_sha=m.prior.REPORT_SHA)
    assert p['schema']==m.NAME+'.prepare' and p['contract']['caps']['encoder']==96
    assert len(p['warm_pixel_proofs'])==4 and len(p['images'])==92 and len(p['rows'])==84
    for proof,item in zip(p['warm_pixel_proofs'],p['images'][:4],strict=True): assert proof['pixel_sha256']==item['checkpoint']['pixel_sha256']
    assert len(m.canonical(p))<2*1024**2 and m.sha(m.prior.__file__)==m.V1_SHA
