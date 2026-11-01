import pytest
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_p2_intersection_rgb_20261029_v1 as m


def test_wrong_external_authority_fails_before_any_file_access():
    with pytest.raises(ValueError, match='authority pins'):
        m.authenticated_roster('/bad','/bad','/bad','0'*64,m.RECEIPT_SHA,m.PREPARE_SHA,m.SOURCE_SHA)


def test_wrong_source_fails_before_any_file_access():
    with pytest.raises(ValueError, match='authority pins'):
        m.authenticated_roster('/bad','/bad','/bad',m.REPORT_SHA,m.RECEIPT_SHA,m.PREPARE_SHA,'0'*64)


def fixture_prepare():
    rows=[]
    for i in range(24):
        rows.append(dict(index=i,slot=i//4,root=i//12,arm='fixture',checkpoint=(0,5,10,15)[i%4],
            prediction_role='observed_root_context_not_forecast' if i%4==0 else 'same_plan_forecast',
            predicted192=[0.0]*192,goal192=[0.0]*192,native_goal=[0.0]*7,frame=None,
            pixel_source=None,cp_public_step=(0,5,10,15)[i%4],native_response=None,
            missing_reason='physical_checkpoint_missing',raw15_bits='',prefix_bits='',selection_digest='fixture',
            status='missing_physical_checkpoint',pixels=None,preprocessed=None,latent=None,modules=[]))
    return dict(rows=rows,contract=m.contract(),expected_model_hash='fixture',bindings={},
        expected_calls={k:0 for k in m.bridge.KINDS}, inherited_physical_counts={})


def test_tempdisk_full_consumer_and_tamper(tmp_path,monkeypatch):
    p=fixture_prepare(); monkeypatch.setattr(m,'verify',lambda path,h:p)
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda bindings:None)
    folder=tmp_path/(m.NAME+'_fixture');budget=m.primitive.Budget();store=m.Store(folder,budget)
    store.put('binding.json',dict(prepared='/fixture',prepare_sha256='fixture'))
    store.put('initial_roster.json',p['rows'])
    seal=dict(rows=deepcopy(p['rows']),modules=[],counts=m.bridge.counts([]),errors=[])
    seal_sha=store.put('native_seal.json',seal,final=True)
    report=dict(schema=m.NAME+'.report',contract=m.contract(),completed=True,prepared='/fixture',prepare_sha256='fixture',
        **seal,native_seal_sha256=seal_sha,host=dict(device='cuda:0'),model_before='fixture',model_after='fixture',
        gpu_peaks=dict(allocated=1,reserved=1,guard_checks=1),resources=m.primitive.accounting(p,seal['rows'],[]),
        event_count=0,artifact_sha256=deepcopy(store.hashes),artifact_order=list(store.order),wall_before_report=0.1)
    store.put('report.json',report,final=True);h=m.sha(folder/'report.json')
    store.put('completion.json',dict(schema=m.NAME+'.completion',report_sha256=h,complete=True,wall_inclusive=0.2,boundary=m.BOUNDARY),final=True)
    result=m.consume(folder,h)
    assert len(result['rows'])==24 and all(row['actual_Q'] is None for row in result['rows'])
    (folder/'initial_roster.json').write_bytes(b'{}')
    with pytest.raises(ValueError,match='census'):m.consume(folder,h)


def test_partial_report_rejected():
    p=fixture_prepare();r=dict(schema=m.NAME+'.report',contract=m.contract(),completed=False,errors=['failed'])
    with pytest.raises(ValueError,match='complete report'):m.validate(p,r,[],lambda x:None)


def test_no_authorization_before_work():
    with pytest.raises(ValueError,match='authorization'):m.run('/missing','bad','/missing',False)


def test_prepare_write_verify_complete_and_mutation(tmp_path, monkeypatch):
    p=fixture_prepare();p.update(schema=m.NAME+'.prepare',arguments={'fixture':True})
    monkeypatch.setattr(m,'build',lambda **kw: deepcopy(p))
    path=tmp_path/(m.NAME+'_prepare.json');h=m.write(path,p)
    assert m.verify(path,h)==p
    with pytest.raises(FileExistsError):m.write(path,p)
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='SHA'):m.verify(path,h)


def test_real_store_trace_consume_with_fake_encoder(tmp_path,monkeypatch):
    p=fixture_prepare();spec=p['rows'][0];spec.update(frame={'path':'fake'},pixel_source={'path':'fake'},
        status='not_started',missing_reason=None)
    p['expected_calls'].update(encoder=1,projector=1)
    monkeypatch.setattr(m,'verify',lambda path,h:p)
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda bindings:None)
    pix=np.zeros((224,224,3),np.uint8);pre=np.zeros((1,1,3,224,224),np.float32)
    monkeypatch.setattr(m.bridge,'load_pixels',lambda source:pix)
    monkeypatch.setattr(m.bridge.pixels,'preprocess_pixels',lambda pixels:pre)
    folder=tmp_path/(m.NAME+'_trace');store=m.Store(folder,m.primitive.Budget());rows=deepcopy(p['rows'])
    store.put('binding.json',dict(prepared='/fixture',prepare_sha256='fixture'));store.put('initial_roster.json',p['rows'])
    mods=[]
    def fake_encoder(pixels,row,scope):
        row['preprocessed']=store.array('pre',pre)
        for i,(kind,inp,out) in enumerate((('encoder',pre.reshape(1,3,224,224),np.zeros((1,257,192),np.float32)),
                                          ('projector',np.zeros((1,192),np.float32),np.zeros((1,192),np.float32)))):
            entry=dict(index=i,scope=deepcopy(scope),resource=kind,completed=False)
            store.event(dict(kind='module_attempt',entry=deepcopy(entry)))
            entry['inputs']=[store.array(f'm{i}_in',inp)];entry['output']=store.array(f'm{i}_out',out);entry['completed']=True
            store.event(dict(kind='module_return',entry=deepcopy(entry)));mods.append(entry)
        row['modules']=deepcopy(mods)
        return np.zeros((192,),np.float32)
    m.primitive.process(p,rows,store,fake_encoder)
    seal=dict(rows=rows,modules=mods,counts=m.bridge.counts(mods),errors=[])
    sh=store.put('native_seal.json',seal,final=True)
    report=dict(schema=m.NAME+'.report',contract=m.contract(),completed=True,prepared='/fixture',prepare_sha256='fixture',
        **seal,native_seal_sha256=sh,host=dict(device='cuda:0'),model_before='fixture',model_after='fixture',
        gpu_peaks=dict(allocated=1,reserved=1,guard_checks=1),resources=m.primitive.accounting(p,rows,mods),
        event_count=store.index,artifact_sha256=deepcopy(store.hashes),artifact_order=list(store.order),wall_before_report=0.1)
    store.put('report.json',report,final=True);h=m.sha(folder/'report.json')
    store.put('completion.json',dict(schema=m.NAME+'.completion',report_sha256=h,complete=True,wall_inclusive=0.2,boundary=m.BOUNDARY),final=True)
    out=m.consume(folder,h)
    assert len(out['rows'])==24 and out['rows'][0]['actual_Q']==0 and out['rows'][1]['actual_Q'] is None
    (folder/'event_0000.json').write_bytes(b'{}')
    with pytest.raises(ValueError,match='census'):m.consume(folder,h)
