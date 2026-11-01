"""Only synthetic RAM environments and temporary disk; no production source/NN/physics."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import agent_lewm_bounded_cem_h3_physical_20261029_v1 as m
from test_agent_allmode_physical_20261028_v1 import fixture as old_fixture, Env, fake_place, h, reseal


def fixture():
    _,data,*_=old_fixture()
    c=dict(schema=m.PRODUCER+'.prepare',expected_model_hash=h('fixture model'),sources={},slots=[])
    rows=[]
    for i in (0,1):
        packet=data['roots'][i]['packets']['off'];ex=data['exports'][i,'off']
        slot=dict(root_slot=i,condition='off',task_id=h(('task',i)),task_sha256=h(('taskfile',i)),
            blind_path=str(ex['path']),blind_sha256=ex['sha256'],warm_raw10=m.moving.warmup_commands().tolist(),
            pixel_sha256=[m.moving.image_sha(x) for x in packet['rgb_history']]+[m.moving.image_sha(packet['goal_rgb'])])
        c['slots'].append(slot)
        a=np.full((15,2),.125*i,np.float32)
        rows.append(dict(root=i,task_id=slot['task_id'],task_sha256=slot['task_sha256'],
            actual_raw15=a.tolist(),actual_raw15_bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            physical_release=False,delivery=dict(returned_mean_evaluated_in_new_job=True)))
    u=dict(schema=m.PRODUCER+'.consume',status='known2_H3_bounded_CEM_actual_raw_saved_verified',
        report_sha256=m.REPORT_SHA,rows=rows,resources=dict(predictor=192,encoder=0,AD=0,physics=0),
        inherited=dict(scope='synthetic'),actual_modules={},fair_ranking=False,official_reproduction=False,
        physical_release=False,scientific_admission=False)
    return m.assemble({},c,u,data),data,c,u


def execute(tmp_path,monkeypatch,*,fault=None,at=11,close_error=False):
    p,data,_,_=fixture();store=m.Store(tmp_path/(m.NAME+'_fixture'))
    envs=[]
    def factory():
        e=Env(fault=fault,at=at,close_error=close_error);envs.append(e);return e
    monkeypatch.setattr(m,'place',fake_place);monkeypatch.setattr(m,'validate_env',lambda _:None)
    r=m.execute(p,data,store,factory,authorization=m.AUTH,prepared=str(tmp_path/'fixture_prepare.json'),prepare_sha256=h('prepare'))
    return r,store,p,data,envs


@pytest.fixture(scope='module')
def complete(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:return execute(tmp_path_factory.mktemp('CEM_FULL15'),mp)


def test_full_real_store_to_reader_consumer_all162(complete,monkeypatch):
    r,s,p,data,envs=complete
    assert r['status']==m.RECORDED,r['errors']
    files=m.saved_artifacts(s.path);events=m.journal_events(files['journal.jsonl'])
    assert len(m.validate_saved(r,events,p))==2
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert sum(n.endswith('.png') for n in files)==162
    monkeypatch.setattr(m,'verify',lambda *a:(p,data))
    v=m.consume(s.path,m.sha(s.path/'report.json'),m.sha(s.path/'completion.json'))
    assert v['saved_PNG_verified']==162 and v['all_two_horizons_completed']
    for e,plan,row in zip(envs,p['plans'],r['rows'],strict=True):
        assert e.closed and np.asarray(e.commands[10:],np.float32).tobytes()==m.raw15(plan['commands']).tobytes()
        assert e.dispatch_flags==[False]*25 and row['prefix51']['complete']
        assert row['missing_checkpoints']==[] and row['own_returned']==15
    assert all(v[k] is False for k in ('P1_admission','P2_admission','scientific_admission','official_reproduction'))


def test_integer_transport_no_arbitrary_copy_factor(complete):
    c=m.shape_census();r,s,*_=complete;files=m.saved_artifacts(s.path)
    assert c['caps']==m.contract()['caps']
    assert c['operation_receipts']==748 and c['operation_events']==1496
    assert c['PNG_count']==162 and c['caps']['physics_step']==508 and c['caps']['public_step']==50
    assert c['working_upper_bytes']<m.CAP-m.FINAL and c['total_upper_bytes']<m.CAP
    assert c['prefix_copies']==102 and c['selected_frames']==54 and c['checkpoint_copies']==8
    assert sum(map(len,files.values()))<=c['total_upper_bytes']
    assert sum(len(v) for n,v in files.items() if not n.endswith('.png'))<=m.META_TOTAL
    assert len(m.raw(r))<m.FINAL


def test_incompressible_all162_png_actual_writer(tmp_path):
    store=m.Store(tmp_path/(m.NAME+'_noise'));rng=np.random.default_rng(123)
    for i in range(162):
        a=rng.integers(0,256,(224,224,3),dtype=np.uint8)
        b=io.BytesIO();Image.fromarray(a).save(b,format='PNG')
        assert len(b.getvalue())<=m.shape_census()['PNG_file_upper_bytes']
        store.write('noise_%03d.png'%i,b.getvalue())
    assert sum(f.stat().st_size for f in store.path.iterdir())<=m.shape_census()['PNG_total_upper_bytes']
    with pytest.raises(ValueError,match='count'):store.write('extra.png',b.getvalue())


@pytest.mark.parametrize('mutation',['order','raw','hash','task','evaluated','partial','denominator','schema','warm','pixels','blind','fair'])
def test_own_actualreceipt_identity_mutations(mutation):
    p,data,c,u=fixture()
    if mutation=='order':u['rows'].reverse()
    elif mutation=='raw':u['rows'][0]['actual_raw15'][0][0]=.125
    elif mutation=='hash':u['rows'][0]['actual_raw15_bytes_sha256']=h('bad')
    elif mutation=='task':u['rows'][0]['task_sha256']=h('other')
    elif mutation=='evaluated':u['rows'][0]['delivery']['returned_mean_evaluated_in_new_job']=False
    elif mutation=='partial':u['status']='partial'
    elif mutation=='denominator':u['rows'].pop()
    elif mutation=='schema':u['schema']='old_grasp.consume'
    elif mutation=='warm':c['slots'][0]['warm_raw10'][0][0]+=.01
    elif mutation=='pixels':c['slots'][0]['pixel_sha256'][0]=h('bad')
    elif mutation=='blind':c['slots'][0]['blind_sha256']=h('bad')
    else:u['fair_ranking']=True
    with pytest.raises((ValueError,KeyError)):m.assemble({},c,u,data)


@pytest.mark.parametrize('fault,at',[('truncate',5),('terminal',11),('truncate',15),('truncate',20),('terminal',25)])
def test_native_early_stop_null_not_padding(tmp_path,monkeypatch,fault,at):
    r,s,p,data,envs=execute(tmp_path,monkeypatch,fault=fault,at=at)
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,m.journal_events((s.path/'journal.jsonl').read_bytes()),p))==2
    monkeypatch.setattr(m,'verify',lambda *a:(p,data))
    v=m.consume(s.path,m.sha(s.path/'report.json'),m.sha(s.path/'completion.json'))
    assert len(v['rows'])==2
    for row,e in zip(v['rows'],envs,strict=True):
        assert len(e.commands)==at and e.closed and row['own_returned']==max(at-10,0)
        assert (15 in row['missing_checkpoints'])==(at<25)
        if at<10:assert row['final_state'] is row['final_metrics'] is row['native_success'] is None


@pytest.mark.parametrize('fault',['throw','guard','substitute','render'])
def test_engineering_failure_not_native_negative(tmp_path,monkeypatch,fault):
    r,s,p,data,envs=execute(tmp_path,monkeypatch,fault=fault,close_error=True)
    assert r['status']=='partial_or_failed' and len(r['rows'])==2 and all(e.closed for e in envs)
    assert all(row['first_error'] for row in r['rows'])
    assert all(row['status']=='engineering_failed' for row in r['rows'])


@pytest.mark.parametrize('field',['event','raw15','checkpoint','reward','goal','count','arm','return'])
def test_saved_resealed_corruption_refused(complete,field):
    orig,s,p,*_=complete;r=deepcopy(orig);events=m.journal_events((s.path/'journal.jsonl').read_bytes())
    if field=='event':events.insert(-1,dict(kind='unknown'))
    elif field=='raw15':
        next(e for e in events if e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step'
            and e['receipt'].get('phase')=='plan' and e['receipt']['step']==15)['receipt']['raw_bits']='00'
    elif field=='checkpoint':r['rows'][0]['checkpoints'].pop()
    elif field=='reward':
        next(e for e in events if e['kind']=='returned_transition' and e['phase']=='plan')['reward']=0.
    elif field=='goal':next(e for e in events if e['kind']=='goal_attached')['goal'][0]+=.5
    elif field=='count':r['counts']['physics_step']['completed']-=1
    elif field=='arm':r['rows'][1]['arm']='old_schema'
    else:events.remove(next(e for e in events if e['kind']=='returned_transition' and e['phase']=='plan'))
    reseal(events,r)
    with pytest.raises((ValueError,KeyError,StopIteration)):m.validate_saved(r,events,p)


@pytest.mark.parametrize('mutation',['png','extra','blank','completion','late','bytes','wall'])
def test_full_saved_consumer_transport_tamper(complete,monkeypatch,mutation):
    r,s,p,data,_=complete;files=m.saved_artifacts(s.path)
    if mutation=='png':
        name=next(n for n in files if n.endswith('.png'));files[name]=files[name][:-1]+b'x'
    elif mutation=='extra':files['unknown']=b'x'
    elif mutation=='blank':files['journal.jsonl']+=b'\n'
    elif mutation=='completion':files['completion.json']+=b' '
    elif mutation=='late':files['completion_failure.json']=m.raw(dict(error='late'))
    else:
        c=m.native.decode(files['completion.json'])
        if mutation=='bytes':c['bytes_before_completion']+=1
        else:c['inclusive_wall_seconds']=m.WALL
        files['completion.json']=m.raw(c)
    monkeypatch.setattr(m,'saved_artifacts',lambda _:files)
    monkeypatch.setattr(m,'verify',lambda *a:(p,data))
    with pytest.raises((ValueError,KeyError)):m.consume(s.path,hashlib.sha256(files['report.json']).hexdigest(),
        hashlib.sha256(files['completion.json']).hexdigest())


def test_journal_exactly_one_newline_real_store(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_codec'));s.emit('first',x=1);s.emit('second',x=2)
    b=(s.path/'journal.jsonl').read_bytes();events=m.journal_events(b)
    assert len(events)==2 and b'\n\n' not in b
    assert events[1]['previous_sha256']==events[0]['event_sha256']
    for e in events:assert e['event_sha256']==m.source.digest({k:v for k,v in e.items() if k!='event_sha256'})
    for bad in (b+b'\n',b.replace(b'\n',b'\r\n'),b'\n'):
        with pytest.raises(ValueError):m.journal_events(bad)


def test_no_auth_no_read_no_factory(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('source opened'))
    with pytest.raises(ValueError):m.run('/RAM/p',h('p'),'/RAM/o',lambda:pytest.fail('env'),authorization='bad')


def test_domain_lossless_no_clipping():
    a=np.zeros((15,2),np.float32);a[0,0]=-0.
    assert m.raw15(a.tolist()).tobytes()==a.tobytes()
    for bad in ([[True,0]]*15,[[1.01,0]]*15,[[0,0]]*10,[[.1,0]]*15):
        with pytest.raises(ValueError):m.raw15(bad)


def test_quota_prewrite_no_metadata_partial(tmp_path,monkeypatch):
    s=m.Store(tmp_path/(m.NAME+'_quota'));monkeypatch.setattr(m,'META_WORK',1)
    with pytest.raises(ValueError,match='metadata'):s.emit('first',x=1)
    assert not list(s.path.iterdir())


def test_postpublication_wall_failure_durable(tmp_path,monkeypatch):
    s=m.Store(tmp_path/(m.NAME+'_late'));s.emit('x')
    r=dict(execution_binding=dict(prepared='/RAM/p',prepare_sha256=h('p')))
    clock=iter([m.WALL-1,m.WALL+1])
    with pytest.raises(ValueError,match='wall failed'):m.publish(s,r,0,clock=lambda:next(clock))
    assert (s.path/'completion_failure.json').is_file()


def temp_build(tmp_path,monkeypatch):
    p,data,c,u=fixture()
    out=tmp_path/'upstream';out.mkdir()
    prep=tmp_path/'upstream_prepare.json';prep.write_bytes(m.raw(c))
    receipt=tmp_path/'upstream_receipt.json'
    report=dict(schema=m.PRODUCER+'.report',prepared=str(prep),prepare_sha256=m.sha(prep),
        status='known2_H3_bounded_CEM_actual_raw_evaluated',errors=[],source_verified=True,
        all_final_predictions_sealed=True,model_before=c['expected_model_hash'],model_after=c['expected_model_hash'],
        actual_modules=u['actual_modules'],rows=[],artifacts={})
    for i in (0,1):
        f=out/('raw%d.npy'%i);np.save(f,np.asarray(u['rows'][i]['actual_raw15'],np.float32))
        rec=dict(file=f.name,file_bytes=f.stat().st_size,sha256=m.sha(f),dtype='<f4',shape=[15,2],nbytes=120)
        report['artifacts'][f.name]={k:rec[k] for k in ('file_bytes','sha256')}
        report['rows'].append(dict(delivery=dict(raw15=rec)))
    (out/'report.json').write_bytes(m.raw(report));u['report_sha256']=m.sha(out/'report.json')
    receipt.write_bytes(m.raw(u))
    monkeypatch.setattr(m,'PLANNER_PREP',prep);monkeypatch.setattr(m,'PREP_SHA',m.sha(prep))
    monkeypatch.setattr(m,'PLANNER_OUTPUT',out);monkeypatch.setattr(m,'REPORT_SHA',m.sha(out/'report.json'))
    monkeypatch.setattr(m,'PLANNER_RECEIPT',receipt);monkeypatch.setattr(m,'RECEIPT_SHA',m.sha(receipt))
    producer_path=m.ROOT/'experiments/spectral_reachability'/(m.PRODUCER+'.py')
    monkeypatch.setattr(m,'PRODUCER_SHA',m.sha(producer_path))
    producer=SimpleNamespace(__file__=str(producer_path),verify=lambda path,expected:m.read(path,m.META))
    monkeypatch.setattr(m.importlib,'import_module',lambda _:producer)
    monkeypatch.setattr(m.native,'source_material',lambda:deepcopy(data))
    return out


def test_complete_temporary_build_prepare_verify(tmp_path,monkeypatch):
    out=temp_build(tmp_path,monkeypatch)
    prepared=tmp_path/(m.NAME+'_prepare_RAM.json')
    meta=m.prepare(prepared);p,data=m.verify(prepared,meta['sha256'])
    assert p['plans'][0]['commands']==m.read(m.PLANNER_RECEIPT,m.META)['rows'][0]['actual_raw15']
    assert len(p['plans'])==2 and m.sha(out/'raw0.npy') in p['bindings'].values()
    # Original source drift detected; no output mutation or re-CEM.
    with (out/'raw0.npy').open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError):m.verify(prepared,meta['sha256'])


def test_full_temporary_build_run_consume_without_verify_stub(tmp_path,monkeypatch):
    temp_build(tmp_path,monkeypatch)
    prepared=tmp_path/(m.NAME+'_prepare_chain.json');meta=m.prepare(prepared)
    monkeypatch.setattr(m,'place',fake_place);monkeypatch.setattr(m,'validate_env',lambda _:None)
    out=tmp_path/(m.NAME+'_execution_chain')
    r=m.run(prepared,meta['sha256'],out,Env,authorization=m.AUTH)
    assert r['status']==m.RECORDED,r['errors']
    v=m.consume(out,m.sha(out/'report.json'),m.sha(out/'completion.json'))
    assert v['saved_PNG_verified']==162 and v['fixed_denominator']==2
    assert v['physical_report_sha256']==m.sha(out/'report.json')
    receipt=tmp_path/(m.NAME+'_consume_chain.json');sealed=m.write_new(receipt,v)
    assert m.raw(m.read(receipt,m.META))==m.raw(v) and sealed['sha256']==m.sha(receipt)


def test_invalid_native_image_saved_as_bounded_incident(tmp_path,monkeypatch):
    p,data,_,_=fixture();s=m.Store(tmp_path/(m.NAME+'_invalid_native'))
    monkeypatch.setattr(m,'place',fake_place);monkeypatch.setattr(m,'validate_env',lambda _:None)
    r=m.execute(p,data,s,lambda:Env(overlay_fault='invalid'),authorization=m.AUTH,
        prepared=str(tmp_path/'fixture_prepare.json'),prepare_sha256=h('prepare'))
    assert r['status']=='partial_or_failed' and len(r['rows'])==2
    assert sum(f.suffix=='.npz' for f in s.path.iterdir())==2
    assert len([e for e in m.journal_events((s.path/'journal.jsonl').read_bytes()) if e['kind']=='invalid_observation'])==2
    assert all(row['own_returned']==0 and row['final_state'] is None for row in r['rows'])


@pytest.mark.parametrize('field',['receipt','raw','report','source'])
def test_temp_upstream_corruption_failclosed(tmp_path,monkeypatch,field):
    out=temp_build(tmp_path,monkeypatch)
    if field=='source':monkeypatch.setattr(m,'PRODUCER_SHA',h('bad'))
    else:
        target=m.PLANNER_RECEIPT if field=='receipt' else out/('raw0.npy' if field=='raw' else 'report.json')
        with target.open('ab') as f:f.write(b' ')
    with pytest.raises(ValueError):m.material()

