"""RAM surfaces and Python protocol doubles only; no native/env/model imports."""
from copy import deepcopy
import gzip
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_raw_native_callback_capture_20261029_v2 as m


class Surface:
    def __init__(self,color):self.data=np.empty((512,512,3),np.uint8);self.data[:]=color
    def get_size(self):return (512,512)


def fixture_api():
    calls=[]
    def draw(surface,color,pos,radius,width=0):
        calls.append((surface,color,pos,radius,width))
        surface.data[int(pos[1]),int(pos[0])]=np.asarray(color[:3],np.uint8)
        return (int(pos[0]),int(pos[1]),1,1)
    pg=NS(draw=NS(**{k:draw for k in m.PRIMITIVES}),surfarray=NS(array3d=lambda s:s.data.transpose(1,0,2).copy()))
    resize_calls=[]
    def resize(src,size):
        resize_calls.append((src.copy(),size));return src[:224,:224].copy() # explicit RAM fixture, NOT native cv2 evidence
    return pg,NS(resize=resize),calls,resize_calls


@pytest.fixture
def work(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=20*1024**3))
    return tmp_path


def actor_fixture(pg,cv,angle=0.):
    ns={'np':np,'pg':pg,'cv':cv,'Surface':Surface}
    text='''class DrawOptions:
    def __init__(self,surface):
        self.surface=surface;self.flags=7;self.transform=(1.,0.,0.,1.,0.,0.)
    def draw_dot(self,size,pos,color):
        pg.draw.circle(self.surface,color,(round(pos[0]),round(pos[1])),round(size),0)
    def draw_polygon(self,verts,radius,outline_color,fill_color):
        for pos in verts:self.draw_dot(radius,pos,fill_color)
'''
    exec(compile(text,str(m.factor.UTILS),'exec'),ns)
    render='''def _render_frame(self,mode):
    self.screen=Surface(self.color)
    options=DrawOptions(self.screen)
    options.draw_polygon([(10.1+self.shift,11.2),(12.3,14.5)],2.,(1,2,3,255),(4,5,6,255))
    img=self.screen.data.copy()
    return cv.resize(img,(224,224))
'''
    exec(compile(render,str(m.factor.ENV),'exec'),ns)
    body=NS(color=(255,255,255),shift=angle,state=[1.,2.,3.],with_target=False,
        variation_space=NS(**{'background':None}))
    body.variation_space={'background':{'color':NS(value=np.array(body.color,np.uint8))}}
    body._render_frame=lambda mode:ns['_render_frame'](body,mode)
    return body


def capture(work,monkeypatch,name='capture',shift=0.):
    store=m.Store(work/'results'/(m.NAME+'_'+name));pg,cv,draw_calls,resize_calls=fixture_api()
    meter=NS(where=dict(branch=0),receipts=[]);body=actor_fixture(pg,cv,shift)
    tap=m.Tap(pg,cv,store,meter,m.factor.UTILS,m.factor.SITE/'pymunk/space_debug_draw_options.py',m.factor.ENV)
    tap.bind(body,0);monkeypatch.setattr(tap,'stamp',lambda:dict(state=body.state.copy(),flag=body.with_target,rng=m.rng_state()))
    return tap,body,pg,cv,store,draw_calls,resize_calls


def test_cold_import_no_native_or_neural():
    assert not any(k in sys.modules for k in ('pymunk','pygame','cv2','torch','gymnasium','stable_worldmodel'))


def test_fixed_schedule_caps_no_unknown_drop():
    c=m.contract();assert c['jobs']==164 and c['pairs']==80 and len(c['pairs_schedule'])==80
    assert c['caps']==m.old.contract()['caps']
    assert c['caps']['physics_step']==33456 and c['render_cap']==11152
    assert c['P1_admission'] is False


def test_transparent_forward_once_callbacks_and_original_raster(work,monkeypatch):
    tap,body,pg,cv,store,draw,resize=capture(work,monkeypatch)
    baseline=body._render_frame('rgb_array');draw.clear();resize.clear()
    original_draw={k:getattr(pg.draw,k) for k in m.PRIMITIVES};original_resize=cv.resize
    before=m.rng_state()
    with tap:actual=body._render_frame('rgb_array')
    assert m.rng_state()==before and np.array_equal(actual,baseline)
    assert len(draw)==2 and len(resize)==1 and tap.total_callbacks==3
    assert all(getattr(pg.draw,k) is v for k,v in original_draw.items()) and cv.resize is original_resize and sys.getprofile() is None
    assert not tap.failed and not tap.restore_errors and len(tap.captures)==1
    shot=tap.captures[0];transcript=m.transcript_load(store.path,shot);m.validate_transcript(transcript)
    assert [e['name'] for e in transcript['events'] if e['kind']=='callback_call']==['draw_polygon','draw_dot','draw_dot']
    assert [e['args'] for e in transcript['events'] if e['kind']=='primitive_intent']==[[[4,5,6,255],[10,11],2,0],[[4,5,6,255],[12,14],2,0]]
    assert np.array_equal(m.load_rgb(store.path,shot['rgb224']),actual)
    assert m.load_rgb(store.path,shot['surface512']).shape==(512,512,3)


@pytest.mark.parametrize('mutation',('state','rng','output','missing_resize','double_resize','foreign_surface','profile_latch'))
def test_bad_render_never_release(work,monkeypatch,mutation):
    tap,body,pg,cv,store,draw,resize=capture(work,monkeypatch,name=mutation)
    old_resize=cv.resize
    if mutation=='state':
        def new_resize(*a):body.state[0]+=1;return old_resize(*a)
        cv.resize=new_resize
    if mutation=='rng':
        def new_resize(*a):np.random.random();return old_resize(*a)
        cv.resize=new_resize
    if mutation=='output':
        fn=body._render_frame
        body._render_frame=lambda mode:fn(mode) # failure injected in returned saved array below
        old_frame=tap.finish
        def bad_finish(value):
            if value is not None:value=value.copy();value[0,0,0]^=1
            return old_frame(value)
        monkeypatch.setattr(tap,'finish',bad_finish)
    with tap:
        if mutation=='missing_resize':
            tap.current=dict(stack=[],resizes=0)
            with pytest.raises(ValueError):tap.finish(np.zeros((224,224,3),np.uint8))
        elif mutation=='double_resize':
            body._render_frame('rgb_array');tap.current=dict(stack=[],resizes=1)
            with pytest.raises(ValueError):cv.resize(np.zeros((512,512,3),np.uint8),(224,224))
        elif mutation=='foreign_surface':
            body.screen=Surface((255,255,255))
            tap.current=dict(events=[],primitives=0,stack=[])
            with pytest.raises(ValueError):pg.draw.circle(Surface((255,255,255)),(1,2,3),(1,1),2)
        elif mutation=='profile_latch':
            tap.failed.append(dict(error='fixture'))
            with pytest.raises(ValueError):body._render_frame('rgb_array')
        else:body._render_frame('rgb_array')
    if mutation in ('state','rng','output'):assert tap.failed and not tap.captures


def test_restore_on_exception(work,monkeypatch):
    tap,body,pg,cv,*_=capture(work,monkeypatch)
    original=cv.resize
    with pytest.raises(RuntimeError):
        with tap:raise RuntimeError('fixture')
    assert cv.resize is original and sys.getprofile() is None


def test_existing_profile_not_destroyed(work,monkeypatch):
    tap,*_=capture(work,monkeypatch);profile=lambda *a:None
    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError):tap.__enter__()
        assert sys.getprofile() is profile
    finally:sys.setprofile(None)


@pytest.mark.parametrize('field',('callback','primitive','resize','before','counter','missing_final','source','late_callback'))
def test_transcript_mutations_rejected(work,monkeypatch,field):
    tap,body,pg,cv,store,*_=capture(work,monkeypatch)
    with tap:body._render_frame('rgb_array')
    t=m.transcript_load(store.path,tap.captures[0])
    if field=='callback':next(e for e in t['events'] if e['kind']=='callback_return')['callback']=999
    if field=='primitive':next(e for e in t['events'] if e['kind']=='primitive_return')['primitive']=999
    if field=='resize':next(e for e in t['events'] if e['kind']=='resize_return')['result']['pixel_sha256']='0'*64
    if field=='before':t['before']['state'][0]=999
    if field=='counter':t['callbacks']+=1
    if field=='missing_final':t['events'].pop()
    if field=='source':next(e for e in t['events'] if e['kind']=='callback_call')['source']='unbound.py'
    if field=='late_callback':
        ev=deepcopy(next(e for e in t['events'] if e['kind']=='callback_call'));ev['index']=len(t['events'])-1
        t['events'].insert(-1,ev);t['events'][-1]['index']+=1
    with pytest.raises(ValueError):m.validate_transcript(t)


@pytest.mark.parametrize('kind',('trailing','member','truncated','cap','semantic_digest'))
def test_saved_transcript_gzip_and_hash_guards(work,monkeypatch,kind):
    tap,body,pg,cv,store,*_=capture(work,monkeypatch)
    with tap:body._render_frame('rgb_array')
    shot=deepcopy(tap.captures[0]);path=store.path/shot['file'];original=path.read_bytes()
    if kind=='trailing':bodybytes=original+b'x'
    elif kind=='member':bodybytes=original+gzip.compress(b'{}')
    elif kind=='truncated':bodybytes=original[:-1]
    elif kind=='cap':bodybytes=gzip.compress(b'x'*(m.SHOT_CAP+1))
    else:bodybytes=original;shot['transcript_digest']='0'*64
    with path.open('wb') as f:f.write(bodybytes)
    shot['sha256']=m.sha(path)
    with pytest.raises(ValueError):m.transcript_load(store.path,shot)


def test_all80_missing_pairs_retained(work):
    rows=m.pair_analysis([],work);assert len(rows)==80
    assert all(x['missing']==['minus','plus'] and x['rgb224_equal'] is None for x in rows)


@pytest.mark.parametrize('shift,same_primitive,same_rgb',( (0.,True,True),(.1,True,True),(1.,False,False) ))
def test_pairs_compare_actual_layers_no_outcome_selection(work,monkeypatch,shift,same_primitive,same_rgb):
    tap,body,pg,cv,store,*_=capture(work,monkeypatch)
    with tap:
        tap.bind(body,1);body._render_frame('rgb_array')
        body.shift=shift;tap.bind(body,2);body._render_frame('rgb_array')
    for s in tap.captures:s['original_observation']=dict(phase='plan',step=10,condition='off',overlay_index=0)
    pair=m.pair_analysis(tap.captures,store.path)[0]
    assert pair['integer_primitives_equal'] is same_primitive and pair['rgb224_equal'] is same_rgb
    assert pair['missing']==[]


def test_meter_intent_return_failure_and_cap(work):
    store=m.Store(work/'results'/(m.NAME+'_meter'));meter=m.Meter(store);meter.where=dict(branch=0)
    assert meter.call('constructor',lambda:'value')=='value'
    with pytest.raises(RuntimeError):meter.call('close',lambda:(_ for _ in ()).throw(RuntimeError('fixture')),cleanup=True)
    assert meter.counts['constructor']==dict(attempted=1,completed=1)
    assert meter.counts['close']==dict(attempted=1,completed=0)
    meter.counts['constructor']['attempted']=164
    with pytest.raises(ValueError):meter.call('constructor',lambda:pytest.fail('must not dispatch'))


def test_authority_before_any_read_or_factory(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('no read without MAIN authority'))
    with pytest.raises(ValueError):m.run('none','none','none')


def test_initial_reset_goal_not_fabricated(monkeypatch):
    pg,cv,*_=fixture_api();b=NS(_goal=None,goal_state=np.arange(7,dtype=float),goal_pose=np.arange(3,dtype=float),
        with_target=False,render_mode='rgb_array',render_size=224,window_size=512,render_action=False,
        latest_action=None,variation_space=NS(value={}),rng=np.random.default_rng(2))
    tap=m.Tap(pg,cv,None,NS(),m.factor.UTILS,'options',m.factor.ENV);tap.bind(b,0)
    monkeypatch.setattr(m.old.moving,'snapshot',lambda b:dict(state=b.goal_state.tolist()))
    assert tap.stamp()['goal_rgb'] is None


def test_output_exclusive_floor_and_cap(work,monkeypatch):
    path=work/'results'/(m.NAME+'_budget');store=m.Store(path)
    store.put('one.json',{})
    with pytest.raises(ValueError):store.put('one.json',{})
    store.bytes=m.CAP-m.FINAL
    with pytest.raises(ValueError):store.write('extra',b'1')
    with pytest.raises(ValueError):m.Store(path)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=m.CAP+m.FLOOR-1))
    with pytest.raises(ValueError):m.Store(work/'results'/(m.NAME+'_floor'))


def test_prepare_exact_semantic_binding(work,monkeypatch):
    q=dict(schema=m.NAME+'.prepare',source='fixture');monkeypatch.setattr(m,'source_material',lambda:(q,None,None,None))
    path=work/'results'/(m.NAME+'_prepare.json');m.prepare(path);m.verify(path,m.sha(path))
    q['source']='changed'
    with pytest.raises(ValueError):m.verify(path,m.sha(path))


def test_pending_stack_has_real_receipt_identity(work,monkeypatch):
    tap,*_=capture(work,monkeypatch)
    ops=[dict(index=0,status='pending',resource='reset'),dict(index=1,status='returned',resource='set_state'),
        dict(index=2,status='pending',resource='goal_render')]
    tap.meter=NS(receipts=ops,active={'reset':1,'goal_render':1,'set_state':0})
    assert tap.pending_operations()==[ops[0],ops[2]]
    tap.meter.active['reset']=2
    with pytest.raises(ValueError):tap.pending_operations()


def test_real_store_two_event_disk_reader_hashchain(work):
    store=m.Store(work/'results'/(m.NAME+'_codec_chain'))
    store.emit('first',value={'z':'escaped\nnewline','a':1})
    store.emit('second',value=[2,3])
    path=store.path/'journal.jsonl';body=path.read_bytes()
    assert body.count(b'\n')==2 and b'\n\n' not in body
    events=list(m.journal_events(store.path));assert len(events)==2
    previous=None
    for index,event in enumerate(events):
        assert event['index']==index and event['previous_sha256']==previous
        unsigned={k:v for k,v in event.items() if k!='event_sha256'}
        assert event['event_sha256']==m.source.digest(unsigned)
        assert body.splitlines(keepends=True)[index]==m.json_record(event)
        previous=event['event_sha256']
    assert previous==store.previous and store.index==2
    assert store.hashes['journal.jsonl'].hexdigest()==m.sha(path)
    assert store.sizes['journal.jsonl']==len(body)


def test_real_put_and_write_new_share_exact_codec(work):
    value={'z':'escaped\nline','a':[1,2]}
    store=m.Store(work/'results'/(m.NAME+'_codec_put'))
    store.put('value.json',value)
    path=store.path/'value.json'
    assert path.read_bytes()==m.json_record(value)
    assert m.read(path,m.sha(path))==value
    prepared=work/'results'/(m.NAME+'_codec_prepare.json')
    m.write_new(prepared,value)
    assert prepared.read_bytes()==m.json_record(value)
    assert m.read(prepared,m.sha(prepared))==value


@pytest.mark.parametrize('body',[
    b'\n',b'{}\n\n',b'{}',b'{}\r\n',b'{\n}\n',b'{ "a":1}\n',
    b'{"z":1,"a":2}\n',b'{"a":1,"a":2}\n',b'{"a":NaN}\n',
    b'{}\ntrailing',b'{} {}\n'])
def test_noncanonical_disk_records_rejected(work,body):
    store=m.Store(work/'results'/(m.NAME+'_bad_codec'))
    store.write('journal.jsonl',body)
    with pytest.raises((ValueError,UnicodeError)):list(m.journal_events(store.path))
    path=store.path/'value.json';store.write('value.json',body)
    with pytest.raises((ValueError,UnicodeError)):m.read(path)


def test_upstream_codec_is_separate_not_v1_partial_repair(work):
    value={'a':1,'b':[2,3]};original=m.old.raw(value)
    assert original.endswith(b'\n') and not original.endswith(b'\n\n')
    assert m.decode_record(original,upstream=True)==value
    assert m.decode_record(m.json_record(value))==value
    with pytest.raises(ValueError):m.decode_record(original)
    with pytest.raises(ValueError):m.decode_record(m.json_record(value),upstream=True)
    store=m.Store(work/'results'/(m.NAME+'_v1_partial_rejected'))
    store.write('journal.jsonl',original+b'\n')
    with pytest.raises(ValueError):list(m.old_events(store.path))


def test_actual_pinned_upstream_journal_first_two_records():
    from itertools import islice
    directory=m.ROOT/'results'/Path(m.factor.PINS['physical_report'][0]).parent
    events=list(islice(m.old_events(directory),2))
    assert len(events)==2
    previous=None
    for index,event in enumerate(events):
        assert event['index']==index and event['previous_sha256']==previous
        assert event['event_sha256']==m.source.digest({k:v for k,v in event.items() if k!='event_sha256'})
        previous=event['event_sha256']


@pytest.mark.parametrize('mutation',('double_lf','missing_lf','spaced','trailing_gzip'))
def test_real_transcript_disk_canonical_rejection(work,mutation):
    value={'a':1,'b':[2,3]};payload=m.json_record(value)
    if mutation=='double_lf':payload+=b'\n'
    if mutation=='missing_lf':payload=payload[:-1]
    if mutation=='spaced':payload=m.old.raw(value)
    body=gzip.compress(payload,mtime=0)
    if mutation=='trailing_gzip':body+=gzip.compress(b'{}\n',mtime=0)
    store=m.Store(work/'results'/(m.NAME+'_transcript_codec'))
    store.write('transcript.json.gz',body)
    rec=dict(file='transcript.json.gz',sha256=m.sha(store.path/'transcript.json.gz'),transcript_digest=m.digest(value))
    with pytest.raises(ValueError):m.transcript_load(store.path,rec)
