"""RAM-only tests: no historical outcome evaluation, environment or model."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import pytest
from experiments.spectral_reachability import root_exclusion_authority_resolution_20261027_v1 as n


def raw(x):return (json.dumps(x,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def reader(files):return n.Reader(Path('/tmp'),lambda p:files[p])
def proj(x,patterns):return n.project(BytesIO(raw(x)),patterns)


def test_inheritance_dedup_preserves_all_960_references():
    unknown=[dict(source='/tmp/f'+str(i)+'.json',reason='old_failure') for i in range(480)]
    x=dict(exclusions=dict(files={'/tmp/f0.json':'a'*64},unknown_sources=unknown))
    p=proj(x,[('exclusions','files','*'),('exclusions','unknown_sources','*','source'),('exclusions','unknown_sources','*','reason')])
    refs=n.inherited([('p0',p),('p1',p)])
    assert len(refs)==480 and sum(map(len,refs.values()))==960
    assert len(refs['/tmp/f0.json'])==2 and refs['/tmp/f0.json'][0]['historical_sha256']=='a'*64
    assert refs['/tmp/f1.json'][0]['historical_sha256'] is None


@pytest.mark.parametrize('field',['success','witness','outcomes','future','cost','selected_actions'])
def test_opaque_outcome_not_decoded(field):
    # Opaque invalid JSON scalar would fail json.loads if decoded.
    data=b'{"root_seed": 17, "'+field.encode()+b'": NaN}'
    p=n.project(BytesIO(data),n.META_PATTERNS)
    assert p.leaves=={('root_seed',):17}


def test_capture_attestation_and_projection():
    data=raw(dict(root_seed=17,root_sha256='b'*64,success=False))
    files={'/tmp/a.json':data,'/tmp/b.json':data}
    refs={'/tmp/a.json':[dict(parent='p',pointer='/x',historical_sha256=sha(data),original_reason='historical_reset_attempt_bound_not_attested')],
          '/tmp/b.json':[dict(parent='p',pointer='/y',historical_sha256=None,original_reason='parsefail')]}
    r=n.capture_history(reader(files),refs)
    assert r[0]['hash_authority']=='external_attested'
    assert r[1]['hash_authority']=='first_capture_not_historical_attestation'
    assert len(r[0]['projection'])==2
    assert any('reset_attempt' in x for x in r[0]['blockers'])


def test_hash_mismatch_is_not_resolved():
    r=n.capture_history(reader({'/tmp/a.json':b'{}'}),{},[dict(path='/tmp/a.json',sha256='b'*64,family='legacy_zero')])
    assert r[0]['status']=='blocked' and 'SHA mismatch' in r[0]['blockers'][0]


def test_conflicting_historical_sha_is_not_opened():
    refs={'/tmp/a.json':[dict(historical_sha256=x,original_reason='old') for x in ('a'*64,'b'*64)]}
    r=n.capture_history(reader({}),refs)
    assert 'conflicting' in r[0]['blockers'][0]


def test_empty_metadata_does_not_certify_identity():
    r=n.capture_history(reader({'/tmp/a.json':raw(dict(success=True))}),{},[dict(path='/tmp/a.json',sha256=sha(raw(dict(success=True))),family='feedback')])
    assert not r[0]['projection'] and r[0]['blockers']


@pytest.mark.parametrize('path',['relative.json','/etc/passwd','/tmp/../etc/passwd'])
def test_path_escape(path):
    with pytest.raises(ValueError):n.safe_path(path,Path('/tmp'))


def test_reader_change_between_reads_rejected():
    files={'/tmp/a.json':b'{}'};r=reader(files);r.read('/tmp/a.json');files['/tmp/a.json']=b'[]'
    with pytest.raises(ValueError,match='changed'):r.read('/tmp/a.json')


def test_reader_unique_bytes_and_caps(monkeypatch):
    files={'/tmp/a.json':b'{}'};r=reader(files);r.read('/tmp/a.json');r.read('/tmp/a.json')
    assert r.used==2
    monkeypatch.setattr(n,'TOTAL_CAP',1)
    with pytest.raises(ValueError,match='budget'):reader(files).read('/tmp/a.json')


def envelope(x):
    d=raw(x)
    return raw(dict(schema=n.v.BASES[0]+'.envelope',plan_sha256=sha(d[:-1]),gzip_base64=n.base64.b64encode(n.gzip.compress(d)).decode()))


def test_prepare_semantic_sha():
    data=envelope(dict(slots=[dict(root_seed=19,reset_seed=20)]))
    assert n.unpack_prepare(data).leaves[('slots',0,'root_seed')]==19
    x=json.loads(data);x['plan_sha256']='a'*64
    with pytest.raises(ValueError,match='semantic'):n.unpack_prepare(raw(x))


def test_prepare_gzip_tail():
    x=json.loads(envelope({}));x['gzip_base64']=n.base64.b64encode(n.gzip.compress(b'{}\n')+b'tail').decode()
    with pytest.raises(ValueError,match='gzip'):n.unpack_prepare(raw(x))


def events(items):
    previous=None;lines=[]
    for i,fields in enumerate(items):
        e=dict(index=i,previous_sha256=previous,**fields);previous=sha(raw(e)[:-1]);e['event_sha256']=previous;lines.append(raw(e))
    return b''.join(lines),previous


def fixture_events():
    items=[]
    for slot in (0,1):
        for step in range(11):
            if step:
                items.extend([dict(kind=k,resource='public_step',slot=slot,phase='replay0',step=step,
                    **(dict(raw_action=[.1,.2]) if k=='operation_attempted' else {})) for k in ('operation_attempted','operation_completed')])
            if step in (0,5,10):
                items.append(dict(kind='fresh_observation',slot=slot,phase='replay0',step=step,state=[slot,2,3,4,5,6,7],
                    snapshot=dict(state=[slot,2,3,4,5,6,7],block_velocity=[0,0],block_angle=0,block_angular_velocity=0),
                    png='s%d_%d.png'%(slot,step),file_sha256='a'*64,pixel_sha256='b'*64))
        items.append(dict(kind='future_outcome',slot=slot,phase='replay0',step=20,witness=dict(secret='not decoded')))
    return events(items)


def test_journal_projection_and_chain():
    rawj,tail=fixture_events();obs,a=n.journal_roots(rawj,tail)
    assert len(obs)==6 and len(a)==20 and all(x['completed'] for x in a.values())
    assert all(x[1] in (0,5,10) for x in obs)


@pytest.mark.parametrize('mutation',['bytes','tail','order','partial'])
def test_journal_corruptions_reject(mutation):
    data,tail=fixture_events()
    if mutation=='bytes':data=data.replace(b'not decoded',b'bad payload')
    elif mutation=='tail':tail='a'*64
    elif mutation=='order':data=b''.join(reversed(data.splitlines(keepends=True)))
    else:data=data.rsplit(b'\n',2)[0]+b'\n'
    with pytest.raises(ValueError):n.journal_roots(data,tail)


def test_physical_bridge_nonzero_raw_and_tasks():
    data,tail=fixture_events();p=proj(dict(journal_chain_tail=tail,rows=[dict(root_sha256='a'*64,blind_payload_sha256=str(i)*64,
        slot=dict(root_seed=10+i,reset_seed=20+i)) for i in (0,1)]),n.v.patterns('sourcepack')+[('journal_chain_tail',)])
    tasks=[('task0',proj(dict(blind_sha256='0'*64,task_id='t0',source_slot=0),n.v.patterns('task')))]
    b=n.physical_bridges(reader({'/tmp/journal.jsonl':data}),'s',Path('/tmp/sourcepack.json'),p,tasks)
    assert b[0]['metadata']['maps_to']==['s#/rows/0','task0#task']
    assert b[0]['metadata']['raw_history_sha256']==sha(n.np.asarray([[.1,.2]]*10,dtype='<f4').tobytes())
    assert b[0]['metadata']['root_seed']==10 and b[1]['metadata']['reset_seed']==21


def test_physical_history_binding_mismatch_rejected():
    data,tail=fixture_events();p=proj(dict(journal_chain_tail=tail,rows=[dict(history_frame_pixel_sha256=['c'*64]*3)]),n.v.patterns('sourcepack')+[('journal_chain_tail',)])
    with pytest.raises(ValueError,match='history binding'):
        n.physical_bridges(reader({'/tmp/journal.jsonl':data}),'s',Path('/tmp/sourcepack.json'),p,[])


def test_pointer_escaping():assert n.pointer(('a/b','c~d',0))=='/a~1b/c~0d/0'


def test_input_binding_links_are_not_physical_equivalence():
    entries=[dict(path='/tmp/a',sha256='a'*64,projection=[dict(pointer='/bindings/source',value='b'*64)]),
        dict(path='/tmp/b',sha256='b'*64,projection=[dict(pointer='/root_sha256',value='c'*64)])]
    n.link_captured_metadata(entries)
    assert entries[0]['input_lineage_links'][0]['target_path']=='/tmp/b'
    assert entries[0]['root_identity_projections']==[]
    assert 'not_physical' in entries[0]['input_lineage_links'][0]['relation']


def test_only_literal_bindings_projected_not_followed():
    p=proj(dict(bindings=dict(parent=dict(path='/outside/report',sha256='a'*64)),outcome=dict(root_seed=99)),n.META_PATTERNS)
    assert p.leaves=={('bindings','parent','path'):'/outside/report',('bindings','parent','sha256'):'a'*64}


def test_duplicate_metadata_key_rejected():
    with pytest.raises(ValueError,match='duplicate'):n.project(BytesIO(b'{"root_seed":1,"root_seed":2}'),n.META_PATTERNS)


def test_stable_root_goal_and_seed_projections_do_not_assign_pose():
    x=dict(path='/tmp/a',sha256='a'*64,projection=[dict(pointer='/roots/0',value='b'*64),
        dict(pointer='/protocol/root_seed',value=13)])
    p=n.stable_projections([x])
    assert p['root_goal_groups'][0]['stable_metadata_id']=='root_goal:'+'b'*64
    assert not p['root_goal_groups'][0]['physical_root_equivalence_proved']
    assert p['seed_projections'][0]['seed']==13


def test_no_directory_discovery_or_neural_imports():
    source=Path(n.__file__).read_text()
    for forbidden in ('glob(', 'iterdir(', 'os.walk(', 'import torch','import gym','resolve_history('):
        assert forbidden not in source


def test_output_reserve_without_write(monkeypatch):
    monkeypatch.setattr(n.shutil,'disk_usage',lambda p:type('Disk',(),dict(free=0))())
    with pytest.raises(ValueError,match='reserve'):n.write_new('/tmp/nonexistent_resolution_ram_test.json',{})
