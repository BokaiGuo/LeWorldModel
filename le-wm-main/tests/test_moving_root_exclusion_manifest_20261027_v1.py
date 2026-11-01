"""RAM files only; no tmp directories, outcomes, models, simulator or census."""
import base64
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import moving_root_exclusion_manifest_20261027_v1 as m


def h(v):return hashlib.sha256(v).hexdigest()
def ident(i=0,kind='observed'):
    return dict(record_id='root'+str(i),record_type=kind,root_seed=900+i,reset_seed=1000+i,
        root_goal_sha256=h(('goal'+str(i)).encode()),root_state=[100.+i,200.,240.,250.,.5,0.,0.],
        root_snapshot_sha256=h(('snap'+str(i)).encode()),history_pixels_sha256=[h(f'{i}frame{j}'.encode()) for j in range(3)],
        raw_history_sha256=h(('raw'+str(i)).encode()),task_id=h(('task'+str(i)).encode()),episode_id='ep'+str(i),parent_refs=[],maps_to=[])


class Inputs:
    def __init__(self):
        self.files={str(m.PROTOCOL):m.PROTOCOL.read_bytes()};self.reads=[]
        self.q=dict(schema=m.NAME+'.request',cutoff_utc='2026-10-27T00:00:00Z',project_root=str(m.ROOT),
            protocol=dict(path=str(m.PROTOCOL),sha256=h(self.files[str(m.PROTOCOL)])),evidence_files=[],
            expected_source_families=list(m.REQUIRED_FAMILIES),declared_unresolved_sources=[],approval_file=None)
        for i,fam in enumerate(m.REQUIRED_FAMILIES):
            r=ident(i,'observed' if fam=='moving' else 'reservation')
            self.add(fam,fam,dict(schema=m.NAME+'.identity_metadata',records=[r]))
    def add(self,sid,family,value,adapter='metadata',format='json',raw=None):
        raw=m.canonical(value) if raw is None else raw
        if format=='gzip_json':raw=gzip.compress(raw,mtime=0)
        path=str(m.ROOT/'results'/('RAM_'+sid+'.json'))
        self.files[path]=raw
        e=dict(source_id=sid,family=family,path=path,sha256=h(raw),format=format,adapter=adapter)
        self.q['evidence_files'].append(e);return e
    def read(self,p):
        assert p in self.files,'attempt to read unlisted path'
        self.reads.append(p);return self.files[p]
    def run(self):return m.build(self.q,read_bytes=self.read)
    def approve(self):
        r=self.run();a=dict(schema=m.NAME+'.MAIN_coverage_approval',authority='MAIN',approved=True,
            coverage_scope=list(m.REQUIRED_FAMILIES),cutoff_utc=self.q['cutoff_utc'],manifest_content_sha256=r['coverage_content_sha256'])
        path=str(m.ROOT/'results/RAM_MAIN_approval.json');raw=m.canonical(a);self.files[path]=raw
        self.q['approval_file']=dict(path=path,sha256=h(raw));return a


def test_no_self_approval_no_model_no_outcomes_or_independence():
    f=Inputs();r=f.run()
    assert r['unresolved_sources']==[] and not r['approval']['approved']
    assert not r['source_run_preconditions_met'] and not r['freshness_certified'] and not r['P1_admission']
    assert len(r['proposed_seed_checks'])==4 and all(not x['independence_certified'] for x in r['proposed_seed_checks'])
    assert r['semantics']['outcome_values_decoded']==0
    assert set(f.reads)==set(f.files)


def test_external_exact_MAIN_approval_only_covers_inventory_not_new_roots():
    f=Inputs();f.approve();r=f.run()
    assert r['approval']['approved'] and r['source_run_preconditions_met']
    assert not r['freshness_certified'] and 'pending_new_root' in r['status']


@pytest.mark.parametrize('change',['content','scope','authority','approved','sha'])
def test_external_approval_tamper_blocks(change):
    f=Inputs();a=f.approve();desc=f.q['approval_file']
    if change=='content':a['manifest_content_sha256']='a'*64
    elif change=='scope':a['coverage_scope']=['moving']
    elif change=='authority':a['authority']='SELF'
    elif change=='approved':a['approved']=False
    else:desc['sha256']='a'*64
    if change!='sha':
        raw=m.canonical(a);f.files[desc['path']]=raw;desc['sha256']=h(raw)
    r=f.run();assert not r['source_run_preconditions_met'] and r['approval_errors']


def test_stable_under_inventory_order_and_repeat_does_not_inflate_roots():
    f=Inputs();a=f.run();f.q['evidence_files'].reverse();b=f.run()
    assert a['coverage_content_sha256']==b['coverage_content_sha256'] and a['stable_roots']==b['stable_roots']
    r=ident(3);r['record_id']='copy';r['task_id']=h(b'new-task-label');r['root_goal_sha256']=h(b'new-goal-label')
    f.add('overlay_copy','overlay',dict(schema=m.NAME+'.identity_metadata',records=[r]))
    c=f.run();assert len(c['observed_roots'])==2 and len(c['stable_roots'])==1 and c['aliases']


def test_state_signed_zero_angle_velocity_goal_do_not_create_fresh_pose():
    a=[0.,200.,240.,250.,.5,0.,0.];b=list(a);b[0]=-0.;b[4]+=2*np.pi;b[5]=10.
    ai=m.state_identity(a);bi=m.state_identity(b)
    assert ai['root_state_sha256']!=bi['root_state_sha256'] and m.same_pose(ai['physical_pose'],bi['physical_pose'])
    assert m.state_identity(a)['root_state_sha256']==h(np.asarray(a,dtype='<f8').tobytes())
    assert m.same_pose([1.,2.,3.,4.,0.],[1.+.5e-9,2.,3.,4.,.5e-12])
    assert not m.same_pose([1.,2.,3.,4.,0.],[1.+2e-9,2.,3.,4.,0.])


@pytest.mark.parametrize('field',['root_state','root_snapshot_sha256','history_pixels_sha256','raw_history_sha256','root_seed','reset_seed'])
def test_every_missing_identity_blocks_even_external_approval(field):
    f=Inputs();r=ident(20);r.pop(field)
    f.add('partial','feedback',dict(schema=m.NAME+'.identity_metadata',records=[r]));f.approve();out=f.run()
    assert out['approval']['approved'] and out['unresolved_sources'] and not out['source_run_preconditions_met']
    assert any(x['source_id']=='partial#root20' for x in out['unresolved_sources'])


def test_failed_unselected_source_rows_never_filtered_by_status_or_cost():
    f=Inputs();rows=[dict(slot=dict(index=i,root_seed=10000+i,reset_seed=20000+i),status=s,reachable=False,
        root_sha256=h(str(i).encode()),cost=-1e9,witness_commands='DO_NOT_DECODE') for i,s in enumerate(['failed','not_started','unselected'])]
    f.add('source','constructive_zero',dict(schema=m.BASES[0]+'.sourcepack',rows=rows),'sourcepack')
    out=f.run();assert len([r for r in out['identity_records'] if r['source_id']=='source'])==3
    assert all(s in out['used_or_reserved_root_seeds'] for s in (10000,10001,10002))
    assert len([x for x in out['unresolved_sources'] if x['source_id'].startswith('source#')])==3


def test_no_heldout_values_decoded_or_embedded_binding_followed(monkeypatch):
    f=Inputs();r=ident(22)
    item=dict(schema=m.NAME+'.identity_metadata',records=[r],heldout_outcomes={'success':'SECRET_FUTURE',
        'witness':['SECRET_FUTURE']},bindings={'/forbidden/outside.json':'a'*64})
    f.add('opaque','overlay',item)
    loads=m.json.loads
    def guarded(raw,*a,**kw):
        assert 'SECRET_FUTURE' not in str(raw),'outcome value decoded'
        return loads(raw,*a,**kw)
    monkeypatch.setattr(m.json,'loads',guarded)
    out=f.run();assert not out['unresolved_sources'] and '/forbidden/outside.json' not in f.reads


@pytest.mark.parametrize('format',['json','gzip_json','prepare_envelope'])
def test_prepare_formats_preserve_all_seed_pairs_and_reservations(format):
    f=Inputs();p=dict(schema=m.BASES[1]+'.prepare',slots=[dict(root_seed=203027101,reset_seed=203027202),
        dict(root_seed=22,reset_seed_span=[203027203,203027204])],exclusions=dict(root_hashes=['b'*64],seed_families=[dict(seed=33)],
        reset_spans=[dict(span=[40,45])],unknown_sources=[dict(source='old',reason='historical_attempt_bound_missing')]))
    raw=None
    if format=='prepare_envelope':
        inner=(json.dumps(p,sort_keys=True,allow_nan=False)+'\n').encode()
        raw=m.canonical(dict(schema=m.BASES[1]+'.envelope',plan_sha256=h(inner[:-1]),gzip_base64=base64.b64encode(gzip.compress(inner,mtime=0)).decode()))
    f.add('planned','reserved_prepares',p,'prepare',format,raw)
    out=f.run();assert {22,33,203027101}<=set(out['used_or_reserved_root_seeds'])
    assert out['proposed_seed_checks'][0]['conflicts'] and out['proposed_seed_checks'][1]['conflicts']
    assert out['proposed_seed_checks'][2]['conflicts'] and not out['proposed_seed_checks'][3]['conflicts']
    assert 'b'*64 in out['root_goal_sha256s'] and [40,45] in out['reserved_reset_spans']
    assert any('inherited_unknown' in x['reason'] for x in out['unresolved_sources'])


def test_task_explicit_metadata_bridge_and_parent_refs():
    f=Inputs();r=ident(32)
    f.add('task','moving',dict(schema='p1_constructive_moving_history_encode_cuda_20261027_v1.task_export',
        source_slot=0,task_id=r['task_id'],blind_sha256='a'*64,task={'goal':'SECRET_FUTURE'}),'task')
    before=f.run();assert any(x['source_id']=='task#task' for x in before['unresolved_sources'])
    r['maps_to']=['task#task'];r['parent_refs']=['moving#root3']
    f.add('bridge','moving',dict(schema=m.NAME+'.identity_metadata',records=[r]))
    out=f.run();assert not out['unresolved_sources']
    assert next(r for r in out['observed_roots'] if r['record_ref']=='task#task')['mapping_evidence']=='bridge#root32'


@pytest.mark.parametrize('bad',['missing_target','missing_parent','conflicting_task','ambiguous'])
def test_mapping_unknown_or_ambiguous_blocks(bad):
    f=Inputs();r=ident(33);r['maps_to']=['absent#root']
    if bad=='missing_parent':r['maps_to']=[];r['parent_refs']=['absent#root']
    if bad in ('conflicting_task','ambiguous'):
        r['maps_to']=['moving#root3']
        if bad=='conflicting_task':r['episode_id']='ep3'  # one key matches, other identities conflict
        else:
            r=ident(3);r['record_id']='other';r['maps_to']=['moving#root3']
            duplicate=deepcopy(r);duplicate['record_id']='third'
            f.add('bridge2','moving',dict(schema=m.NAME+'.identity_metadata',records=[duplicate]))
    f.add('bridge','moving',dict(schema=m.NAME+'.identity_metadata',records=[r]))
    assert f.run()['unresolved_sources']


def test_legacy_unknown_cannot_be_self_upgraded():
    f=Inputs();f.add('index','legacy_zero',dict(schema=m.lexical.NAME+'.authority_evidence',root_hashes=['c'*64],
        seed_families=[dict(seed=8)],reset_spans=[dict(span=[9,15])],confirmation_authority=dict(status='complete'),unknown_sources=[]),'legacy_index')
    r=f.run();assert 'c'*64 in r['root_goal_sha256s'] and 8 in r['used_or_reserved_root_seeds']
    assert any('legacy_index_coverage_unknown' in x['reason'] for x in r['unresolved_sources'])


@pytest.mark.parametrize('bad',['sha','schema','malformed','gzip','envelope'])
def test_evidence_failure_is_visible_not_silently_dropped(bad):
    f=Inputs();e=f.q['evidence_files'][0]
    if bad=='sha':e['sha256']='f'*64
    else:
        raw=b'{"schema": "unknown"}' if bad=='schema' else b'{' if bad=='malformed' else b'not-gzip'
        if bad=='gzip':e['format']='gzip_json'
        if bad=='envelope':e['format']='prepare_envelope';raw=m.canonical(dict(schema=m.BASES[1]+'.envelope',plan_sha256='a'*64,gzip_base64='@@@@'))
        f.files[e['path']]=raw;e['sha256']=h(raw)
    r=f.run();assert any(x['source_id']==e['source_id'] for x in r['unresolved_sources'])
    assert len(r['evidence_files'])==len(f.q['evidence_files'])


def test_missing_family_declared_unknown_and_half_open_span():
    f=Inputs();f.q['evidence_files']=f.q['evidence_files'][1:];f.q['declared_unresolved_sources']=['undocumented old reset ledger']
    r=f.run();assert len(r['unresolved_sources'])==2 and not r['source_run_preconditions_met']


@pytest.mark.parametrize('bad',['duplicate_path','duplicate_id','outside','glob','extra','heldout_adapter'])
def test_invalid_request_rejected_before_file_reads(bad):
    f=Inputs();e=f.q['evidence_files'][0]
    if bad=='duplicate_path':f.q['evidence_files'][1]['path']=e['path']
    elif bad=='duplicate_id':f.q['evidence_files'][1]['source_id']=e['source_id']
    elif bad=='outside':e['path']='/etc/passwd'
    elif bad=='glob':e['path']=str(m.ROOT/'results/*.json')
    elif bad=='extra':f.q['allow_outcomes']=True
    else:e['adapter']='heldout_outcomes'
    with pytest.raises(ValueError):f.run()
    assert f.reads==[]


def test_duplicate_metadata_keys_and_changed_state_hash_fail():
    f=Inputs();r=ident(35);r['root_state_sha256']='a'*64
    f.add('badstate','feedback',dict(schema=m.NAME+'.identity_metadata',records=[r]))
    assert any(x['source_id']=='badstate' for x in f.run()['unresolved_sources'])
    with pytest.raises(ValueError):m.project(BytesIO(b'{"schema":"x","schema":"y"}'),'task')


def test_same_root_goal_hash_cannot_hide_conflicting_physical_identity():
    f=Inputs();r=ident(3);r['record_id']='inconsistent';r['root_state'][0]+=100
    f.add('inconsistent','feedback',dict(schema=m.NAME+'.identity_metadata',records=[r]))
    out=f.run();assert any('hash_conflicts_with_physical_pose' in x['reason'] for x in out['unresolved_sources'])


def test_numeric_alias_with_changed_goal_and_velocity_is_conservatively_one_root():
    f=Inputs();r=ident(3);r['record_id']='alias';r['root_state'][0]+=.5e-9;r['root_state'][4]+=2*np.pi;r['root_state'][5]=123.
    r['root_goal_sha256']=h(b'differentgoal');r['root_snapshot_sha256']=h(b'differentsnapshot');r['history_pixels_sha256']=[h(f'other{i}'.encode()) for i in range(3)]
    f.add('alias','overlay',dict(schema=m.NAME+'.identity_metadata',records=[r]))
    out=f.run();assert len(out['stable_roots'])==1
    assert 'physical_pose_numeric_alias' in out['aliases'][0]['reasons']


def test_naive_UTC_cutoff_rejected_before_reads():
    f=Inputs();f.q['cutoff_utc']='2026-10-27T00:00:00'
    with pytest.raises(ValueError):f.run()
    assert not f.reads


def test_blind_identity_retained_without_reading_npz():
    f=Inputs();r=ident(44);r['blind_sha256']='a'*64
    f.add('blindtask','moving',dict(schema='p1_constructive_moving_history_encode_cuda_20261027_v1.task_export',
        task_id=r['task_id'],blind_sha256='a'*64),'task')
    r['maps_to']=['blindtask#task'];f.add('blindbridge','moving',dict(schema=m.NAME+'.identity_metadata',records=[r]))
    out=f.run();assert not out['unresolved_sources']
    assert next(x for x in out['identity_records'] if x['source_id']=='blindtask')['blind_sha256']=='a'*64


def test_global_metadata_budget_stops_later_files_without_opening(monkeypatch):
    f=Inputs();monkeypatch.setattr(m,'TOTAL_CAP',1)
    out=f.run();assert not out['source_run_preconditions_met']
    assert len(out['evidence_files'])==7 and len(set(f.reads))==2
    assert any('budget_exhausted_not_opened' in x['reason'] for x in out['unresolved_sources'])


@pytest.mark.parametrize('failure',[None,'floor','fsync'])
def test_NEW_durable_writer_in_RAM_only(monkeypatch,failure):
    written=[];calls=[]
    class Stream:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def write(self,raw):written.append(raw);return len(raw)
        def flush(self):calls.append('flush')
        def fileno(self):return 7
    class P:
        parent='RAM_PARENT'
        def is_absolute(self):return True
        def exists(self):return False
        def is_symlink(self):return False
        def open(self,mode):assert mode=='xb';return Stream()
        def __str__(self):return '/RAM/new.json'
    monkeypatch.setattr(m,'Path',lambda p:P())
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:type('Space',(),dict(free=m.FLOOR-1 if failure=='floor' else m.FLOOR+m.OUTPUT_CAP))())
    monkeypatch.setattr(m,'file_sha',lambda p:h(written[0]))
    monkeypatch.setattr(m.os,'open',lambda *a:8)
    monkeypatch.setattr(m.os,'close',lambda fd:calls.append('close'))
    def fsync(fd):
        calls.append('fsync')
        if failure=='fsync':raise OSError('durable fsync failed; preserve partial')
    monkeypatch.setattr(m.os,'fsync',fsync)
    result=dict(status='blocked')
    if failure:
        with pytest.raises((ValueError,OSError)):m.write_new('/RAM/new.json',result)
        assert bool(written)==(failure=='fsync')
    else:
        receipt=m.write_new('/RAM/new.json',result)
        assert receipt['sha256']==h(m.canonical(result)) and calls==['flush','fsync','fsync','close']


def test_imports_and_cli_have_no_census_neural_or_old_resolver_calls():
    source=Path(m.__file__).read_text()
    assert 'resolve_history(' not in source and '.glob(' not in source and '.iterdir(' not in source
    assert 'import torch' not in source and 'load_task(' not in source and 'stable_worldmodel' not in source
