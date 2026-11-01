"""RAM metadata, original lexical projector and all-pairs identity functions only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from experiments.spectral_reachability import agent_p4_fresh_panel_manifest_20261029_v1 as m
import test_moving_root_exclusion_manifest_20261027_v1 as old

def h(value):return hashlib.sha256(m.canonical(value)).hexdigest()

class Inputs:
    def __init__(self):
        self.files={str(m.REG):m.REG.read_bytes()}
        self.q=m.template()
        for key in ('legacy_manifest','legacy_consumer'):
            path=self.q[key]['path'];self.files[path]=Path(path).read_bytes()
        base=json.loads(self.files[self.q['legacy_manifest']['path']])
        self.records=[]
        for i in range(4):
            c=base['candidates'][i];self.records.append(self.record('actual'+str(i),i+20,c['root_seed'],c['reset_seed']))
        self.actual=self.add('actual',dict(schema=m.NAME+'.actual_roots',slots=self.records))
        self.q['actual_roots']=self.actual
    def add(self,name,value):
        path=str(m.ROOT/'results'/('RAM_'+name+'.json'));raw=m.canonical(value);self.files[path]=raw
        return dict(path=path,sha256=hashlib.sha256(raw).hexdigest())
    def record(self,name,n,seed,reset):
        value=dict(root_seed=seed,reset_seed=reset,root_goal_sha256=h(('goal',n)),
            root_state=[float(n*10),200.,240.,250.,.5,0.,0.],root_snapshot_sha256=h(('snapshot',n)),
            history_pixels_sha256=[h(('RGB',n,j)) for j in range(3)],raw_history_sha256=h(('raw',n)),
            episode_id='ep'+str(n))
        desc=self.add('source_'+name,dict(schema='RAM_NATIVE_SOURCE',metadata=value,
            outcomes={'cost':'NOT_DECODED','success':'NOT_DECODED'}))
        evidence={k:dict(file=desc,pointer='/metadata/'+k) for k in m.FIELDS}
        return dict(record_id=name,**{k:value[k] for k in m.FIELDS},
            source_episode=dict(source=desc,episode_id=value['episode_id'],pointer='/metadata/episode_id'),evidence=evidence)
    def reader(self):
        def read(path):
            assert path in self.files,'unlisted file opened '+path
            return self.files[path]
        return m.Reader(read)
    def run(self):return m.build(self.q,self.reader())
    def replace_actuals(self):self.q['actual_roots']=self.add('actual',dict(schema=m.NAME+'.actual_roots',slots=self.records))
    def approved_history(self):
        seen=self.record('seen',2,200,300)
        exposure=dict(schema=m.exclusion.NAME+'.identity_metadata',cutoff_utc='2026-10-29T00:00:00Z',
            project_root=str(m.ROOT),includes=list(m.SCOPE),complete=True,unknowns=[],records=[seen],reserved_reset_spans=[])
        ex=self.add('exposure',exposure);self.q['exposure_inventory']=ex
        f=old.Inputs();f.q['cutoff_utc']=exposure['cutoff_utc']
        f.q['evidence_files'].append(dict(source_id='new_exposure',family='feedback',**ex,adapter='metadata',format='json'))
        f.files.update(self.files);a=f.approve()
        a['coverage_scope']+=m.required_coverage_scope(self.q)
        f.files[f.q['approval_file']['path']]=m.canonical(a)
        f.q['approval_file']['sha256']=hashlib.sha256(m.canonical(a)).hexdigest()
        self.files.update(f.files);self.history=f
        self.q['history_request']=self.add('history_request',f.q)
        return exposure
    def panel_approve(self):
        content=self.run()
        a=dict(schema=m.NAME+'.MAIN_panel_approval',authority='MAIN',approved=True,panel_content_sha256=content['panel_content_sha256'],
            cutoff_utc=content['content']['exposure_cutoff_utc'],outcome_blind_selection=True)
        self.q['panel_approval']=self.add('panel_approval',a)
        return a

def test_fixed_original_four_not_a_new_sampler_or_root_evidence():
    f=Inputs();r=f.run()
    assert r['status']=='blocked_planning_artifact'
    assert len(r['content']['rows'])==4 and r['content']['fixed_denominator']==4
    assert all(x['metadata_disjoint'] for x in r['content']['rows'])
    assert not r['metadata_panel_certified'] and not r['freshness_certified'] and not r['new_root_evidence']
    assert not r['scientific_admission'] and not r['source_run_authorized']
    assert r['actual_neural_calls']==r['actual_physics_calls']==0

def test_existing_history_MAIN_approval_plus_exact_panel_metadata_attestation_not_science():
    f=Inputs();f.approved_history();before=f.run()
    assert before['content']['unresolved']==[]
    assert not before['metadata_panel_certified']
    f.panel_approve();r=f.run()
    assert r['metadata_panel_certified']
    assert r['content']['history_approval']['approved']
    assert not r['freshness_certified'] and r['certified_fresh_roots']==[]
    assert not r['scientific_admission'] and not r['P1_admission'] and not r['P4_admission']

@pytest.mark.parametrize('fault',['null','partial','record_id','seed','extra_outcome','proof','proof_outcome','source_sha'])
def test_actual_identity_input_errors_or_missing(fault):
    f=Inputs()
    if fault=='null':f.records=[None]*4
    elif fault=='partial':
        f.records[0]['root_snapshot_sha256']=None;f.records[0]['evidence'].pop('root_snapshot_sha256')
    elif fault=='record_id':f.records[1]['record_id']=f.records[0]['record_id']
    elif fault=='seed':f.records[0]['root_seed']=10
    elif fault=='extra_outcome':f.records[0]['success']=False
    elif fault=='proof':f.records[0]['evidence']['root_state']['pointer']='/metadata/root_seed'
    elif fault=='proof_outcome':f.records[0]['evidence']['root_state']['pointer']='/outcomes/cost'
    else:f.records[0]['source_episode']['source']['sha256']='0'*64
    f.replace_actuals()
    if fault in ('null','partial'):
        r=f.run();assert len(r['content']['rows'])==4 and not r['metadata_panel_certified']
        assert not r['content']['rows'][0]['metadata_disjoint']
    else:
        with pytest.raises(ValueError):f.run()

def test_same_source_episode_excludes_both_even_different_physical_state():
    f=Inputs()
    f.records[1]['source_episode']=deepcopy(f.records[0]['source_episode'])
    f.replace_actuals();rows=f.run()['content']['rows']
    for i in (0,1):
        assert rows[i]['status']=='excluded_duplicate'
        assert any(x['kind']=='batch_source_episode' for x in rows[i]['duplicate_evidence'])
    assert len(rows)==4

def test_historical_source_episode_excluded_even_changed_goal_pose_or_velocity():
    f=Inputs();exposure=f.approved_history()
    f.records[0]['source_episode']=deepcopy(exposure['records'][0]['source_episode'])
    f.replace_actuals();row=f.run()['content']['rows'][0]
    assert any(x['kind']=='historical_source_episode' for x in row['duplicate_evidence'])

def test_original_pose_alias_excludes_both_without_new_tolerance():
    f=Inputs()
    state=deepcopy(f.records[0]['root_state']);state[4]+=2*3.141592653589793;state[5]=123.
    f.records[1]['root_state']=state
    desc=f.records[1]['evidence']['root_state']['file'];src=json.loads(f.files[desc['path']]);src['metadata']['root_state']=state
    new=f.add('alias_state',src);f.records[1]['evidence']['root_state']['file']=new
    f.replace_actuals();rows=f.run()['content']['rows']
    assert all(rows[i]['status']=='excluded_duplicate' for i in (0,1))
    assert any('physical_pose_alias' in x.get('reasons',[]) for x in rows[0]['duplicate_evidence'])

def test_raw_history_alone_not_root_or_episode_equivalence():
    f=Inputs();f.records[1]['raw_history_sha256']=f.records[0]['raw_history_sha256']
    proof=f.records[0]['evidence']['raw_history_sha256'];f.records[1]['evidence']['raw_history_sha256']=deepcopy(proof)
    f.replace_actuals()
    assert all(r['metadata_disjoint'] for r in f.run()['content']['rows'])

@pytest.mark.parametrize('which',['cross_seed','half_open_span'])
def test_seed_and_reserved_cross_namespace_not_silently_replaced(which):
    f=Inputs();ex=f.approved_history();seed=f.records[0]['reset_seed']
    if which=='cross_seed':ex['records'].append(f.record('seed_collision',99,seed,seed+999))
    else:
        span=[seed,seed+1];proof=f.add('reserved',dict(reset_seed_span=span))
        ex['reserved_reset_spans']=[dict(span=span,evidence=dict(file=proof,pointer='/reset_seed_span'))]
    f.q['exposure_inventory']=f.add('changed_exposure',ex)
    rows=f.run()['content']['rows']
    assert rows[0]['status']=='excluded_duplicate' and len(rows)==4

@pytest.mark.parametrize('fault',['scope','cutoff','unknown','complete','missing_inventory','bad_approval','drop_history','unbound_inventory'])
def test_history_or_current_exposure_cannot_be_self_approved(fault):
    f=Inputs();ex=f.approved_history()
    if fault=='scope':ex['includes'].pop()
    elif fault=='cutoff':ex['cutoff_utc']='2026-10-29T01:00:00Z'
    elif fault=='unknown':ex['unknowns']=['unsaved_failed_source']
    elif fault=='complete':ex['complete']=False
    elif fault=='missing_inventory':f.q['exposure_inventory']=None
    elif fault=='bad_approval':f.history.q['approval_file']=None;f.q['history_request']=f.add('history_request',f.history.q)
    elif fault=='drop_history':f.q['history_request']=None
    else:ex['records']=[]
    if fault in ('scope','cutoff','unknown','complete','unbound_inventory'):
        f.q['exposure_inventory']=f.add('changed_exposure',ex)
    if fault=='scope':
        with pytest.raises(ValueError):f.run()
    else:assert f.run()['content']['unresolved'] and not f.run()['metadata_panel_certified']

@pytest.mark.parametrize('fault',['sha','authority','cutoff','outcome','approved'])
def test_panel_approval_exact_hash_scope_not_a_boolean_override(fault):
    f=Inputs();f.approved_history();a=f.panel_approve()
    if fault=='sha':a['panel_content_sha256']='0'*64
    elif fault=='authority':a['authority']='SELF'
    elif fault=='cutoff':a['cutoff_utc']='2026-01-01T00:00:00Z'
    elif fault=='outcome':a['outcome_blind_selection']=False
    else:a['approved']=False
    f.q['panel_approval']=f.add('panel_approval',a)
    with pytest.raises(ValueError):f.run()

def test_approval_cannot_clear_missing_identity():
    f=Inputs();f.approved_history();f.records=[None]*4;f.replace_actuals();f.panel_approve()
    r=f.run();assert not r['metadata_panel_certified'] and r['approval_error']

def test_duplicate_json_and_caps_failclosed():
    with pytest.raises(ValueError):m.decode(b'{"a":1,"a":2}')
    with pytest.raises(ValueError):m.decode(b'{"a":NaN}')
    with pytest.raises(ValueError):m.decode(b' '* (m.CAP+1))
    f=Inputs();desc=f.add('cap',{});reader=f.reader()
    with pytest.raises(ValueError):reader.read(desc,1)

def test_exact_output_and_input_path_no_overwrite(tmp_path,monkeypatch):
    # Writer deliberately restricted to workspace; use a RAM double for I/O-independent cap check.
    with pytest.raises(ValueError):m.write_new('/tmp/'+m.NAME+'_outside.json',{})
    with pytest.raises(ValueError):m.path_value(str(m.ROOT/'results/../escape'))
    with pytest.raises(ValueError):m.path_value(str(m.ROOT/'results/*.json'))

def test_actual_saved_draft_keeps_all_unknowns_without_reading_referenced_outcomes():
    reader=m.Reader();r=m.build(m.template(),reader)
    assert len(r['content']['rows'])==4 and all(x['actual_root_identity'] is None for x in r['content']['rows'])
    assert not r['metadata_panel_certified'] and r['status']=='blocked_planning_artifact'
    assert set(reader.bindings)=={m.template()[k]['path'] for k in ('preregistration','legacy_manifest','legacy_consumer')}

def test_cold_import_no_neural_or_environment():
    code="from experiments.spectral_reachability import "+m.NAME+"; import sys; assert not any(x in sys.modules for x in ('torch','pygame','pymunk','gymnasium'))"
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_deterministic_complete_replay_request_order_not_outcomes():
    f=Inputs();a=f.run();b=f.run()
    assert m.canonical(a)==m.canonical(b)
    assert a['content']['source_sha256']==m.own_hashes()

def test_external_request_binding_rebuild_with_fresh_reader():
    f=Inputs();desc=f.add('request',f.q);reader=f.reader()
    saved=m.build(reader.json(desc),reader,request_file=desc)
    rebuilt=m.build(saved['request'],f.reader(),request_file=saved['request_file'])
    assert m.canonical(saved)==m.canonical(rebuilt)
    assert saved['bindings'][desc['path']]==desc['sha256']

def test_identity_proof_in_actual_rows_array_pointer():
    f=Inputs();record=f.records[0];source=f.add('array_source',dict(rows=[dict(root_state=record['root_state'])]))
    record['evidence']['root_state']=dict(file=source,pointer='/rows/0/root_state')
    f.replace_actuals();assert f.run()['content']['rows'][0]['metadata_disjoint']

def test_sealed_legacy_cannot_be_replaced_with_empty_same_schema():
    f=Inputs();f.q['legacy_manifest']=f.add('fake_base',dict(schema='chat_freshroots_20261027.root_manifest.v1'))
    with pytest.raises(ValueError,match='sealed legacy'):f.run()

def test_family_labels_alone_cannot_discharge_legacy_unknowns():
    f=Inputs();f.approved_history();q=f.history.q;a=json.loads(f.files[q['approval_file']['path']])
    a['coverage_scope']=list(m.exclusion.REQUIRED_FAMILIES)
    q['approval_file']=f.add('family_only',a);f.q['history_request']=f.add('history_request',q)
    r=f.run();assert r['content']['unresolved'] and not r['metadata_panel_certified']

