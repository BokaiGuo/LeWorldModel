"""Only RAM fixtures; transport traces are not scientific result fixtures."""
import copy
import gzip
import base64
from pathlib import Path

import pytest
from experiments.spectral_reachability import agent_freshroots_resume_20261027_next as m
from experiments.spectral_reachability import agent_freshroots_resume_20261027_resolver as old


def event(kind,cid,line,**values):
    return dict(log='RAM',line=line,timestamp=str(line),line_sha256='line'+str(line),
                payload=dict(type=kind,call_id=cid,**values))


def call(cid,line,args):
    return event('function_call',cid,line,arguments=args)


def output(cid,line,*blocks):
    return event('custom_tool_call_output',cid,line,output=[
        dict(type='input_text',text='Script completed\nWall time 1\nOutput:\n')]+[
        dict(type='input_text',text=x) for x in blocks])


def cmd(sid='E'):
    directory=next(x[1] for x in old.SOURCES if x[0]==sid)
    return '.venv/bin/python -m experiments.spectral_reachability.preflight_pusht_d2p_p1_reconfiguration --output results/'+directory


def prior():
    return dict(recovery=dict(dispatches=[dict(call_id='start',source_id='E',line=1)],edits=[]),
                sources=[dict(id='E',sha256='report')])


def wrapper(**values):
    return m.encode(dict(chunk_id='fixture',**values)).decode()


@pytest.mark.parametrize('key',['cmd','"cmd"'])
def test_actual_JS_and_JSON_fields(key):
    a='text(await tools.exec_command({'+key+':'+m.encode(cmd()).decode()+'}));'
    assert m.operations(a,old)[0]['command']==cmd()


def test_operations_ignore_parentheses_inside_command_string():
    a='text(await tools.exec_command({cmd:"echo (not evaluated)"}));'
    assert m.operations(a,old)[0]['command']=='echo (not evaluated)'


def test_truncated_call_rejected():
    with pytest.raises(ValueError,match='truncated'):m.operations('tools.write_stdin({session_id:3}',old)


def test_complete_session_chain_clears_only_specific_atomic_request():
    args='text(await tools.exec_command({cmd:'+m.encode(cmd()).decode()+'}));'
    records=[call('start',1,args),output('start',2,wrapper(session_id=31,output='')),
        call('poll',3,'text(await tools.write_stdin({session_id:31,chars:""}));'),
        output('poll',4,wrapper(exit_code=0,output='future labels must not be parsed'))]
    j=m.join_records(records,prior(),old);s=m.summarize(prior(),j)
    assert j['open_sessions']==[] and len(s['atomic_requests_cleared'])==1
    assert s['sources'][0]['dispatch_status'][0]['exit_code']==0
    assert s['inherited_reset_blockers_after']==16 and not s['source_run_authorized']


def test_wrong_session_does_not_close_owner():
    a='tools.exec_command({cmd:'+m.encode(cmd()).decode()+'})'
    j=m.join_records([call('start',1,a),output('start',2,wrapper(session_id=31)),
        call('poll',3,'tools.write_stdin({session_id:99})'),output('poll',4,wrapper(exit_code=0))],prior(),old)
    assert len(j['dispatch_receipts'])==1 and len(j['open_sessions'])==1


def test_wrong_call_id_does_not_join_by_proximity():
    j=m.join_records([call('start',1,'tools.exec_command({cmd:'+m.encode(cmd()).decode()+'})'),
        output('other',2,wrapper(exit_code=0))],prior(),old)
    assert j['dispatch_receipts']==[]


def test_arity_mismatch_failclosed_not_assign_first_response():
    j=m.join_records([call('start',1,'tools.exec_command({cmd:'+m.encode(cmd()).decode()+'})'),
        output('start',2,wrapper(exit_code=0),wrapper(exit_code=0))],prior(),old)
    assert j['dispatch_receipts']==[] and len(j['unassigned'])==1


def test_failed_dispatch_retained_not_replaced_by_later_success():
    p=prior();p['recovery']['dispatches'].append(dict(call_id='second',source_id='E',line=3))
    a='tools.exec_command({cmd:'+m.encode(cmd()).decode()+'})'
    j=m.join_records([call('start',1,a),output('start',2,'ValueError: fixed failure\n'),
        call('second',3,a),output('second',4,wrapper(exit_code=0))],p,old)
    s=m.summarize(p,j)
    assert [r['state'] for r in s['sources'][0]['dispatch_status']]==[
        'traceback_failure_recorded','terminal_exit_recorded']
    assert len(s['atomic_requests_cleared'])==2


def test_empty_patch_response_not_bytes_attestation():
    p=prior();p['recovery']['edits']=[dict(call_id='edit')]
    j=m.join_records([call('edit',1,'tools.apply_patch("patch text")'),output('edit',2,'{}')],p,old)
    assert j['patch_receipts'][0]['empty_response']
    assert not j['patch_receipts'][0]['actual_source_bytes_attested']


def test_script_completed_not_process_exit():
    r=m.receipt('Script completed\nroot_count=4',old)
    assert r['exit_code'] is None


def test_transport_output_is_opaque_not_decoded_labels():
    r=m.receipt(wrapper(exit_code=0,output='{"native_success":true,"cost":12345}'),old)
    assert r['exit_code']==0 and set(r)=={'kind','exit_code','session_id','exceptions','producer_traceback_frames'}


def test_traceback_producer_frame_and_error_preserved():
    text='Traceback (most recent call last):\n  File "'+str(m.ROOT/old.P0)+'", line 293, in main\nValueError: wrong dims\n'
    r=m.receipt(text,old)
    assert r['exceptions']==['ValueError: wrong dims']
    assert r['producer_traceback_frames'][0]['line']==293


def test_unrelated_malformed_poll_not_parsed():
    j=m.join_records([call('unrelated',1,'tools.exec_command({cmd:"bad"broken}); tools.write_stdin({session_id:99})')],prior(),old)
    assert j['dispatch_receipts']==[]


@pytest.mark.parametrize('name,expected',[('report.json',True),('plan.json',True),('sourcepack.json',True),
    ('x_prepare01.json',True),('x_request01.json',True),('native_success.json',False),
    ('episode.png',False),(m.NAME+'_scope.json',False)])
def test_predeclared_metadata_filename_scope(name,expected):
    assert m.eligible(Path('/RAM')/name)==expected


def test_delta_ignores_physical_future_labels():
    raw=m.encode(dict(schema='fixture',native_success=True,outcomes=dict(root_seed=999),
        protocol=dict(root_seed=42),trajectory=[dict(state=[1]*7,goal=[2]*7)]))
    leaves,n=m.project_delta(raw,old)
    assert leaves=={('schema',):'fixture',('protocol','root_seed'):42} and n==0


def test_delta_links_require_exact_path_and_hash_pair():
    h='a'*64
    leaves={('bindings',0,'path'):'/A',('bindings',0,'sha256'):h,
            ('bindings',1,'path'):'/B',('bindings',1,'sha256'):'wrong'}
    r=m.delta_row(dict(path='/fixture',sha256='bytes'),leaves)
    assert r['exact_links']==[dict(path='/A',sha256=h)]
    assert r['metadata_role']=='bound_reference_only'


def test_missing_origin_fields_not_called_an_origin_blocker():
    r=m.delta_row(dict(path='/fixture',sha256='bytes'),{('schema',):'derived-analysis'})
    assert r['metadata_role']=='no_origin_fields_in_whitelist'
    assert r['physical_reset_coverage_certified'] is False


def test_declared_span_including_reserved_not_actual_attempt_count():
    r=m.delta_row(dict(path='/fixture',sha256='bytes'),{
        ('plan','slots',0,'reset_seed_span',0):10,('plan','slots',0,'reset_seed_span',1):20})
    assert r['declared_reset_spans']==[dict(pointer='/plan/slots/0/reset_seed_span',half_open=[10,20],usage='declared_not_observed')]


@pytest.mark.parametrize('bad_end',[10,9,True,None])
def test_invalid_span_not_silently_certified(bad_end):
    r=m.delta_row(dict(path='/fixture',sha256='bytes'),{
        ('slots',0,'reset_seed_span',0):10,('slots',0,'reset_seed_span',1):bad_end})
    assert r['declared_reset_spans']==[] and not r['physical_reset_coverage_certified']


def test_gzip_prepare_projected_with_exact_inner_pointer():
    raw=m.encode(dict(schema='fixture-envelope',gzip_base64=base64.b64encode(
        gzip.compress(m.encode(dict(schema='fixture-plan',slots=[dict(root_seed=42)])),mtime=0)).decode()))
    leaves,n=m.project_delta(raw,old)
    assert leaves[('decoded_envelope','slots',0,'root_seed')]==42 and n>0
    assert leaves[('schema',)]=='fixture-envelope'


@pytest.mark.parametrize('suffix',[b'garbage',gzip.compress(b'{}',mtime=0)])
def test_gzip_trailing_and_multimember_rejected(suffix):
    raw=m.encode(dict(gzip_base64=base64.b64encode(gzip.compress(b'{}',mtime=0)+suffix).decode()))
    with pytest.raises(ValueError,match='single-member'):m.project_delta(raw,old)


def test_inflate_cap_is_not_disk_cap(monkeypatch):
    monkeypatch.setattr(m,'INFLATE_CAP',3)
    raw=m.encode(dict(gzip_base64=base64.b64encode(gzip.compress(b'{"a":123}',mtime=0)).decode()))
    with pytest.raises(ValueError,match='bounded'):m.project_delta(raw,old)


def test_duplicate_key_rejected():
    with pytest.raises(ValueError,match='duplicate'):m.parse('{"x":1,"x":2}')


def test_fake_global_scope_rejected_before_file_reads():
    scope=dict(schema=m.NAME+'.scope.v1',previous_sha256=m.PREVIOUS_SHA,
               inventory_sha256=m.INVENTORY_SHA,complete_global_history=True,files=[])
    with pytest.raises(ValueError,match='global'):m.analyze_delta(scope,old)


def test_reordered_scope_rejected_before_file_reads():
    scope=dict(schema=m.NAME+'.scope.v1',previous_sha256=m.PREVIOUS_SHA,
               inventory_sha256=m.INVENTORY_SHA,complete_global_history=False,
               files=[dict(path='/B'),dict(path='/A')])
    with pytest.raises(ValueError,match='sorted'):m.analyze_delta(scope,old)


@pytest.mark.parametrize('key,value',[('previous_sha256','wrong'),('inventory_sha256','wrong'),('schema','other')])
def test_scope_authority_mutations(key,value):
    scope=dict(schema=m.NAME+'.scope.v1',previous_sha256=m.PREVIOUS_SHA,
               inventory_sha256=m.INVENTORY_SHA,complete_global_history=False,files=[])
    scope[key]=value
    with pytest.raises(ValueError,match='authority'):m.analyze_delta(scope,old)


def test_no_receipts_keeps_all_eight_original_reference_blockers():
    s=m.summarize(prior(),dict(dispatch_receipts=[]))
    assert not s['atomic_requests_cleared'] and s['inherited_reset_blockers_before']==s['inherited_reset_blockers_after']==16


def test_nonzero_exit_still_factual_terminal_not_scientific_rejection():
    a='tools.exec_command({cmd:'+m.encode(cmd()).decode()+'})'
    j=m.join_records([call('start',1,a),output('start',2,wrapper(exit_code=2))],prior(),old)
    s=m.summarize(prior(),j)
    assert s['sources'][0]['dispatch_status'][0]['exit_code']==2 and not s['P1_admitted']


def test_lossless_large_result_all_rows_preserved(monkeypatch):
    monkeypatch.setattr(m,'OUTPUT_CAP',10)
    result=dict(rows=[dict(path='/RAM/'+str(i),sha256='a'*64) for i in range(100)])
    raw=m.pack_result(result)
    assert m.unpack_result(raw)==result
    env=m.parse(raw);env['semantic_sha256']='b'*64
    with pytest.raises(ValueError,match='semantic binding'):m.unpack_result(m.encode(env))


def test_result_envelope_trailing_gzip_rejected(monkeypatch):
    monkeypatch.setattr(m,'OUTPUT_CAP',1)
    env=m.parse(m.pack_result(dict(rows=[1,2,3])))
    env['gzip_base64']=base64.b64encode(base64.b64decode(env['gzip_base64'])+b'x').decode()
    with pytest.raises(ValueError,match='single result gzip'):m.unpack_result(m.encode(env))


def test_result_semantic_cap_not_silently_truncated(monkeypatch):
    monkeypatch.setattr(m,'SEMANTIC_CAP',3)
    with pytest.raises(ValueError,match='semantic result cap'):m.pack_result(dict(rows=[1]))


def test_large_authority_closure_preserves_all_bindings():
    leaves={('bindings',i,'path'):'/RAM/'+str(i) for i in range(9000)}
    leaves.update({('bindings',i,'sha256'):'a'*64 for i in range(9000)})
    r=m.delta_row(dict(path='/fixture',sha256='bytes'),leaves)
    assert len(r['exact_links'])==9000 and not r['physical_reset_coverage_certified']
