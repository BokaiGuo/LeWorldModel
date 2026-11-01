"""RAM-only mutation fixtures. No neural/physics or historical result fabrication."""
import copy
from io import BytesIO
import json

import pytest
from experiments.spectral_reachability import agent_freshroots_resume_20261027_resolver as m


def record(arguments, line=1):
    return dict(log='RAM.jsonl', line=line, timestamp=f'2026-10-24T10:00:{line:02d}Z',
                call_id=f'call{line}', arguments=arguments)


def add(source='SCHEMA = "v2"\n', path=m.P1):
    return '*** Begin Patch\n*** Add File: '+path+'\n'+''.join('+'+s+'\n' for s in source.splitlines())+'*** End Patch'


def command(sid='F'):
    directory = next(s[1] for s in m.SOURCES if s[0] == sid)
    return '.venv/bin/python -m experiments.spectral_reachability.preflight_pusht_d2p_p1_reconfiguration --output results/'+directory+' --roots 4 --seed 202710251'


def registration():
    return dict(candidate_order=[dict(slot=i,root_seed=203027101+i,reset_seed=203027201+i,
        witness_sign=1 if i % 2 == 0 else -1) for i in range(4)], sampling_denominator=4,
        attempt_per_slot=1,proposal_per_slot=1,replacement_or_retry=False)


def test_replay_recovers_old_version_not_later_version():
    update = '*** Begin Patch\n*** Update File: '+m.P1+'\n@@\n-SCHEMA = "v2"\n+SCHEMA = "v3"\n*** End Patch'
    x = m.replay([record(add()), record(json.dumps(dict(cmd=command())),2),
                  record(update,3), record(json.dumps(dict(cmd=command())),4)])
    assert [d['recovered_producer_metadata']['schema'] for d in x['dispatches']] == ['v2','v3']
    assert x['dispatches'][0]['recovered_producer_sha256'] != x['dispatches'][1]['recovered_producer_sha256']
    assert all(not d['execution_source_attested'] for d in x['dispatches'])


def test_repeated_failed_intents_not_deduplicated_by_directory():
    x = m.replay([record(add()),*[record(json.dumps(dict(cmd=command())),i) for i in range(2,5)]])
    assert len(x['dispatches']) == 3
    assert all(not d['process_return_attested'] for d in x['dispatches'])


@pytest.mark.parametrize('args', [lambda c:json.dumps(dict(cmd=c)),
    lambda c:'text(await tools.exec_command({cmd:'+json.dumps(c)+'}));',
    lambda c:'const r = await tools.exec_command({"cmd":'+json.dumps(c)+'});'])
def test_actual_JS_and_JSON_cmd_forms(args):
    assert len(m.replay([record(args(command()))])['dispatches']) == 1


@pytest.mark.parametrize('cmd', ['cat results/pusht_d2p_p1_reconfiguration_replay_bound_20261024_cuda/report.json',
    'rg preflight_pusht_d2p_p1_reconfiguration .', 'echo outcome',
    command().replace('--output','--p0-evidence'), command().replace('_cuda','_cuda_wrong')])
def test_mentions_are_not_launches(cmd):
    assert m.launch_matches(cmd) == []


def test_original_shell_default_only_not_current_source():
    cmd = 'bash '+m.SHELL
    assert m.launch_matches(cmd) == []
    old = 'OUTPUT="${1:-results/'+m.SOURCES[0][1]+'}"\n'
    assert m.launch_matches(cmd,old)[0][0] == 'A'


def test_missing_base_never_substitute_current_file():
    patch = '*** Begin Patch\n*** Update File: '+m.P1+'\n@@\n-old\n+new\n*** End Patch'
    x=m.replay([record(patch),record(json.dumps(dict(cmd=command())),2)])
    assert x['recovery_errors']
    assert x['dispatches'][0]['recovered_producer_sha256'] is None


def test_unbounded_while_parsed_never_executed_or_certified():
    code='SCHEMA = "v2"\ndef f(env):\n    while True:\n        env.reset(seed=3)\n'
    x=m.replay([record(add(code)),record(json.dumps(dict(cmd=command())),2)])
    d=x['dispatches'][0]
    assert d['recovered_producer_metadata']['reset_and_collector_sites'][0]['line'] == 4
    assert not d['execution_source_attested']


def test_no_exec_of_historical_source():
    code='SCHEMA = "v2"\nraise RuntimeError("must not execute")\n'
    assert not m.replay([record(add(code))])['recovery_errors']


def test_ambiguous_context_receipt_does_not_claim_unique():
    choices=[]
    out=m.apply_section('x=1\nx=1\n',['@@','-x=1','+x=2'],choices=choices)
    assert out == 'x=2\nx=1\n' and choices[0]['matches'] == [0,1]


def test_partial_patch_rejected():
    with pytest.raises(ValueError,match='partial'):
        m.sections(add()[:-8])


@pytest.mark.parametrize('key,value', [('sampling_denominator',3),('attempt_per_slot',2),
    ('proposal_per_slot',2),('replacement_or_retry',True)])
def test_fixed_denominator_mutations(key,value):
    reg=registration();reg[key]=value
    with pytest.raises(ValueError):m.fixed_candidates(reg)


def test_candidate_order_mutation():
    reg=registration();reg['candidate_order'].reverse()
    with pytest.raises(ValueError):m.fixed_candidates(reg)


def test_unknown_or_fake_complete_never_promotes_candidates():
    row=dict(id='B',conditional_family_span=[0,1],coverage_complete=True,
             original_execution_reset_bound_certified=True)
    result=m.candidates(registration(),[row])
    assert len(result)==4 and all(not r['certifiable'] and not r['dispatch_authorized'] for r in result)


def test_cross_namespace_reserved_span_conflict():
    row=dict(id='A',conditional_family_span=[203027201,203027201],
             reserved_family_span=[203027101,203027101])
    assert m.candidates(registration(),[row])[0]['conditional_seed_conflicts']==['A','A:reservation']


@pytest.mark.parametrize('field,value',[(('schema',),'v3'),(('protocol','root_seed'),999),
    (('protocol','root_count'),3),(('protocol','root_oversample_factor'),8),
    (('protocol','root_count'),True)])
def test_protocol_version_seed_and_bound_mutations(field,value):
    p={('schema',):'v2',('protocol','root_seed'):10,('protocol','root_count'):4,
       ('protocol','root_oversample_factor'):4}
    p[field]=value
    with pytest.raises(ValueError):m.validate_protocol(p,'F',10,4,'v2')


def test_duplicate_json_key_and_nonfinite_rejected():
    for data in ('{"x":1,"x":2}','{"x":NaN}'):
        with pytest.raises(ValueError):m.strict_json(data)


class RamPath:
    def __init__(self,data):self.data=data
    def stat(self):
        class Size:st_size=len(self.data)
        return Size()
    def read_bytes(self):return self.data
    def open(self,mode):return BytesIO(self.data)
    def __str__(self):return 'RAM.jsonl'


def test_sha_mutation_rejected():
    with pytest.raises(ValueError,match='SHA mismatch'):m.pinned(RamPath(b'a'),m.sha(b'b'))


def event(kind,args):
    return m.canonical(dict(timestamp='2026-10-24T10:00:00Z',payload=dict(type=kind,
        name='exec',call_id='c',arguments=args)))+b'\n'


def test_log_only_decodes_tool_arguments_not_results_or_messages(monkeypatch):
    raw=event('function_call',json.dumps(dict(cmd=command())))
    raw+=event('function_call_output','secret outcome '+command())
    raw+=event('message','future label '+command())
    seen=[];original=m.strict_json
    def decode(x):
        seen.append(x)
        assert b'secret outcome' not in x if isinstance(x,bytes) else 'secret outcome' not in x
        return original(x)
    monkeypatch.setattr(m,'strict_json',decode)
    x,bind=m.scan_logs([RamPath(raw)])
    assert len(x['dispatches'])==1 and len(seen)==2 # event + inner literal cmd object
    assert bind[0]['sha256']==m.sha(raw)


@pytest.mark.parametrize('raw',[b'{"broken":',b'x'*20])
def test_truncated_log_not_coverage(raw):
    with pytest.raises(ValueError,match='partial'):m.scan_logs([RamPath(raw)])


def test_log_budget_not_silently_partial(monkeypatch):
    monkeypatch.setattr(m,'LOG_FILE_CAP',1)
    with pytest.raises(ValueError,match='bounded'):m.scan_logs([RamPath(b'aa\n')])


def test_projection_whitelist_skips_native_outcomes():
    raw=b'{"schema":"v2","protocol":{"root_seed":10},"native_success":true,"trajectory":[{"state":[1,2,3]}]}'
    assert m.project(raw,[('schema',),('protocol','root_seed')]) == {
        ('schema',):'v2',('protocol','root_seed'):10}


def test_later_seal_or_inventory_is_not_an_execution_attestation():
    x=m.replay([record(json.dumps(dict(cmd='cat later_prerun_seal.json '+command())))])
    assert x['edits']==[] and x['dispatches']==[]
    result=m.candidates(registration(),[])
    assert all('complete_historical_scope' in x['missing'] for x in result)


def test_wrong_schema_not_counted_as_recovered_match():
    row=dict(id='F',sha256='report')
    d=dict(source_id='F',log='RAM',line=1,call_id='c',recovered_producer_sha256='hash',
           recovered_producer_metadata=dict(schema='v3'))
    m.summarize_dispatches([row],[d])
    assert row['schema_matched_recovered_versions']==[]
    assert row['coverage_request']['at_least_dispatch_intents_to_cover']==1


@pytest.mark.parametrize('mutation',['drop_failed_launch','fake_complete','swap_version','change_tolerance'])
def test_saved_closure_mutation_fails_full_recomputation(mutation):
    actual=dict(dispatches=[dict(line=1),dict(line=2)],coverage_complete=False,
                producer_sha256='v2',tolerances=dict(xy=1e-9,angle=1e-12))
    bad=copy.deepcopy(actual)
    if mutation=='drop_failed_launch':bad['dispatches'].pop(0)
    if mutation=='fake_complete':bad['coverage_complete']=True
    if mutation=='swap_version':bad['producer_sha256']='v3'
    if mutation=='change_tolerance':bad['tolerances']['xy']=1e-3
    with pytest.raises(ValueError,match='drift'):m.verify_saved(m.canonical(bad),actual)


def test_registration_exclusion_tolerance_mutation_rejects_external_SHA():
    original=m.canonical(dict(exclude='XY1e-9 angle1e-12',**registration()))
    changed=original.replace(b'1e-9',b'1e-3')
    with pytest.raises(ValueError,match='SHA'):m.pinned(RamPath(changed),m.sha(original))
