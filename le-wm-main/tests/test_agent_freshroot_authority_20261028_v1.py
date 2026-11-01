"""RAM examples only. Accepted submissions are NOT historical attestations."""
import copy
import pytest
from experiments.spectral_reachability import agent_freshroot_authority_20261028_v1 as m


def catalog():
    sources=[];dispatches=[]
    for sid in 'ABCDEFGH':
        sources.append(dict(id=sid,path='/RAM/'+sid+'/report.json',sha256=m.sha(sid.encode()),
            references=[dict(parent=p) for p in ('reachable_prepare','moving_prepare')]))
        for i in range(3 if sid=='B' else 1):
            dispatches.append(dict(source_id=sid,call_id=sid+str(i),command='python frozen.py',
                producer='preflight_pusht_d2p_p0.py',log='/RAM/original.jsonl',line=i+1))
    return dict(sources=sources,recovery=dict(dispatches=dispatches))


def fixture(sid='A',route='exhaustive_ledger'):
    request=m.requests(catalog());r=next(x for x in request['sources'] if x['source_id']==sid)
    blobs={};runs=[]
    def asset(path,value):
        blobs[path]=m.canonical(value)
        return dict(path=path,sha256=m.sha(blobs[path]),pointer='')
    for d in r['dispatches']:
        cid=d['call_id']
        if route=='exhaustive_ledger':
            reset=dict(call_id=cid,attempts=[dict(index=0,reset_kind='random',
                disposition='failed',identity_sha256=None),dict(index=1,reset_kind='explicit_state',
                disposition='selected',identity_sha256='b'*64)])
            loaded=[]
        else:
            reset=dict(call_id=cid,upper_bound=2,derivation='RAM only, not a historical bound')
            loaded=[]
            for role in sorted(m.load_roles(d)):
                path='/RAM/'+cid+'/'+role+'.py';blobs[path]=b'# fixture, never executed'
                loaded.append(dict(role=role,path=path,sha256=m.sha(blobs[path])))
        a=dict(call_id=cid,report_sha256=r['report']['sha256'],route=route,attempts=2,
            attempts_complete=True,restarts_complete=True,
            reset_evidence=asset('/RAM/'+cid+'/reset.json',reset),loaded=loaded)
        runs.append(dict(assertion=a,origin_citation=asset('/RAM/'+cid+'/origin.json',a)))
    response=dict(schema=m.NAME+'.response',request_sha256=m.sha(m.canonical(request)),
        sources=[dict(source_id=sid,report_sha256=r['report']['sha256'],runs=runs,
                      unresolved_reason=None)],historical_scope=None)
    return request,response,blobs


def evaluate(f):
    req,res,blobs=f
    return m.validate(res,req,m.Reader(provider=blobs.__getitem__))


def rebind(f):
    _,res,blobs=f
    run=res['sources'][0]['runs'][0];ref=run['origin_citation']
    blobs[ref['path']]=m.canonical(run['assertion']);ref['sha256']=m.sha(blobs[ref['path']])


def test_fixed_16_refs_are_shared_eight_sources_ten_dispatches():
    req=m.requests(catalog())
    assert sum(len(x['inherited_refs']) for x in req['sources'])==16
    assert sum(len(x['dispatches']) for x in req['sources'])==10
    assert not req['coverage_complete'] and req['certified_fresh_candidates']==0


@pytest.mark.parametrize('route',['exhaustive_ledger','loaded_closure_and_exhaustive_envelope'])
def test_acceptable_RAM_submission_never_issues_authority(route):
    out=evaluate(fixture(route=route))
    assert out['origin_review_required'] and out['reset_blockers_cleared']==0
    assert out['certified_fresh_candidates']==0 and not out['P1_admitted']
    assert len(out['missing_sources'])==7


def test_B_requires_all_three_dispatches_not_just_final():
    f=fixture('B');f[1]['sources'][0]['runs']=f[1]['sources'][0]['runs'][-1:]
    f[1]['sources'][0]['unresolved_reason']='two earlier failed launches still missing'
    row=evaluate(f)['sources'][0]
    assert row['status']=='partial' and row['missing_dispatches']==['B0','B1']


def test_B_all_three_are_only_reviewable_not_certified():
    out=evaluate(fixture('B'))
    assert out['sources'][0]['submitted_dispatches']==3 and out['reset_blockers_cleared']==0


@pytest.mark.parametrize('key,value',[('request_sha256','0'*64),('schema','other'),
    ('externally_reviewed_complete',True)])
def test_unknown_authority_or_wrong_request_rejected(key,value):
    f=fixture();f[1][key]=value
    with pytest.raises(ValueError):evaluate(f)


@pytest.mark.parametrize('key,value',[('report_sha256','a'*64),('source_id','Z')])
def test_wrong_source_report_join(key,value):
    f=fixture();f[1]['sources'][0][key]=value
    with pytest.raises(ValueError):evaluate(f)


def test_duplicate_source_rejected():
    f=fixture();f[1]['sources']*=2
    with pytest.raises(ValueError,match='source identity'):evaluate(f)


def test_duplicate_run_rejected():
    f=fixture('B');f[1]['sources'][0]['runs'][1]=f[1]['sources'][0]['runs'][0]
    with pytest.raises(ValueError,match='dispatch identity'):evaluate(f)


def test_origin_payload_not_same_as_claim_rejected():
    f=fixture();f[1]['sources'][0]['runs'][0]['assertion']['attempts']=1
    with pytest.raises(ValueError,match='citation payload'):evaluate(f)


@pytest.mark.parametrize('field,value',[('attempts',True),('attempts',-1),
    ('attempts',3),('attempts_complete',False),('restarts_complete',False),('route','current_source')])
def test_bad_attempt_claims(field,value):
    f=fixture();f[1]['sources'][0]['runs'][0]['assertion'][field]=value;rebind(f)
    with pytest.raises(ValueError):evaluate(f)


def test_failed_attempt_without_observed_root_retained():
    f=fixture();out=evaluate(f)
    assert out['sources'][0]['status']=='origin_review_required'


def edit_reset(f, edit):
    a=f[1]['sources'][0]['runs'][0]['assertion'];ref=a['reset_evidence']
    v=m.parse(f[2][ref['path']]);edit(v);f[2][ref['path']]=m.canonical(v)
    ref['sha256']=m.sha(f[2][ref['path']]);rebind(f)


def test_attempt_gap_rejected():
    f=fixture();edit_reset(f,lambda v:v['attempts'][1].update(index=3))
    with pytest.raises(ValueError,match='contiguous'):evaluate(f)


def test_selected_root_null_identity_rejected():
    f=fixture();edit_reset(f,lambda v:v['attempts'][1].update(identity_sha256=None))
    with pytest.raises(ValueError,match='identity'):evaluate(f)


def test_current_loaded_hash_cannot_override_origin_receipt():
    f=fixture(route='loaded_closure_and_exhaustive_envelope')
    path=f[1]['sources'][0]['runs'][0]['assertion']['loaded'][0]['path']
    f[2][path]=b'# changed since origin receipt'
    with pytest.raises(ValueError,match='SHA mismatch'):evaluate(f)


def test_missing_factory_role_rejected():
    f=fixture(route='loaded_closure_and_exhaustive_envelope')
    f[1]['sources'][0]['runs'][0]['assertion']['loaded'].pop();rebind(f)
    with pytest.raises(ValueError,match='closure roles'):evaluate(f)


def test_envelope_bound_is_not_guess_from_root_count():
    f=fixture(route='loaded_closure_and_exhaustive_envelope')
    edit_reset(f,lambda v:v.update(upper_bound=1))
    with pytest.raises(ValueError,match='upper bound'):evaluate(f)


def test_partial_without_reason_rejected():
    f=fixture('B');f[1]['sources'][0]['runs'].pop()
    with pytest.raises(ValueError,match='text'):evaluate(f)


def test_empty_submission_valid_but_all_missing():
    f=fixture();f[1]['sources']=[]
    assert evaluate(f)['missing_sources']==list('ABCDEFGH')


def test_scope_requires_bound_origin_not_just_complete_true():
    f=fixture();f[1]['historical_scope']=dict(complete=True)
    with pytest.raises(ValueError,match='fields'):evaluate(f)


def test_scope_literal_payload_can_be_submitted_not_approved():
    f=fixture();scope=dict(machines=['RAM'],worktrees=['/RAM'],time_interval='RAM only',
        unsaved_failures='declared none, requires origin review')
    raw=m.canonical(scope);f[2]['/RAM/scope.json']=raw
    f[1]['historical_scope']=dict(**scope,origin_citation=dict(path='/RAM/scope.json',
        sha256=m.sha(raw),pointer=''))
    out=evaluate(f)
    assert out['historical_scope_submitted'] and out['reset_blockers_cleared']==0


@pytest.mark.parametrize('p',['/../bad','relative.json'])
def test_nonliteral_path_rejected(p):
    with pytest.raises(ValueError,match='literal'):m.Reader(provider=lambda _:b'').read(p,'a'*64)


def test_asset_cap_before_acceptance(monkeypatch):
    monkeypatch.setattr(m,'TOTAL_CAP',1)
    with pytest.raises(ValueError,match='total byte cap'):evaluate(fixture())


@pytest.mark.parametrize('p',['/a~2b','/missing','/a/01'])
def test_invalid_pointer_rejected(p):
    with pytest.raises(ValueError):m.pointer({'a':[1]},p)


def test_escaped_pointer():
    assert m.pointer({'a/b':{'~':[7]}},'/a~1b/~0/0')==7


def test_duplicate_json_and_nonfinite_rejected():
    for raw in ('{"a":1,"a":2}','{"a":NaN}'):
        with pytest.raises(ValueError):m.parse(raw)


def test_shell_and_P1_require_their_own_closure_roles():
    d=dict(command='bash launch.sh',producer='preflight_pusht_d2p_p1_reconfiguration.py')
    assert {'shell_launcher','p0_dependency'}<=m.load_roles(d)
