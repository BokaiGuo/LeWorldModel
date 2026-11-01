"""RAM metadata fixtures; no resolver, model, environment or evidence acquisition."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from experiments.spectral_reachability import moving_authority_identity_link_closure_20261027_v1 as c


def node(name, identities=()):
    fields=[dict(pointer=p,value=v) for p,v in identities]
    return dict(path=str(c.ROOT/'results'/name),sha256=hashlib.sha256(name.encode()).hexdigest(),
                status='metadata_captured_not_coverage_certified',hash_authority='external_attested',
                inventory_family='unverified',projection=copy.deepcopy(fields),
                root_identity_projections=fields,input_lineage_links=[],blockers=[],inherited_refs=[])


def link(a,b):
    pointer='/bindings/'+b['path'].replace('~','~0').replace('/','~1')
    a['projection'].append(dict(pointer=pointer,value=b['sha256']))
    a['input_lineage_links'].append(dict(pointer=pointer,target_path=b['path'],target_sha256=b['sha256'],relation=c.RELATION))


def report(nodes):
    return dict(schema=c.INPUT_SCHEMA,history_file_requests=nodes,summary=dict(inherited_reset_bound_unattested_refs=sum(
        r['original_reason']==c.RESET_REASON for n in nodes for r in n['inherited_refs'])),
        authority_requests=[dict(family='original_four',inventory_paths=[],status='blocked',remaining='MAIN required')],
        inventory_binding=dict(path='/saved/inventory.json',sha256='a'*64),
        MAIN_coverage_approval=False,freshness_certified=False,P1_admission=False)


def run(nodes):
    raw=json.dumps(report(nodes)).encode()
    return c.convert(raw,hashlib.sha256(raw).hexdigest())


def test_transitive_proof_chain_not_identity_transfer(monkeypatch):
    a,b,d=node('a'),node('b'),node('d',[('/roots/0/root_sha256','f'*64)])
    link(a,b);link(b,d)
    monkeypatch.setattr(Path,'open',lambda *a,**k:pytest.fail('evidence opened'))
    out=run([a,b,d]); source=out['sources'][0]
    assert source['direct_identity_projections']==[]
    assert source['associations'][0]['target_path']==d['path']
    assert len(source['associations'][0]['witness_edge_ids'])==2
    assert out['summary']['sources_with_upstream_identity_association']==2
    assert out['physical_root_equivalence_proved'] is False
    assert out['freshness_certified'] is False


def test_no_edge_no_filename_or_shared_identity_inference():
    a=node('same_root_report.json');b=node('same_root_args.json',[('/root_seed',17)])
    out=run([a,b])
    assert out['sources'][0]['associations']==[]
    assert 'no_bound_upstream_identity' in out['sources'][0]['unknown']


def test_sha_alias_is_unknown_not_path_identity():
    a,b,d=node('a'),node('b'),node('alias',[('/root_seed',17)])
    d['sha256']=b['sha256'];link(a,b)
    a['input_lineage_links'][0]['target_path']=d['path']
    out=run([a,b,d])
    assert out['sources'][0]['associations']==[]
    assert out['edges'][0]['status']=='unknown'
    assert out['edges'][0]['reason']=='path_alias_not_explicit_same_path'


@pytest.mark.parametrize('kind',['self','two','upstream'])
def test_cycles_failclosed(kind):
    a,b,d=node('a',[('/root_seed',1)]),node('b',[('/root_seed',2)]),node('d')
    if kind=='self':link(a,a)
    else:link(a,b);link(b,a)
    if kind=='upstream':link(d,a)
    out=run([d,a,b])
    affected=out['sources'][0] if kind=='upstream' else out['sources'][1]
    assert affected['associations']==[]
    assert 'reachable_cycle' in affected['unknown']


@pytest.mark.parametrize('kind',['target_sha','projection_sha','missing_projection','relation'])
def test_conflicting_edge_blocks_source_and_dependents(kind):
    a,b,d=node('a'),node('b'),node('d',[('/root_seed',12)])
    link(a,b);link(b,d)
    edge=b['input_lineage_links'][0]
    if kind=='target_sha':edge['target_sha256']='a'*64
    elif kind=='projection_sha':b['projection'][0]['value']='a'*64
    elif kind=='missing_projection':b['projection']=[]
    else:edge['relation']='physical_same_root'
    out=run([a,b,d])
    assert out['sources'][0]['associations']==[]
    assert 'reachable_conflict' in out['sources'][0]['unknown']


def test_missing_target_and_missing_edge_explicit_unknown():
    a,b=node('a'),node('b',[('/root_seed',2)]);link(a,b)
    out=run([a]);assert out['edges'][0]['reason']=='target_not_captured'
    a['input_lineage_links']=[]
    out=run([a,b]);assert out['sources'][0]['associations']==[]
    assert out['sources'][0]['unlinked_bindings']


def test_diamond_multiple_roots_are_set_of_upstream_evidence_not_conflict():
    a,b,d=node('a'),node('b',[('/root_seed',1)]),node('d',[('/root_seed',2)])
    link(a,b);link(a,d)
    out=run([a,b,d]);assert len(out['sources'][0]['associations'])==2
    assert out['sources'][0]['status']=='upstream_metadata_associated_not_physical_identity'


def test_original_blockers_and_16_refs_preserved():
    a,b=node('a'),node('b',[('/root_seed',7)]);link(a,b)
    a['blockers']=['old unknown',c.RESET_REASON]
    a['inherited_refs']=[dict(parent='old',pointer=f'/unknown/{i}',historical_sha256=a['sha256'],original_reason=c.RESET_REASON) for i in range(16)]
    out=run([a,b]);source=out['sources'][0]
    assert source['original_blockers']==a['blockers']
    assert source['inherited_refs']==a['inherited_refs']
    assert out['summary']['preserved_reset_bound_unattested_refs']==16
    assert out['reset_attempt_bound_certified'] is False
    assert out['authority_requests']==report([a,b])['authority_requests']
    assert out['summary']['inherited_refs_without_direct_identity']==16
    assert out['summary']['inherited_refs_without_direct_but_associated']==16


def test_ancestor_with_unlinked_binding_stays_unknown():
    a,b,d=node('a'),node('b'),node('d',[('/root_seed',2)])
    link(a,b);link(b,d);b['input_lineage_links']=[]
    out=run([a,b,d])
    assert out['sources'][0]['associations']==[]
    assert 'reachable_unlinked_bindings' in out['sources'][0]['unknown']


def test_cycle_does_not_poison_disjoint_valid_branch():
    a,b,d=node('a'),node('b'),node('d',[('/root_seed',2)])
    link(a,a);link(b,d)
    out=run([a,b,d])
    assert out['sources'][1]['associations']
    assert out['sources'][1]['status']!='blocked_graph'


@pytest.mark.parametrize('field', ['history_file_requests','summary','authority_requests','inventory_binding'])
def test_required_capture_metadata_missing(field):
    payload=report([node('a')]);del payload[field]
    raw=json.dumps(payload).encode()
    with pytest.raises(ValueError):c.convert(raw,hashlib.sha256(raw).hexdigest())


def test_no_self_approval_and_reset_count_mismatch():
    for field,value in [('freshness_certified',True),('summary',dict(inherited_reset_bound_unattested_refs=16))]:
        payload=report([node('a')]);payload[field]=value
        raw=json.dumps(payload).encode()
        with pytest.raises(ValueError):c.convert(raw,hashlib.sha256(raw).hexdigest())


def test_limits_prevent_output_creation(tmp_path,monkeypatch):
    source=tmp_path/'capture.json';raw=json.dumps(report([node('a')])).encode();source.write_bytes(raw)
    target=tmp_path/'new.json'
    monkeypatch.setattr(c,'OUTPUT_CAP',10)
    with pytest.raises(ValueError):c.main(['--input',str(source),'--input-sha256',hashlib.sha256(raw).hexdigest(),'--output',str(target)])
    assert not target.exists()
    monkeypatch.setattr(c,'NODE_CAP',1)
    with pytest.raises(ValueError):run([node('a'),node('b')])
    monkeypatch.setattr(c,'NODE_CAP',1024);monkeypatch.setattr(c,'EDGE_CAP',0)
    a,b=node('a'),node('b');link(a,b)
    with pytest.raises(ValueError):run([a,b])


@pytest.mark.parametrize('kind',['duplicate_path','duplicate_projection','identity_mismatch','identity_bool','historical_sha'])
def test_bad_nodes_failclosed(kind):
    a=node('a',[('/root_seed',1)]);nodes=[a]
    if kind=='duplicate_path':nodes.append(copy.deepcopy(a))
    elif kind=='duplicate_projection':a['projection']*=2
    elif kind=='identity_mismatch':a['root_identity_projections'][0]['value']=2
    elif kind=='identity_bool':a['projection'][0]['value']=True;a['root_identity_projections'][0]['value']=True
    else:a['inherited_refs']=[dict(parent='p',pointer='/x',historical_sha256='f'*64,original_reason='unknown')]
    with pytest.raises(ValueError):run(nodes)


def test_opaque_outcome_skipped_and_duplicate_keys_rejected():
    raw=json.dumps(report([node('a')])).encode()
    raw=raw.replace(b'"history_file_requests":',b'"outcome": {"success":NaN}, "history_file_requests":')
    assert c.convert(raw,hashlib.sha256(raw).hexdigest())['summary']['sources']==1
    raw=raw.replace(b'"schema":',b'"schema":"bad", "schema":',1)
    with pytest.raises(ValueError):c.convert(raw,hashlib.sha256(raw).hexdigest())


def test_budgets_and_input_sha(monkeypatch):
    raw=json.dumps(report([node('a')])).encode()
    with pytest.raises(ValueError):c.convert(raw,'f'*64)
    monkeypatch.setattr(c,'INPUT_CAP',10)
    with pytest.raises(ValueError):c.convert(raw,hashlib.sha256(raw).hexdigest())


def test_cli_only_capture_and_new_output(tmp_path,capsys):
    source=tmp_path/'capture.json';raw=json.dumps(report([node('a')])).encode();source.write_bytes(raw)
    target=tmp_path/'new.json'
    args=['--input',str(source),'--input-sha256',hashlib.sha256(raw).hexdigest(),'--output',str(target)]
    c.main(args);receipt=json.loads(capsys.readouterr().out)
    assert receipt['sha256']==hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):c.main(args)
    dangling=tmp_path/'dangling';dangling.symlink_to(tmp_path/'absent');args[-1]=str(dangling)
    with pytest.raises(FileExistsError):c.main(args)
    assert not (tmp_path/'absent').exists()
