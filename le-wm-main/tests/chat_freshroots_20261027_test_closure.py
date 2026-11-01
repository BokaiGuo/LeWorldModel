"""RAM-only consumer tests; no models, producer calls or disk fixtures."""
import pytest
from experiments.spectral_reachability import chat_freshroots_20261027_closure as c


def test_plan_sealed():
    raw = c.bound_read(c.OUTER/'docs/chat_freshroots_20261027_preregistration.json',
                       '79e1f45e79d692cae72376c91a4830c94b6ac1f53bb346eeaa085715e4197d7e')
    c.validate_plan(c.strict_json(raw))


@pytest.mark.parametrize('field,value', [('sampling_denominator',3),
    ('replacement_or_retry',True), ('new_outcomes_opened',True)])
def test_plan_no_override(field,value):
    plan = c.strict_json((c.OUTER/'docs/chat_freshroots_20261027_preregistration.json').read_bytes())
    plan[field] = value
    with pytest.raises(ValueError): c.validate_plan(plan)


def test_no_candidate_reorder():
    plan = c.strict_json((c.OUTER/'docs/chat_freshroots_20261027_preregistration.json').read_bytes())
    plan['candidate_order'].reverse()
    with pytest.raises(ValueError): c.validate_plan(plan)


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_strict_json(raw):
    with pytest.raises(ValueError): c.strict_json(raw)


@pytest.mark.parametrize('value,expected', [(9,0),(10,1),(19,1),(20,0)])
def test_half_open(value,expected):
    candidate = dict(root_seed=value,reset_seed=100)
    assert len(c.seed_conflicts(candidate,[],[dict(half_open=[10,20])])) == expected


def test_cross_namespace_and_preserved_denominator():
    hits = c.seed_conflicts(dict(root_seed=101,reset_seed=201),
                           [dict(seed=101,kind='reset_seed'),dict(seed=201,kind='root_seed')],[])
    assert len(hits) == 2


@pytest.mark.parametrize('span', [[2,2],[3,1],[True,3],[0,2**63+1]])
def test_bad_span(span):
    with pytest.raises(ValueError):
        c.seed_conflicts(dict(root_seed=1,reset_seed=2),[],[dict(half_open=span)])


def graph():
    return dict(sources=[dict(path='/a',sha256='a'*64,associations=[
        dict(target_path='/b',target_sha256='b'*64,witness_edge_ids=[0])],direct_identity_projections=[]),
        dict(path='/b',sha256='b'*64,associations=[],direct_identity_projections=[1])],
        edges=[dict(id=0,status='bound',source_path='/a',source_sha256='a'*64,
                    target_path='/b',target_sha256='b'*64,pointer='/bindings/~1b')])


def test_witness():
    assert c.audit_witnesses(graph()) == 1


def test_multihop_and_actual_cycle():
    g = graph()
    g['sources'].append(dict(path='/c',sha256='c'*64,associations=[],direct_identity_projections=[1]))
    g['edges'].append(dict(id=1,status='bound',source_path='/b',source_sha256='b'*64,
                           target_path='/c',target_sha256='c'*64,pointer='/bindings/~1c'))
    assoc = g['sources'][0]['associations'][0]
    assoc.update(target_path='/c',target_sha256='c'*64,witness_edge_ids=[0,1])
    assert c.audit_witnesses(g) == 1
    g['edges'][1].update(target_path='/a',target_sha256='a'*64,pointer='/bindings/~1a')
    with pytest.raises(ValueError): c.audit_witnesses(g)


@pytest.mark.parametrize('key,value', [('status','unknown'),('source_path','/c'),
    ('source_sha256','x'*64),('target_sha256','x'*64),('pointer','/bindings/~1alias')])
def test_witness_tampering(key,value):
    g = graph(); g['edges'][0][key] = value
    with pytest.raises(ValueError): c.audit_witnesses(g)


def test_empty_witness():
    g = graph(); g['sources'][0]['associations'][0]['witness_edge_ids'] = []
    with pytest.raises(ValueError): c.audit_witnesses(g)


def test_no_terminal_identity():
    g = graph(); g['sources'][1]['direct_identity_projections'] = []
    with pytest.raises(ValueError): c.audit_witnesses(g)


def test_huge_span_constant_arithmetic():
    assert len(c.seed_conflicts(dict(root_seed=1,reset_seed=2),[],
                               [dict(half_open=[0,2**63])])) == 2
