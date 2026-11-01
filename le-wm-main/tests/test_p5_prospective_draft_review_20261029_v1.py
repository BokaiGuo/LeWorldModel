import json

import pytest

from experiments.spectral_reachability import p5_prospective_draft_review_20261029_v1 as review


def test_empty_draft_explicitly_unknown_and_never_authorizes():
    result = review.review({})
    assert result['sections_missing'] == list(review.SECTIONS)
    assert all(c['declared'] == 'unknown' for c in result['separation_claims'].values())
    assert not any(result[k] for k in ('protocol_frozen', 'source_run_authorized',
                                     'freshness_certified', 'P1_admission', 'P5_admission'))


def test_independent_claims_do_not_promote_each_other():
    citation = dict(path='/independent/source', sha256='a'*64, pointer='/statement')
    draft = {'sections': {s: 'a draft commitment' for s in review.SECTIONS},
             'separation_claims': {'event_new': {'declaration': 'asserted', 'citations': [citation]}}}
    result = review.review(draft)
    assert result['sections_missing'] == []
    assert result['separation_claims']['event_new'] == dict(declared='asserted', citations=1, evidence_verified=False)
    assert all(result['separation_claims'][s]['declared'] == 'unknown' for s in review.CLAIMS[1:])
    assert not result['freshness_certified'] and not result['source_run_authorized']


@pytest.mark.parametrize('draft', [
    {'sections': {'sample_count': '4'}},
    {'sections': {'training_cost': '   '}},
    {'separation_claims': {'event_new': {'declaration': 'certified', 'citations': []}}},
    {'separation_claims': {'event_new': {'declaration': 'asserted', 'citations': [{'path': '/x', 'sha256': 'a'*64}]}}},
    {'separation_claims': {'event_new': {'declaration': 'asserted', 'citations': [{'path': 'relative', 'sha256': 'a'*64, 'pointer': '/x'}]}}},
    {'ready': True},
])
def test_reject_bad_draft_without_promoting_claims(draft):
    with pytest.raises(ValueError):
        review.review(draft)


@pytest.mark.parametrize('raw', [b'{"sections":{},"sections":{}}', b'{"x":NaN}', b' '* (review.CAP+1)])
def test_reject_duplicate_nonfinite_and_unbounded_json(raw):
    with pytest.raises(ValueError):
        review.parse(raw)


def test_roundtrip_no_citations_are_opened(tmp_path):
    fake = tmp_path / 'does-not-exist'
    draft = {'separation_claims': {'training_lineage': {'declaration': 'asserted',
             'citations': [dict(path=str(fake), sha256='b'*64, pointer='/claim')]}}}
    result = review.review(review.parse(json.dumps(draft).encode()))
    assert not fake.exists()
    assert result['separation_claims']['training_lineage']['evidence_verified'] is False
