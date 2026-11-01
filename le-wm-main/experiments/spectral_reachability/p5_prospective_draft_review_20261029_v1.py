"""Review a P5 prospective protocol *draft*, never certify freshness.

The proposed input format is a local review aid, not a frozen P5 ABI.  It does
not dereference evidence, infer claims, approve a protocol, or authorize runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

NAME = 'p5_prospective_draft_review_20261029_v1'
SECTIONS = (
    'cohort_order_seed_distribution', 'source_episode_split_exclusion',
    'model_task_observation_action_history_labels', 'arms_resource_caps',
    'failure_denominator_stopping', 'training_cost', 'statistical_commitments',
)
CLAIMS = ('event_new', 'state_set_disjoint', 'sampling_independence', 'training_lineage')
HEX = re.compile(r'^[0-9a-f]{64}$')
CAP = 1024 * 1024


def fail(message):
    raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail('duplicate JSON key')
        result[key] = value
    return result


def parse(data):
    if len(data) > CAP:
        fail('draft too large')
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda value: fail('non-finite JSON'))


def descriptor(value):
    # A citation is only a locator and digest; its provenance is not verified.
    if not isinstance(value, dict) or set(value) != {'path', 'sha256', 'pointer'}:
        fail('citation must contain path, sha256, pointer')
    if (not isinstance(value['path'], str) or not Path(value['path']).is_absolute()
            or not isinstance(value['sha256'], str) or not HEX.fullmatch(value['sha256'])
            or not isinstance(value['pointer'], str) or not value['pointer'].startswith('/')):
        fail('invalid citation locator')


def review(draft):
    if not isinstance(draft, dict) or set(draft) - {'sections', 'separation_claims'}:
        fail('unknown draft keys')
    sections = draft.get('sections', {})
    claims = draft.get('separation_claims', {})
    if not isinstance(sections, dict) or set(sections) - set(SECTIONS):
        fail('unknown sections')
    if not isinstance(claims, dict) or set(claims) - set(CLAIMS):
        fail('unknown separation claims')
    present, missing = [], []
    for key in SECTIONS:
        value = sections.get(key)
        if value is None or value == '':
            missing.append(key)
        elif not isinstance(value, str) or not value.strip():
            fail('section must be a nonblank string or null')
        else:
            present.append(key)
    reviewed = {}
    for key in CLAIMS:
        claim = claims.get(key)
        if claim is None:
            reviewed[key] = {'declared': 'unknown', 'citations': 0, 'evidence_verified': False}
            continue
        if not isinstance(claim, dict) or set(claim) != {'declaration', 'citations'}:
            fail('claim requires declaration and citations')
        declaration, citations = claim['declaration'], claim['citations']
        if declaration not in ('unknown', 'asserted') or not isinstance(citations, list):
            fail('invalid declaration or citations')
        for citation in citations:
            descriptor(citation)
        reviewed[key] = {'declared': declaration, 'citations': len(citations), 'evidence_verified': False}
    return {
        'schema': NAME + '.review', 'status': 'advisory_only_not_certification',
        'sections_present': present, 'sections_missing': missing, 'separation_claims': reviewed,
        'protocol_frozen': False, 'source_run_authorized': False, 'freshness_certified': False,
        'P1_admission': False, 'P5_admission': False, 'neural_calls': 0, 'physics_calls': 0,
        'limitations': ['draft text and citation locators only; no cited evidence checked',
                        'no conclusion about history, independence, training, or state overlap',
                        'not the future approved protocol ABI or an execution gate'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', help='optional local review draft JSON; omitted means no draft')
    args = parser.parse_args()
    if args.draft:
        with Path(args.draft).open('rb') as stream:
            data = stream.read(CAP + 1)
    else:
        data = b'{}'
    result = review(parse(data))
    result['draft_sha256'] = hashlib.sha256(data).hexdigest() if args.draft else None
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
