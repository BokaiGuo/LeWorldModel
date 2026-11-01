"""Saved-only fixed-panel accounting; never executes or reconstructs a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import agent_p3_c3_common_cap_cuda_20261029_v1 as original

ROOT = original.ROOT
NAME = 'agent_p3_c3_preregistered_analysis_20261029_v1'
HOST = 'agent_p3_c3_compact_host_cuda_20261029_v1'
PINS = {
    ROOT/'experiments/spectral_reachability'/f'{HOST}.py': 'b30304e823f8922f22ef3ec1dec96437bcaec3eb6e757fbb56a9fa481814240b',
    ROOT/'tests'/f'test_{HOST}.py': '6eade1c8d4c355e264e24d2bb3cd8c1b8e7960c619c636774085498691485d84',
    ROOT/'innerdocs'/f'{HOST}.md': '8795cd3ddde0a8f78f6639aaee5aa9679a795cf31a92e2404f90e0e5a881a31f',
    original.PROTOCOL: original.PROTOCOL_SHA,
    Path(original.__file__).resolve(): '7419b72f12ea81f90703d2c075143aa46f07c8fbe47ba0313c488239797520a7',
    Path(original.tree.__file__).resolve(): 'da00e09c89ed8b5be5399952c2ec328a98e036db08f764b61342d69a18f5e6a6',
    Path(original.parent.__file__).resolve(): '150a1c4189a68f9b7c71479e5aa6980cea490fdeed0d5524b7bfad0cc78c2558',
    Path(original.core.__file__).resolve(): '72f076106fb8f6c2aa73386c06fecb912f033e81a5108aa1f36885bd0342c2bc',
}
AXES = original.core.AXES
SETTLED = original.SETTLED


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def unique(pairs):
    result = {}
    for k, v in pairs:
        need(k not in result, 'duplicate JSON key')
        result[k] = v
    return result


def read_pinned(path, expected):
    need(type(expected) is str and len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), 'SHA256 syntax')
    p = Path(path)
    need(p.is_file() and not p.is_symlink() and p.stat().st_size <= 16*1024**2, 'bounded regular receipt')
    raw = p.read_bytes()
    need(sha(raw) == expected, 'receipt SHA256 mismatch')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def analyze_receipt(receipt, receipt_sha):
    need(receipt.get('schema') == HOST+'.consume' and receipt.get('status') == 'saved12_common_cap_engineering_verified', 'HOST strict receipt only')
    need(receipt.get('denominator') == 12 and receipt.get('total_request_denominator') == 1728 and receipt.get('request_slots_per_arm') == 144, 'fixed denominators')
    for field in ('production_ready', 'full_C3_demonstrated', 'physical_release', 'same_actual_resources', 'same_FLOPs'):
        need(receipt.get(field) is False, 'scientific boundary: '+field)
    for field in ('prepare_sha256', 'manifest_sha256', 'report_sha256', 'completion_sha256'):
        s = receipt.get(field)
        need(type(s) is str and len(s) == 64 and all(c in '0123456789abcdef' for c in s), 'upstream identity: '+field)
    arms = receipt.get('arms')
    need(type(arms) is list and len(arms) == 12, 'all fixed slots')
    rows = []
    for identity, arm in zip(original.roster(), arms, strict=True):
        need(all(arm.get(k) == v and type(arm.get(k)) is type(v) for k, v in identity.items()), 'ordered slot identity')
        status = arm.get('status'); plan = arm.get('plan_available')
        need(status in SETTLED and type(plan) is bool and plan == (status == 'tree_complete_not_admitted'), 'complete plan/status')
        resources = arm.get('resources'); need(type(resources) is dict, 'resources')
        vectors = {}
        for key in ('caps', 'attempted', 'completed'):
            v = resources.get(key)
            need(type(v) is dict and set(v) == set(AXES), 'complete 27-axis '+key)
            need(all(type(v[a]) is int and v[a] >= 0 for a in AXES), 'nonnegative integer axes')
            vectors[key] = {a:v[a] for a in AXES}
        need(vectors['caps'] == original.caps(identity['K']), 'original common-vector cap')
        need(all(vectors['completed'][a] <= vectors['attempted'][a] <= vectors['caps'][a] for a in AXES), 'completed <= attempted <= cap')
        need(resources.get('FLOPs') is None and resources.get('backward_engine_calls') is None and resources.get('cuda_ns') is None, 'unknown resource axes not zero')
        count = arm.get('R_groups'); eligible = arm.get('eligible_counts')
        need(type(count) is int and count >= 0 and type(eligible) is list and len(eligible) == count and all(type(x) is int and 0 <= x <= 8 for x in eligible), 'R group counts')
        for field in ('AD_primal_predict_API', 'Jacobian_elements', 'native_calls'):
            need(type(arm.get(field)) is int and arm[field] >= 0, 'recorded '+field)
        need(arm.get('scalar_backward_calls') is None and arm.get('total_FLOPs') is None, 'unmeasured nested costs')
        event_sha = resources.get('events_semantic_sha256')
        need(type(resources.get('events_count')) is int and resources['events_count'] >= 0 and type(event_sha) is str and len(event_sha) == 64 and all(c in '0123456789abcdef' for c in event_sha), 'saved ledger provenance')
        need(type(resources.get('wall_ns')) is int and resources['wall_ns'] >= 0 and type(resources.get('inclusive_operation_wall_ns')) is dict and all(type(v) is int and v >= 0 for v in resources['inclusive_operation_wall_ns'].values()), 'saved inclusive wall observations')
        rows.append(dict(**identity, status=status, plan_available=plan, R_groups=count, eligible_counts=eligible,
                         resources=vectors, AD_primal_predict_API=arm['AD_primal_predict_API'],
                         Jacobian_elements=arm['Jacobian_elements'], native_calls=arm['native_calls'],
                         events_count=resources['events_count'], events_semantic_sha256=resources['events_semantic_sha256'],
                         wall_ns=resources.get('wall_ns'), inclusive_operation_wall_ns=resources.get('inclusive_operation_wall_ns')))
    return dict(schema=NAME+'.analysis', receipt_sha256=receipt_sha,
        source_receipt_identity={k:receipt[k] for k in ('prepare_sha256','manifest_sha256','report_sha256','completion_sha256')},
        denominator=12, request_identity_denominator=1728, rows=rows,
        fixed_metrics=['status','plan_available','R_groups','eligible_counts','27_axis_caps_attempted_completed','AD_primal_predict_API','Jacobian_elements','native_calls','wall_ns'],
        interpretation=dict(common_cap='same vector caps within K, not equal actual spend',
            timing='inclusive nested operation times are not additive or FLOPs',
            unknown=['total_FLOPs','scalar_backward_calls','cuda_ns'],
            outcome_selection=False, cross_K_aggregation=False, partial_plan_as_complete=False,
            modelQ_or_physical_benefit=False, full_C3_demonstrated=False, physical_release=False),
        trust='inherits externally SHA-pinned MAIN strict receipt; no independent full reconstruction')


def build(path, expected):
    for file, pin in PINS.items():
        need(sha(file.read_bytes()) == pin, 'frozen source/protocol SHA: '+str(file))
    return analyze_receipt(read_pinned(path, expected), expected)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('analyze','verify'))
    p.add_argument('--receipt', required=True); p.add_argument('--receipt-sha', required=True)
    p.add_argument('--analysis', required=True); p.add_argument('--analysis-sha')
    a = p.parse_args(argv); result = build(a.receipt, a.receipt_sha); path = Path(a.analysis)
    if a.command == 'analyze':
        need(a.analysis_sha is None and path.parent.resolve() == ROOT/'results' and path.name.startswith(NAME), 'exclusive output')
        raw = canonical(result)
        with path.open('xb') as f: f.write(raw)
        print(json.dumps(dict(path=str(path.absolute()), sha256=sha(raw))))
    else:
        need(a.analysis_sha is not None and read_pinned(path, a.analysis_sha) == result, 'entire analysis reconstruction')
        print(json.dumps(dict(verified=True, analysis_sha256=a.analysis_sha)))


if __name__ == '__main__': main()
