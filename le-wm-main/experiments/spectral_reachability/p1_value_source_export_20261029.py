"""Read-only P1 exports, gated by the existing saved-only strict consumers."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import agent_p1_pair_margin_decomposition_20261028_v1 as fixed
from experiments.spectral_reachability import agent_p1_joint_vreach_authority_prepare_20261029_v1 as joint
from experiments.spectral_reachability import agent_p1_joint_vreach_saved_audit_20261029_v1 as arm_audit

FIXED_SHA = 'f723c5eb73ecebd0615a2ded0aa4b741c6f132ca5f6063c673f12d5e98539143'
JOINT_SHA = '997289d3ca4ae173f5ece36e7c6a54fb021a306d99b8f07222b747df3fd148bd'
FIXED_PATH = fixed.ROOT / 'results/agent_p1_pair_margin_decomposition_20261028_v1_MAIN01.json'
JOINT_PATH = fixed.ROOT / 'results/agent_p1_joint_vreach_authority_prepare_20261029_v1_MAIN01.json'


def missing(reason):
    return {'status': 'missing', 'reason': reason, 'value': None}


def present(value, source, step):
    return {'status': 'available', 'value': value, 'source': source, 'absolute_step': step}


def fixed_rows(rebuilt, sealed):
    fixed.require(len(rebuilt['slots']) == len(sealed['candidates']) == 40, 'fixed 40')
    rows = []
    for i, (s, candidate) in enumerate(zip(rebuilt['slots'], sealed['candidates'], strict=True)):
        p = candidate['original_plan']
        fixed.require(s['index'] == i and s['plan_digest'] == p['plan_digest'], 'fixed plan join')
        prefix = fixed.literal(p['prefix_commands'], (5, 2)).tobytes()
        suffix = fixed.literal(p['suffix_commands'], (5, 2)).tobytes()
        fixed.require(hashlib.sha256(prefix).hexdigest() == s['prefix_sha256'] and
                      hashlib.sha256(suffix).hexdigest() == s['suffix_sha256'] and
                      (prefix + suffix).hex() == candidate['full10_raw_bits'], 'fixed action bits')
        rows.append(dict(index=i, identity={k: s[k] for k in ('task_id', 'anchor_id', 'context_sha256',
            'goal_sha256', 'plan_digest', 'job_digest', 'actual_return_event_sha256')},
            prefix_hex=prefix.hex(), suffix_hex=suffix.hex(),
            root_step=10, prefix_end_step=15, terminal_step=20,
            immediate=missing('no authenticated per-candidate prefix score in fixed40 consumer'),
            suffix=present(s['predicted_Q'], 'strict fixed40 predicted terminal identity192 MSE', 20),
            vreach=missing('no authenticated conditional Vreach in fixed40 consumer'),
            physical=present(s['physical_metrics'], 'strict fixed40 original native return', 20),
            observed_terminal=(present(s['actual_Q'], 'strict fixed40 actual terminal identity192 MSE', 20)
                if s['actual_Q'] is not None else missing('original terminal latent absent'))))
    return rows


def export_fixed():
    rebuilt = fixed.build()
    fixed.verify(FIXED_PATH, FIXED_SHA, rebuilt)
    old = fixed.upstream()
    sealed = old.teacher.read(old.SELECTOR, old.SELECTOR_SHA, fixed.AUTHORITY_CAP)
    return dict(source='fixed40', strict='full fixed40 saved reconstruction',
                source_sha256=FIXED_SHA, rows=fixed_rows(rebuilt, sealed), scientific_admission='unchanged')


def prepared_rows(prepared):
    rows = []
    for i, row in enumerate(prepared['authority']['rows']):
        q = row['request']
        fixed.require(q['slot'] == i, 'joint prepared slot')
        joint.method.original.request(q)
        rows.append(dict(slot=i, root_slot=q['root_slot'], condition=q['condition'],
            task_id=q['task']['binding']['task_id'], winner_authority=q['winner_authority'],
            context_digest=joint.core.digest(q['task']['root_context']),
            goal_digest=joint.core.digest(q['task']['goal']), winner_raw_bits=q['winner_raw_bits'],
            immediate=missing('prepare contains no completed joint arm'),
            suffix=missing('prepare contains no completed joint arm'),
            vreach=missing('prepare contains no completed joint arm'),
            physical=missing('no joint physical outcome in prepare')))
    fixed.require(len(rows) == 4, 'joint fixed four')
    return rows


def export_joint_prepare():
    prepared = joint.verify(str(JOINT_PATH), JOINT_SHA)
    return dict(source='joint_prepare', strict='full original winner authority reconstruction',
                source_sha256=JOINT_SHA, rows=prepared_rows(prepared), scientific_admission='unchanged')


def export_joint_arm(prepared, bundle):
    """Audit a completed arm against strict prepared request; no journal origin attestation."""
    fixed.require(set(bundle) == {'slot', 'arm', 'result', 'scalar_receipt', 'events'}, 'arm bundle keys')
    slot = bundle['slot']
    fixed.require(type(slot) is int and 0 <= slot < 4, 'arm slot')
    q = prepared['authority']['rows'][slot]['request']
    summary = arm_audit.audit(q, bundle['arm'], bundle['result'], bundle['scalar_receipt'], bundle['events'])
    result = bundle['result']
    fixed.require(len(result['candidates']) == 20, 'complete candidates')
    rows = []
    for c in result['candidates']:
        rows.append(dict(slot=slot, arm=bundle['arm'], start=c['start'], iteration=c['iteration'],
            raw_float32_bits=c['raw_float32_bits'], parent_context_digest=joint.core.digest(c['mid']),
            immediate=present(c['stage_Q'], 'joint audited method replay', None),
            suffix=present(c['terminal_Q'], 'joint audited method replay', None),
            vreach=present(c['own_reach_value'], 'joint audited conditional signal and ridge', None),
            physical=missing('joint arm audit has no physical outcome')))
    return dict(source='joint_completed_arm', strict='prepared request plus complete saved arm replay',
                journal_origin='not authenticated by arm audit', summary=summary, rows=rows,
                scientific_admission='unchanged')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', choices=('fixed40', 'joint-prepare', 'joint-arm'))
    p.add_argument('--bundle', help='completed arm JSON, externally acquired')
    p.add_argument('--bundle-sha', help='expected SHA256 of complete bundle')
    a = p.parse_args()
    if a.source == 'fixed40': out = export_fixed()
    elif a.source == 'joint-prepare': out = export_joint_prepare()
    else:
        fixed.require(a.bundle and a.bundle_sha, 'bundle and expected sha required')
        bundle = fixed.read(Path(a.bundle), a.bundle_sha, 512 * 1024**2)
        prepared = joint.verify(str(JOINT_PATH), JOINT_SHA)
        out = export_joint_arm(prepared, bundle)
        out['bundle_sha256'] = a.bundle_sha
    print(json.dumps(out, sort_keys=True, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__': main()
