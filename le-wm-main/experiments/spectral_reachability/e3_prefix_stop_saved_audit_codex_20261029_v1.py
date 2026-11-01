"""Audit recorded E3 prefix-stop terminal consistency, not native full strict.

This is a saved-only addition to the E3 transport and native semantics audits.
It does not replay physics, authenticate model/source guards, or authorize runs.
"""
import argparse
import json
import math
from pathlib import Path
import struct

from experiments.spectral_reachability import agent_e3_native_saved_semantics_20261029_v1 as native
from experiments.spectral_reachability import agent_e3_terminal_row_20261029_v1 as terminal

need = native.need
pub = native.pub


def raw_bits(raw):
    need(type(raw) is list and len(raw) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in raw), 'finite raw pair')
    try:
        return struct.pack('<ff', *raw).hex()
    except (OverflowError, struct.error) as exc:
        raise ValueError('raw float32 range') from exc


def check(events, row, episode):
    """Check prefix-stop terminal against actual recorded return and observation."""
    need(type(events) is list and type(row) is dict and row.get('episode_id') == episode, 'episode')
    def selected(resource):
        return [e for e in events if e['origin'] == 'physical' and e['resource'] == resource]
    dispatch = selected('native_authorized_dispatch')
    returns = selected('native_control_return')
    observations = selected('native_fresh_observation')
    n = row.get('control_returned')
    need(type(n) is int and 1 <= n <= 5 and len(dispatch) == len(returns) == n, 'returned prefix count')
    need(row.get('status') == 'native_stopped_prefix' and row.get('warmup_returned') == 10, 'prefix status')
    need(type(row.get('terminated')) is bool and type(row.get('truncated')) is bool and
         (row['terminated'] or row['truncated']), 'native stop flags')
    actions = row.get('executed_raw')
    need(type(actions) is list and len(actions) == n, 'record actual raw prefix')
    for i, (d, r, action) in enumerate(zip(dispatch, returns, actions, strict=True), 1):
        x, y = d['detail'], r['detail']
        need(x['episode_id'] == y['episode_id'] == episode and x['step'] == y['step'] == i and
             d['index'] < r['index'] and x['raw_bits'] == y['raw_bits'] == raw_bits(action) and
             raw_bits(x['command']) == raw_bits(y['command']) == raw_bits(action), 'dispatch/return/raw identity')
        need(type(y['terminated']) is bool and type(y['truncated']) is bool, 'literal stop flags')
        need(not (y['terminated'] or y['truncated']) if i < n else
             (y['terminated'] == row['terminated'] and y['truncated'] == row['truncated']), 'first stop preserved')
    last = row.get('last_observation')
    matching = [e for e in observations if e['detail'].get('episode_id') == episode and
                e['detail'].get('receipt') == last]
    need(all(e['detail']['receipt'].get('phase') != 'plan' or
             (type(e['detail']['receipt'].get('step')) is int and e['detail']['receipt']['step'] <= n)
             for e in observations), 'no future plan observation')
    need(type(last) is dict and len(matching) == 1 and last.get('phase') == 'plan' and last.get('step') == n and
         dispatch[-1]['index'] < matching[0]['index'] < returns[-1]['index'] and
         last.get('observed_event_index') == matching[0]['index'], 'last actual observation receipt')
    need(row.get('final_state') == last.get('state'), 'final state from actual observation')
    success = row.get('native_success')
    metrics = row.get('final_metrics')
    need(type(success) is bool and type(metrics) is dict and metrics.get('native_success') is success,
         'reported native success consistency; not recomputed')
    checkpoints = row.get('checkpoints')
    need(type(checkpoints) is list and all(type(c) is dict and c.get('phase') == 'plan' and
         type(c.get('step')) is int and 0 <= c['step'] <= n for c in checkpoints), 'no future checkpoints')
    need(not selected('native_e3_encode_intent') and not any(e['origin'] in ('encoding', 'numerical') for e in events),
         'no feedback or numerical events after prefix stop')
    return dict(episode=episode, terminal_status=row['status'], returned_prefix=n,
                terminated=row['terminated'], truncated=row['truncated'], recorded_native_success=success,
                feedback_boundaries_completed=0, native_success_independently_recomputed=False,
                no_future_checkpoints_in_terminal=True, full_saved_strict=False, scientific_admission=False)


def audit(path, report_sha, completion_sha, episode):
    p = Path(path)
    transport = pub.consume(p, report_sha, completion_sha)
    need(transport['complete'], 'complete transport required')
    report = pub.base.read_bytes(p/'report.json', pub.FINAL_CAP)
    journal = pub.base.read_bytes(p/'events.jsonl', pub.base.CAP)
    need(pub.base.fsha(report) == report_sha and pub.base.fsha(journal) == pub.base.decode(report)['journal_sha256'], 'publication binding')
    events = [pub.base.decode(line) for line in journal.splitlines()]
    semantic = native.audit(events, episode, True)
    value = check(events, terminal.read(events), episode)
    need(semantic['cleanup'] == 'returned' and semantic['native_stop_recorded'], 'existing native semantics')
    need(pub.base.fsha(pub.base.read_bytes(p/'events.jsonl', pub.base.CAP)) == pub.base.fsha(journal) and
         pub.base.fsha(pub.base.read_bytes(p/'report.json', pub.FINAL_CAP)) == report_sha and
         pub.base.fsha(pub.base.read_bytes(p/'completion.json', pub.FINAL_CAP)) == completion_sha, 'source drift')
    return dict(schema='e3_prefix_stop_saved_audit_codex_20261029_v1.audit', report_sha256=report_sha,
                completion_sha256=completion_sha, transport_verified=True, source_guard_reverified=False,
                physics_replayed=False, neural_executed=False, result=value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('output', 'report-sha256', 'completion-sha256', 'episode'):
        p.add_argument('--'+field, required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.output, a.report_sha256, a.completion_sha256, a.episode), sort_keys=True))


if __name__ == '__main__':
    main()
