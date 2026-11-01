"""Full-commit P3 preparation and RAM control seam; no production worker loader.

The existing first-five physical release cannot satisfy this protocol. This
module deliberately cannot turn a callback/mock receipt into CUDA admission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core

NAME = 'agent_p3_prospective_equal_horizon_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
MODES = ('fixed', 'adaptive')
TASK_AUTHORITY = ROOT/'results/agent_p3_paired_quota_runtime_20261028_v1_prepare_MAIN01.json'
TASK_AUTHORITY_SHA = '46e191695e6e121324daf691bafe1b2862a36349e11232822957d8ca6785e6d8'
FLAGS = dict(P1_admission=False, scientific_admission=False, C3_demonstrated=False,
             freshness=False, production_ready=False)
REUSE = (
    'agent_p3_runtime_contract_20261028_v1.py',
    'agent_p3_child_signal_cuda_20261028_v1.py',
    'agent_p3_child_program_cuda_20261028_v1.py',
    'agent_p3_paired_quota_runtime_20261028_v1_compiler.py',
    'agent_p3_paired_quota_physical_handoff_20261028_v1.py',
    'agent_p3_child_scheduler_contract_20261028_v1.py',
    'agent_p4_moving_multiboundary_20261028_v1.py',
    'p1_moving_live_feedback_physical_20261027_v1.py',
)
BLOCKERS = (
    'NativeChildSignal/NativeProgram validate old saved child roster: need reviewed live-context adapter, not forged ancestry',
    'paired compiler execute_branch releases raw5 only: need full raw(5*H) release, whole/segmented and selected full-H replay consumer',
    'reconfiguration/composition compiler not certified by existing progress-only compiler; retain refusal, no silent fallback',
    'P4 RollingSuffix.compile_suffix is H1-only; cannot substitute it for H3',
    'need MAIN-certified stateful native worker and separate CUDA worker with durable predispatch hook IPC; existing inspected P3/P4 modules do not implement this live IPC',
    'need actual per-prefix source/goal/warm10/history/RGB gates and native attempt-return profile/strict consumer for this new full-commit design',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


canonical = core.canonical
digest = core.digest
sha = core.sha


def protocol():
    return dict(schema=NAME+'.protocol', public_horizon=20, common_prefix_steps=5,
        remaining_steps=15, warmup_steps=10, modes=list(MODES),
        execution='full_commit_5H; never selected_first5 for H3',
        feedback='own actual RGB at 5/10/15/20; paid own local H1/H3 signal at 5/10/15, including inside committed H3',
        commitment='schedule only when previous full program ended; intermediate feedback does not truncate/replace committed actions',
        terminal='observe/encode/metrics only; no terminal scheduling signal',
        tail='refuse whole node if 5*H exceeds remaining; no shortening, padding, forced H1 or rescheduling',
        scheduler='core.choose depth=0 at each physical decision; not a recursive C3 tree',
        natural_candidates=[4, 8], quota_override=False,
        signal='own fresh H1/H3 nominal+AD-forward, two J, 384 scalar VJPs and two array SVD per decision',
        compiler='original progress raw-coordinate bounded-ridge/Q; independent whole and segmented candidate paths; selected full-H replay',
        normalization='saved float32 subtract/divide; no /100, clipping, tanh or padding',
        comparison='equal intended public20 only; equal realized horizon requires every slot complete',
        FLOPs=None, total_compute=None, walltime_equivalence=None,
        **FLAGS)


def predict_axes(calls):
    out = dict.fromkeys(core.AXES, 0)
    out.update(predict_API=calls, predict_samples=calls, predict_context_tokens=3*calls,
        action_encoder_calls=calls, action_encoder_samples=calls, action_encoder_context_tokens=3*calls,
        predictor_calls=calls, predictor_samples=calls, predictor_context_tokens=3*calls,
        pred_proj_calls=calls, pred_proj_rows=3*calls)
    return out


def program_caps(h, n):
    require(type(h) is int and h in (1, 3) and type(n) is int and n in (4, 8), 'natural H/N only')
    out = predict_axes(h*(2*n+1))
    out.update(array_SVD_calls=1, solve_calls=1, program_compile_calls=1,
               replay_calls=2*n+1, metric_calls=n)
    return out


def arm_caps(mode):
    require(mode in MODES, 'fixed/adaptive')
    # Sum committed H <= 3, <=3 decisions; n<=4 fixed / <=8 adaptive.
    n = 4 if mode == 'fixed' else 8
    out = predict_axes(3*8 + 3*(2*n+1))
    out.update(encoder_calls=4, encoder_samples=4, projector_calls=4, projector_samples=4,
        jacobian_API=6, VJP_directions=1152, grad_API=1152, raw_backward_callbacks=1152,
        array_SVD_calls=9, solve_calls=3, signal_compile_calls=6,
        program_compile_calls=3, replay_calls=3*(2*n+1), metric_calls=3*n, schedule_decisions=3)
    return out


def validate_sources(sources):
    require(type(sources) is list and len(sources) == 2, 'exact two known-root sources')
    require(sha(TASK_AUTHORITY) == TASK_AUTHORITY_SHA, 'sealed original task authority')
    tasks = core.read(TASK_AUTHORITY)['wiring']['tasks']
    ids = set()
    for s in sources:
        core.keys(s, ('source_id', 'root_slot', 'task', 'recipe_sha256', 'prefix_sha256', 'authority_sha256'))
        require(type(s['source_id']) is str and s['source_id'] and s['source_id'] not in ids, 'unique source id')
        ids.add(s['source_id'])
        require(type(s['root_slot']) is int and s['root_slot'] in (0, 1), 'known two roots only')
        t = core.validate_task(s['task'])
        require(t == tasks[s['root_slot']], 'exact original task/root/Q/normalizer, no new root')
        require(len(t['goal']) == 192, 'original native latent192')
        for k in ('recipe_sha256', 'prefix_sha256', 'authority_sha256'):
            core.hash_value(s[k])
    require({s['root_slot'] for s in sources} == {0, 1}, 'one source per original root')
    return deepcopy(sources)


def material(sources):
    sources = validate_sources(sources)
    paths = [Path(__file__), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md', TASK_AUTHORITY]
    paths += [ROOT/'experiments/spectral_reachability'/p for p in REUSE]
    return dict(schema=NAME+'.prepare', status='blocked_missing_live_full_H_workers',
        protocol=protocol(), sources=sources,
        slots=[dict(source_id=s['source_id'], mode=m, source_sha256=digest(s),
            caps=arm_caps(m), status='not_started') for s in sources for m in MODES],
        fixed_denominator=len(sources)*2, blockers=list(BLOCKERS),
        file_sha256={str(p.resolve()): sha(p) for p in paths},
        native_caps=None, native_actual=None, inherited_acquisition_cost=None,
        wall_seconds_cap=900, output_bytes_cap=256*1024**2, disk_floor_bytes=200*1024**2,
        FLOPs=None, **FLAGS)


def write_new(path, value):
    p = Path(path).absolute()
    require(p.parent == ROOT/'results' and p.name.startswith(NAME+'_'), 'exclusive new result stem')
    raw = canonical(value)
    require(len(raw) <= 8*1024**2, 'bounded preparation')
    with p.open('xb') as f:
        require(f.write(raw) == len(raw), 'short write')
        f.flush()
        os.fsync(f.fileno())
    return dict(path=str(p), sha256=sha(p))


def verify(path, expected):
    core.hash_value(expected)
    require(sha(path) == expected, 'external actual prepare SHA')
    p = core.read(path)
    authority = p.pop('roster_authority')
    core.keys(authority, ('path', 'sha256'))
    require(sha(authority['path']) == authority['sha256'] and
            core.read(authority['path']) == p['sources'], 'external roster authority')
    require(p == material(p['sources']), 'exact declaration/source closure; no hand release')
    return p


def run(prepared, *, authorization=None):
    # No callable registration or arbitrary module import can elevate this prep.
    require(prepared.get('protocol') == protocol(), 'exact prospective protocol')
    raise ValueError('production blocked: '+ '; '.join(BLOCKERS))


def observed(value, source, mode, boundary):
    core.keys(value, ('source_sha256', 'mode', 'boundary', 'context', 'rgb_sha256',
        'pixel_sha256', 'encoding_sha256', 'raw_history10', 'observed_index', 'encoded_index'))
    require(value['source_sha256'] == digest(source) and value['mode'] == mode and
            type(value['boundary']) is int and value['boundary'] == boundary, 'own actual feedback identity')
    c = core.context(value['context'])
    raw = core.finite(value['raw_history10'], (10, 2), True).astype(np.float32)
    mean, std = (np.asarray(source['task'][k], np.float32) for k in ('action_mean', 'action_std'))
    hist = ((raw-mean)/std).reshape(2, 10)
    require(np.array_equal(hist, np.asarray(c['action_history'], np.float32)), 'exact actual raw/token history')
    for k in ('rgb_sha256', 'pixel_sha256', 'encoding_sha256'):
        core.hash_value(value[k])
    require(type(value['observed_index']) is int and type(value['encoded_index']) is int and
            0 <= value['observed_index'] < value['encoded_index'], 'observe before encode')
    return deepcopy(value)


class RamDriver:
    """Executable orchestration tests ONLY, not a trusted tensor/native runner.

    Callbacks receive separate native/model arguments. A real worker must enforce
    these in separate interpreters and validate full saved arrays, not fake hooks.
    """
    def __init__(self, sources, backend, sink):
        self.sources = validate_sources(sources)
        require(getattr(backend, 'kind', None) == 'RAM_FIXTURE', 'RAM-only seam, no production callback elevation')
        require(callable(sink), 'explicit event sink')
        self.backend, self.sink = backend, sink
        self.rows = [dict(source_id=s['source_id'], mode=m, status='not_started',
            actual_public_steps=0, rounds=[], metrics=[], errors=[], resources=None,
            native_actual=None, FLOPs=None) for s in self.sources for m in MODES]
        self.sink(dict(kind='denominator', rows=deepcopy(self.rows), evidence='RAM_ONLY'))

    def execute(self):
        for row in self.rows:
            source = next(s for s in self.sources if s['source_id'] == row['source_id'])
            mode = row['mode']
            ledger = core.Ledger(arm_caps(mode), sink=self.sink)
            ledger.scope = dict(source_id=source['source_id'], mode=mode, boundary=5)
            try:
                # start must fresh reset/warm10/prefix5 through exact native gates.
                initial = self.backend.start(deepcopy(source), mode)
                row['actual_public_steps'] = 5
                obs = self._encode(initial, source, mode, 5, ledger)
                history = deepcopy(obs['raw_history10'])
                pending = []
                committed = None
                while row['actual_public_steps'] < 20:
                    b = row['actual_public_steps']
                    ledger.scope = dict(source_id=source['source_id'], mode=mode, boundary=b)
                    node = dict(context=deepcopy(obs['context']), depth=0)
                    backend = self.backend.local(deepcopy(source['task']), deepcopy(obs))
                    before = deepcopy(ledger.attempted)
                    signals, receipt = core.local_signal(source['task'], node, backend, ledger, core.policy(source['task'], node))
                    r = dict(boundary=b, observation_sha256=digest(obs), signal=receipt,
                        signal_attempted={k: ledger.attempted[k]-before[k] for k in core.AXES},
                        decision=None, status='paid_feedback', request=None, program=None)
                    row['rounds'].append(r)
                    if not pending:
                        decision = ledger.call('schedule', lambda: core.choose(source['task'], node, signals, mode))
                        r['decision'] = decision
                        h, n = decision['horizon'], decision['candidates']
                        if b+5*h > 20:
                            r['status'] = row['status'] = 'refused_full_H_exceeds_remaining'
                            break
                        if decision['extension_type'] != 'progress' or not decision['continue_expanding']:
                            r['status'] = row['status'] = 'refused_uncertified_program_type'
                            break
                        req = dict(source_sha256=digest(source), binding=deepcopy(source['task']['binding']),
                            observation=deepcopy(obs), decision=deepcopy(decision), signal_sha256=digest(receipt),
                            local_response=deepcopy(receipt['responses']))
                        r['request'] = deepcopy(req)
                        old = deepcopy(ledger.attempted)
                        old_completed = deepcopy(ledger.completed)
                        pc = program_caps(h, n)
                        require(all(old[k]+pc[k] <= ledger.caps[k] for k in core.AXES), 'whole-node predispatch cap')
                        plan = self.backend.compile(deepcopy(req), ledger)
                        core.keys(plan, ('request_sha256', 'actions', 'full_replay_sha256'))
                        require(plan['request_sha256'] == digest(req), 'sealed current local request')
                        core.hash_value(plan['full_replay_sha256'])
                        raw = core.finite(plan['actions'], (5*h, 2), True)
                        require(np.max(np.abs(raw)) <= 1, 'original raw action domain')
                        require(all(ledger.attempted[k]-old[k] == pc[k] for k in core.AXES), 'full-H metering, not prefix-only replay')
                        require(not ledger.failed and all(ledger.completed[k]-old_completed[k] == pc[k] for k in core.AXES), 'full compiler returns before physical dispatch')
                        r['program'] = deepcopy(plan)
                        self.sink(dict(kind='sealed_full_program', row=deepcopy(r)))
                        pending = deepcopy(plan['actions'])
                        committed = r
                    else:
                        r['status'] = 'paid_feedback_inside_full_commit'
                    committed['status'] = 'executing'
                    chunk, pending = pending[:5], pending[5:]
                    for i, action in enumerate(chunk):
                        # No outcome/native state is ever supplied to model callbacks.
                        step = self.backend.step(deepcopy(action), b+i+1)
                        core.keys(step, ('done', 'metric', 'action_sha256'))
                        require(type(step['done']) is bool and step['action_sha256'] == digest(action), 'actual transition/action identity')
                        row['actual_public_steps'] += 1
                        history = history[1:]+[deepcopy(action)]
                        row['metrics'].append(deepcopy(step))
                        if step['done']:
                            row['status'] = r['status'] = 'native_early_stop'
                            break
                    if row['status'] == 'native_early_stop':
                        break
                    if not pending:
                        committed['status'] = 'full_program_returned'
                    previous = obs
                    obs = self._encode(self.backend.observe(), source, mode, row['actual_public_steps'], ledger)
                    require(obs['raw_history10'] == history, 'feedback history must be own actually executed actions')
                    require(obs['observed_index'] > previous['encoded_index'], 'fresh observation after previous feedback')
                if row['actual_public_steps'] == 20 and row['status'] != 'native_early_stop':
                    row['status'] = 'complete'
            except Exception as exc:
                row['status'] = 'engineering_failure'
                row['errors'].append(type(exc).__name__+': '+str(exc))
            finally:
                try:
                    self.backend.close()
                except Exception as exc:
                    row['status'] = 'engineering_failure'
                    row['errors'].append('close: '+str(exc))
                row['resources'] = ledger.receipt()
        return dict(schema=NAME+'.ram_result', rows=deepcopy(self.rows), fixed_denominator=len(self.rows),
            complete_public_horizon=all(r['status'] == 'complete' for r in self.rows),
            evidence='RAM_ONLY_NOT_SCIENCE', FLOPs=None, **FLAGS)

    def _encode(self, frame, source, mode, boundary, ledger):
        ledger.scope = dict(source_id=source['source_id'], mode=mode, boundary=boundary)
        before = deepcopy(ledger.attempted)
        value = self.backend.encode(deepcopy(frame), ledger)
        expected = dict.fromkeys(core.AXES, 0)
        expected.update(encoder_calls=1, encoder_samples=1, projector_calls=1, projector_samples=1)
        require(not ledger.failed and all(ledger.attempted[k]-before[k] == expected[k] for k in core.AXES), 'paid own single RGB encoder/projector; no hidden predict or batching')
        return observed(value, source, mode, boundary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--sources', required=True, help='MAIN predeclared outcome-free source roster JSON')
    a.add_argument('--sources-sha', required=True)
    a.add_argument('--output', required=True)
    for name in ('verify', 'run'):
        a = sub.add_parser(name)
        a.add_argument('--prepared', required=True)
        a.add_argument('--prepare-sha', required=True)
    args = p.parse_args()
    if args.command == 'prepare':
        require(sha(args.sources) == args.sources_sha, 'external roster SHA')
        value = material(core.read(args.sources))
        value['roster_authority'] = dict(path=str(Path(args.sources).resolve()), sha256=args.sources_sha)
        # Roster binding is checked separately; material remains deterministic.
        print(json.dumps(write_new(args.output, value)))
    else:
        value = verify(args.prepared, args.prepare_sha)
        if args.command == 'run':
            run(value)
        print(json.dumps(dict(status=value['status'], **FLAGS)))


if __name__ == '__main__':
    main()
