"""Live v2: immutable v1 orchestration, robust cleanup and circular diagnostics.

Saved prepare/verify and mocks only for agent use. MAIN exclusively owns run.
No old source/result mutations; the existing full v1 closure is hash-verified,
not recopied. This module owns its run/finalization, never patches v1 globals.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import signal
import numpy as np
from experiments.spectral_reachability import p4_feedback_live_cuda_20261027_v1 as v1

NAME = 'p4_feedback_live_cuda_20261027_v2'
REPO, io, require = v1.REPO, v1.io, v1.require
RESERVE, OUTPUT_CAP, PREPARE_CAP = v1.RESERVE, v1.OUTPUT_CAP, v1.PREPARE_CAP
PARENT = REPO / 'results/p4_feedback_live_cuda_20261027_v1_prepare02'
PARENT_SHA = '2762a6d9ee3107da0d27519d5a37aba8c8378b11e615cb6ec4526d81ef548481'
caps = v1.caps


def semantics():
    return dict(predict_unit='model.predict API entry, NOT model.predictor.forward entry or FLOPs',
        internal_predictor_entries_instrumented=False,
        action_encoder_unit='model.action_encoder.forward entry',
        bypass_policy='sealed Native callgraph uses model.predict; API hooks alone are not a universal direct-predictor bypass detector',
        overlap=dict(rtol=0., atol=2e-5, meaning='batch-encoder numerical consistency only; NOT bitwise equality or exact observed value identity',
                     exact_identity='own live uint8 RGB pixel SHA; original root whole-stack SHA; native array SHA separately'),
        terminal=dict(native_official='installed evaluator record only; legacy angle bug may produce false positives',
                      diagnostic='independent abs(atan2(sin(delta),cos(delta))); no planner input or scientific admission'),
        cleanup='append journal-close error; preserve first_error; any close failure prohibits completion')


def revision_sources():
    return [Path(__file__).resolve(), REPO / 'tests' / ('test_' + NAME + '.py'), REPO / 'innerdocs' / (NAME + '.md')]


def material(limits):
    parent = v1.verify(PARENT, PARENT_SHA)
    require(limits == parent['limits'], 'engineering caps unchanged from sealed v1')
    p = deepcopy(parent)
    p.update(schema=NAME + '.protocol', parent_prepared=str(PARENT), parent_prepare_sha256=PARENT_SHA,
             parent_protocol_sha256=io.sha(PARENT / 'protocol.json'), semantics=semantics(),
             source_closure_policy='verify existing parent seals plus archive only explicit v2 increment')
    p['revision_sources_sha256'] = {str(f): io.sha(f) for f in revision_sources()}
    return p


def prepare(output, limits):
    output = Path(output).absolute()
    require(not output.exists() and shutil.disk_usage(output.parent).free >= RESERVE + PREPARE_CAP + OUTPUT_CAP,
            'NEW prepare / full8+16MiB headroom above200MiB floor')
    p = material(limits); files = revision_sources()
    require(sum(f.stat().st_size for f in files) <= 256 * 1024, 'small v2 source increment')
    store = v1.Store(output, PREPARE_CAP); store.write('protocol.json', p)
    seal = io.create_seal(files=files, args_json=io.io.canonical(dict(protocol_sha256=io.sha(output / 'protocol.json'),
        parent_prepare_sha256=PARENT_SHA)), output_dir=output / 'revision_source_seal')
    store.write('prepare.json', dict(schema=NAME + '.prepare', status='prepared_not_run',
        protocol_sha256=io.sha(output / 'protocol.json'), parent_prepare_sha256=PARENT_SHA, revision_source_seal=seal))
    return io.sha(output / 'prepare.json')


def verify(folder, expected):
    folder = Path(folder).resolve(); require(io.sha(folder / 'prepare.json') == expected, 'external v2 prepare SHA')
    r = io.read(folder / 'prepare.json'); p = io.read(folder / 'protocol.json')
    require(r['schema'] == NAME + '.prepare' and r['status'] == 'prepared_not_run' and
            r['parent_prepare_sha256'] == PARENT_SHA and io.sha(folder / 'protocol.json') == r['protocol_sha256'], 'v2 prepare identity')
    seal = r['revision_source_seal']
    io.verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    require(io.read(Path(seal['output_dir']) / 'args.json') == dict(protocol_sha256=r['protocol_sha256'], parent_prepare_sha256=PARENT_SHA), 'v2 seal args')
    require(io.same(material(p['limits']), p), 'parent full source closure plus exact v2 reconstruction')
    return p


def wrapped_diagnostic(state, goal, native):
    """Independent evaluator bookkeeping, never policy selection or admission."""
    state = np.asarray(state, float); goal = np.asarray(goal, float)
    require(state.shape == goal.shape == (7,) and np.isfinite([state, goal]).all(), 'finite terminal state/goal')
    raw = float(abs(state[4] - goal[4])); angle = float(abs(math.atan2(math.sin(state[4]-goal[4]), math.cos(state[4]-goal[4]))))
    full_position = float(np.linalg.norm(state[:4] - goal[:4])); object_position = float(np.linalg.norm(state[2:4] - goal[2:4]))
    wrapped_success = bool(full_position < 20 and angle < math.pi/9)
    return dict(raw_absolute_angle_delta=raw, wrapped_angle_error=angle,
        native_min_angle_domain_valid=raw <= 2*math.pi, native_reported_angle_error=float(native['angle_error']),
        native_reported_success=bool(native['official_success']),
        wrapped_official_position_error=full_position, wrapped_object_position_error=object_position,
        wrapped_official_success=wrapped_success, wrapped_object_only_success=bool(object_position < 20 and angle < math.pi/9),
        native_wrapped_success_agree=bool(native['official_success']) == wrapped_success,
        controls_actions=False, changes_goal_Q=False, P1_admission=False, P4_admission=False)


class Physical(v1.Physical):
    def evaluate(self):
        result = super().evaluate()
        # Deliberately retain native labels, including a potential historical bug,
        # under an explicit name. Never relabel them as robust terminal success.
        native = result.pop('official')
        result.update(official_native=native, wrapped_diagnostic=wrapped_diagnostic(result['state'], result['goal'], native),
                      official_native_scope='installed evaluator record, not circular-success validation')
        return result


def cleanup(stage, callback, errors):
    try:
        callback(); return True
    except BaseException as exc:
        errors.append(io.error(stage, exc)); return False


def finish_journal(journal, complete, errors):
    if not cleanup('final_journal', lambda: journal(dict(event='run_finalizing', pre_close_complete=complete, errors=deepcopy(errors))), errors):
        complete = False
    if not cleanup('journal_close', journal.close, errors): complete = False
    return complete


def run(folder, expected, output):
    p = verify(folder, expected); output = Path(output).absolute()
    require(not output.exists() and shutil.disk_usage(output.parent).free >= RESERVE + OUTPUT_CAP, 'NEW run /200MiB floor plus16MiB output')
    store = v1.Store(output); store.write('execution_binding.json', dict(prepared=str(Path(folder).resolve()),
        prepare_sha256=expected, parent_prepare_sha256=PARENT_SHA))
    journal = v1.response.Journal(store); ledger = v1.Ledger(p['limits'], journal); rows = []; errors = []
    hooks = io.Hooks(); model = env = None; before = after = host = None; old_alarm = None; armed = False; source_ok = False
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'MAIN dedicated process; no existing alarm')
        def deadline(signum, frame): raise TimeoutError('live wall cap')
        old_alarm = signal.signal(signal.SIGALRM, deadline); signal.setitimer(signal.ITIMER_REAL, p['limits']['wall_seconds']); armed = True
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
        host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0'); torch.cuda.reset_peak_memory_stats(device)
        ledger.sync = lambda: torch.cuda.synchronize(device)
        ledger.sample = lambda: (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
        total = torch.cuda.get_device_properties(device).total_memory
        require(p['limits']['reserved_bytes'] <= total, 'reserved cap exceeds device')
        torch.cuda.set_per_process_memory_fraction(p['limits']['reserved_bytes'] / total, device)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        policy = Path(next(x for x in p['model_bindings'] if Path(x).name == 'weights.pt')).parent
        model = load_compatible_pretrained(str(policy), device=device).eval().requires_grad_(False)
        io.frozen(model); require(all(b.device == device for b in model.buffers()), 'CUDA model buffers')
        before = io.tensor_hash(model); require(before == p['expected_model_hash'], 'same actual frozen response model state')
        v1.install_hooks(torch, model, ledger, hooks)
        env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
        require(env.unwrapped.variation_space['agent']['color'].value.tolist() == [65, 105, 225], 'exact original visible agent color')
        physical = Physical(p, env, ledger, store, base)
        with sdpa_kernel(SDPBackend.MATH): v1.workflow(p, v1.Native(torch, model, device, p), physical, ledger, store, rows)
    except BaseException as exc: errors.append(io.error('live_execution', exc))
    finally:
        elapsed = ledger.clock() - ledger.started
        if armed: cleanup('alarm_disable', lambda: signal.setitimer(signal.ITIMER_REAL, 0), errors)
        if old_alarm is not None: cleanup('alarm_restore', lambda: signal.signal(signal.SIGALRM, old_alarm), errors)
        cleanup('hooks_restore', lambda: hooks.restore(errors), errors)
        if env is not None: cleanup('env_close', env.close, errors)
        if model is not None:
            try: io.frozen(model); after = io.tensor_hash(model); ledger.check()
            except BaseException as exc: errors.append(io.error('final_model_resources', exc))
        try: verify(folder, expected); source_ok = True
        except BaseException as exc: errors.append(io.error('final_source', exc))
        complete = (not errors and not ledger.failed and source_ok and before is not None and before == after and
                    len(rows) == 2 and all(r['status'] == 'completed20' for r in rows) and elapsed <= p['limits']['wall_seconds'])
        complete = finish_journal(journal, complete, errors)
        receipt = dict(schema=NAME + '.receipt', status='two_live_20step_3round_arms_completed' if complete else 'partial_or_failed',
            root_sha256=p['root_sha256'], prepare_sha256=expected, parent_prepare_sha256=PARENT_SHA,
            response_receipt_sha256=p['response_receipt_sha256'], semantics=p['semantics'],
            model_bindings=p['model_bindings'], model_hash_before=before, model_hash_after=after, source_verified=source_ok,
            rows=rows, errors=errors, first_error=deepcopy(errors[0]) if errors else None,
            counts=ledger.counts(), entries=ledger.entries, round_ledgers=ledger.rounds,
            physics_attempts=ledger.physics, physics_returns=ledger.physics_returns, limits=p['limits'],
            peaks=ledger.peaks, wall_seconds=elapsed, host=host, journal_failed=journal.failed,
            journal_close_failed=any(e['stage'] == 'journal_close' for e in errors),
            journal_sha256=io.sha(store.path / 'events.jsonl'), training=False, known_roots_only=True,
            independent_confirmation=False, P1_admission=False, P4_admission=False,
            scope='one known-root pair; native labels and independent wrapped diagnostic separately recorded; no scientific admission')
        receipt['output_files_sha256'] = {f.name: io.sha(f) for f in store.path.iterdir() if f.is_file()}
        store.write('receipt.json', receipt, final=True)
    require(complete, 'incomplete live run; preserved first error/receipt/journal; no retry'); return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run')); parser.add_argument('--output', type=Path)
    parser.add_argument('--prepared', type=Path); parser.add_argument('--prepare-sha')
    parser.add_argument('--wall-seconds', type=float); parser.add_argument('--allocated-bytes', type=int); parser.add_argument('--reserved-bytes', type=int)
    a = parser.parse_args()
    if a.command == 'prepare': print(prepare(a.output, caps(a.wall_seconds, a.allocated_bytes, a.reserved_bytes)))
    elif a.command == 'verify': print(json.dumps(verify(a.prepared, a.prepare_sha)['limits'], sort_keys=True))
    else: print(run(a.prepared, a.prepare_sha, a.output)['status'])


if __name__ == '__main__': main()
