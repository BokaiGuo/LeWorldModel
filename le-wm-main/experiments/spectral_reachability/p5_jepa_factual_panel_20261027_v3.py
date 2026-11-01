"""Prepare-only by default. Main explicitly runs three fixed native factual roots.

No physics/neural imports in prepare. No previous outcome reports are read.
Only v1 helpers and v2 numeric/schema functions are dynamically SHA-bound.
"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np

NAME = 'p5_jepa_factual_panel_20261027_v3'
V2 = 'p5_jepa_factual_nonzero_20261027_v2'
V2_MANIFEST_SHA = '4decc107d6e8ce71263f996ff99a5b177c99bd71391d06d29f909365a1c44210'
V2_SOURCE_SHA = '73bd811ff678ad2fe6a77dd11f97ef7c4404875c9896df3f50737c28e8f1eda8'
ROOT = Path(__file__).resolve().parents[2]
SEEDS = (2026102702, 2026102703, 2026102704)
FRAMES = [0, 5, 10, 15]
ARMS = ['actual', 'normalized_zero', 'negative_normalized', 'reverse_steps']
MAX_DISK_BYTES = 50_000_000
MAX_PAYLOAD_BYTES = 40_000_000
FINAL_RESERVE = 2_000_000
MAX_NODES = 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular non-symlink path required')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read_manifest(path):
    require(Path(path).stat().st_size < 2_000_000, 'manifest size bound')
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def bound(root):
    path = root / f'results/{V2}/manifest_frozen.json'
    require(sha(path) == V2_MANIFEST_SHA, 'v2 manifest identity rejected')
    manifest = read_manifest(path)
    for p, h in manifest['seals'].items():
        require(sha(p) == h, 'v2 inherited source drift: ' + p)
    source = root / f'experiments/spectral_reachability/{V2}.py'
    require(sha(source) == V2_SOURCE_SHA, 'v2 source identity rejected')
    spec = importlib.util.spec_from_file_location(NAME + '_sealed_v2', source)
    v2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2)
    base = v2.helpers(root)
    _, runtime = base.runtime_manifest(root)
    require(runtime['fresh_calibration_proposal']['seeds'] == [2026102701, *SEEDS], 'predeclared runtime seed list differs')
    require(manifest == v2.prepare(root), 'v2 frozen protocol differs')
    return v2, base, manifest


def protocol(v2):
    p = v2.protocol()
    p.update(schema=NAME, seeds=list(SEEDS), root='ALL remaining three predeclared runtime seeds, fixed order; distinct from already opened first seed',
             selection='no v11/outcome reads, no seed/action/contact filtering, no retries/replacement; ordinary root failures retain assets and do not exclude later roots',
             schedule_selection='exact sealed v2 nonzero schedule, never selected using v11 ranking',
             directory_schema='panel/seed_<seed>/{journal.jsonl,report.json,schedule_env_u.npy,a|b/frame_00..15/*,a|b/command_01..15.json,a|b/consumer/*}',
             per_root_expected=p.pop('expected'), panel_expected=dict(env_construct=6, seed=12, reset=6, control_step=90, space_step=906, native_render=96, native_get_obs=96, close=6),
             panel_failure='global import/source/budget failure leaves every remaining root not_run; all root journals/reports retained; no interrupted run resume',
             budget=dict(max_disk_bytes=MAX_DISK_BYTES, max_payload_bytes=MAX_PAYLOAD_BYTES, final_receipt_reserve=FINAL_RESERVE, max_nodes=MAX_NODES,
                         conservative_node_overhead_bytes=8192, maximum_accounted_bytes=MAX_PAYLOAD_BYTES + 8192 * MAX_NODES),
             source_boundary='unmodified native physics/render, sealed v1 journal/profile/public-state/numeric helpers; v2 stats/action-input functions only; seed-parameterized capture/export implemented here',
             prospective_control=False, training=False)
    del p['seed']
    return p


def prepare(root):
    v2, _, parent = bound(root)
    stats = v2.native_stats(root)
    inputs = v2.action_inputs(stats)
    seals = dict(parent['seals'])
    for p in (root / f'results/{V2}/manifest_frozen.json', Path(__file__).resolve(),
              root / f'tests/test_{NAME}.py', root.parent / f'docs/{NAME}.md'):
        seals[str(p)] = sha(p)
    return dict(schema=NAME, status='prepared_no_physics', protocol=protocol(v2), seals=seals,
                parent_manifest_sha256=V2_MANIFEST_SHA, native_stats={k: v.tolist() for k, v in stats.items()},
                action_inputs={k: dict(dtype=v.dtype.str, shape=list(v.shape), values=v.tolist()) for k, v in inputs.items()},
                physics_executed=False, neural_executed=False, P5_admission=False)


def verify(root, manifest):
    require(manifest == prepare(root), 'panel protocol/source seal rejected')


class BudgetIO:
    """Only writer: exact payload admission before writes plus bounded node overhead.

    40MB payload + at most1024 nodes x8KiB conservative overhead <50MB.
    2MB payload is reserved for final failure/notrun/terminal receipts.
    """
    def __init__(self, base):
        self.base = base
        self.payload = 0
        self.nodes = 0

    def charge(self, size, nodes=0, final=False):
        limit = MAX_PAYLOAD_BYTES if final else MAX_PAYLOAD_BYTES - FINAL_RESERVE
        node_limit = MAX_NODES if final else MAX_NODES - 32
        require(self.payload + size <= limit and self.nodes + nodes <= node_limit, 'panel storage budget exhausted; no retry')
        self.payload += size
        self.nodes += nodes

    def mkdir(self, path):
        self.charge(0, 1)
        self.base.mkdir(path)

    def save_json(self, path, value, final=False):
        self.charge(len(self.base.canonical(value)) + 1, 1, final)
        self.base.save_json(path, value)

    def save_array(self, path, value):
        a = np.asarray(value)
        require(a.dtype.kind in 'biuf' and a.nbytes < 200_000, 'bounded numeric array required')
        buffer = io.BytesIO()
        np.save(buffer, a, allow_pickle=False)
        self.charge(buffer.tell(), 1)
        self.base.save_array(path, a)

    def journal(self, path):
        self.charge(0, 1)
        return BudgetJournal(self, self.base.Journal(path))


class BudgetJournal:
    def __init__(self, writer, journal):
        self.writer = writer
        self.native = journal

    def append(self, event, final=False):
        row = dict(sequence=self.native.sequence, previous_sha256=self.native.previous, event=event, entry_sha256='0' * 64)
        self.writer.charge(len(self.writer.base.canonical(row)) + 1, final=final)
        return self.native.append(event)

    def close(self):
        self.native.close()


def capture(base, writer, env, obs, state, transition, path):
    """v1 capture schema with every write routed through the panel budget."""
    writer.mkdir(path)
    visual, state, proprio = np.asarray(obs['visual']), np.asarray(state), np.asarray(obs['proprio'])
    for name, a in (('rgb', visual), ('state', state), ('proprio', proprio)):
        writer.save_array(path / (name + '.npy'), a)
    writer.save_json(path / 'transition.json', base.numeric_json(transition))
    writer.save_json(path / 'full_public_state.json', base.public_state(env))
    require(visual.shape == (224, 224, 3) and visual.dtype == np.uint8 and state.shape == (7,) and state.dtype == np.float32
            and proprio.shape == (4,) and proprio.dtype == np.float32 and np.isfinite(state).all() and np.isfinite(proprio).all(), 'native observation contract rejected; partial raw returns retained')
    require(np.array_equal(proprio, state[[0, 1, 5, 6]]), 'native proprio/state mismatch')
    return dict(frame=int(path.name.split('_')[-1]), files={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(path.iterdir())}, state=state.tolist(), proprio=proprio.tolist())


def collect_trajectory(base, writer, cls, seed, role, out, journal, calls, inputs, records):
    env = None
    branch = out / role
    writer.mkdir(branch)
    try:
        env = base.operation(journal, calls, role, 'construct', dict(config=base.CTOR), lambda: cls(**base.CTOR))
        writer.save_json(branch / 'constructor_rng.json', base.rng_state(env))
        base.operation(journal, calls, role, 'seed', dict(seed=seed), lambda: env.seed(seed))
        writer.save_json(branch / 'pre_reset_rng.json', base.rng_state(env))
        obs, state = base.operation(journal, calls, role, 'reset', dict(seed=seed, expected_space_steps=1, dt=.01), env.reset)
        record = capture(base, writer, env, obs, state, dict(kind='reset'), branch / 'frame_00')
        records.append(record)
        journal.append(dict(phase='assets_durable', role=role, record=record))
        for frame, u in enumerate(inputs['raw_env_commands'], start=1):
            agent = np.array(env.agent.position, np.float64)
            command = dict(frame=frame, action_index=frame-1, packet_index=(frame-1)//5, step_within_packet=(frame-1)%5,
                           raw_env_argument=u.tolist(), pixel_displacement=(100*u).tolist(), agent_xy_before=agent.tolist(),
                           absolute_target=(agent+100*u).tolist(), expected_substeps=10, dt=.01,
                           pre_public_sha256=record['files']['full_public_state.json']['sha256'])
            writer.save_json(branch / f'command_{frame:02d}.json', command)
            obs, reward, done, info = base.operation(journal, calls, role, 'step', command, lambda: env.step(u.copy()))
            record = capture(base, writer, env, obs, info['state'], dict(kind='step', reward=reward, done=done, info=info), branch / f'frame_{frame:02d}')
            records.append(record)
            journal.append(dict(phase='assets_durable', role=role, record=record))
            require(not done, 'unexpected native done; retain root without replacement')
    finally:
        if env is not None:
            base.operation(journal, calls, role, 'close', {}, env.close)


def export_inputs(v2, writer, branch, seed, records, inputs, stats):
    dest = branch / 'consumer'
    writer.mkdir(dest)
    for name, a in inputs.items():
        writer.save_array(dest / (name + '.npy'), a)
    states, proprio, frames = [], [], []
    for frame in FRAMES:
        raw = branch / f'frame_{frame:02d}'
        rec = records[frame]
        require(rec['frame'] == frame, 'consumer frame identity rejected')
        states.append(v2.bounded_array(raw/'state.npy', (7,), np.float32, rec['files']['state.npy']['sha256']))
        proprio.append(v2.bounded_array(raw/'proprio.npy', (4,), np.float32, rec['files']['proprio.npy']['sha256']))
        v2.bounded_array(raw/'rgb.npy', (224,224,3), np.uint8, rec['files']['rgb.npy']['sha256'])
        frames.append(dict(frame=frame, rgb=f'{branch.name}/frame_{frame:02d}/rgb.npy', state=f'{branch.name}/frame_{frame:02d}/state.npy',
                           proprio=f'{branch.name}/frame_{frame:02d}/proprio.npy', full_public_state=f'{branch.name}/frame_{frame:02d}/full_public_state.json', files=rec['files']))
    states, proprio = np.stack(states), np.stack(proprio)
    normalized = (proprio-stats['PROPRIO_MEAN'])/stats['PROPRIO_STD']
    for name, a in dict(states_frames=states, raw_proprio_frames=proprio, normalized_proprio_frames=normalized, context_proprio=normalized[:3].copy()).items():
        writer.save_array(dest/(name+'.npy'), a)
    projected = inputs['dataset_pixel_displacements']/np.float32(100)
    inverse = inputs['normalized_packets'].reshape(15,2)*stats['ACTION_STD']+stats['ACTION_MEAN']
    value = dict(schema=NAME, layout_schema=V2, role=branch.name, seed=seed, frame_order=FRAMES, context_frames=[0,5,10], target_frame=15,
                 frames=frames, context_proprio_path=f'{branch.name}/consumer/context_proprio.npy', normalized_packets_path=f'{branch.name}/consumer/normalized_packets.npy',
                 controls_packets_path=f'{branch.name}/consumer/controls_packets.npy', arms=ARMS,
                 action_packet_layout='shape3x10, packet row-major five steps xy; row0[0:5],row1[5:10],row2[10:15]',
                 file_hashes={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(dest.iterdir())}, native_stats={k:v.tolist() for k,v in stats.items()},
                 precision=dict(dataset_projection_env_u_max_abs=float(np.max(np.abs(projected.astype(np.float64)-inputs['raw_env_commands']))), inverse_normalization_env_u_max_abs=float(np.max(np.abs(inverse.astype(np.float64)-inputs['raw_env_commands'])))),
                 target_scope='frame15 label only; target state/proprio/RGB must not select schedule or enter predictor history',
                 controls_not_executed_physically=True, neural_executed=False, action_fidelity_claim=False, P5_admission=False)
    writer.save_json(dest/'inputs_manifest.json', value)
    return dict(path=f'{branch.name}/consumer/inputs_manifest.json', sha256=sha(dest/'inputs_manifest.json'))


def collect_root(root, manifest, base, v2, writer, cls, pymunk, seed, out, journal, inputs, stats):
    collected = {r: [] for r in ('a', 'b')}
    role_status = {r: 'not_run' for r in ('a', 'b')}
    errors, consumer = [], {}
    calls, owned = None, False
    rows = [dict(frame=i, status='not_run') for i in range(16)]
    try:
        writer.save_array(out/'schedule_env_u.npy', inputs['raw_env_commands'])
        journal.append(dict(phase='preflight_intent', seed=seed, protocol=manifest['protocol'], schedule_file_sha256=sha(out/'schedule_env_u.npy')))
        calls = base.NativeCalls(cls, pymunk)
        calls.start()
        owned = True
        for role in ('a', 'b'):
            try:
                collect_trajectory(base, writer, cls, seed, role, out, journal, calls, inputs, collected[role])
                consumer[role] = export_inputs(v2, writer, out/role, seed, collected[role], inputs, stats)
                role_status[role] = 'completed'
                journal.append(dict(phase='consumer_assets_durable', role=role, consumer=consumer[role]))
            except Exception as exc:
                role_status[role] = 'failed'
                errors.append(role+': '+type(exc).__name__+': '+str(exc))
                journal.append(dict(phase='role_failed', role=role, error=errors[-1], interrupted_native_intents_not_resumed=True), final=True)
        if all(len(v)==16 for v in collected.values()):
            rows = base.compare_records(collected['a'], collected['b'])
        require(not errors, 'one or more native roles failed')
        require(all(r.get('all_exact',False) for r in rows), 'exact paired frame gate failed')
        require(all(calls.counts.get(k,0)==v for k,v in manifest['protocol']['per_root_expected'].items()), 'native call count mismatch')
        require(all(e['dt']==.01 for e in calls.space_steps), 'native physics timing mismatch')
    except Exception as exc:
        errors.append(type(exc).__name__+': '+str(exc))
    finally:
        if owned:
            calls.stop()
    unchanged = False
    try:
        verify(root, manifest)
        unchanged = True
    except Exception as exc:
        errors.append('source_check: '+str(exc))
    return dict(schema=NAME, layout_schema=V2, seed=seed, status='fresh_native_panel_root_recorded' if not errors and unchanged else 'fresh_native_panel_root_failed',
                protocol=manifest['protocol'], errors=errors, rows=rows, role_status=role_status, consumer_inputs=consumer,
                trajectory_frames={k:len(v) for k,v in collected.items()}, native_call_receipts=calls.receipt() if calls else None,
                source_hashes_unchanged=unchanged, physics_executed=bool(calls and calls.counts.get('space_step',0)), neural_executed=False,
                action_fidelity_claim=False, control_benefit_claim=False, P5_admission=False)


def initial_rows():
    return [dict(seed=s, root=f'seed_{s}', status='not_run', report=f'seed_{s}/report.json', journal=f'seed_{s}/journal.jsonl') for s in SEEDS]


def dispatch(rows, execute, finalize, checkpoint, precheck):
    """Outcome-independent census. Global source/budget failures block later roots."""
    blocked = None
    for i, row in enumerate(rows):
        result = None
        if blocked is None:
            try:
                precheck()
            except Exception as exc:
                blocked = type(exc).__name__+': '+str(exc)
            else:
                try:
                    result = execute(row['seed'])
                except Exception as exc:
                    blocked = type(exc).__name__+': '+str(exc)
                    result = dict(schema=NAME, seed=row['seed'], status='fresh_native_panel_root_failed', errors=[blocked],
                                  execution_attempted=True, physics_executed=None, neural_executed=False, P5_admission=False)
        if result is None:
            result = dict(schema=NAME, seed=row['seed'], status='not_run', reason=blocked, physics_executed=False, neural_executed=False, P5_admission=False)
        row['status'] = result['status']
        finalize(row, result)
        checkpoint(i+1, rows)


def run(root, manifest, manifest_path, expected_sha, out):
    require(sha(manifest_path)==expected_sha, 'explicit manifest SHA rejected')
    verify(root, manifest)
    v2, base, _ = bound(root)
    out = Path(out)
    require(not out.exists() and out.is_absolute() and out.parent.resolve()==root/f'results/{NAME}' and out.name not in ('.','..'), 'exclusive direct panel result child required')
    for p in (out.parent, *out.parent.parents):
        require(not p.is_symlink(), 'output symlink rejected')
    writer = BudgetIO(base)
    writer.mkdir(out)
    rows = initial_rows()
    journals = {}
    for row in rows:
        writer.mkdir(out/row['root'])
        journals[row['seed']] = writer.journal(out/row['journal'])
    inputs, stats = v2.action_inputs(v2.native_stats(root)), v2.native_stats(root)
    writer.save_json(out/'panel_intent.json', dict(protocol=manifest['protocol'], manifest_file_sha256=expected_sha, rows=rows))
    writer.save_array(out/'schedule_env_u.npy', inputs['raw_env_commands'])
    writer.save_json(out/'progress_00.json', dict(rows=rows, completed_roots=0))
    ready, problem, cls, pymunk = False, None, None, None
    try:
        for seed, journal in journals.items():
            journal.append(dict(phase='panel_registered_before_runtime', seed=seed, manifest_file_sha256=expected_sha))
        runtime, parent = base.load_runtime(root)
        probe = runtime.probe(root, parent, root/f'results/{base.RUNTIME}/manifest_frozen.json')
        writer.save_json(out/'runtime_probe.json', probe)
        require(probe['status']=='native_import_ready_physics_not_executed', 'strict native runtime probe rejected')
        source = root/runtime.ENV_REL
        spec = importlib.util.spec_from_file_location(NAME+'_official_native', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = module.PushTEnv
        require(Path(cls.__init__.__code__.co_filename).resolve()==source, 'native constructor provenance rejected')
        import pymunk  # Main-only confirmed run path, never prepare/test.
        ready = True
    except Exception as exc:
        problem = type(exc).__name__+': '+str(exc)

    def precheck():
        require(ready, problem or 'runtime unavailable')
        verify(root, manifest)
        writer.charge(0)

    def execute(seed):
        return collect_root(root, manifest, base, v2, writer, cls, pymunk, seed, out/f'seed_{seed}', journals[seed], inputs, stats)

    def finalize(row, result):
        journal = journals[row['seed']]
        result['manifest_file_sha256'] = expected_sha
        journal.append(dict(phase='terminal', status=result['status'], summary_sha256=hashlib.sha256(base.canonical(result)).hexdigest()), final=True)
        journal.close()
        result['journal_sha256'] = sha(out/row['journal'])
        writer.save_json(out/row['report'], result, final=True)
        row['report_sha256'] = sha(out/row['report'])
        row['journal_sha256'] = result['journal_sha256']
        row['physics_executed'] = result['physics_executed']

    dispatch(rows, execute, finalize, lambda i,r: writer.save_json(out/f'progress_{i:02d}.json', dict(rows=r, processed_roots=i), final=True), precheck)
    source_after, source_error = False, None
    try:
        verify(root, manifest)
        source_after = True
    except Exception as exc:
        source_error = str(exc)
    exact = source_after and all(r['status']=='fresh_native_panel_root_recorded' for r in rows)
    result = dict(schema=NAME, status='fresh_native_nonzero_panel_recorded' if exact else 'fresh_native_nonzero_panel_failed',
                  protocol=manifest['protocol'], manifest_file_sha256=expected_sha, rows=rows, source_hashes_unchanged=source_after,
                  runtime_error=problem, source_error=source_error, storage=dict(payload_bytes_before_final_report=writer.payload, nodes_before_final_report=writer.nodes,
                  max_disk_bytes=MAX_DISK_BYTES, max_payload_bytes=MAX_PAYLOAD_BYTES, max_nodes=MAX_NODES),
                  physics_executed=any(r.get('physics_executed',False) for r in rows), physics_execution_indeterminate=any(r.get('physics_executed') is None for r in rows),
                  neural_executed=False, action_fidelity_claim=False, control_benefit_claim=False, P5_admission=False)
    writer.save_json(out/'report.json', result, final=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare','run'], nargs='?', default='prepare')
    p.add_argument('--manifest', type=Path, default=ROOT/f'results/{NAME}/manifest_frozen.json')
    p.add_argument('--manifest-sha256')
    p.add_argument('--confirm-physics', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/f'results/{NAME}/factual_main_v1')
    args = p.parse_args()
    if args.command=='prepare':
        value = prepare(ROOT)
    else:
        if not args.confirm_physics or not args.manifest_sha256:
            p.error('main-only run requires --confirm-physics and --manifest-sha256')
        value = run(ROOT, read_manifest(args.manifest), args.manifest, args.manifest_sha256, args.output)
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    if value['status']=='fresh_native_nonzero_panel_failed':
        raise SystemExit(2)


if __name__=='__main__':
    main()
