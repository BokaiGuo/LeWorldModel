"""Fixed-length beam vs flat search; no training, outcome selection or admission.

Reusable runner: provide a frozen raw bank, a full witness root/provenance/domain,
and the saved-goal metric. Neural use MUST supply RootAnchoredCudaReplay below.
Fixtures use run_fixture_pair explicitly, never a CPU neural fallback. Prediction
and action-encoding attempts are debited before dispatch, including failures.
Unknown FLOPs and upstream encoding costs are not fabricated as zero.
"""
from copy import deepcopy
import time
from functools import wraps

import numpy as np

from experiments.spectral_reachability.d2p_composition_witness_v2 import compose, digest
from experiments.spectral_reachability.d2p_kinodynamic_program_search import (
    _bank, _cost, _validated_context,
)
from experiments.spectral_reachability.pusht_composition_cuda_replay_v2 import CudaReplay


class RootAnchoredCudaReplay(CudaReplay):
    """Existing correct between-boundary recurrence; full incoming history is NOT used.

    [z(t-2),z(t-1),z(t)] pairs with [a(t-2),a(t-1),new a(t)].
    The first two actions are context.action_history, NOT incoming history[-2:]
    from an unrelated time convention. No bootstrap prediction is made.
    """


def schedule(bank_size=16, beam_width=4):
    if any(type(v) is not int or v < 1 for v in (bank_size, beam_width)):
        raise ValueError('positive integer bank size and beam width required')
    if beam_width > bank_size or bank_size * (1 + beam_width) % 2:
        raise ValueError('beam must fit bank and planning token budget must be even')
    planning = bank_size * (1 + beam_width)
    if planning // 2 > bank_size ** 2:
        raise ValueError('flat schedule exceeds unique ordered pairs')
    return dict(horizon_tokens=2, primitive_steps=10,
                planning_predict_calls=planning, planning_predict_samples=planning,
                replay_predict_calls=4, replay_predict_samples=4,
                flat_programs=planning // 2, composition_terminal_attempts=bank_size * beam_width)


class _Ledger:
    def __init__(self, replay, arm):
        self.replay = replay
        self.arm = arm
        self.events = []

    def call(self, context, actions, phase, path):
        # One packet per dispatch: even partial full-program failures are visible.
        state = deepcopy(context)
        for start in range(0, len(actions), 5):
            event = dict(phase=phase, path=list(path), packet=start // 5,
                         predict_calls=1, predict_samples=1, action_encoder_calls=1,
                         completed=False)
            self.events.append(event)
            before = time.perf_counter()
            try:
                if type(self.replay) is RootAnchoredCudaReplay:
                    self.replay._p2_scope = dict(arm=self.arm, phase=phase, path=list(path))
                state = _validated_context(self.replay(
                    deepcopy(state), deepcopy(actions[start:start + 5])), context)
                event['completed'] = True
            except Exception as exc:
                event.update(error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                event['wall_time_s'] = time.perf_counter() - before
                if type(self.replay) is RootAnchoredCudaReplay:
                    self.replay._p2_scope = None
        return state


def _run_pair(*, primitive_bank, root, provenance, action_domain, target_metric,
              replay, beam_width, flat_seed, fixture):
    root, provenance, action_domain = deepcopy((root, provenance, action_domain))
    bank = _bank(primitive_bank)
    if any(p.shape != (5, 2) for p in bank):
        raise ValueError('matched runner requires exactly one five-step packet per primitive')
    if type(flat_seed) is not int or flat_seed < 0:
        raise ValueError('explicit nonnegative pre-outcome flat seed required')
    config = schedule(len(bank), beam_width)
    initial = _validated_context(root['context'])
    # Validate full witness metadata and raw action domain BEFORE any model work.
    def edge(context, output, actions, identity):
        return dict(program_id=identity, kind='progress', root_id=root['root_id'],
                    provenance=deepcopy(provenance), action_domain=deepcopy(action_domain),
                    input_context=deepcopy(context), output_context=deepcopy(output),
                    actions=deepcopy(actions))
    compose(root=root, provenance=provenance, action_domain=action_domain,
            edges=[edge(initial, initial, bank[0].tolist(), 'metadata_check')],
            replay=lambda c, a: c, max_edges=1, max_actions=5)
    rng = np.random.default_rng(flat_seed)
    pairs = [(i, j) for i in range(len(bank)) for j in range(len(bank))]
    flat_paths = [pairs[int(i)] for i in rng.permutation(len(pairs))[:config['flat_programs']]]
    declaration = dict(schema='p2_matched_composition_20261027.v1', root=root,
                       provenance=provenance, action_domain=action_domain,
                       bank=[p.tolist() for p in bank], beam_width=beam_width,
                       flat_seed=flat_seed, flat_paths=flat_paths, schedule=config,
                       selection='minimum finite depth2 metric; lexicographic ties',
                       root_selection='caller must externally seal before outcomes')
    rows = []
    for arm in ('composition', 'flat_search'):
        ledger = _Ledger(replay, arm)
        attempts, terminals = [], []
        started = time.perf_counter()

        def attempt(parent, actions, path):
            event = dict(path=list(path), completed=False)
            attempts.append(event)
            try:
                successor = ledger.call(parent, actions, 'planning', path)
                value = _cost(target_metric(deepcopy(successor)))
                event.update(completed=True, cost=value)
                return value, tuple(path), successor
            except Exception as exc:
                event.update(error_type=type(exc).__name__, error=str(exc))
                return None

        if arm == 'composition':
            parents = []
            for i, primitive in enumerate(bank):
                result = attempt(initial, primitive.tolist(), (i,))
                if result is not None:
                    parents.append(result)
            parents.sort(key=lambda x: (x[0], x[1]))
            for _, path, state in parents[:beam_width]:
                for j, primitive in enumerate(bank):
                    result = attempt(state, primitive.tolist(), (*path, j))
                    if result is not None:
                        terminals.append(result)
        else:
            for path in flat_paths:
                actions = np.concatenate([bank[i] for i in path]).tolist()
                result = attempt(initial, actions, path)
                if result is not None:
                    terminals.append(result)
        terminals.sort(key=lambda x: (x[0], x[1]))
        selected = terminals[0] if terminals else None
        witness = None
        replay_error = None
        if selected is not None:
            cost, path, terminal = selected
            actions = np.concatenate([bank[i] for i in path]).tolist()
            try:
                # Independently regenerate every edge, not cached search states.
                mid = ledger.call(initial, actions[:5], 'replay', path)
                end = ledger.call(mid, actions[5:], 'replay', path)
                full = ledger.call(initial, actions, 'replay', path)
                if full != end or full != terminal:
                    raise ValueError('segmented/search/end-to-end root replay mismatch')
                edges = ([edge(initial, mid, actions[:5], 'selected_0'),
                          edge(mid, end, actions[5:], 'selected_1')]
                         if arm == 'composition' else
                         [edge(initial, end, actions, 'selected_flat')])
                # States were independently replayed above; compose only binds
                # these verified transitions, with no hidden extra predictions.
                witness = compose(root=root, provenance=provenance, action_domain=action_domain,
                    edges=edges, replay=lambda c, a: deepcopy(next(
                        e['output_context'] for e in edges if e['input_context'] == c and e['actions'] == a)),
                    max_edges=len(edges), max_actions=10)
                witness = dict(receipt=witness, edges=edges, independent_full_replay_exact=True)
            except Exception as exc:
                replay_error = dict(error_type=type(exc).__name__, error=str(exc))
        counts = {phase: sum(e['predict_samples'] for e in ledger.events if e['phase'] == phase)
                  for phase in ('planning', 'replay')}
        rows.append(dict(arm=arm, attempts=attempts, selected_path=list(selected[1]) if selected else None,
                         selected_model_cost=selected[0] if selected else None,
                         witness=witness, replay_error=replay_error,
                         planning_predict_samples=counts['planning'], replay_predict_samples=counts['replay'],
                         predict_calls=len(ledger.events), action_encoder_calls=len(ledger.events),
                         failed_predict_attempts=sum(not e['completed'] for e in ledger.events),
                         attempted_expansions=len(attempts),
                         candidate_metric_evaluations=sum(e['completed'] for e in attempts),
                         failed_search_attempts=sum(not e['completed'] for e in attempts),
                         ledger=ledger.events, actual_wall_time_s=time.perf_counter()-started))
    equal = all(r['planning_predict_samples'] == config['planning_predict_samples'] and
                r['replay_predict_samples'] == 4 and r['witness'] is not None and
                not r['failed_search_attempts'] and not r['failed_predict_attempts'] for r in rows)
    return deepcopy(dict(declaration=declaration, declaration_sha256=digest(declaration), rows=rows,
        status='prediction_matched_only' if equal else 'incomplete_or_failed_pair',
        fixture_only=fixture, prediction_budget_matched=equal, fair_total_budget_comparison=False,
        unmeasured_costs=['upstream_bank_construction', 'upstream_image_encoding', 'flops',
                          'peak_cuda_memory', 'wall_time_cap_not_enforced'],
        online_image_encoder_calls=0, backward_calls=0, optimizer_steps=0, physical_env_steps=0,
        P1_admission=False, P2_admission=False, physical_efficacy_verified=False))


def run_cuda_pair(*, replay, **kwargs):
    """Caller schedules CUDA and retains externally sealed declaration and inputs.

    Both arms reuse pre-encoded frozen inputs. No GPU allocation is made here.
    Pass target_metric from run_pusht_program_search_cuda.metric_for(compiled).
    Full model provenance, source seals and before/after integrity remain the
    host integration runner's responsibility; never infer them from this API.
    """
    if type(replay) is not RootAnchoredCudaReplay:
        raise ValueError('explicit RootAnchoredCudaReplay required; no neural CPU fallback')
    replay._check_model()
    model = replay.model
    actual = []
    # Instrument actual entries, not the dispatch estimates in the generic
    # ledger. Restore instance attributes exactly, including on exceptions.
    restored = []
    for owner, name, kind in ((model, 'predict', 'predict'),
                              (model.action_encoder, 'forward', 'action_encoder')):
        original = getattr(owner, name)
        restored.append((owner, name, name in vars(owner), vars(owner).get(name)))

        def wrap(original, kind):
            @wraps(original)
            def counted(*args, **kw):
                event = dict(kind=kind, samples=int(args[0].shape[0]), completed=False,
                             scope=deepcopy(replay._p2_scope))
                actual.append(event)
                try:
                    result = original(*args, **kw)
                    event['completed'] = True
                    return result
                except Exception as exc:
                    event.update(error_type=type(exc).__name__, error=str(exc))
                    raise
            return counted
        setattr(owner, name, wrap(original, kind))
    try:
        result = _run_pair(replay=replay, fixture=False, **kwargs)
    finally:
        for owner, name, existed, value in reversed(restored):
            if existed:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
    result['actual_model_entries'] = actual
    expected = sum(r['predict_calls'] for r in result['rows'])
    result['actual_entries_match_dispatch'] = all(
        sum(e['samples'] for e in actual if e['kind'] == kind) == expected
        and sum(1 for e in actual if e['kind'] == kind) == expected
        for kind in ('predict', 'action_encoder'))
    for row in result['rows']:
        row['actual_model_entries'] = [e for e in actual if e['scope']['arm'] == row['arm']]
        for phase in ('planning', 'replay'):
            spent = [e for e in row['actual_model_entries']
                     if e['kind'] == 'predict' and e['scope']['phase'] == phase]
            row['actual_' + phase + '_predict_calls'] = len(spent)
            row['actual_' + phase + '_predict_samples'] = sum(e['samples'] for e in spent)
            if len(spent) != row[phase + '_predict_samples']:
                result['actual_entries_match_dispatch'] = False
    if not result['actual_entries_match_dispatch'] or any(not e['completed'] for e in actual):
        result.update(prediction_budget_matched=False, status='incomplete_or_failed_pair')
    return result


def run_fixture_pair(**kwargs):
    """Non-neural deterministic fixtures ONLY; not a CUDA evidence path."""
    return _run_pair(fixture=True, **kwargs)


def run_calibration_cli(compilation, policy, output, flat_seed=20261027):
    """ALL four existing calibration roots, not a prospective efficacy cohort.

    Writes only new p2_matched-named artifacts. Never executes physics. Source
    snapshots are stored inline in one bounded JSON, not duplicated weights.
    """
    import json
    from pathlib import Path
    import torch
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.run_pusht_program_search_cuda import metric_for
    from experiments.spectral_reachability.d2p_composition_witness_v2 import PROVENANCE_KEYS
    from experiments.spectral_reachability.d2p_composition_witness_v2 import _sha

    source, bindings, config = checked.validate_inputs(compilation, policy)
    policy, output = Path(policy).resolve(), Path(output).resolve()
    if 'p2_matched' not in output.name.lower():
        raise ValueError('new output directory name must contain p2_matched')
    if type(flat_seed) is not int or flat_seed < 0:
        raise ValueError('nonnegative flat seed required')
    upstream = [json.loads(Path(p).read_text()) for p in bindings if Path(p).name == 'report.json']
    image_sources = [r for r in upstream if len(r.get('rows', [])) == 4 and
                     all('history_rgb_sha256' in row and 'goal_rgb_sha256' in row for row in r['rows'])]
    if not image_sources:
        raise ValueError('sealed complete root/goal RGB bindings missing')
    images = {r['root_sha256']: r for r in image_sources[0]['rows']}
    if set(images) != {r['root_sha256'] for r in source['rows']}:
        raise ValueError('all calibration roots must be preserved')
    prepared = []
    for row in source['rows']:
        c = row['compiled']
        initial = c['input_context']
        context = dict(layout='states_with_between_state_actions_v1', context_length=3,
                       latent_dim=192, action_dim=10, latent_history=initial['emb'][0],
                       action_history=initial['actions'][0][1:])
        _validated_context(context)
        metric_for(c)  # validate saved Q, NOT task efficacy
        bank = _bank([np.repeat(np.asarray(r['program_live_commands'][0])[None], 5, axis=0)
                      for r in c['receipts']])
        image = images[row['root_sha256']]
        root = dict(root_id=row['root_sha256'], context=context,
                    root_rgb_sha256=image['history_rgb_sha256'], goal_rgb_sha256=image['goal_rgb_sha256'])
        for value in (root['root_id'], root['root_rgb_sha256'], root['goal_rgb_sha256'],
                      c['scoring']['metric']['sha256']):
            _sha(value)
        prepared.append(dict(root=root, bank=[p.tolist() for p in bank],
                             metric_sha256=c['scoring']['metric']['sha256']))
    paths = checked.sources_before_allocate(source, config)
    paths = sorted(set(paths + [Path(__file__).resolve(), Path(metric_for.__code__.co_filename).resolve(),
        Path(__file__).resolve().parents[2] / 'tests/test_p2_matched_composition_20261027.py',
        Path(__file__).resolve().parents[3] / 'docs/D2P_P2_MATCHED_COMPOSITION_20261027.md']))
    snapshots = []
    for path in paths:
        if path.stat().st_size > 1024 * 1024:
            raise ValueError('source snapshot exceeds 1 MiB per file')
        snapshots.append(dict(path=str(path), sha256=base.digest(path), source=path.read_text()))
    protocol = dict(schema='p2_matched_calibration_20261027.v1', bindings=bindings,
                    prepared=prepared, source_manifest_sha256=digest(snapshots), flat_seed=flat_seed,
                    schedule=schedule(), expected_predict_samples=672,
                    scope='known calibration roots; Q interface/cost only; not task efficacy',
                    P1_admission=False, P2_admission=False, fair_total_budget_comparison=False)
    # Exclusive new artifacts before GPU allocation; no old report is rewritten.
    output.mkdir(exist_ok=False)
    base.write_new(output/'p2_matched_sources.json', snapshots)
    base.write_new(output/'p2_matched_protocol.json', protocol)
    print('P2_MATCHED pre-CUDA protocol SHA256 ' + digest(protocol), flush=True)
    host = base.require_cuda_allocation('cuda')
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    model = base.load_compatible_pretrained(policy, device=device).eval().requires_grad_(False)
    before = base.p0._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    rows = []
    domain = dict(name='raw', units='official PushT live command', low=[-1., -1.], high=[1., 1.],
                  action_block=5, normalization_sha256=digest(dict(
                      mean=base.p0.PUBLIC_ACTION_MEAN.tolist(), std=base.p0.PUBLIC_ACTION_STD.tolist())))
    for row, ready in zip(source['rows'], prepared, strict=True):
        provenance = {k: protocol['source_manifest_sha256'] for k in PROVENANCE_KEYS}
        provenance.update(checkpoint_sha256=checked.OFFICIAL_SHA, encoder_sha256=checked.OFFICIAL_SHA,
                          protocol_sha256=digest(protocol), metric_sha256=ready['metric_sha256'],
                          candidate_bank_sha256=digest(ready['bank']))
        result = run_cuda_pair(replay=RootAnchoredCudaReplay(model, device),
            primitive_bank=ready['bank'], root=ready['root'], provenance=provenance,
            action_domain=domain, target_metric=metric_for(row['compiled']), beam_width=4, flat_seed=flat_seed)
        result.update(root_sha256=row['root_sha256'], root_index=row['root_index'])
        base.write_new(output/f'p2_matched_root_{row["root_index"]}.json', result)
        rows.append(result)
        print(f'P2_MATCHED root {row["root_index"]}: {result["status"]}', flush=True)
    gates = dict(all_four_roots=len(rows)==4,
                 all_prediction_pairs_matched=all(r['prediction_budget_matched'] for r in rows),
                 actual_predict_samples_672=sum(e['samples'] for r in rows for e in r['actual_model_entries']
                                                if e['kind']=='predict') == 672,
                 parameters_unchanged=before == base.p0._parameter_hash(model),
                 buffers_unchanged=all(torch.equal(buffers[k], v) for k, v in model.named_buffers()),
                 frozen_eval=all(not m.training for m in model.modules()) and
                             all(not p.requires_grad for p in model.parameters()),
                 inputs_unchanged=all(base.digest(Path(p))==d for p, d in bindings.items()),
                 sources_unchanged=all(base.digest(Path(s['path']))==s['sha256'] for s in snapshots))
    report = dict(schema='p2_matched_cuda_calibration_20261027.v1', protocol_sha256=digest(protocol),
                  status='calibration_completed' if all(gates.values()) else 'integrity_fail',
                  host=host, gates=gates, rows=rows, elapsed_seconds=time.perf_counter()-started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  P1_admission=False, P2_admission=False, physical_efficacy_verified=False,
                  task_metric_repaired=False, fair_total_budget_comparison=False)
    base.write_new(output/'p2_matched_report.json', report)
    if not all(gates.values()):
        raise RuntimeError('P2_MATCHED calibration integrity failed; retain artifacts')


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Host CUDA calibration only; no P1/P2 admission or task efficacy')
    for name in ('compilation', 'policy', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--flat-seed', type=int, default=20261027)
    run_calibration_cli(**vars(parser.parse_args()))
