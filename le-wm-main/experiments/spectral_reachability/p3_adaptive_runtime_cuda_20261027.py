"""Parent-launched HOST CUDA paired runtime probe; never P3 admission.

Saved calibration cohort only. No encoder/readout training, physical evaluation,
root selection, CPU neural fallback, or writes to upstream files.
"""
import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.spectral_reachability import p3_adaptive_runtime_20261027 as runtime
from experiments.spectral_reachability.d2p_adaptive_budget_v2 import PredictionBudget
from experiments.spectral_reachability.d2p_planning_compute_ledger import Binding, BudgetContract, Resources
from experiments.spectral_reachability.d2p_p3_paired_accounting import (
    BranchReceipt, PairedPlan, RootPlan, RunReceipt, compare_paired,
)


def zero_resources(**updates):
    return Resources(**({f.name: 0 for f in fields(Resources)} | updates))


class ReplayAccounting:
    """Attempted replay work; encoder hooks count actual entries separately.

    One rollout is an attempted edge OR full verification trajectory. Imagined
    tokens count NEW outgoing tokens B (including subsequent predict blocks).
    Each action_encoder entry processes B*T context tokens, counted separately;
    its two historical tokens must not become extra future predictions.
    max horizon counts requested full prefix, not historical context length.
    """
    def __init__(self, replay):
        self.replay = replay
        self.rollouts = self.horizon = self.tokens = 0
        self.encoder_forwards = self.encoder_samples = 0
        self.action_encoder_forwards = self.action_encoder_samples = 0
        self.processed_action_context_tokens = 0
        self.depth = 1

    def action_entry(self, module, args):
        batch = args[0].shape[0]
        self.action_encoder_forwards += 1
        self.action_encoder_samples += batch
        self.processed_action_context_tokens += batch * args[0].shape[1]
        self.tokens += batch

    def encoder_entry(self, module, args):
        self.encoder_forwards += 1
        self.encoder_samples += args[0].shape[0]

    def __call__(self, context, actions):
        self.rollouts += 1
        self.horizon = max(self.horizon, self.depth, len(actions) // 5)
        return self.replay(context, actions)

    def resources(self, receipt):
        return zero_resources(predict_calls=sum(p['predict_calls'] for p in receipt['phases']),
            predict_samples=receipt['attempted_predict_samples'],
            encoder_forwards=self.encoder_forwards, encoder_samples=self.encoder_samples,
            rollout_samples=self.rollouts, imagined_action_tokens=self.tokens,
            max_planning_horizon_tokens=self.horizon)


def run_arm(*, model, replay, row, binding, policy, mode, synchronize):
    """Keep failure receipts and reserve max_depth predictions for full replay."""
    method = 'p3_adaptive_runtime_' + mode
    compiled = row['compiled']
    initial = compiled['input_context']
    context = dict(layout='states_with_between_state_actions_v1', context_length=3,
        latent_dim=192, action_dim=10, latent_history=initial['emb'][0],
        action_history=initial['actions'][0][1:])
    bank = [np.repeat(np.asarray(r['program_live_commands'][0])[None], 5, axis=0)
            for r in compiled['receipts']]
    work = ReplayAccounting(replay)
    hooks = []
    failure, result, exact = None, None, False
    synchronize()
    started = perf_counter()
    with PredictionBudget(model, binding=binding, method=method,
                          sample_limit=policy.max_expansions + policy.max_depth) as budget:
        try:
            hooks.append(model.action_encoder.register_forward_pre_hook(work.action_entry))
            # CudaReplay uses cached latents and never calls the visual encoder.
            # Hook it anyway: accidental work is observed, not silently free.
            hooks.append(model.encoder.register_forward_pre_hook(work.encoder_entry))
            with budget.phase('planning', method=method, horizon_tokens=policy.max_depth):
                result = runtime.run_search(primitive_bank=bank, root_context=context,
                    replay=work, goal=compiled['goal_vectors'][0],
                    projection=compiled['scoring']['metric']['projection'], policy=policy, mode=mode)
            # Full horizon is recoverable from attempted kernel parent paths,
            # including failed edges; verification may use a shorter best plan.
            work.horizon = max(work.horizon, max((l['depth'] for l in result['layers']), default=0))
            best = result['best_plan']
            if best is not None:
                with budget.phase('evaluation', method=method, horizon_tokens=best['depth']):
                    exact = work(context, best['actions']) == best['context']
                if not exact:
                    raise ValueError('whole-program replay differs from incremental search')
            if best is None or result['status'] == 'failed':
                raise ValueError('search failed; retained attempted resources')
        except Exception as exc:
            failure = dict(type=type(exc).__name__, message=str(exc))
        finally:
            for hook in hooks:
                hook.remove()
        synchronize()
        receipt = budget.receipt()
    elapsed = perf_counter() - started
    resources = work.resources(receipt)
    branch = BranchReceipt('search_and_verification', 'failed' if failure else 'completed',
                           False, resources)  # No measured physical task success.
    run = RunReceipt(binding, method, (branch,), zero_resources(), receipt, elapsed)
    return run, dict(root_sha256=binding.root_id, method=method, search=result,
        prediction_receipt=receipt, resources=asdict(resources), failure=failure,
        full_replay_exact=exact, walltime_seconds=elapsed,
        action_encoder_forwards=work.action_encoder_forwards,
        action_encoder_samples=work.action_encoder_samples,
        processed_action_context_tokens=work.processed_action_context_tokens,
        physical_success_measured=False)


def run(*, compilation, policy, output, preflight_only=False):
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import pusht_composition_cuda_replay_v2 as adapter
    source, bindings, config = checked.validate_inputs(compilation, policy)
    # Import every runtime dependency BEFORE fingerprinting current sources.
    import torch
    from experiments.gpu_preflight import require_cuda_allocation
    from experiments.checkpoint_compat import load_compatible_pretrained
    files = set(checked.sources_before_allocate(source, config))
    files.update((Path(__file__).resolve(), Path(runtime.__file__).resolve(),
        Path(__file__).resolve().parents[2] / 'tests/test_p3_adaptive_runtime_20261027.py',
        Path(__file__).resolve().parents[3] / 'docs/P3_ADAPTIVE_RUNTIME_20261027.md'))
    hashes = {str(p): checked.digest(p) for p in sorted(files)}
    protocol = dict(schema='p3_adaptive_runtime_protocol_20261027.v1', bindings=bindings,
        roots=source['protocol']['roots'], sources=hashes,
        schedule=asdict(runtime.RuntimePolicy('preseal')) | {'source_seal': 'protocol_id'},
        bank='all 16 saved commands, each five raw steps',
        methods=['p3_adaptive_runtime_fixed', 'p3_adaptive_runtime_adaptive'],
        predict_cap=147, planning_edge_cap=144, verification_reserve=3,
        scope='cached-latent online search plus signals plus full replay; no physical execution',
        shared_cost_policy='prior compilation/checkpoint loading/source hashing excluded from per-root vectors; measured once separately; no total-pipeline claim',
        signal_policy='both arms compute own finite response SVD; only adaptive uses it',
        action_encoder_sample_cap=147, processed_action_context_token_cap=441,
        action_accounting='imagined new tokens B; processed action context tokens B*T; T=3, not three future predictions',
        numpy_signal_math='CPU non-neural SVD, timed not FLOPs',
        derivative_policy='torch.no_grad throughout; no VJP, backward, fitting or training',
        torch_version=torch.__version__, p1_dependency='not_passed', p3_admitted=False)
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    frozen = runtime.RuntimePolicy(protocol_id)
    limits = zero_resources(predict_calls=147, predict_samples=147, rollout_samples=145,
                            imagined_action_tokens=147, max_planning_horizon_tokens=3)
    roots = tuple(RootPlan(BudgetContract(Binding(protocol_id, 'pusht_cached_latent_runtime',
        r, checked.OFFICIAL_SHA + ':' + checked.CONFIG_SHA), limits), ('search_and_verification',))
        for r in protocol['roots'])
    plan = PairedPlan(protocol_id, roots, *protocol['methods'])
    if preflight_only:
        print(json.dumps(dict(status='preflight_only_no_cuda_allocation', protocol_id=protocol_id,
            roots=len(roots), sources=len(hashes), source_bytes=sum(Path(p).stat().st_size for p in hashes),
            upstream_seal_verified=True, p3_admitted=False), indent=2))
        return
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    # Hash manifest, not a replacement for the existing archive seal. Keeping
    # names scoped avoids creating generic manifest/args/source files.
    checked.write_new(output / 'P3_ADAPTIVE_RUNTIME_protocol_20261027.json',
                      dict(protocol=protocol, protocol_id=protocol_id, paired_plan=asdict(plan)))
    def recheck():
        checked.validate_inputs(compilation, policy)  # Existing seal stays strict.
        if any(checked.digest(p) != sha for p, sha in hashes.items()):
            raise ValueError('current source fingerprint changed; do not reseal or relax')
    recheck()
    started = perf_counter()
    host = require_cuda_allocation('cuda')
    device = torch.device('cuda:0')
    model = load_compatible_pretrained(str(Path(policy).resolve()), device=device).eval().requires_grad_(False)
    replay = adapter.CudaReplay(model, device)
    torch.cuda.synchronize()
    setup_seconds = perf_counter() - started
    before = checked.tensor_state_hash(model)
    fixed, adaptive, rows = [], [], []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for index, (row, root) in enumerate(zip(source['rows'], roots, strict=True)):
            # Deterministically alternate order without selecting a root.
            for mode in (('fixed', 'adaptive') if index % 2 == 0 else ('adaptive', 'fixed')):
                run_receipt, detail = run_arm(model=model, replay=replay, row=row,
                    binding=root.contract.binding, policy=frozen, mode=mode,
                    synchronize=torch.cuda.synchronize)
                (fixed if mode == 'fixed' else adaptive).append(run_receipt)
                rows.append(detail)
                checked.write_new(output / f'P3_ADAPTIVE_RUNTIME_{root.contract.binding.root_id}_{mode}.json', detail)
                print(f'{root.contract.binding.root_id} {mode}: {detail["resources"]["predict_samples"]} attempted samples; failure={detail["failure"]}', flush=True)
    after = checked.tensor_state_hash(model)
    pair = compare_paired(plan, fixed, adaptive)
    source_failure = None
    try:
        recheck()
    except Exception as exc:
        source_failure = str(exc)
    gates = dict(model_unchanged=before == after, sources_unchanged=source_failure is None,
        all_frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        full_cohort=len(rows) == 8, all_replays=all(r['full_replay_exact'] for r in rows),
        no_failures=all(r['failure'] is None for r in rows),
        action_encoder_caps=all(r['action_encoder_samples'] <= 147 for r in rows),
        processed_action_context_caps=all(r['processed_action_context_tokens'] <= 441 for r in rows),
        action_context_accounting=all(r['processed_action_context_tokens'] == 3 * r['action_encoder_samples']
            and r['resources']['imagined_action_tokens'] == r['action_encoder_samples'] for r in rows),
        vector_caps=pair['all_pairs_within_budget'])
    checked.write_new(output / 'P3_ADAPTIVE_RUNTIME_report_20261027.json', dict(
        status='runtime_integration_completed' if all(gates.values()) else 'integrity_fail',
        gates=gates, rows=rows, paired=pair, host=host, shared_model_setup_seconds=setup_seconds,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), source_failure=source_failure,
        protocol_id=protocol_id, p1_dependency='not_passed', p3_admitted=False,
        compute_advantage=False, walltime_advantage=False, efficacy_claim=False))
    if not all(gates.values()):
        raise RuntimeError('runtime integration failed; preserve every receipt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('compilation', 'policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    run(**vars(parser.parse_args()))
