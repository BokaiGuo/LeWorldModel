"""Frozen-bank CUDA search integration; not official baseline or efficacy proof."""
import argparse
import json
from pathlib import Path
import numpy as np


def metric_for(compiled):
    goals = np.asarray(compiled['goal_vectors'], dtype=np.float64)
    projection = np.asarray(compiled['scoring']['metric']['projection'], dtype=np.float64)
    if (goals.ndim != 2 or goals.shape[1] != 192 or projection.shape != (192, 192)
            or not np.isfinite(goals).all() or not np.isfinite(projection).all()):
        raise ValueError('finite saved 192-dimensional metric and goals required')
    # Variants define the nuisance projector; goal[0] is the actual task goal.
    def cost(context):
        terminal = np.asarray(context['latent_history'][-1], dtype=np.float64)
        if terminal.shape != (192,) or not np.isfinite(terminal).all():
            raise ValueError('finite terminal required')
        delta = (terminal - goals[0]) @ projection.T
        return float(np.mean(delta * delta))
    return cost


def run(compilation, policy, output):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import d2p_kinodynamic_program_search as search
    from experiments.spectral_reachability import pusht_composition_cuda_replay_v2 as adapter
    from experiments.spectral_reachability.d2p_adaptive_budget_v2 import PredictionBudget, Binding
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    import torch
    source, bindings, config = checked.validate_inputs(compilation, policy)
    output = Path(output)
    output.mkdir(exist_ok=False)
    method = 'fixed_bank_depth2_beam4_cuda_v1'
    protocol = dict(bindings=bindings, max_depth=2, beam_width=4,
                    search_prediction_cap=80, verification_prediction_cap=2,
                    max_attempted_expansions=80, bank='16 saved prefix raw commands repeated five times',
                    selector='best nonempty visited; variable horizon 1 or 2; no padding',
                    physical_outcomes_used=False, P1_admission=False,
                    official_paper_faithfulness=False, fair_total_budget_comparison=False)
    files = checked.sources_before_allocate(source, config)
    files = sorted(set(files + [Path(__file__).resolve()]))
    # Separate manifests only to respect the seal utility's file-count limit.
    seals = [create_seal(files=files[i:i+50], args_json=json.dumps(protocol),
                         output_dir=output/f'source_seal_{i//50}')
             for i in range(0, len(files), 50)]
    base.write_new(output/'protocol.json', protocol)
    host = base.require_cuda_allocation('cuda')
    device = torch.device('cuda:0')
    model = base.load_compatible_pretrained(Path(policy), device=device).eval().requires_grad_(False)
    before = base.p0._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    replay = adapter.CudaReplay(model, device)
    rows = []
    for row in source['rows']:
        compiled = row['compiled']
        initial = compiled['input_context']
        context = dict(layout='states_with_between_state_actions_v1', context_length=3,
                       latent_dim=192, action_dim=10, latent_history=initial['emb'][0],
                       action_history=initial['actions'][0][1:])
        bank = [np.repeat(np.asarray(r['program_live_commands'][0])[None], 5, axis=0)
                for r in compiled['receipts']]
        binding = Binding(seals[0]['manifest_sha256'], 'pusht_fixed_bank_search',
                          row['root_sha256'], checked.OFFICIAL_SHA)
        with PredictionBudget(model, binding=binding, method=method, sample_limit=82) as budget:
            with budget.phase('planning', method=method, horizon_tokens=2):
                result = search.search_programs(primitive_bank=bank, root_context=context,
                    replay=replay, target_metric=metric_for(compiled), max_depth=2,
                    beam_width=4, max_attempted_expansions=80)
            planning_spent = budget.receipt()['attempted_predict_samples']
            if planning_spent != 80 or result['status'] != 'completed':
                raise ValueError('fixed 16+4*16 search work did not complete')
            best = result['best_plan']
            if best is None:
                raise ValueError('no finite search witness')
            with budget.phase('evaluation', method=method, horizon_tokens=best['depth']):
                full = replay(context, best['actions'])
            exact = full == best['context']
            if not exact:
                raise ValueError('whole-action replay differs from search recurrence')
            receipt = budget.receipt()
        value = dict(root_sha256=row['root_sha256'], root_index=row['root_index'],
                     search=result, budget=receipt, full_replay_exact=exact,
                     planning_predict_samples=planning_spent)
        base.write_new(output/f'root_{row["root_index"]}.json', value)
        rows.append(value)
        print(f'root {row["root_index"]}: 80 expansions, best depth {best["depth"]}, exact replay', flush=True)
    gates = dict(four_roots=len(rows)==4, all_full_replays=all(r['full_replay_exact'] for r in rows),
                 actual_planning_samples_320=sum(r['planning_predict_samples'] for r in rows)==320,
                 actual_budget_caps=all(r['budget']['attempted_predict_samples']<=82 for r in rows),
                 parameters_unchanged=before==base.p0._parameter_hash(model),
                 buffers_unchanged=all(torch.equal(buffers[k],v) for k,v in model.named_buffers()),
                 eval=all(not m.training for m in model.modules()),
                 inputs_unchanged=all(base.digest(Path(p))==d for p,d in bindings.items()))
    verified = [verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256']) for s in seals]
    base.write_new(output/'report.json', dict(schema='pusht_program_search_cuda.v1',
        status='integration_completed' if all(gates.values()) else 'integrity_fail',
        protocol=protocol, host=host, rows=rows, gates=gates, source_seals=seals,
        source_verification=verified, P1_admission=False, P2_admission=False, P3_admission=False))
    if not all(gates.values()):
        raise RuntimeError('search integration failed integrity')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('compilation', 'policy', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(**vars(parser.parse_args()))
