"""Offline source-anchored Hi FIRST/scaler and prospective three-arm ABI.

No imports of native/model code, no fitting, inference, physics or network.
This is not an authority to run or a certificate of historical scaler values.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path

NAME = 'agent_e1_hilewm_time_scaler_contract_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
TEXT = 'results/p1_hilewm_text_contract_20261027_v1/texts/'
SOURCES = {
    'eval': (TEXT+'package__code__h_le_wm__eval__hierarchical.py.source.txt', '0a2c2b2688b73a388ff6011c4b32081704a7641b31efa1645a47b2f1d34eab3c'),
    'policy': (TEXT+'package__code__h_le_wm__planning__policies.py.source.txt', 'd6e7d476b45e47c5015494775b4d96727b8ebb8c801f719a100ba32878f288d1'),
    'model': (TEXT+'package__code__h_le_wm__models__jepa.py.source.txt', 'bb5338bbb4f9e76b0bd15b94767624168883c994ab4c2ad8209337a3d70022a2'),
    'sampling': (TEXT+'package__code__h_le_wm__models__waypoint_sampling.py.source.txt', 'ae072ca39f6c257cc8c2db543998f8eda0d99a6ad235d871cdab00993180817b'),
    'training_config': (TEXT+'package__checkpoints__pusht__main__config.yaml.source.txt', 'dff6ac0dc9e7c4cfb7d76b268ceb568f6fc0916eaa4a725e6e1a5b53587e9075'),
    'eval_config': (TEXT+'package__code__config__eval__hi_pusht.yaml.source.txt', '35b02a4f6e22dd9fecf034d664e799b2c5c85f157750ba53e0d69ba0be1091cc'),
    'matrix': (TEXT+'package__code__h_le_wm__experiments__matrix__pusht__hierarchical_matrix.csv.source.txt', 'b02675192db8ce6613ea9885ffaf2db8492e00bf77bbdb80a5db5f1690ad0dc9'),
    'mirror_config': ('results/agent_hilewm_normalizer_asset_route_20261029_v2_small_metadata01/metadata_02.bin', '2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09'),
    'mirror_report': ('results/agent_hilewm_normalizer_asset_route_20261029_v2_mirror01/report.json', 'f8eeae02c04ef66449243f687c199799c117fb438da9922451bdd1853d6d2256'),
    'small_report': ('results/agent_hilewm_normalizer_asset_route_20261029_v2_small_metadata01/report.json', '1de9c82ac42613de2973d0d5cdb4d83b34239612dfab6b92aba85f889f423034'),
    'first_integration': ('results/agent_hilewm_rgb_high_low_20261028_v1_consume_MAIN01.json', '40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0'),
}
ARMS = ('arbitrary_latent_waypoint', 'existing_learned_prior', 'dynamics_compiled_witness')
SHARED = ('dataset_revision', 'episode_start', 'goal_frame_offset', 'goal_bytes_sha256',
          'history_bytes_sha256', 'model_loaded_bytes_sha256', 'preprocess_sha256',
          'action_scaler_export_sha256', 'controller_source_sha256', 'controller_config_sha256',
          'raw_action_domain', 'native_environment_sha256', 'feedback_schedule', 'eval_budget')
CAP = 4 * 1024 * 1024


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def decode(data):
    def pairs(values):
        out = {}
        for key, value in values:
            need(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def bad(_):
        raise ValueError('nonfinite JSON')
    need(len(data) <= CAP, 'metadata cap')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def pinned_sources(root=ROOT):
    result = {}
    for key, (relative, expected) in SOURCES.items():
        path = root / relative
        need(path.is_file() and path.stat().st_size <= CAP, 'missing or oversized source: '+key)
        data = path.read_bytes()
        need(sha(data) == expected, 'source SHA mismatch: '+key)
        result[key] = data
    return result


def function(tree, name, owner=None):
    nodes = tree.body
    if owner:
        cls = [n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner]
        need(len(cls) == 1, 'class '+owner)
        nodes = cls[0].body
    found = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    need(len(found) == 1, 'function '+name)
    return found[0]


def statement(fn, code):
    wanted = ast.dump(ast.parse(code).body[0], include_attributes=False)
    found = [n for n in ast.walk(fn) if ast.dump(n, include_attributes=False) == wanted]
    need(len(found) == 1, 'semantic statement mismatch: '+code)
    return found[0].lineno


def source_proof(data):
    ev, po, mo, sa = (ast.parse(data[k].decode('utf-8')) for k in ('eval', 'policy', 'model', 'sampling'))
    process = function(ev, 'build_process_map')
    builder = function(ev, 'build_policy')
    policy = 'HierarchicalWorldModelPolicy'
    high = function(po, '_plan_high', policy)
    low = function(po, '_plan_low', policy)
    action = function(po, 'get_action', policy)
    prep = function(po, '_prepare_info', policy)
    rollout = function(mo, 'rollout_high', 'HiJEPA')
    sampler = function(sa, '_sample_random_sorted')
    # Pin entire native files as well as checking the semantically relevant AST.
    checks = {
        'scaler_construct': statement(process, 'processor = preprocessing.StandardScaler()'),
        'source_column': statement(process, 'col_data = dataset.get_col_data(col)'),
        'remove_nan_rows': statement(process, 'col_data = col_data[~np.isnan(col_data).any(axis=1)]'),
        'fit': statement(process, 'processor.fit(col_data)'),
        'store_process': statement(process, 'process[col] = processor'),
        'high_solve': statement(high, 'outputs = self.high_solver(high_info, init_action=high_init_action)'),
        'high_inputs': statement(high, 'high_info = {"planner_level": "high", "z_init": z_init, "z_goal": z_goal}'),
        'returned_actions': statement(high, 'actions_solver = outputs["actions"]'),
        'high_rollout': statement(high, 'pred = self.model.rollout_high(z_init, macro_seq)'),
        'first_target': statement(high, 'self._z_subgoal = pred[:, 0, 0, :]'),
        'low_uses_first': statement(low, 'outputs = self.low_solver(low_info, init_action=low_init_action)'),
        'low_inputs': statement(low, 'low_info = {"planner_level": "low", "z_hist": z_init.unsqueeze(1), "a_hist": a_hist, "z_subgoal": self._z_subgoal}'),
        'prepare_normalize': statement(prep, 'v = self.process[k].transform(v)'),
        'action_inverse': statement(action, 'action_np = self.process["action"].inverse_transform(action_np)'),
        'replan_condition': statement(action, 'need_high = self._z_subgoal is None or self._steps_since_high >= self.macro_replan_interval'),
        'variable_span': statement(sampler, 'total_span = torch.randint(low=min_total_span, high=max_available_span + 1, size=(batch_size,), device=device)'),
        'random_allocation': statement(sampler, 'residual_alloc = torch.multinomial(probs, num_samples=int(residual.max().item()) if int(residual.max().item()) > 0 else 1, replacement=True)'),
        'model_predict': statement(rollout, 'z_next = self.predict_high(z_ctx, a_ctx)[:, -1:, :]'),
    }
    need([x.arg for x in rollout.args.args] == ['self', 'z_init', 'latent_actions'], 'high rollout ABI')
    policy_calls = [x for x in ast.walk(builder) if isinstance(x, ast.Call)
                    and isinstance(x.func, ast.Name) and x.func.id == 'HierarchicalWorldModelPolicy']
    need(len(policy_calls) == 1, 'native hierarchical constructor')
    kw = {k.arg: ast.unparse(k.value) for k in policy_calls[0].keywords}
    need(kw.get('process') == 'process' and kw.get('macro_replan_interval') == 'int(cfg.planning.high.replan_interval)',
         'process/replan policy propagation')
    checks['policy_process_binding'] = policy_calls[0].lineno
    # Capture actual main-scope propagation, rather than assuming process is used.
    main = next((n for n in ev.body if isinstance(n, ast.FunctionDef) and any(
        isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == 'evaluate_from_dataset'
        for x in ast.walk(n))), None)
    need(main is not None, 'eval entry')
    checks.update({
        'evaluation_dataset': statement(main, 'dataset = get_dataset(cfg, cfg.eval.dataset_name)'),
        'evaluation_stats': statement(main, 'stats_dataset = dataset'),
        'evaluation_process': statement(main, 'process = build_process_map(cfg, stats_dataset)'),
        'evaluation_policy': statement(main, 'policy = build_policy(cfg, model, dataset, process, transform)'),
    })
    calls = [x for x in ast.walk(main) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == 'evaluate_from_dataset']
    need(len(calls) == 1, 'single eval call')
    kws = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    need(kws.get('goal_offset_steps') == 'cfg.eval.goal_offset_steps' and kws.get('eval_budget') == 'cfg.eval.eval_budget', 'goal/eval ABI')
    checks['dataset_evaluation'] = calls[0].lineno
    return checks


def matrix(data):
    rows = list(csv.DictReader(io.StringIO(data.decode()), delimiter=';'))
    result = []
    for index, row in enumerate(rows):
        if row['goal_offset'] != 'D50':
            continue
        r = {k: (v if k in ('goal_offset', 'config_note') else int(v)) for k, v in row.items()}
        need(r['eval_budget'] == 100, 'D50 eval budget')
        r['source_csv_line'] = index + 2
        # A per-solve configured sampling ceiling, not a measured dispatch.
        r['high_samples_per_solve'] = r['high_num_samples'] * r['high_n_steps']
        r['low_samples_per_solve'] = r['low_num_samples'] * r['low_n_steps']
        result.append(r)
    need(len(result) == 8, 'all D50 rows')
    return result


def reference_ceiling(row):
    """Prospective six independent nonstaged full episodes, no unmeasured costs inferred."""
    need(row['config_note'] == '' and row['high_replan_interval'] == 5 and
         row['low_receding_horizon'] == 1 and row['low_action_block'] == 5, 'reference schedule')
    episodes, steps = 6, row['eval_budget']
    high_calls = episodes * ((steps + row['high_replan_interval'] - 1) // row['high_replan_interval'])
    low_interval = row['low_receding_horizon'] * row['low_action_block']
    low_calls = episodes * ((steps + low_interval - 1) // low_interval)
    return dict(episodes=episodes, raw_control_calls=episodes*steps, per_episode_raw_control_cap=steps,
        high_solver_invocation_cap=high_calls, low_solver_invocation_cap=low_calls,
        high_configured_candidate_sequences_cap=high_calls*row['high_samples_per_solve'],
        low_configured_candidate_sequences_cap=low_calls*row['low_samples_per_solve'],
        high_candidate_macro_slots_cap=high_calls*row['high_samples_per_solve']*row['high_horizon']*row['high_action_block'],
        low_candidate_grouped_action_slots_cap=low_calls*row['low_samples_per_solve']*row['low_horizon'],
        # This is a model of a future schedule, not a count of native module calls.
        winner_replay_module_calls=None, proposal_generator_calls=None, encoder_calls=None,
        native_reset_and_warmup=None, physics_substeps=None, output_bytes=None, gpu_bytes=None,
        wall_seconds=None, status='conditional_reference_not_run_authority')


def compare_packets(packets):
    """Strict structural ABI check only; a self-declared hash is not evidence."""
    need(type(packets) is list and len(packets) == 6, 'six fixed slots')
    expected = [(root, arm) for root in (0, 1) for arm in ARMS]
    need([(p.get('root_slot'), p.get('arm')) for p in packets] == expected, 'root/arm ordering')
    out = []
    for root in (0, 1):
        group = packets[root*3:root*3+3]
        fields = {}
        for key in SHARED:
            vals = [p.get('shared', {}).get(key) for p in group]
            present = [v for v in vals if v is not None]
            status = ('mismatch' if present and any(encoded(v) != encoded(present[0]) for v in present[1:])
                      else 'unknown' if len(present) != 3 else 'match')
            fields[key] = status
        # Matching a deliberately wrong goal horizon must not pass the D50 ABI.
        for p in group:
            shared = p.get('shared', {})
            need(shared.get('goal_frame_offset') in (None, 50), 'D50 goal offset')
            need(shared.get('eval_budget') in (None, 100), 'D50 eval budget')
        out.append(dict(root_slot=root, fields=fields))
    return out


def build(root=ROOT):
    data = pinned_sources(root)
    proof = source_proof(data)
    config = decode(data['mirror_config'])
    need(config['projector']['norm_fn']['_target_'] == 'torch.nn.BatchNorm1d', 'mirror config ABI')
    need(config['pred_proj']['norm_fn']['_target_'] == 'torch.nn.BatchNorm1d', 'mirror config ABI')
    rows = matrix(data['matrix'])
    reference = rows[0]
    return dict(schema=NAME+'.prepare',
        sources={k: dict(path=str((root/v[0]).resolve()), sha256=v[1]) for k,v in SOURCES.items()},
        self_sources={str(p): sha(p.read_bytes()) for p in (
            root/'experiments/spectral_reachability'/ (NAME+'.py'), root/'tests'/('test_'+NAME+'.py'), root/'innerdocs'/(NAME+'.md'))},
        ast_lines=proof, d50_matrix=rows,
        semantics=dict(scaler='eval action column non-NaN rows -> StandardScaler.fit -> same process inverse_transform',
            original_scaler_values=None, original_scaler_byte_authority=False,
            candidate_bn_is_action_scaler=False, source_proof_is_historical_loaded_bytes=False,
            first='high solver returned actions -> paid rollout_high -> pred[:,0,0,:]',
            first_physical_offset=None, goal_dataset_offset=50, official_d50_eval_budget=100,
            replan_interval_is_hold_not_prediction_duration=True, hi_is_only_h50=False,
            training_waypoint_spacing='random_sorted variable gaps; frameskip=5; no deterministic FIRST offset',
            staged_policy_is_separate=True, h10_compatibility=None),
        prospective=dict(arms=list(ARMS), root_slots=[0,1], all_six_required=True,
            shared_fields=list(SHARED), goal_offset=50, eval_budget=100,
            control_contract='same model coordinates, scaler, task, history, feedback, raw domain and controller for all arms',
            first_target='learned FIRST is a target, not a timestamped physical state; no implicit projection or substitution',
            raw_inverse='original authenticated scaler for all arms or separately approved common new scaler for all arms; do not call either historical parity without proof',
            witness='model action witness differs from real physical arrival; no unapproved arrival threshold',
            reference_choice='first D50 nonstaged CSV row solely as prospective concrete envelope; MAIN must approve before use',
            reference_csv_line=reference['source_csv_line'], reference_ceiling=reference_ceiling(reference),
            remaining_before_dispatch=['independent approved cohort with real dataset offset50 goal and full history',
                'actual scaler/process bytes and provenance or explicitly approved new common-domain scaler',
                'same frozen model/encoder/controller loaded bytes and coordinate receipts for all arms',
                'presealed proposal and replay module/byte/wall/physics caps for every arm',
                'approved arrival metric and proof of task/target identity',
                'native action domain and reset/warmup accounting'],
            physical_release=False, scientific_admission=False, historical_parity=False, ready=False),
        execution=dict(fit=False, network=False, inference=False, physics=False, fullstrict=False))


def read_exact(path, digest):
    p = Path(path)
    need(p.is_absolute() and p.is_file() and not p.is_symlink(), 'absolute regular file')
    data = p.read_bytes()
    need(sha(data) == digest, 'external SHA mismatch')
    return decode(data)


def write_new(path, value):
    p = Path(path)
    need(p.is_absolute() and not p.is_symlink(), 'absolute new output')
    data = encoded(value)
    need(len(data) <= CAP, 'output cap')
    with p.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return dict(path=str(p), sha256=sha(data))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('prepare', 'verify', 'check-abi'))
    ap.add_argument('--output')
    ap.add_argument('--manifest')
    ap.add_argument('--manifest-sha256')
    ap.add_argument('--packets')
    ap.add_argument('--packets-sha256')
    args = ap.parse_args()
    if args.command == 'prepare':
        need(args.output is not None, 'output required')
        result = write_new(args.output, build())
    elif args.command == 'verify':
        need(args.manifest and args.manifest_sha256, 'manifest and SHA required')
        need(read_exact(args.manifest, args.manifest_sha256) == build(), 'whole prepare reconstruction')
        result = dict(status='offline_source_contract_verified_not_authorized')
    else:
        need(args.packets and args.packets_sha256, 'packet file and SHA required')
        # Require the same pinned source proof before comparing future packets.
        build()
        result = dict(schema=NAME+'.abi_check', fields=compare_packets(read_exact(args.packets, args.packets_sha256)),
                      identity_evidence_authenticated=False, dispatch_allowed=False)
        if args.output:
            result = write_new(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
