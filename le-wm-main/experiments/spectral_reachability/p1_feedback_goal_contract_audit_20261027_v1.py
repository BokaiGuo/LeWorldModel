"""Offline, fixed-root goal/Q contract audit. Never imports a model or simulator.

Only saved vectors, images, geometry receipts and source text are consumed.
This checks contracts, not physical benefit or checkpoint-training conventions.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

TAG = 'p1_feedback_goal_contract_audit_20261027_v1'
REPO = Path(__file__).resolve().parents[2]
ROOT = '23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050'
ORDER = [(r, a) for r in ('selected', 'greedy', 'zero') for a in ('imagined', 'actual')]
DIRS = dict(physical='pusht_feedback_suffix_physical_20261027_v1_prepare_v1',
    compiler='pusht_feedback_suffix_compile_20261027_v1_prepare_v1',
    feedback='pusht_geometry_feedback_response_20261027_v1_prepare_v1',
    geometry='pusht_goal_variant_actual_geometry20_20261027',
    intervention='pusht_d2p_goal_agent_intervention4_v2_20261026_cuda',
    metric='pusht_d2p_metric_program_preflight4_v1_20261026_cuda',
    original='pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda',
    representation='pusht_representation_runtime_audit4_v1_20261027_cuda')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def array(value, shape):
    a = np.asarray(value, dtype=float)
    require(a.shape == shape and np.isfinite(a).all(), f'finite {shape} required')
    return a


def close(a, b, name, atol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=atol, rtol=0), name)


def projected_cost(q, goal, vector):
    return float(np.mean((q @ (goal-vector))**2))


def projection_from_goals(goals):
    goals = array(goals, (5, 192))
    directions = (goals[1:]-goals[0]).T
    u, s, _ = np.linalg.svd(directions, full_matrices=False)
    rank = int(np.sum(s > s[0]*max(directions.shape)*np.finfo(float).eps))
    return np.eye(192)-u[:, :rank]@u[:, :rank].T, rank


def official(goal, state):
    """Exact installed evaluator formula; AST equivalence checked before use."""
    pos_diff = np.linalg.norm(goal[:4] - state[:4])
    angle_diff = np.abs(goal[4] - state[4])
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    success = pos_diff < 20 and angle_diff < np.pi / 9
    state_dist = np.linalg.norm(goal - state)
    return bool(success), float(state_dist)


def finite_witnesses(goals, geometry, q):
    require(len(geometry) == 5, 'all five geometry variants required')
    offsets = [[0., 0.], [-32., 0.], [32., 0.], [0., -32.], [0., 32.]]
    target = array(geometry[0]['requested_goal'], (7,))
    rows = []
    for i, v in enumerate(geometry):
        require(v['offset'] == offsets[i], 'fixed goal variant order')
        requested = array(v['requested_goal'], (7,))
        actual = array(v['actual_goal_render_state'], (7,))
        close(requested, actual, 'actual goal render geometry differs', atol=0)
        close(requested[:2], target[:2]+offsets[i], 'agent goal offset differs', atol=0)
        close(requested[2:], target[2:], 'object/velocity goal changed', atol=0)
        success, dist = official(target, actual)
        rows.append(dict(offset=v['offset'], Q_goal_MSE=projected_cost(q, goals[0], goals[i]),
            raw_goal_MSE=float(np.mean((goals[0]-goals[i])**2)),
            official_success_vs_original=success, official_state_distance_vs_original=dist,
            object_xy_error=float(np.linalg.norm(actual[2:4]-target[2:4])),
            object_angle_error=float(abs(np.arctan2(np.sin(actual[4]-target[4]), np.cos(actual[4]-target[4])))),
            actual_equals_requested=True))
    require(rows[0]['official_success_vs_original'] and
            all(not v['official_success_vs_original'] for v in rows[1:]), 'official witness contract')
    require(max(v['Q_goal_MSE'] for v in rows) < 1e-24, 'goal directions not nulled')
    return rows


def audit_plan(plan, case, q, goal):
    for k in ('root_sha256', 'root_index', 'role', 'arm', 'prefix', 'context'):
        require(plan[k] == case[k], f'plan/case {k} binding')
    require(plan['root_sha256'] == ROOT and plan['root_index'] == 1, 'fixed root')
    close(plan['q'], q, 'plan Q source differs', atol=0)
    close(plan['goal'], goal, 'plan goal source differs', atol=0)
    candidates = plan['candidates']
    require(len(candidates) == 10 and [c['candidate_id'] for c in candidates] ==
            [f'suffix_{i}' for i in range(10)], 'complete unique ordered candidates')
    costs = []
    for c in candidates:
        commands = array(c['commands'], (10, 2))
        close(commands[:5], plan['prefix'], 'candidate prefix binding', atol=0)
        # GPU float32 tanh; this is a coordinate check, not a new prediction.
        close(commands[5:], np.tanh(array(c['suffix_params'], (10,))).reshape(5, 2),
              'candidate raw/tanh binding', atol=1e-7)
        require(np.max(np.abs(commands)) <= 1, 'raw commands out of bounds')
        value = projected_cost(q, goal, array(c['predicted_terminal'], (192,)))
        close(value, c['nonlinear_terminal_cost'], 'candidate objective differs')
        costs.append(value)
    require(candidates[int(np.argmin(costs))]['candidate_id'] == plan['selected_candidate_id'],
            'not stable model-only argmin')
    require(plan['independent_replay_exact'], 'independent prediction replay missing')
    return dict(method=plan['role']+'_'+plan['arm'], candidate_costs=costs,
                selected_candidate_id=plan['selected_candidate_id'], costs_exact=True)


def method_source(path, name):
    text = Path(path).read_text()
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1, f'unique source method {name}')
    return ast.get_source_segment(text, nodes[0]), nodes[0]


def public_action_stats(path):
    values = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('PUBLIC_ACTION_MEAN', 'PUBLIC_ACTION_STD'):
                require(isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'np.asarray',
                        'literal public stats required')
                values[name] = array(ast.literal_eval(node.value.args[0]), (2,)).astype(np.float32)
    require(len(values) == 2 and np.all(values['PUBLIC_ACTION_STD'] > 0), 'public stats missing/invalid')
    return np.tile(values['PUBLIC_ACTION_MEAN'], 5), np.tile(values['PUBLIC_ACTION_STD'], 5)


def check_evaluator(path):
    _, installed = method_source(path, 'eval_state')
    _, local = method_source(__file__, 'official')
    # Compare arithmetic AST after renaming only local variables and casting
    # wrappers on the return. No untrusted source code is executed.
    class Names(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id={'goal_state':'goal', 'cur_state':'state'}.get(node.id, node.id),
                                             ctx=node.ctx), node)
    installed_body = [ast.dump(Names().visit(n), include_attributes=False) for n in installed.body[:-1]]
    local_body = [ast.dump(n, include_attributes=False) for n in local.body[1:-1]]
    require(installed_body == local_body, 'installed official evaluator formula changed')
    require(ast.dump(installed.body[-1].value) == ast.dump(ast.Tuple(elts=[ast.Name(id='success',ctx=ast.Load()),
            ast.Name(id='state_dist',ctx=ast.Load())],ctx=ast.Load())), 'evaluator return changed')


class Inputs:
    def __init__(self):
        self.bindings = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.bindings:
            self.bindings[key] = sha(path)
        require(expected is None or self.bindings[key] == expected, f'input hash mismatch: {path}')
        return path

    def load(self, path):
        return read(self.bind(path))

    def seal(self, seal):
        directory = Path(seal['output_dir'])
        m = self.load(self.bind(directory/'manifest.json', seal['manifest_sha256']))
        self.bind(directory/'args.json', m['args_sha256'])
        for e in m['files']:
            self.bind(e['source'], e['sha256'])
            self.bind(directory/e['archive'], e['sha256'])

    def unchanged(self):
        require(all(sha(p) == h for p, h in self.bindings.items()), 'inputs changed during audit')


def collect():
    inputs = Inputs()
    dirs = {k: REPO/'results'/v for k, v in DIRS.items()}
    reports = {k: inputs.load(d/'report.json') for k, d in dirs.items()}
    protocols = {}
    for k, d in dirs.items():
        protocols[k] = inputs.load(d/'protocol.json') if (d/'protocol.json').exists() else reports[k]['protocol']
        for p, h in protocols[k].get('bindings', {}).items():
            inputs.bind(p, h)
        for s in reports[k].get('source_seals', []):
            inputs.seal(s)
        for field in ('source_seal', 'evaluator_source_seal'):
            if reports[k].get(field):
                inputs.seal(reports[k][field])
        if (d/'prepare_receipt.json').exists():
            receipt = inputs.load(d/'prepare_receipt.json')
            inputs.bind(d/'protocol.json', receipt['protocol_sha256'])
            for s in receipt['source_seals']:
                inputs.seal(s)
    for e in reports['physical']['encodings']:
        inputs.bind(e['rgb_path'], e['rgb_sha256'])
    for p in protocols['physical']['plans']:
        inputs.bind(p['expected_prefix_rgb'])
    return inputs, reports, protocols


def audit(inputs, reports, protocols):
    def rootrow(key):
        rows = [r for r in reports[key]['rows'] if r['root_sha256'] == ROOT]
        require(len(rows) == 1, f'unique {key} root')
        return rows[0]
    require(reports['compiler']['status'] == 'plans_sealed' and reports['compiler']['failure'] is None,
            'compiler not complete')
    require(reports['physical']['status'] == 'calibration_recorded' and not reports['physical']['failures'],
            'physical not complete')
    for key in ('geometry', 'metric', 'representation'):
        require(all(reports[key]['gates'].values()), f'{key} evidence gates')
    require(all(reports['representation']['checks'].values()), 'representation runtime evidence checks')
    compiled = rootrow('metric')['compiled']
    variants = rootrow('intervention')['variants']
    require(len(variants) == 5, 'five original encoded goals required')
    goals = array([v['goal_vector'] for v in variants], (5, 192))
    close(goals, compiled['goal_vectors'], 'metric/intervention goal lineage', atol=0)
    q = array(compiled['scoring']['metric']['projection'], (192, 192))
    rebuilt, rank = projection_from_goals(goals)
    close(q, rebuilt, 'Q differs from original finite-goal SVD')
    close(q, q.T, 'Q not symmetric')
    close(q@q, q, 'Q not idempotent')
    require(rank == compiled['scoring']['metric']['rank'] == 4, 'original nuisance rank')
    close(protocols['feedback']['goal'], goals[0], 'feedback goal swap', atol=0)
    close(protocols['feedback']['q'], q, 'feedback Q swap', atol=0)
    geometry = rootrow('geometry')['variants']
    for a, b in zip(variants, geometry, strict=True):
        require(a['offset'] == b['offset'] and a['goal_rgb_sha256'] == b['goal_rgb_sha256'],
                'goal image/geometry lineage')
    require(rootrow('original')['goal_rgb_sha256'] == geometry[0]['goal_rgb_sha256'], 'original RGB goal differs')
    evaluator = REPO/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py'
    inputs.bind(evaluator); check_evaluator(evaluator)
    witnesses = finite_witnesses(goals, geometry, q)
    plans = reports['compiler']['plans']; cases = protocols['compiler']['cases']
    physical_plans = protocols['physical']['plans']; rows = reports['physical']['rows']
    for sequence in (plans, cases, physical_plans, rows):
        require([(p['role'], p['arm']) for p in sequence] == ORDER, 'six fixed ordered contexts required')
    costs = []; physical_checks = []; encoding_checks = []
    contexts = {r['role']:r for r in reports['feedback']['contexts']}
    source_cases = {r['role']:r for r in protocols['feedback']['cases']}
    mean, std = public_action_stats(inputs.bind(REPO/'experiments/run_pusht_counterfactual.py'))
    require(protocols['compiler']['original_initial_context'] == protocols['feedback']['initial_context'] and
            protocols['compiler']['original_root_rgb'] == protocols['feedback']['root_rgb'], 'initial root context binding')
    for plan, case, pp, row in zip(plans, cases, physical_plans, rows, strict=True):
        costs.append(audit_plan(plan, case, q, goals[0]))
        method = plan['role']+'_'+plan['arm']; role = plan['role']; arm = plan['arm']
        require(row['root_sha256'] == ROOT and row['root_index'] == 1 and row['method'] == method,
                'physical row/root binding')
        close(plan['prefix'], source_cases[role]['prefix'], 'original prefix binding', atol=0)
        require(plan['context'] == contexts[role][arm] == pp['predictor_context'], 'context provenance')
        close(contexts[role]['imagined']['latent_history'][:2], contexts[role]['actual']['latent_history'][:2],
              'older latent isolation', atol=0)
        close(contexts[role]['imagined']['action_history'], contexts[role]['actual']['action_history'],
              'action history isolation', atol=0)
        raw = array(source_cases[role]['between_raw'], (2, 5, 2))
        expected = np.stack([raw[1], array(plan['prefix'], (5, 2))]).astype(np.float32).reshape(2,10)
        close((expected-mean)/std, plan['context']['action_history'], 'incoming packet order/normalization', atol=1e-6)
        winner = next(c for c in plan['candidates'] if c['candidate_id'] == plan['selected_candidate_id'])
        require(pp['commands'] == winner['commands'] and pp['selected_candidate_id'] == winner['candidate_id'],
                'physical winner binding')
        close(pp['goal'], goals[0], 'physical latent goal binding', atol=0)
        close(pp['q'], q, 'physical Q binding', atol=0)
        close(pp['saved_candidate']['predicted_terminal'], winner['predicted_terminal'], 'physical terminal binding', atol=0)
        close(pp['saved_candidate']['predicted_prefix'], plan['context']['latent_history'][-1], 'prefix prediction binding', atol=0)
        close(row['context_identity']['goal'], geometry[0]['requested_goal'], 'physical state goal swap', atol=0)
        require(row['context_identity']['goal_rgb_sha256'] == geometry[0]['goal_rgb_sha256'], 'physical goal RGB swap')
        require(row['status'] == 'completed' and row['completed_commands'] == 10 and len(row['trace']) == 10,
                'complete ten-step branch required')
        for i, event in enumerate(row['trace']):
            require(event['primitive_step'] == i+1 and event['command'] == winner['commands'][i], 'executed command binding')
        require(row['prefix_identity']['pixels_exact'], 'prefix pixels mismatch')
        target = array(geometry[0]['requested_goal'], (7,))
        success, distance = official(target, array(row['final_state'], (7,)))
        require(success == row['official']['official_success'], 'official final label differs')
        close(distance, row['official']['official_state_distance'], 'official final distance differs')
        state = array(row['final_state'], (7,))
        position = float(np.linalg.norm(state[2:4]-target[2:4]))
        angle = float(abs(np.arctan2(np.sin(state[4]-target[4]), np.cos(state[4]-target[4]))))
        close(position, row['pose']['xy_distance'], 'original pose distance differs')
        close(angle, row['pose']['angle_distance'], 'original pose angle differs')
        close(position+20*angle, row['pose']['pose_cost'], 'original pose cost differs')
        physical_checks.append(dict(method=method, original_goal_exact=True, commands_exact=True,
                                    older_context_isolated=True, official_formula_exact=True, pose_formula_exact=True))
        enc = [e for e in reports['physical']['encodings'] if e['method'] == method]
        require(len(enc) == 2 and {e['kind'] for e in enc} == {'prefix','terminal'}, 'paired endpoints required')
        for e in enc:
            require(e['status'] == 'completed' and e['root_sha256'] == ROOT, 'endpoint incomplete/wrong root')
            vector = array(e['vector'], (192,))
            pred = array(pp['saved_candidate']['predicted_'+e['kind']], (192,))
            observed = projected_cost(q, goals[0], vector)
            projected = projected_cost(q, pred, vector)
            close(observed, e['observed_Q_goal_cost'], 'physical Q goal cost differs')
            close(projected, e['projected_prediction_mse'], 'physical paired projection differs')
            close(float(np.mean((pred-vector)**2)), e['prediction_mse'], 'physical paired MSE differs')
            from PIL import Image
            rgb = np.asarray(Image.open(inputs.bind(e['rgb_path'], e['rgb_sha256'])).convert('RGB'))
            require(rgb.shape == (224,224,3) and rgb.dtype == np.uint8, 'endpoint image frame differs')
            if e['kind'] == 'prefix':
                old = np.asarray(Image.open(inputs.bind(pp['expected_prefix_rgb'])).convert('RGB'))
                require(np.array_equal(rgb, old), 'exact prefix image mismatch')
            encoding_checks.append(dict(method=method, kind=e['kind'], observed_Q_goal_cost=observed,
                                       projected_prediction_mse=projected, pairing_exact=True))
    require(reports['physical']['new_predict_calls'] == 0 and reports['physical']['physical_env_step_returns'] == 60
            and reports['physical']['attempted_encode_samples'] == 12, 'saved physical work census')
    return dict(status='offline_contract_audited', root_sha256=ROOT, root_index=1,
        concrete_finding='Original finite-goal Q removes agent-goal directions required by official success; intentional surrogate is not the official task metric.',
        projection=dict(nuisance_rank=rank, retained_rank=192-rank,
            reconstruction_max_error=float(np.max(np.abs(q-rebuilt))),
            idempotence_max_error=float(np.max(np.abs(q@q-q)))),
        finite_goal_witnesses=witnesses, candidate_checks=costs, physical_contract_checks=physical_checks,
        endpoint_checks=encoding_checks,
        exclusions=['No goal variant swap/averaging: all six use original offset-zero 192-vector and original Q.',
            'Root1 actual goal render state equals requested physical goal in all five saved variants; no shifted object goal at render for this root.',
            'All 60 candidate costs equal Q-goal MSE with no vector renormalization; stable nonlinear argmin preserved.',
            'All 12 physical costs and prediction residuals use the corresponding saved endpoint and prediction.',
            'Six exact prefix RGB gates, commands, older latent isolation and incoming packet order verified.'],
        unresolved=['Q only removes four finite goal-image directions; object-only task fidelity and global agent invariance not established.',
            'Absolute checkpoint training action normalization and training-camera provenance are not proved by runtime consistency.',
            'Shared render/preprocessing source excludes a separate runtime goal-vs-observation path here, not absolute visual calibration.',
            'No causal explanation of six physical outcomes, efficacy comparison, theoretical negative or code-wide exclusion.'],
        new_predict_calls=0, new_encode_frames=0, new_physical_steps=0, training=False,
        outcome_selection=False, P1_admission=False, efficacy_verified=False)


def source_evidence(inputs):
    sources = dict(transform=('experiments/run_pusht_counterfactual.py','image_transform'),
        goal_tensor=('experiments/spectral_reachability/preflight_pusht_d2p_p0.py','_goal_latent'),
        image_tensor=('experiments/spectral_reachability/preflight_pusht_d2p_p0.py','_image_tensor'),
        frames_tensor=('experiments/spectral_reachability/preflight_pusht_d2p_p0.py','_frames_tensor'),
        packet=('experiments/spectral_reachability/pusht_d2p_packet_contract.py','normalize_live_packet'),
        transition=('experiments/spectral_reachability/pusht_full_packet_compiler_20261027.py','make_transition'),
        endpoint_setup=('experiments/spectral_reachability/pusht_feedback_suffix_physical_20261027_v1.py','run'),
        criterion=('.venv/lib/python3.12/site-packages/stable_worldmodel/wm/lewm/lewm.py','criterion'),
        reset=('.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py','reset'),
        render=('.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py','_render_frame'))
    result = {}
    for key,(path,name) in sources.items():
        p = inputs.bind(REPO/path); text, node = method_source(p, name)
        result[key] = dict(path=str(p), sha256=sha(p), function=name, line=node.lineno, source=text)
    texts = {k:v['source'] for k,v in result.items()}
    require('Normalize(**spt.data.dataset_stats.ImageNet)' in texts['transform'] and
            'scale=True' in texts['transform'] and 'Resize(size=224)' in texts['transform'], 'image transform changed')
    require('_image_tensor(goal_rgb, transform=transform' in texts['goal_tensor'] and
            '_image_tensor(frame, transform=transform' in texts['frames_tensor'] and
            'base.p0._image_tensor(rgb,transform=transform' in texts['endpoint_setup'] and
            'base.p0.image_transform()' in texts['endpoint_setup'], 'shared image preprocessing changed')
    require('normalize_live_packet(raw' in texts['transition'] and 'normalized_action_sequence' not in texts['transition']
            and '/ 100' not in texts['transition'] and '/100' not in texts['transition'], 'current packet scaling changed')
    require('stat.repeat(ACTION_BLOCK)' in texts['packet'] and
            '(commands.reshape(*commands.shape[:-2], 10) - stats[0]) / stats[1]' in texts['packet'], 'packet normalization changed')
    require('self._goal = self.render()' in texts['reset'] and 'axes=(1, 0, 2)' in texts['render'] and
            'cv2.resize(img, (self.render_size, self.render_size))' in texts['render'], 'shared renderer path changed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive NEW output required')
    print('Offline goal contract audit: saved six plans and five original goal variants; zero neural/physics calls.', flush=True)
    inputs, reports, protocols = collect()
    evidence = source_evidence(inputs)
    # Bind the new audit implementation/tests/doc before any audit result is produced.
    for p in (Path(__file__), REPO/f'tests/test_{TAG}.py', REPO/f'innerdocs/{TAG}.md'):
        inputs.bind(p)
    args.output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (args.output/name).open('x') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
    protocol = dict(schema=TAG, bindings=inputs.bindings, fixed_root=ROOT, fixed_cases=ORDER,
                    source_evidence=evidence, policy='offline contract checks only; no selection or fit',
                    P1_admission=False)
    write('protocol.json', protocol)
    result = audit(inputs, reports, protocols)
    inputs.unchanged()
    result.update(schema=TAG, protocol_sha256=sha(args.output/'protocol.json'),
                  input_files_verified=len(inputs.bindings), inputs_sources_unchanged=True,
                  shared_preprocessing_and_render_source_checked=True,
                  saved_representation_runtime_checks=reports['representation']['checks'],
                  source_evidence_scope='text/AST and previously sealed runtime evidence; no fresh model execution')
    require(protocol['bindings'] == read(args.output/'protocol.json')['bindings'], 'late unsealed input')
    write('input_verification.json', dict(bindings=inputs.bindings, unchanged=True))
    write('report.json', result)
    print(json.dumps({k:result[k] for k in ('status','concrete_finding','projection','input_files_verified')}, indent=2))


if __name__ == '__main__':
    main()
