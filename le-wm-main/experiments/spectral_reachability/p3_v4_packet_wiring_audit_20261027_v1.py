"""Frozen saved-artifact identity audit, never imports planner/model/simulator.

Deliberately excludes cost/SVD/allocation/argmin/budget arithmetic (separate audit).
Prepare snapshots inputs before --audit. A source witness is not execution proof.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

NAME = 'p3_v4_packet_wiring_audit_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
SRC = 'experiments/spectral_reachability/'
PHYSICAL = 'results/p3_horizon_residual_physical_20261027_v4_main01'
PLANNER = 'results/p3_horizon_residual_budget_20261027_v4_cuda_main01'
COMPILED = 'results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json'
INTERVENTION = 'results/pusht_d2p_goal_agent_intervention4_v2_20261026_cuda/report.json'
MEAN = np.array([-.0087, .0068], dtype=np.float32)
STD = np.array([.2019, .2002], dtype=np.float32)
METHODS = ('uniform', 'dynamic')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def same(a, b, label):
    require(canonical(a) == canonical(b), label)


def index(rows, keys):
    out = {tuple(r[k] for k in keys): r for r in rows}
    require(len(out) == len(rows), 'duplicate identities '+str(keys))
    return out


def tokens(commands):
    a = np.asarray(commands)
    require(a.ndim == 2 and a.shape[1] == 2 and a.shape[0] in (10, 15)
            and a.dtype.kind in 'fi' and np.isfinite(a).all() and (np.abs(a) <= 1).all(),
            'bounded time-major H2/H3 commands')
    # Independent NumPy float32 expression, not sealed normalizer execution.
    return ((a.astype(np.float32).reshape(-1, 10)-np.tile(MEAN, 5))/np.tile(STD, 5))


def context_witness(initial, commands, context):
    t = tokens(commands)
    for field, expected in [('layout', 'states_with_between_state_actions_v1'),
                            ('context_length', 3), ('latent_dim', 192), ('action_dim', 10)]:
        require(initial[field] == context[field] == expected, 'context layout '+field)
    latent = np.asarray(context['latent_history'])
    require(latent.shape == (3, 192) and np.isfinite(latent).all(), 'finite latent3')
    require(np.array_equal(np.asarray(context['action_history']), t[-2:]),
            'exact last-two normalized packets')
    if len(t) == 2:
        same(context['latent_history'][0], initial['latent_history'][-1],
             'H2 retained oldest latent must be original decision latent')


def warmup_witness(identity, initial):
    same(identity['raw_history'], np.zeros((3, 2), dtype=np.float32).tolist(), 'three zero warmup packets')
    raw = np.repeat(np.asarray(identity['raw_history'], dtype=np.float32)[:, None, :], 5, axis=1)
    normalized = (raw.reshape(3, 10)-np.tile(MEAN, 5))/np.tile(STD, 5)
    same(normalized[1:].tolist(), initial['action_history'], 'warmup incoming packet alignment')
    same(identity['normalized_between_history'], initial['action_history'], 'receipt warmup history')


def branch_witness(plan, row, events, encoding):
    for k in ('root_index', 'root_sha256', 'method', 'selection_sha256', 'planned_horizon_tokens', 'planned_commands'):
        same(row[k], plan[k], 'branch binding '+k)
    h = plan['planned_horizon_tokens']
    require(h in (2, 3) and plan['planned_commands'] == 5*h, 'selected horizon, not search max')
    tokens(plan['commands'])
    require(row['status'] == 'completed' and row['full_horizon_completed'] and row['endpoint_fresh'], 'full fresh endpoint')
    require(row['endpoint_scope'] == 'full_H'+str(h), 'endpoint horizon')
    for k in ('physical_env_step_attempts', 'physical_env_step_returns', 'completed_commands',
              'attempted_commands', 'actual_endpoint_primitive_steps'):
        require(row[k] == 5*h, 'selected path count '+k)
    require(len(events) == 1+3*5*h and events[0]['kind'] == 'initial', 'full ordered event stream')
    same(events[0]['state'], row['context_identity']['state'], 'initial state anchored after warmup')
    observations = []
    for i, command in enumerate(plan['commands'], 1):
        attempt, returned, observed = events[3*i-2:3*i+1]
        for event, kind in zip((attempt, returned, observed), ('attempt', 'step_returned', 'observation')):
            require(event['kind'] == kind and event['primitive_step'] == i, 'event order/primitive time')
        same(attempt['command'], command, 'attempt exact raw XY order')
        same(observed['command'], command, 'observation exact raw XY order')
        require(not observed['terminated'] and not observed['truncated'], 'no ignored early termination')
        observations.append({k: v for k, v in observed.items() if k not in ('rgb_path', 'rgb_file_sha256')})
    same(observations, row['trace'], 'journal/trace identity')
    same(events[-1]['state'], row['final_state'], 'endpoint state is last observation')
    require(events[-1]['rgb_path'] == row['terminal_rgb_path']
            and events[-1]['rgb_file_sha256'] == row['terminal_rgb_sha256'], 'last observed image is endpoint')
    require(row['guard']['root_id'] == plan['root_sha256'] and not row['guard']['abstained'], 'guard root binding')
    require([x['primitive_step'] for x in row['guard']['observations']] == list(range(5*h+1)), 'fresh guard observation before each command')
    require(row['guard']['primitive_commands_authorized'] == 5*h, 'guard authorizations')
    for k in ('root_sha256', 'method', 'selection_sha256', 'planned_horizon_tokens', 'goal_vector_sha256', 'projection_sha256'):
        same(encoding[k], plan[k], 'encoding binding '+k)
    require(encoding['status'] == 'completed' and encoding['attempted_encode_samples'] == 1
            and encoding['endpoint_scope'] == row['endpoint_scope']
            and encoding['terminal_rgb_sha256'] == row['terminal_rgb_sha256'], 'encoding uses actual final PNG')
    v = np.asarray(encoding['terminal_vector'])
    require(v.shape == (192,) and np.isfinite(v).all(), 'saved endpoint vector only, no encode rerun')


def source_witness(path, function, fragments):
    text = Path(path).read_text()
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function]
    require(len(nodes) == 1, 'unique source function '+function)
    node = nodes[0]
    segment = '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])
    # Whitespace-independent token fragments, explicitly weaker than AST semantic equivalence.
    compact = ''.join(segment.split())
    for fragment in fragments:
        require(''.join(fragment.split()) in compact, 'source wiring witness '+function+': '+fragment)
    return dict(path=str(path), function=function, start_line=node.lineno, end_line=node.end_lineno,
                sha256=sha(path), ast_sha256=hashlib.sha256(ast.dump(node).encode()).hexdigest(), fragments=fragments)


SOURCE_SPECS = [
    ('p3_horizon_residual_budget_20261027_v4.py', 'compile_cuda', ['torch.tanh(u).reshape(h,5,2)', 'state,between=transition(state,between,raw)', "return state[0,-1]"]),
    ('p3_horizon_residual_budget_20261027_v4.py', 'search', ["replay(deepcopy(p['root_context']),actions)", "q=np.asarray(p['projection']);goal=np.asarray(p['goal'])", 'q@(terminal-goal)']),
    ('pusht_full_packet_compiler_20261027.py', 'make_transition', ['torch.cat((history,token[None,None]),dim=1)', 'model.predict(emb,model.action_encoder(outgoing))[:,-1]', 'emb[:,1:]', 'outgoing[:,1:]']),
    ('pusht_composition_cuda_replay_v2.py', '__call__', ['torch.cat((history, token[None, None]), dim=1)', 'self.model.predict(latent, self.model.action_encoder(outgoing))[:, -1]', 'latent[:, 1:]', 'history = outgoing[:, 1:]']),
    ('pusht_composition_cuda_replay_v2.py', 'packets', ['source.astype(np.float32).reshape(-1, 5, 2)']),
    ('pusht_d2p_packet_contract.py', 'normalize_live_packet', ['stat.repeat(ACTION_BLOCK)', '(commands.reshape(*commands.shape[:-2], 10) - stats[0]) / stats[1]']),
    ('preflight_pusht_d2p_p0.py', '_collect_causal_rgb_context', ['np.zeros((int(history_len), 2), dtype=np.float32)', 'recorder.execute(packets.repeat_command(action))', 'recorder.frames[:-1]']),
    ('preflight_pusht_d2p_p0.py', '_collect_root_anchored_rgb_context', ['frames[1:]+[current]']),
    ('run_pusht_d2p_p1_physical_calibration.py', '_reset_and_warmup', ['_collect_root_anchored_rgb_context']),
    ('p3_matched_physical_20261027_v1.py', 'guarded_execute', ['raw = np.asarray(commands)', 'for index, command in enumerate(raw):', 'authorized = guard.dispatch(command)', 'env.step(authorized)']),
    ('pusht_rgb_observability_guard.py', 'dispatch', ['value = np.asarray(command)', 'return value.copy()']),
    ('p3_horizon_residual_physical_20261027_v4.py', 'execute_physics', ["base.physical._reset_and_warmup(env,root,history_len=3)", "verify_rgb_context(frames,goal,protocol['references'][plan['root_sha256']])", "official_checked(env,state,root['goal'])", "base.physical._pose_label(state,root['goal'])"]),
    ('p3_horizon_residual_physical_20261027_v4.py', 'optional_encodings', ["encode(path)", "np.asarray(p['projection'])@(vector-np.asarray(p['goal']))"]),
]


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        current = sha(path)
        require(expected is None or current == expected, 'input SHA '+str(path))
        require(str(path) not in self.hashes or self.hashes[str(path)] == current, 'input changed during discovery')
        self.hashes[str(path)] = current
        return path

    def json(self, path, expected=None):
        return read(self.bind(path, expected))

    def unchanged(self):
        for path, expected in self.hashes.items():
            require(sha(path) == expected, 'frozen input changed '+path)


def prepared(inputs, directory, expected, seen):
    directory = Path(directory).resolve()
    if str(directory) in seen:
        inputs.bind(directory/'prepare.json', expected)
        return
    seen.add(str(directory))
    receipt = inputs.json(directory/'prepare.json', expected)
    envelope = inputs.json(directory/'protocol.json', receipt['protocol_sha256'])
    body = envelope.get('protocol', envelope)
    if 'protocol_id' in envelope:
        require(digest(body) == envelope['protocol_id'] == receipt['protocol_id'], 'prepared protocol content identity')
    for field in ('sources_sha256', 'inputs_sha256'):
        for path, h in body.get(field, {}).items():
            inputs.bind(path, h)
    for seal in receipt['source_seals']:
        d = Path(seal['output_dir'])
        manifest = inputs.json(d/'manifest.json', seal['manifest_sha256'])
        inputs.bind(d/'args.json', manifest['args_sha256'])
        for entry in manifest['files']:
            inputs.bind(entry['source'], entry['sha256'])
            inputs.bind(d/entry['archive'], entry['sha256'])
    for prefix in ('prior', 'source', 'planner'):
        if prefix+'_prepared' in body:
            prepared(inputs, body[prefix+'_prepared'], body[prefix+'_prepare_sha256'], seen)


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).absolute()
    require(not output.exists(), 'exclusive new audit directory')
    inputs = Inputs()
    for f in (Path(__file__), root/'tests'/('test_'+NAME+'.py'), root/'innerdocs'/(NAME+'.md')):
        inputs.bind(f)
    seen = set()
    for dirname in (PHYSICAL, PLANNER):
        report = inputs.json(root/dirname/'report.json')
        execution = inputs.json(root/dirname/'execution_protocol.json')
        prepared(inputs, execution['prepared'], execution['prepare_sha256'], seen)
        for row in report.get('rows', [])+report.get('roots', []):
            if 'journal_path' not in row:
                continue
            path = inputs.bind(row['journal_path'], row['journal_sha256'])
            for line in path.read_text().splitlines():
                e = json.loads(line)
                if 'rgb_path' in e:
                    inputs.bind(e['rgb_path'], e['rgb_file_sha256'])
    # Bind all per-arm durable row / encoding / endpoint receipts, not just report copies.
    for path in sorted((root/PHYSICAL).glob('*.json')):
        inputs.bind(path)
    inputs.bind(root/PHYSICAL/'encoding.events.jsonl')
    inputs.bind(root/COMPILED)
    intervention = inputs.json(root/INTERVENTION)
    bank = inputs.json(intervention['protocol']['bank_path'], intervention['protocol']['bank_sha256'])
    inputs.bind(bank['upstream_path'], bank['upstream_sha256'])
    for file, _, _ in SOURCE_SPECS:
        inputs.bind(root/SRC/file)
    inputs.bind(root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py')
    inputs.unchanged()
    output.mkdir(exist_ok=False)
    protocol = dict(schema=NAME+'.protocol', project_root=str(root), physical=PHYSICAL, planner=PLANNER,
                    scope='identity/source-wiring only; no cost, SVD, allocation, argmin or compute-budget recomputation',
                    input_sha256=dict(sorted(inputs.hashes.items())), new_predict_calls=0, new_frame_encodes=0, new_physical_steps=0)
    write(output/'protocol.json', protocol)
    receipt = dict(status='prepared_offline_only', protocol_sha256=sha(output/'protocol.json'), files=len(inputs.hashes))
    write(output/'prepare.json', receipt)
    return receipt


def audit(output, expected):
    output = Path(output).absolute()
    require(sha(output/'prepare.json') == expected, 'audit prepare SHA')
    receipt = read(output/'prepare.json')
    require(sha(output/'protocol.json') == receipt['protocol_sha256'], 'audit protocol SHA')
    protocol = read(output/'protocol.json')
    inputs = Inputs()
    inputs.hashes = protocol['input_sha256']
    inputs.unchanged()
    root = Path(protocol['project_root'])

    def load(path):
        path = Path(path).resolve()
        require(str(path) in inputs.hashes, 'undeclared read '+str(path))
        return read(path)

    physical = load(root/PHYSICAL/'report.json')
    planner = load(root/PLANNER/'report.json')
    execution = load(root/PHYSICAL/'execution_protocol.json')
    consumer = load(Path(execution['prepared'])/'protocol.json')
    envelope = load(Path(consumer['planner_prepared'])/'protocol.json')
    body = envelope['protocol']
    require(physical['status'] == 'guarded_calibration_recorded' and planner['status'] == 'cuda_completed', 'complete actual reports')
    require(physical['planning_report_sha256'] == consumer['planning_report_sha256'] == sha(root/PLANNER/'report.json'), 'physical bound exact planner report')
    require(physical['consumer_prepare_sha256'] == execution['prepare_sha256'], 'physical consumer receipt')
    require(physical['planner_protocol_id'] == consumer['planner_protocol_id'] == planner['protocol_id'] == envelope['protocol_id'], 'protocol chain')
    require(sha(root/PHYSICAL/'physical_report.json') == physical['physical_report_sha256'], 'durable physical report hash')
    durable = load(root/PHYSICAL/'physical_report.json')
    same(durable['rows'], physical['rows'], 'physical rows retained across encoding')
    require(physical['new_predict_calls'] == 0 and physical['source_verified'] and physical['model_unchanged'] and not physical['failures'], 'runtime integrity receipts')
    roots = body['roots']
    require(len(roots) == len(set(roots)) == 4, 'four unselected roots')
    expected_order = [(r, m) for i, r in enumerate(roots) for m in (METHODS if i % 2 == 0 else METHODS[::-1])]
    for rows, method in ((consumer['plans'], 'method'), (physical['rows'], 'method')):
        require([(r['root_sha256'], r[method]) for r in rows] == expected_order, 'all-eight counterbalanced execution order')
    plan_index = index(consumer['plans'], ('root_sha256', 'method'))
    physical_index = index(physical['rows'], ('root_sha256', 'method'))
    encoding_index = index(physical['encodings'], ('root_sha256', 'method'))
    row_index = index(planner['rows'], ('root_sha256', 'mode'))
    require(set(plan_index) == set(physical_index) == set(encoding_index) == set(row_index) == set(expected_order), 'complete eight-arm census')
    compilation = index(load(root/COMPILED)['rows'], ('root_sha256',))
    intervention = load(root/INTERVENTION)
    variants = index(intervention['rows'], ('root_sha256',))
    bank = load(intervention['protocol']['bank_path'])
    upstream = load(bank['upstream_path'])
    references = index(upstream['rows'], ('root_sha256',))
    root_records = index(planner['roots'], ('root_sha256',))
    require(set(root_records) == set(compilation) == set(variants) == set(references) == {(r,) for r in roots}, 'original root census')
    candidates_checked = 0
    arms = []
    for i, packet in enumerate(body['packets']):
        r = roots[i]
        require(packet['root_sha256'] == r, 'packet root order')
        source = compilation[r,]['compiled']
        initial = packet['root_context']
        same(initial['latent_history'], source['input_context']['emb'][0], 'original observed latents')
        same(initial['action_history'], source['input_context']['actions'][0][1:], 'drop oldest outgoing token only')
        same(packet['goal'], source['goal_vectors'][0], 'original goal0 vector')
        same(packet['projection'], source['scoring']['metric']['projection'], 'original inherited Q matrix')
        same(packet['goal'], variants[r,]['variants'][0]['goal_vector'], 'unshifted goal intervention vector')
        ref = references[r,]
        same(consumer['references'][r], {k: ref[k] for k in ('root_sha256', 'history_rgb_sha256', 'goal_rgb_sha256')}, 'original RGB references')
        require(ref['goal_rgb_sha256'] == variants[r,]['variants'][0]['goal_rgb_sha256'], 'goal image/vector provenance')
        require(packet['input_sha256'] == digest({k: v for k, v in packet.items() if k != 'input_sha256'}), 'packet content digest')
        identities = []
        for mode in METHODS:
            searchrow = row_index[r, mode]
            require(searchrow['input_sha256'] == packet['input_sha256'], 'search initial input')
            search = searchrow['search']
            require(searchrow['status'] == search['status'] == 'completed', 'completed saved search')
            cidx = index(search['candidates'], ('horizon', 'index'))
            for c in search['candidates']:
                require(c['completed'] and len(c['actions']) == 5*c['horizon'], 'candidate packet shape')
                context_witness(initial, c['actions'], c['context'])
                if c['index'] == 0:
                    require(np.count_nonzero(c['actions']) == 0, 'zero candidate identity')
                    same(c['context']['latent_history'][-1], root_records[r,]['compilation'][str(c['horizon'])]['nominal'], 'zero candidate vs matching-horizon nominal')
                candidates_checked += 1
            winner = search['winner']
            same(winner, cidx[winner['horizon'], winner['index']], 'saved winner belongs to complete candidate set, not reselected')
            same(winner['context'], search['whole_replay_context'], 'whole winner context')
            same(winner['context'], search['segmented_replay_context'], 'segmented winner context')
            plan, row, enc = (idx[r, mode] for idx in (plan_index, physical_index, encoding_index))
            require(plan['root_index'] == i and plan['input_sha256'] == packet['input_sha256'], 'plan root/index/input')
            require(plan['selection_sha256'] == digest({k: v for k, v in plan.items() if k != 'selection_sha256'}), 'plan content digest')
            for k, v in [('commands', winner['actions']), ('planned_horizon_tokens', winner['horizon']),
                         ('predictor_context', initial), ('selected_predicted_context', winner['context']),
                         ('goal', packet['goal']), ('projection', packet['projection'])]:
                same(plan[k], v, 'export transcription '+k)
            require(plan['goal_vector_sha256'] == digest(packet['goal']) and plan['projection_sha256'] == digest(packet['projection']), 'goal/Q content hashes')
            identity = row['context_identity']
            warmup_witness(identity, initial)
            for k in ('history_rgb_sha256', 'goal_rgb_sha256'):
                require(identity[k] == ref[k], 'actual reset RGB reference '+k)
            identities.append(identity)
            journal = Path(row['journal_path'])
            require(sha(journal) == row['journal_sha256'], 'selected path journal hash')
            events = [json.loads(line) for line in journal.read_text().splitlines()]
            branch_witness(plan, row, events, enc)
            # Compare durable per-branch artifacts, endpoint and encoding receipt to final report.
            stem = f'root_{i}_{mode}_replay0'
            stored = load(root/PHYSICAL/(stem+'.physical.json'))
            for k, v in stored.items():
                same(row[k], v, 'durable physical row '+k)
            same(load(root/PHYSICAL/(stem+'.encoding.json')), enc, 'durable endpoint encoding')
            endpoint = load(root/PHYSICAL/(stem+'.endpoint.json'))
            for k, v in endpoint.items():
                same(row[k], v, 'endpoint receipt '+k)
            max_roundoff = float(np.max(np.abs(np.asarray(plan['commands'])-np.asarray(plan['commands'], dtype=np.float32).astype(float))))
            arms.append(dict(root_index=i, root_sha256=r, method=mode, horizon=winner['horizon'],
                             selection_sha256=plan['selection_sha256'], endpoint_png_sha256=row['terminal_rgb_sha256'],
                             model_float32_vs_physical_float64_max_command_delta=max_roundoff, wiring='exact_except_declared_float32_cast'))
        same(identities[0], identities[1], 'paired physical decision state/history/goal identical')
    sources = [source_witness(root/SRC/f, fn, fragments) for f, fn, fragments in SOURCE_SPECS]
    env = root/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py'
    sources.append(source_witness(env, 'step', ['action = self.agent.position + action * self.action_scale',
                                               'for _ in range(n_steps):', 'self.space.step(self.dt)']))
    inputs.unchanged()
    result = dict(schema=NAME+'.report', status='offline_identity_audit_complete', roots=4, arms=arms,
                  candidates_context_checked=candidates_checked, source_witnesses=sources,
                  prepare_sha256=expected, input_files=len(inputs.hashes),
                  new_predict_calls=0, new_frame_encodes=0, new_physical_steps=0,
                  reproducible_causal_code_bug_established=False, P1_admission=False, P3_admission=False, efficacy_claim=False,
                  exclusions=['time-major XY packet ordering and normalized last-two action histories, all saved candidates',
                              'H2 retained decision latent, matching-horizon zero nominal, whole/segmented winner context identity',
                              'all four original cached context/goal0/Q vector identities and goal RGB receipt provenance',
                              'all eight selected commands, chronological journal events, guard roots, final PNG to encoded-vector receipt'],
                  timeline=dict(warmup_boundary_steps=[5, 10, 15], context_between_steps=[[6, 10], [11, 15]],
                                decision_relative_H2_steps=[1, 10], reset_warmup_relative_H2_steps=[16, 25],
                                scope='source-derived timeline, not independent warmup trajectory replay'),
                  limitations=['15 primitive warmup steps per branch are outside durable selected-path journals; full intermediate trajectory unavailable',
                               'warmup RGB stack and goal images are hash receipts, not independently reencoded or reconstructed here',
                               'command float64 physical vs float32 model discrepancy quantified; no causal attribution',
                               'source seals bind bytes, not proof of exact historical execution; model unchanged is runtime receipt only',
                               'checkpoint training normalization/camera provenance not certified by runtime agreement',
                               'Q is inherited goal-agent nuisance projection, not an official/pose metric equivalence',
                               'no cost/SVD/allocation/argmin/compute-budget arithmetic repeated; delegated audit remains independent',
                               'no global code exclusion, theoretical negative, physical benefit or novel-root efficacy conclusion'])
    write(output/'report.json', result)
    inputs.unchanged()
    return dict(status=result['status'], report_sha256=sha(output/'report.json'), arms=len(arms), candidates=candidates_checked)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--audit', action='store_true')
    p.add_argument('--project-root', type=Path, default=PROJECT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-sha256')
    args = p.parse_args()
    if args.prepare:
        result = prepare(args.project_root, args.output)
    else:
        require(args.prepare_sha256 is not None, 'explicit prepare digest required')
        result = audit(args.output, args.prepare_sha256)
    print(canonical(result))


if __name__ == '__main__':
    main()
