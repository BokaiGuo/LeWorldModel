"""Execute four sealed gradient-MPC witnesses; no new planning or CPU models."""
import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from experiments.spectral_reachability import p2_full20_gradient_mpc_20261027_v1 as planner
from experiments.spectral_reachability import p2_no_composition_sra_physical_20261027_v2 as provenance
from experiments.spectral_reachability import p2_full20_physical_20261027_v2 as helpers

TAG = 'p2_gradient_mpc_physical_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
require, digest, sha, write = provenance.require, provenance.digest, provenance.sha, provenance.write
ACCOUNTING = dict(total_predict_samples=336, candidate_predict_samples=320,
    replay_predict_samples=16, autograd_grad_calls=156, optimizer_steps=156,
    scalar_objective_vjp_directions=156, processed_context_token_samples=1008,
    fair_total_budget_comparison=False, prediction_samples_match_sampling=True,
    unknown_is_not_zero=True, unmeasured=deepcopy(planner.BOUNDARY['unmeasured']))


def plan_for(ready, actions, selected_candidate_id, receipt_sha256):
    commands = np.asarray(actions)
    require(commands.shape == (10, 2) and commands.dtype.kind in 'fi'
        and np.isfinite(commands).all() and np.all(np.abs(commands) <= 1), 'bounded exact ten-command witness required')
    require(ready['root']['root_id'] == ready['root_sha256'], 'stable root mismatch')
    value = dict(root_index=ready['root_index'], root_sha256=ready['root_sha256'],
        method=planner.ARM, commands=commands.tolist(), planned_horizon_tokens=2,
        planned_commands=10, corrected_search_horizon=2,
        root=deepcopy(ready['root']), action_domain=deepcopy(ready['action_domain']),
        selected_candidate_id=selected_candidate_id, receipt_sha256=receipt_sha256)
    value['selection_sha256'] = digest(value)
    return value


def check_plan(plan):
    require(plan['method'] == planner.ARM and plan['planned_horizon_tokens'] == 2
        and plan['planned_commands'] == 10, 'fixed gradient H2 plan required')
    require(digest({k:v for k,v in plan.items() if k != 'selection_sha256'}) == plan['selection_sha256'],
        'selection digest changed')
    raw = np.asarray(plan['commands'])
    require(raw.shape == (10,2) and raw.dtype.kind in 'fi' and np.isfinite(raw).all()
        and np.all(np.abs(raw) <= 1), 'finite bounded ten-step commands required')


def encoding_contract(rows, encodings):
    keys = {(r['root_sha256'], r['method']) for r in rows}
    recorded = {(e['root_sha256'], e['method']):e for e in encodings}
    if len(recorded) != len(encodings) or set(recorded) != keys:
        return False
    for row in rows:
        entry = recorded[row['root_sha256'], row['method']]
        full = row['full_horizon_completed'] and row['completed_commands'] == 10
        if entry['status'] != ('completed' if full else 'blocked_fixed_horizon'):
            return False
        if entry['attempted_encode_samples'] != int(full):
            return False
    return True


def inputs(planning):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    directory, protocol, prepared = planner.load_prepared(planning)
    exported = planner.selected_actions(directory)
    report = json.loads((directory/'report.json').read_text())
    require(sha(directory/'report.json') == exported['report_sha256'], 'planning report changed during export')
    # Exporter must already belong to the archived pre-CUDA source, not be
    # appended after observing a result and retrospectively resealed.
    sealed = []
    for seal in prepared['source_seals']:
        archive_dir = Path(seal['output_dir'])
        manifest = json.loads((archive_dir/'manifest.json').read_text())
        for entry in manifest['files']:
            if entry['source'] == str(Path(planner.__file__).resolve()):
                archive = archive_dir/entry['archive']
                require(sha(archive) == entry['sha256'] == sha(planner.__file__)
                    and '\ndef selected_actions(output):' in archive.read_text(), 'original archived exporter required')
                sealed.append(entry)
    require(len(sealed) == 1, 'unique original planner source required')
    rows = provenance.indexed(report['rows'], 'root_sha256')
    cases = provenance.indexed(protocol['context']['cases'], 'root_sha256')
    require(rows.keys() == cases.keys()
        and set(exported['actions']) == {(key, planner.ARM) for key in cases}, 'four-root exporter census mismatch')
    plans = []
    for key, ready in sorted(cases.items(), key=lambda item:item[1]['root_index']):
        result = rows[key]['result']
        plans.append(plan_for(ready, exported['actions'][key, planner.ARM],
            result['selected_candidate_id'], result['witness']['receipt']['receipt_sha256']))
    bindings = deepcopy(protocol['bindings'])
    for name in ('protocol.json', 'prepared.json', 'report.json'):
        bindings[str(directory/name)] = sha(directory/name)
    for ready in cases.values():
        for suffix in ('.json', '_journal.jsonl'):
            path = directory/f'root_{ready["root_index"]}{suffix}'
            bindings[str(path)] = sha(path)
    reports = [json.loads(Path(p).read_text()) for p in protocol['bindings'] if Path(p).name == 'report.json']
    origin = [r['protocol'] for r in reports if 'bank_path' in r.get('protocol', {})]
    require(len(origin) == 1, 'unique original physical root provenance required')
    bank_path = Path(origin[0]['bank_path']).resolve()
    require(sha(bank_path) == origin[0]['bank_sha256'], 'physical bank binding rejected')
    bank = json.loads(bank_path.read_text())
    upstream_path = Path(bank['upstream_path']).resolve()
    require(sha(upstream_path) == origin[0]['upstream_sha256'], 'original physical preflight rejected')
    upstream = base.physical._load_preflight(upstream_path)
    refs = provenance.indexed(upstream['rows'], 'root_sha256')
    require(refs.keys() == cases.keys(), 'physical root census differs')
    for plan in plans:
        ref = refs[plan['root_sha256']]
        require(plan['root']['root_rgb_sha256'] == ref['history_rgb_sha256']
            and plan['root']['goal_rgb_sha256'] == ref['goal_rgb_sha256'], 'physical root/goal RGB binding rejected')
    for path in (bank_path, upstream_path):
        bindings[str(path)] = sha(path)
    source, _, config = checked.validate_inputs(protocol['context']['compilation'], protocol['context']['policy'])
    return base, checked, source, config, bindings, plans, upstream, refs, protocol


def prepare(planning, output):
    output = Path(output).resolve()
    require(TAG in output.name and not output.exists(), 'exclusive tagged preparation required')
    context = inputs(planning)
    base, checked, source, config, bindings, plans, *_ = context
    creator = provenance.environment_creator(base)
    files = checked.sources_before_allocate(source, config)
    files += [Path(m.__file__) for n,m in tuple(sys.modules.items())
        if n.startswith('experiments.') and getattr(m, '__file__', '').endswith('.py')]
    files += [Path(__file__), PROJECT/'tests'/f'test_{TAG}.py', PROJECT/'innerdocs'/f'{TAG}.md']
    env_paths = sorted({str(Path(inspect.getfile(x)).resolve()) for x in
        (creator, *[getattr(creator,n) for n in ('eval_state','reset','render','step')])})
    files += [Path(p) for p in env_paths]
    files = sorted({p.resolve() for p in files})
    for path in env_paths:
        bindings[path] = sha(path)
    protocol = dict(schema=TAG+'.protocol', planning=str(Path(planning).resolve()), bindings=bindings,
        plans=plans, environment_source_paths=env_paths, source_paths=list(map(str,files)),
        planning_accounting=ACCOUNTING, P1_admission=False, P2_admission=False,
        scope='four already-used calibration roots, one exact H2 execution each; preserve every guard partial',
        physical_before_optional_encoding=True, extra_planning=False, root_replacement=False)
    output.mkdir(exist_ok=False)
    write(output/'protocol.json', protocol)
    seals = [provenance.create_seal(files=files[i:i+40], args_json=json.dumps(dict(protocol_sha256=digest(protocol))),
        output_dir=output/f'source_seal_{i//40}') for i in range(0,len(files),40)]
    write(output/'prepared.json', dict(schema=TAG+'.prepared', protocol_sha256=digest(protocol), source_seals=seals))
    load_prepared(output)


def load_prepared(output):
    output = Path(output).resolve()
    protocol = json.loads((output/'protocol.json').read_text())
    prepared = json.loads((output/'prepared.json').read_text())
    require(protocol['schema'] == TAG+'.protocol' and prepared['schema'] == TAG+'.prepared'
        and digest(protocol) == prepared['protocol_sha256'] and protocol['planning_accounting'] == ACCOUNTING
        and protocol['P1_admission'] is protocol['P2_admission'] is False, 'physical protocol changed')
    paths = []
    require(bool(prepared['source_seals']), 'source closure missing')
    for seal in prepared['source_seals']:
        provenance.verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
        directory = Path(seal['output_dir'])
        require(json.loads((directory/'args.json').read_text()) == dict(protocol_sha256=digest(protocol)), 'source args binding rejected')
        paths += [e['source'] for e in json.loads((directory/'manifest.json').read_text())['files']]
    required = {str(Path(__file__).resolve()), str(Path(planner.__file__).resolve()),
        str(PROJECT/'tests'/f'test_{TAG}.py'), str(PROJECT/'innerdocs'/f'{TAG}.md')}
    require(sorted(paths) == protocol['source_paths'] and required <= set(paths), 'source closure differs')
    require(all(sha(p) == h for p,h in protocol['bindings'].items()), 'physical bound input changed')
    context = inputs(protocol['planning'])
    require(context[5] == protocol['plans'] and all(protocol['bindings'].get(p) == h for p,h in context[4].items()),
        'exported selected actions or binding closure changed')
    return protocol, prepared, context


def run(prepared, output, encode_terminals=False):
    protocol, sealed, context = load_prepared(prepared)
    base, checked, source, config, bindings, plans, upstream, refs, planning_protocol = context
    output = Path(output).resolve()
    require(TAG in output.name, 'dedicated new output required')
    output.mkdir(exist_ok=False)
    write(output/'protocol.json', dict(prepared=str(Path(prepared).resolve()), prepared_protocol_sha256=sealed['protocol_sha256'],
        plans=plans, encode_terminals=encode_terminals))
    rows, failures, encodings = [], [], []
    env = model = before = host = None
    try:
        env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
        require(str(Path(inspect.getfile(type(env.unwrapped))).resolve()) in protocol['environment_source_paths'], 'unsealed environment class')
        require(tuple(env.unwrapped.variation_space['agent']['color'].value.tolist()) == (65,105,225), 'unsupported renderer color')
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4, seed=int(upstream['protocol']['root_seed']), oversample_factor=4)
        roots = {base.root_fingerprint(r):r for r in roots}
        require(roots.keys() == refs.keys(), 'root census differs; no replacement')
        for plan in plans:
            check_plan(plan)
            root = roots[plan['root_sha256']]
            def reset():
                require(base.root_fingerprint(root) == plan['root_sha256'], 'reset root changed')
                frames, history, goal = base.physical._reset_and_warmup(env, root, history_len=3)
                identity = provenance.verify_rgb_context(frames, goal, refs[plan['root_sha256']])
                identity.update(history=np.asarray(history).tolist(), state=base.p0.packets.state_of(env).tolist(), goal=np.asarray(root['goal']).tolist())
                return identity
            def evaluate(state):
                return dict(official=provenance.official_checked(env,state,root['goal']), pose=base.physical._pose_label(state,root['goal']))
            rows.append(helpers.physical_branch(plan=plan, output=output, reset=reset, env=env,
                state_of=base.p0.packets.state_of, evaluate=evaluate))
    except (Exception, KeyboardInterrupt) as exc:
        failures.append(provenance.durable.failure(exc, 'physical_setup_or_branch'))
    finally:
        if env is not None:
            try: env.close()
            except Exception as exc: failures.append(provenance.durable.failure(exc, 'env_close'))
    helpers.materialize_missing(plans, rows, output)
    write(output/'physical_report.json', dict(schema=TAG+'.physical_before_encoding', rows=rows, failures=failures,
        planning_accounting=ACCOUNTING, new_predict_calls=0, P1_admission=False, P2_admission=False))
    if encode_terminals:
        def setup():
            nonlocal model, before, host
            import torch
            from PIL import Image
            host = base.require_cuda_allocation('cuda')
            device = torch.device('cuda:0')
            model = base.load_compatible_pretrained(Path(planning_protocol['context']['policy']), device=device).eval().requires_grad_(False)
            before = checked.tensor_state_hash(model)
            def forbid(*a, **kw): raise RuntimeError('endpoint encoding must not predict')
            model.predict = forbid
            transform = base.p0.image_transform()
            def encode(path):
                with Image.open(path) as image: rgb = np.asarray(image.convert('RGB')).copy()
                with torch.no_grad():
                    value = model.encode({'pixels':base.p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
                require(value.device.type == 'cuda' and torch.isfinite(value).all(), 'host CUDA finite endpoint required')
                torch.cuda.synchronize(device)
                return value.cpu().tolist()
            return encode
        encodings = helpers.optional_encodings(rows, output, setup)
    gates = dict(all_four_retained=len(rows)==4 and len({r['root_sha256'] for r in rows})==4,
        trace_matches_selected=all(len(r['trace']) == r['completed_commands'] == r['physical_env_step_returns']
            and [e['primitive_step'] for e in r['trace']] == list(range(1,r['completed_commands']+1))
            and [e['command'] for e in r['trace']] == next(p['commands'] for p in plans
            if p['root_sha256']==r['root_sha256'])[:r['completed_commands']] for r in rows),
        branches_durable=(output/'physical_report.json').is_file())
    try:
        load_prepared(prepared)
        gates['sources_inputs_unchanged'] = True
    except Exception as exc:
        gates['sources_inputs_unchanged'] = False
        failures.append(provenance.durable.failure(exc, 'final_source_check'))
    if model is not None:
        gates.update(model_unchanged=before==checked.tensor_state_hash(model), frozen_eval=all(not m.training for m in model.modules())
            and all(not p.requires_grad for p in model.parameters()))
    guards_only = all(r['status']=='completed' or r['status']=='observation_unavailable' and r['guard']['abstained'] for r in rows)
    full = all(r['full_horizon_completed'] and r['completed_commands']==10 for r in rows)
    encoding_ok = not encode_terminals or encoding_contract(rows, encodings)
    valid = all(gates.values()) and guards_only and encoding_ok and not failures
    status = ('calibration_recorded' if full else 'guarded_calibration_recorded') if valid else 'integrity_fail'
    write(output/'report.json', dict(schema=TAG+'.report', status=status, rows=rows, encodings=encodings, gates=gates,
        failures=failures, host=host, source_seals=sealed['source_seals'], prepared_protocol_sha256=sealed['protocol_sha256'],
        actual_physical_steps=sum(r['physical_env_step_returns'] for r in rows),
        attempted_terminal_encode_samples=sum(e['attempted_encode_samples'] for e in encodings),
        planning_accounting=ACCOUNTING, new_predict_calls=0, full_H2_all=full,
        P1_admission=False, P2_admission=False, physical_efficacy_verified=False))
    if not valid: raise RuntimeError('physical consumer integrity failed; retain all partial evidence')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--planning', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--prepared', type=Path, required=True)
    execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--encode-terminals', action='store_true')
    args = vars(parser.parse_args())
    command = args.pop('command')
    prepare(**args) if command=='prepare' else run(**args)
