"""Independent NumPy-only raw receipt audit; never imports the CUDA runner/model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import zlib

import numpy as np

NAME = 'p1_line_search_gn_saved_audit_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/'results/p1_line_search_gn_cuda_20261027_v1_cuda_main01/report.json'
INPUT_SHA = '5f9825f98ed40f637c9d48e49c7c53428d1380e4234e3a1e5503a76f74c4bf1e'
PREPARE_SHA = 'e1b6d9783da6a4e32d360326869fe1b64fef8c3f4f168095be4b7f8c215b26d9'
SCALES = [1., .5, .25, .125, .0625]
BUDGET = dict(predict=78, action_encoder=78, jacobian=4, vjp_directions=768,
              svd=4, linear_solve=4, proposal=4, selection=4)


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(v):
    return (json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(path):
    path = Path(path); require(path.is_file() and not path.is_symlink(), 'regular nonsymlink file: '+str(path))
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def raw(path):
    require(Path(path).stat().st_size <= 32*1024**2, 'bounded compressed input')
    value = Path(path).read_bytes()
    if str(path).endswith('.gz'):
        d = zlib.decompressobj(31); value = d.decompress(value, 32*1024**2+1)
        require(d.eof and not d.unused_data and not d.unconsumed_tail, 'complete single gzip stream, no trailing data')
    require(len(value) <= 32*1024**2, 'bounded expanded input')
    return value


def read(path): return json.loads(raw(path))


def array(v, shape):
    a = np.asarray(v)
    require(a.shape == shape and a.dtype.kind in 'fi' and np.isfinite(a).all(), 'finite aligned array '+str(shape))
    return a.astype(float)


def context(c):
    require(set(c) == {'layout', 'context_length', 'latent_dim', 'action_dim', 'latent_history', 'action_history'}, 'context fields')
    require((c['layout'], c['context_length'], c['latent_dim'], c['action_dim']) ==
            ('states_with_between_state_actions_v1', 3, 192, 10), 'context layout')
    array(c['latent_history'], (3, 192)); array(c['action_history'], (2, 10))


def cost(t, c):
    residual = array(t['projection'], (192, 192))@(array(c['latent_history'], (3, 192))[-1]-array(t['goal'], (192,)))
    return float(np.mean(residual**2))


def commands(logits, cap):
    x = array(logits, (2, 10))
    require(np.array_equal(x, x.astype(np.float32)) and (np.linalg.norm(x, axis=1) <= cap).all(), 'feasible float32 logits')
    return np.tanh(x.astype(np.float32)).reshape(10, 2).astype(float)


def history(t, previous, c, action):
    context(c)
    tokens = (np.asarray(action, np.float32).reshape(10)-np.tile(np.asarray(t['action_mean'], np.float32), 5))/np.tile(np.asarray(t['action_std'], np.float32), 5)
    require(c['latent_history'][:2] == previous['latent_history'][1:], 'latent chronology')
    require(np.array_equal(c['action_history'], np.vstack((previous['action_history'], tokens))[-2:]), 'normalized token/history content')


def projected(x, cap):
    x = array(x, (2, 10)).copy()
    for i in range(2):
        n = np.linalg.norm(x[i])
        if n > cap: x[i] *= cap/n
    x = x.astype(np.float32)
    for i in range(2):
        if np.linalg.norm(x[i].astype(float)) > cap: x[i] *= np.float32(1-4*np.finfo(np.float32).eps)
    return x.astype(float)


class Cursor:
    """Strict total event order incl nested J; no missing, extra or refunded calls."""
    def __init__(self, events, root):
        self.events, self.root, self.i = events, root, 0
        self.group = None; self.attempts = []; self.completed = set(); self.counts = Counter()

    def take(self, kind):
        require(self.i < len(self.events), 'truncated journal at '+kind)
        e = self.events[self.i]; self.i += 1
        require(e['kind'] == kind and e['root_index'] == self.root['root_index'], 'ordered event '+kind)
        if kind != 'root_started':
            require(e['root_sha256'] == self.root['root_sha256'] and e['evaluation_id'] == self.group, 'root/evaluation lineage')
        return e

    def begin(self, resource):
        e = self.take('operation_attempted')['event']; index = len(self.attempts)
        require(e == dict(index=index, phase='direct', evaluation_id=self.group, resource=resource, samples=1, completed=False), 'exact predispatch attempt')
        self.attempts.append(e); self.counts[resource] += 1
        require(resource in BUDGET and self.counts[resource] <= BUDGET[resource], 'nonrefundable per-root resource cap')
        return index

    def end(self, index):
        e = self.take('operation_completed')['event']
        require(index not in self.completed and e == dict(self.attempts[index], completed=True), 'unique matching return')
        self.completed.add(index)

    def op(self, resource):
        index = self.begin(resource); self.end(index)

    def packets(self, t, rawcommands, kind, mode=None):
        previous = t['root_context']; contexts = []
        for k in range(2):
            self.op('action_encoder'); self.op('predict')
            e = self.take(kind)
            require(e['packet'] == k and (mode is None or e['mode'] == mode), 'two packet order/mode')
            history(t, previous, e['context'], rawcommands[5*k:5*k+5]); previous = e['context']
            contexts.append(previous)
        return contexts

    def evaluate(self, t, logits, mode):
        rawcommands = commands(logits, t['packet_logit_cap'])
        request = self.take('evaluation_request')
        require(request['mode'] == mode and request['logits'] == np.asarray(logits).tolist() and
                np.array_equal(request['commands'], rawcommands), 'fresh request before actual forward')
        contexts = self.packets(t, rawcommands, 'native_packet', mode)
        value = dict(task_id=self.root['task_id'], logits=np.asarray(logits).tolist(), commands=rawcommands.tolist(),
                     contexts=contexts, cost=cost(t, contexts[-1]))
        e = self.take('evaluation_complete')
        require(e['mode'] == mode and e['prediction'] == value, 'complete same-Q prediction reconstructed')
        return value

    def triple(self, t, logits):
        values = [self.evaluate(t, logits, mode) for mode in ('nominal', 'whole_replay', 'segmented_replay')]
        require(values[0] == values[1] == values[2], 'three independent exact fullH2 replays')
        return dict(values[0], whole_replay=values[1]['contexts'][-1], segmented_replay=values[2]['contexts'])


def audit_root(t, root, saved_root, top_root, events):
    context(t['root_context'])
    q = array(t['projection'], (192, 192)); goal = array(t['goal'], (192,))
    require(np.allclose(q, q.T, atol=1e-8, rtol=0) and np.allclose(q@q, q, atol=1e-8, rtol=0) and np.trace(q) > .5, 'original orthogonal Q')
    require(t['horizon'] == 2 and t['packet_steps'] == 5 and t['feedback'] == 'open_loop_model_only', 'wholeH2 fixed feedback')
    require(t['representation'] == 'native_projected_cls' and t['latent_dim'] == 192 and t['control_dim'] == 2, 'full native192 action space')
    mean, std = array(t['action_mean'], (2,)), array(t['action_std'], (2,))
    require((std > 0).all(), 'positive original normalizer')
    rootraw = array(t['root_raw_action_history'], (10, 2))
    tokens = (rootraw.astype(np.float32).reshape(2, 10)-np.tile(mean.astype(np.float32), 5))/np.tile(std.astype(np.float32), 5)
    require(np.array_equal(tokens, t['root_context']['action_history']), 'root raw action/token history binding')
    require(hashlib.sha256(json.dumps(t, sort_keys=True, allow_nan=False).encode()).hexdigest() == root['task_id'], 'literal task identity with original JSON spacing')
    c = Cursor(events, root); start = c.take('root_started'); require(start['task_id'] == root['task_id'], 'root task start')
    c.group = 'initial'; incumbent = c.triple(t, np.zeros((2, 10)))
    require(c.take('initial_verified')['incumbent'] == incumbent, 'initial verified exact')
    steps = []; diagnostics = []; initial_cost = incumbent['cost']
    for iteration in range(1, 5):
        c.group = f'iter_{iteration}_AD'; jac_index = c.begin('jacobian')
        path = c.packets(t, np.asarray(incumbent['commands']), 'jacobian_forward_packet')
        fwd = c.take('jacobian_forward_complete')
        require(path == incumbent['contexts'] and fwd['logits'] == incumbent['logits'] and
                fwd['terminal'] == path[-1]['latent_history'][-1], 'fresh J point entire incumbent path')
        rows = []
        for row_index in range(192):
            c.op('vjp_directions'); e = c.take('jacobian_row')
            require(e['row'] == row_index, 'ALL192 ordered raw derivative rows')
            rows.append(array(e['raw_control_derivative'], (20,)))
        raw_j = np.stack(rows); ucommands = np.asarray(incumbent['commands']).reshape(20)
        j = raw_j*(1-ucommands**2)[None]
        joint = c.take('joint_logit_jacobian')
        require(np.array_equal(joint['jacobian'], j) and joint['terminal'] == fwd['terminal'], 'single tanh chain exact full192x20 J')
        c.end(jac_index)
        residual = q@(np.asarray(fwd['terminal'])-goal); a = q@j; gradient = 2*a.T@residual/192
        d = c.take('derivative_complete')['derivative']
        require(d['task_id'] == root['task_id'] and d['at_logits'] == incumbent['logits'] and
                d['at_terminal'] == fwd['terminal'] and np.array_equal(d['raw_terminal_jacobian'], j), 'derivative fresh-point binding')
        require(np.allclose(d['objective_gradient'], gradient, rtol=1e-8, atol=1e-10), 'same-Q objective gradient arithmetic')
        c.op('svd'); c.op('linear_solve'); c.op('proposal')
        proposal = c.take('line_proposals_frozen')
        delta = array(proposal['delta'], (20,))
        # Independent normal equations, not producer SVD or validation function.
        matrix = a.T@a/192+.002*np.eye(20); rhs = -a.T@residual/192
        solved = np.linalg.solve(matrix, rhs)
        require(np.allclose(delta, solved, rtol=1e-8, atol=1e-10), 'original .002 ridge GN direction')
        endpoint = projected(np.asarray(incumbent['logits'])+delta.reshape(2, 10), t['packet_logit_cap'])
        require(proposal['iteration'] == iteration and proposal['incumbent_logits'] == incumbent['logits'] and
                proposal['scales'] == SCALES and np.array_equal(proposal['endpoint'], endpoint), 'original full GN projected endpoint and fixed scales')
        trials = [(np.asarray(incumbent['logits'])+s*(endpoint-incumbent['logits'])).astype(np.float32).astype(float) for s in SCALES]
        require(np.array_equal(proposal['logits'], trials), 'ALL five float32 logit segments')
        candidates = []
        for i, x in enumerate(trials):
            c.group = f'iter_{iteration}_scale_{i}'; candidates.append(c.evaluate(t, x, 'whole_replay'))
        c.op('selection')
        choices = [incumbent]+candidates; index = min(range(6), key=lambda i:(choices[i]['cost'], i)); chosen = choices[index]
        row = dict(iteration=iteration, incumbent_cost=incumbent['cost'], trial_costs=[v['cost'] for v in candidates],
                   selected_index=index, selected_scale=None if index == 0 else SCALES[index-1], selected_cost=chosen['cost'])
        selection = c.take('selection_complete')
        require(all(selection[k] == v for k,v in row.items()), 'stable complete-five same-Q selection')
        c.group = f'iter_{iteration}_selected_replay'; fresh = c.triple(t, chosen['logits'])
        require(fresh['contexts'] == chosen['contexts'] and fresh['cost'] == chosen['cost'], 'selected fresh replay exact')
        update = c.take('update_verified')
        require(update['iteration'] == iteration and update['prediction'] == fresh, 'next J receives verified incumbent')
        diagnostics.append(dict(iteration=iteration, raw_rows=192, chained_j_sha256=hashlib.sha256(canonical(j.tolist())).hexdigest(),
            ridge_delta_max_abs_error=float(np.max(np.abs(delta-solved))), normal_equation_max_abs_residual=float(np.max(np.abs(matrix@delta-rhs))),
            gradient_max_abs_error=float(np.max(np.abs(gradient-d['objective_gradient'])))))
        steps.append(row); incumbent = fresh
    counts = {k:dict(attempted=c.counts[k], completed=sum(i in c.completed and e['resource'] == k for i,e in enumerate(c.attempts))) for k in BUDGET}
    require(c.i == len(events) and len(c.completed) == len(c.attempts) and counts == {k:dict(attempted=v, completed=v) for k,v in BUDGET.items()}, 'complete exact raw budget, no extra events')
    require(saved_root == dict(root_index=root['root_index'], root_sha256=root['root_sha256'], task_id=root['task_id'],
                              steps=steps, final_prediction=incumbent, resources=counts), 'root gzip exact reconstructed result')
    require(top_root == dict(root_sha256=root['root_sha256'], steps=steps, final_cost=incumbent['cost']), 'top report exact reconstructed result')
    return dict(root_index=root['root_index'], root_sha256=root['root_sha256'], task_id=root['task_id'],
        initial_cost=initial_cost, final_cost=incumbent['cost'], ownQ_cost_reduction=initial_cost-incumbent['cost'],
        steps=steps, derivatives=diagnostics, resources=counts, raw_j_rows=768, fullH2_rollouts=35, AD_fullH2_forwards=4)


def load_bundle(report_path=INPUT, expected=INPUT_SHA):
    report_path = Path(report_path).resolve(); bindings = {}
    def bound(path, h=None):
        path = Path(path).resolve(); actual = sha(path)
        require(h is None or h == actual, 'SHA mismatch: '+str(path))
        bindings[str(path)] = actual; return path
    top = read(bound(report_path, expected)); folder = report_path.parent
    require(top['schema'] == 'p1_line_search_gn_cuda_20261027_v1.report' and
            top['status'] == 'cuda_line_search_gn_completed' and top['errors'] == [] and top['first_error'] is None, 'actual complete no errors')
    expected_files = {'binding.json', 'events.jsonl.gz', 'root_0.json.gz', 'root_1.json.gz'}
    require(set(top['artifact_sha256']) == expected_files and {p.name for p in folder.iterdir()} == expected_files|{'report.json'}, 'exact output file census')
    for name, h in top['artifact_sha256'].items(): bound(folder/name, h)
    binding = read(folder/'binding.json')
    require(top['prepare_sha256'] == binding['prepare_sha256'] == PREPARE_SHA, 'original protocol seal')
    p = read(bound(binding['prepared'], PREPARE_SHA))
    require(p['schema'] == 'p1_line_search_gn_cuda_20261027_v1.prepare' and p['scales'] == SCALES and p['updates'] == 4 and p['damping'] == .002 and
            p['per_root_budget'] == BUDGET and p['total_budget'] == {k:2*v for k,v in BUDGET.items()} and
            p['initial_logits'] == np.zeros((2, 10)).tolist(), 'fixed protocol full budget')
    for path, h in p['bindings_sha256'].items(): bound(path, h)
    require(top['model_hash_before'] == top['model_hash_after'] == p['expected_model_hash'] and
            top['model_unchanged'] is True and top['source_verified'] is True, 'recorded frozen tensor state')
    tasks = [read(r['task_path'])['task'] for r in p['roots']]
    for r,t in zip(p['roots'], tasks):
        require(str(Path(r['task_path']).resolve()) in bindings, 'task source hash bound')
        for name, key in (('weights.pt', 'model_sha256'), ('config.json', 'model_config_sha256')):
            require(t[key] == next(h for path,h in p['bindings_sha256'].items() if Path(path).name == name), 'task model/config identity')
    require(len(p['roots']) == 2 and [r['root_index'] for r in p['roots']] == [0,1] and len({r['root_sha256'] for r in p['roots']}) == 2, 'fixed two-root census')
    events = [json.loads(line) for line in raw(folder/'events.jsonl.gz').splitlines()]
    roots = [read(folder/f'root_{i}.json.gz') for i in range(2)]
    return dict(top=top, prepare=p, tasks=tasks, events=events, roots=roots, bindings=bindings)


def audit(bundle):
    b = bundle; events = b['events']; rows = []; start = 0
    starts = [i for i,e in enumerate(events) if e['kind'] == 'root_started']
    require(len(starts) == 2 and starts[0] == 0, 'root chronology census')
    require(len(b['top']['roots']) == len(b['roots']) == len(b['tasks']) == 2, 'all two-root objects')
    for index, root in enumerate(b['prepare']['roots']):
        end = starts[index+1] if index+1 < len(starts) else len(events)
        rows.append(audit_root(b['tasks'][index], root, b['roots'][index], b['top']['roots'][index], events[start:end])); start = end
    counts = {k:dict(attempted=sum(r['resources'][k]['attempted'] for r in rows), completed=sum(r['resources'][k]['completed'] for r in rows)) for k in BUDGET}
    require(counts == b['top']['resources'], 'top raw budget exact')
    require(b['top']['new_images'] == b['top']['new_physics'] == 0 and b['top']['training'] is False and b['top']['new_Q'] is False and
            b['top']['P1_admission'] is False and b['top']['independent_sample_increment'] == 0, 'scope unchanged')
    for path, expected in b['bindings'].items(): require(sha(path) == expected, 'post-read source/input TOCTOU')
    return dict(schema=NAME+'.report', status='saved_arithmetic_verified_not_admitted', all_checks=True, roots=rows,
        resources=counts, event_count=len(events), raw_derivative_rows=1536, fullH2_rollouts=70, AD_fullH2_forwards=8,
        input_sha256=b['bindings'], training=False, new_neural_calls=0, new_physical_steps=0, P1_admission=False,
        limitations=['saved receipt arithmetic, not a new execution or independent AD correctness proof',
                    'frozen tensor identity is recorded by CUDA producer, not independently reloaded here',
                    'two calibration roots, not physical efficacy or matched B8 FLOPs; no cross-Q subtraction',
                    'ridge numeric tolerances 1e-8 relative/1e-10 absolute are receipt checks, not scientific thresholds'])


def execute(report, expected, output):
    result = audit(load_bundle(report, expected))
    result['audit_source_sha256'] = {str(p):sha(p) for p in (Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md'))}
    data = canonical(result); output = Path(output).resolve()
    require(len(data) < 256*1024 and not output.exists(), 'exclusive small new audit report')
    require(shutil.disk_usage(output.parent).free >= 200*1024**2+len(data), '200MiB floor')
    with output.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    return dict(output=str(output), sha256=sha(output), bytes=len(data), status=result['status'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=INPUT); parser.add_argument('--report-sha', default=INPUT_SHA)
    parser.add_argument('--output', type=Path, required=True); a = parser.parse_args()
    print(canonical(execute(a.report, a.report_sha, a.output)).decode().strip())
