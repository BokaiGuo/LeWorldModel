"""MAIN-only central differences of saved root/selected-suffix Jacobians.

All four conditions, ten coordinates and two fixed epsilons are retained.
No new AD, optimization, image encoding, physics, fitted threshold or admission.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.spectral_reachability import agent_mixedmode_prefix_20261028_v1 as source

NAME = 'audit_mixedmode_fd_MAIN_20261028_v1'
REPORT = source.ROOT / 'results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01/report.json'
REPORT_SHA = 'd6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2'
ANALYSIS = source.ROOT / 'results/analyze_mixedmode_MAIN_20261028_v1_MAIN01.json'
ANALYSIS_SHA = '0788833b58af36e49c4794be90def317688c3590c2d873f81cfc631f2226c830'
EPS = (.01, .005)
TOTAL = 8 * 41
require, sha, canonical = source.require, source.sha, source.canonical


def compare(jacobian, coordinate, epsilon, minus, plus):
    j = np.asarray(jacobian, np.float64)
    a, b = np.asarray(minus, np.float64), np.asarray(plus, np.float64)
    require(j.shape == (192, 10) and a.shape == b.shape == (192,), 'complete coordinate/latent dimensions')
    require(np.isfinite(j).all() and np.isfinite(a).all() and np.isfinite(b).all(), 'finite full vectors')
    fd = (b - a) / (2 * float(np.float32(epsilon)))
    reference = j[:, coordinate]
    absolute = float(np.linalg.norm(fd - reference))
    scale = max(float(np.linalg.norm(fd)), float(np.linalg.norm(reference)), 1e-12)
    return dict(fd=fd.tolist(), saved_AD=reference.tolist(), absolute_l2=absolute,
                relative_l2=absolute / scale)


def write_new(path, value):
    payload = canonical(value)
    require(len(payload) <= 8 * 1024**2, 'bounded metadata')
    with Path(path).open('xb') as stream:
        require(stream.write(payload) == len(payload), 'full metadata write')
        stream.flush()
        os.fsync(stream.fileno())


def prepare(output):
    require(sha(REPORT) == REPORT_SHA and sha(ANALYSIS) == ANALYSIS_SHA, 'actual completed evidence seals')
    r = json.loads(REPORT.read_bytes())
    p = json.loads(Path(r['prepared']).read_bytes())
    require(r['commands_released'] and not r['errors'] and len(r['rows']) == 4, 'complete producer')
    points = []
    for slot, row in zip(p['slots'], r['rows'], strict=True):
        c = row['result']['winner']
        suffix_context = row['forwards'][2 + 6*c['index']]['output']
        for label, context, nominal, jacobian in (
            ('root', slot['packet']['root_context'], row['result']['root_nominal'], row['result']['root_jacobian']),
            ('selected_suffix', suffix_context, c['nominal_terminal'], c['jacobian'])):
            points.append(dict(index=len(points), condition=row['condition'], root_slot=row['root_slot'],
                               label=label, context=context, nominal=nominal, jacobian=jacobian,
                               selected_index=c['index']))
    files = dict(p['bindings'])
    for path in (REPORT, ANALYSIS, Path(__file__).resolve(),
                 source.ROOT / 'tests/test_audit_mixedmode_fd_MAIN_20261028_v1.py', Path(r['prepared'])):
        files[str(path)] = sha(path)
    value = dict(schema=NAME+'.prepare', bindings=files, points=points, epsilons=list(EPS),
                 policy=p['policy'], expected_model_hash=p['expected_model_hash'],
                 total_predict=TOTAL, new_AD=0, new_physics=0, scientific_threshold=None,
                 P1_admission=False, selection='all4conditions root and model-selected suffix; no physical outcomes')
    verify_bindings(value)
    write_new(output, value)
    return dict(prepared=str(output), sha256=sha(output), points=8, predict=TOTAL)


def verify_bindings(p):
    require(p['schema'] == NAME+'.prepare' and len(p['points']) == 8 and p['total_predict'] == TOTAL
            and p['epsilons'] == list(EPS), 'fixed full diagnostic census')
    for path, expected in p['bindings'].items():
        require(sha(path) == expected, 'unchanged bound input/source: '+path)


def run(prepared, expected, output, authorization):
    require(authorization == 'MAIN_HOST_CUDA_FD_ONLY', 'explicit MAIN CUDA authorization')
    require(sha(prepared) == expected, 'external prepare SHA')
    p = json.loads(Path(prepared).read_bytes())
    verify_bindings(p)
    folder = Path(output)
    require(folder.parent.resolve() == source.ROOT/'results' and not folder.exists()
            and folder.name.startswith(NAME), 'new exclusive result directory')
    folder.mkdir()
    write_new(folder/'binding.json', dict(prepared=str(Path(prepared).resolve()), sha256=expected))
    started = time.monotonic()
    errors, evaluations, comparisons, handles = [], [], [], []
    model = before = after = host = None
    counts = {key: dict(attempted=0, completed=0) for key in ('predictor_module', 'action_encoder', 'pred_proj')}
    active = None
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host = require_cuda_allocation('cuda:0')
        device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
        source.overlay.moving.low.frozen(model)
        before = source.overlay.moving.low.tensor_hash(model)
        require(before == p['expected_model_hash'], 'original frozen model')
        transition = source.compiler.make_transition(model, device)
        def pre(key):
            def hook(module, args):
                require(not torch.is_grad_enabled() and active is not None
                        and counts[key]['attempted'] < TOTAL, 'no AD, retry or excess call')
                counts[key]['attempted'] += 1
            return hook
        def post(key):
            def hook(module, args, result):
                require(counts[key]['completed'] < counts[key]['attempted'], 'matched native return')
                counts[key]['completed'] += 1
            return hook
        for key, module in [('predictor_module', model.predictor), ('action_encoder', model.action_encoder), ('pred_proj', model.pred_proj)]:
            handles.extend([module.register_forward_pre_hook(pre(key)), module.register_forward_hook(post(key))])
        with (folder/'events.jsonl').open('xb') as journal:
            def evaluate(point, coordinate=None, epsilon=None, sign=0):
                nonlocal active
                require(len(evaluations) < TOTAL and time.monotonic()-started < 900, 'bounded predispatch')
                logits = np.zeros(10, np.float32)
                if coordinate is not None:
                    logits[coordinate] = np.float32(epsilon)*sign
                raw = np.tanh(logits).reshape(5, 2)
                active = dict(order=len(evaluations), point=point['index'], coordinate=coordinate,
                              epsilon=epsilon, sign=sign, commands=raw.tolist())
                journal.write(canonical(dict(kind='attempt', request=active))); journal.flush()
                c = point['context']
                latent, _ = transition(torch.tensor(c['latent_history'], device=device, dtype=torch.float32)[None],
                    torch.tensor(c['action_history'], device=device, dtype=torch.float32)[None],
                    torch.tensor(raw, device=device))
                terminal = latent[0, -1].detach().cpu().tolist()
                require(np.isfinite(terminal).all(), 'finite actual output')
                row = dict(active, terminal=terminal)
                evaluations.append(row)
                journal.write(canonical(dict(kind='returned', row=row))); journal.flush()
                active = None
                return terminal
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type='cuda', enabled=False):
                for point in p['points']:
                    nominal = evaluate(point)
                    require(nominal == point['nominal'], 'exact saved nominal, no retry')
                    for coordinate in range(10):
                        for epsilon in EPS:
                            minus = evaluate(point, coordinate, epsilon, -1)
                            plus = evaluate(point, coordinate, epsilon, 1)
                            comparisons.append(dict(point=point['index'], coordinate=coordinate, epsilon=epsilon,
                                **compare(point['jacobian'], coordinate, epsilon, minus, plus)))
            journal.flush(); os.fsync(journal.fileno())
        require(len(evaluations) == TOTAL and len(comparisons) == 160
                and all(v == dict(attempted=TOTAL, completed=TOTAL) for v in counts.values()), 'full actual ledger')
    except BaseException as error:
        errors.append(type(error).__name__+': '+str(error))
    finally:
        for handle in handles:
            handle.remove()
        if model is not None:
            try:
                source.overlay.moving.low.frozen(model)
                after = source.overlay.moving.low.tensor_hash(model)
                require(before == after == p['expected_model_hash'], 'unchanged weights/buffers')
            except BaseException as error:
                errors.append('final_model: '+str(error))
        try:
            verify_bindings(p)
        except BaseException as error:
            errors.append('final_sources: '+str(error))
        report = dict(schema=NAME+'.report', status='complete_diagnostic_only' if not errors else 'failed_preserved',
            errors=errors, host=host, model_before=before, model_after=after, evaluations=evaluations,
            comparisons=comparisons, resources=counts, wall_seconds=time.monotonic()-started,
            new_AD=0, new_physics=0, P1_admission=False, scientific_threshold=None,
            source_binding_sha256=expected, interrupted_request=active,
            journal_sha256=sha(folder/'events.jsonl') if (folder/'events.jsonl').exists() else None)
        write_new(folder/'report.json', report)
    require(not errors, 'diagnostic failed; retain partial report: '+str(errors))
    return dict(status=report['status'], report_sha256=sha(folder/'report.json'), predict=len(evaluations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    for key in ('output', 'prepared', 'sha256', 'authorization'):
        parser.add_argument('--'+key)
    args = parser.parse_args()
    result = prepare(args.output) if args.command == 'prepare' else run(args.prepared, args.sha256, args.output, args.authorization)
    print(canonical(result).decode())


if __name__ == '__main__':
    main()
