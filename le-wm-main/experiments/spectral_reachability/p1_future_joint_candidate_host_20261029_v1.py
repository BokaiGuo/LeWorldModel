"""Fixed two-packet differentiable lookahead candidates, known roots only.

Prepare is CPU/saved-only. Run requires explicit HOST CUDA authorization.
This is a proposal producer, not a P1 gate or native resource attestation.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
from experiments.spectral_reachability import agent_p1_joint_vreach_host_cuda_20261029_v1 as host
from experiments.spectral_reachability import agent_p1_two_packet_uphill_generator_20261029_v1 as parent

NAME = 'p1_future_joint_candidate_host_20261029_v1'
AUTH = 'P1_FUTURE_JOINT_KNOWN_ROOT_HOST_CUDA_ONLY'
ROOT = host.ROOT
UPDATES = 8
PENALTY = 1.0
STARTS = ('winner', 'zero', 'positive_e0', 'negative_e0')
PREDICT_CAP = 296
REVERSE_CAP = 128


def contract():
    return dict(slots=4, known_roots=2, starts=list(STARTS), updates=UPDATES,
        candidates_per_slot=36, candidate_denominator=144, baseline_replays=4,
        predict_calls=PREDICT_CAP, action_encoder_calls=PREDICT_CAP, scalar_reverse_calls=REVERSE_CAP,
        objective='terminal_mean_Q + relu(initial_Q-stage_mean_Q)',
        update='simultaneous two-packet raw gradient, tanh chain, eta=1, original separate packet balls',
        eligibility='diagnostic strict uphill and terminal improvement only; original Vreach gate unevaluated',
        equal_FLOPs=False, FLOPs=None, physics=False, training=False, freshness=False,
        P1_admission=False, physical_release=False)


def starts(winner):
    winner = np.asarray(winner, dtype=np.float64)
    host.require(winner.shape == (10,) and np.isfinite(winner).all() and np.linalg.norm(winner) <= host.F.CAP, 'winner prefix domain')
    e = np.zeros(10); e[0] = .25
    return [winner.copy(), np.zeros(10), e, -e]


def update(x, gradient, action):
    x = np.asarray(x, np.float64); g = np.asarray(gradient, np.float64); a = np.asarray(action, np.float32).astype(np.float64)
    host.require(x.shape == (10,) and g.shape == (10,) and a.shape == (10,) and
                 all(np.isfinite(v).all() for v in (x,g,a)), 'finite projected update')
    return host.F.proposal(x, g * (1-a*a))


def two_step(model, z, history, prefix, suffix, mean, std, torch):
    """Keep the predicted midpoint and action history on the AD graph."""
    def step(z0, h0, action):
        token = ((action.reshape(5,2)-mean)/std).reshape(1,1,10)
        outgoing = torch.cat((h0, token), 1)
        predicted = model.predict(z0, model.action_encoder(outgoing))
        host.require(tuple(predicted.shape) == (1,3,192), 'predict output ABI')
        return torch.cat((z0[:,1:], predicted[:,-1:]),1), outgoing[:,1:]
    mid, h1 = step(z, history, prefix)
    end, _ = step(mid, h1, suffix)
    return mid[0,-1], end[0,-1]


def prepare(source, source_sha, output):
    p = host.verify(source, source_sha)
    for i, q in enumerate(p['requests']):
        parent.request(q); host.require(q['slot'] == i, 'fixed known-root slots')
    own = str(Path(__file__).resolve())
    bindings = dict(p['bindings'])
    host.authority.add(bindings, own, host.sha(own))
    host.F.check_bindings(bindings)
    value = dict(schema=NAME+'.prepare', source=source, source_sha256=source_sha,
                 source_bindings=bindings, contract=contract(), known_roots=True,
                 scientific_admission=False)
    return write_file(output, value)


def verify(path, expected):
    p = Path(path)
    host.require(p.is_absolute() and p.resolve() == p and host.sha(p) == expected, 'prepared path/hash')
    value = host.signal.read(p, 256*1024**2)
    host.require(value['schema'] == NAME+'.prepare' and value['contract'] == contract() and
                 value['known_roots'] is True and value['scientific_admission'] is False, 'fixed contract')
    source = host.verify(value['source'], value['source_sha256'])
    bindings = dict(source['bindings']); own = str(Path(__file__).resolve())
    host.authority.add(bindings, own, host.sha(own))
    host.require(value['source_bindings'] == bindings, 'exact source bindings')
    host.F.check_bindings(bindings)
    host.require(host.sha(p) == expected, 'stable prepared bytes')
    return source, value


def write_file(path, value):
    p = Path(path); data = host.raw(value)
    host.require(p.is_absolute() and p.resolve() == p and p.parent == ROOT/'results' and
                 p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and
                 len(data) <= 256*1024**2, 'exclusive bounded output')
    with p.open('xb') as f:
        host.require(f.write(data) == len(data), 'complete write'); f.flush(); os.fsync(f.fileno())
    host.signal.fsync_dir(p.parent)
    return dict(path=str(p), sha256=hashlib.sha256(data).hexdigest())


def run(prepared, prepare_sha, output, authorization):
    host.require(authorization == AUTH, 'explicit CUDA authorization')
    source, seal = verify(prepared, prepare_sha)
    p = Path(output)
    host.require(p.is_absolute() and p.resolve() == p and p.parent == ROOT/'results' and
                 p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink(), 'exclusive run directory')
    p.mkdir(); host.signal.fsync_dir(p.parent)
    journal_path = p/'events.jsonl'; report_path = p/'report.json'
    started = time.monotonic(); calls = dict(predict=0, action_encoder=0, reverse=0)
    rows = []; errors = []; before = after = device_info = None
    def emit(event):
        data = host.raw(event) + b'\n'
        with journal_path.open('ab') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    try:
        os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        device_info = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            model = load_compatible_pretrained(source['policy'], device=device).eval().requires_grad_(False)
            before = host.signal.model_hash(model)
            host.require(before['original_state_dict'] == source['expected_model_hash'], 'original model hash')
            for module in model.modules():
                host.require(str(Path(inspect.getfile(type(module))).resolve()) in seal['source_bindings'], 'model class source binding')
            def hook(name, cap):
                def count(_module, _args):
                    host.require(calls[name] < cap and time.monotonic()-started < 7200, 'fixed call/wall cap')
                    host.require(torch.cuda.memory_allocated(device) <= 3*1024**3 and torch.cuda.memory_reserved(device) <= 4*1024**3, 'CUDA cap')
                    calls[name] += 1
                return count
            hooks = [model.predictor.register_forward_pre_hook(hook('predict', PREDICT_CAP)),
                     model.action_encoder.register_forward_pre_hook(hook('action_encoder', PREDICT_CAP))]
            try:
                for q in source['requests']:
                    task, wx, wy = parent.request(q); host.require(np.linalg.norm(wy) <= host.F.CAP, 'winner suffix domain')
                    tensor = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
                    z = tensor(task['root_context']['latent_history'])[None]; h = tensor(task['root_context']['action_history'])[None]
                    goal = tensor(task['goal']); mean = tensor(task['action_mean']); std = tensor(task['action_std'])
                    initial = host.F.cost(task['goal'], task['root_context'])
                    with torch.no_grad():
                        _, b = two_step(model,z,h,tensor(host.F.chart(wx).reshape(10)),tensor(host.F.chart(wy).reshape(10)),mean,std,torch)
                    baseline = float(np.mean((b.cpu().numpy().astype(np.float64)-np.asarray(task['goal'],np.float64))**2))
                    for start, x0 in enumerate(starts(wx)):
                        x = x0.copy(); y = wy.copy()
                        for iteration in range(UPDATES+1):
                            ax = host.F.chart(x).reshape(10); ay = host.F.chart(y).reshape(10)
                            a = tensor(ax).requires_grad_(iteration < UPDATES); b = tensor(ay).requires_grad_(iteration < UPDATES)
                            mid, end = two_step(model,z,h,a,b,mean,std,torch)
                            stage_loss = ((mid-goal)**2).mean(); terminal_loss = ((end-goal)**2).mean()
                            stage = float(np.mean((mid.detach().cpu().numpy().astype(np.float64)-np.asarray(task['goal'],np.float64))**2))
                            terminal = float(np.mean((end.detach().cpu().numpy().astype(np.float64)-np.asarray(task['goal'],np.float64))**2))
                            host.require(np.isfinite(stage) and np.isfinite(terminal), 'finite model scores')
                            row = dict(slot=q['slot'], root_slot=q['root_slot'], condition=q['condition'], start=start,
                                iteration=iteration, prefix_logits=x.tolist(), suffix_logits=y.tolist(),
                                commands=np.concatenate((ax.reshape(5,2),ay.reshape(5,2))).tolist(),
                                initial_Q=initial, stage_Q=stage, terminal_Q=terminal, baseline_Q=baseline,
                                stage_uphill=stage>initial, terminal_better=terminal<baseline,
                                original_vreach_gate='not_evaluated', P1_admission=False)
                            emit(dict(kind='candidate', row=row)); rows.append(row)
                            if iteration < UPDATES:
                                host.require(calls['reverse'] < REVERSE_CAP, 'reverse cap')
                                loss = terminal_loss + PENALTY*torch.relu(tensor(initial)-stage_loss)
                                ga, gb = torch.autograd.grad(loss, (a,b)); calls['reverse'] += 1
                                x = update(x,ga.detach().cpu().numpy(),ax); y = update(y,gb.detach().cpu().numpy(),ay)
                                host.require(np.linalg.norm(x) <= host.F.CAP and np.linalg.norm(y) <= host.F.CAP, 'packet domains')
            finally:
                for hook_handle in hooks: hook_handle.remove()
            torch.cuda.synchronize(device); after = host.signal.model_hash(model)
            host.require(before == after and calls == dict(predict=PREDICT_CAP,action_encoder=PREDICT_CAP,reverse=REVERSE_CAP)
                         and len(rows) == 144, 'frozen model and fixed denominators')
        host.F.check_bindings(seal['source_bindings']); host.require(host.sha(prepared) == prepare_sha, 'stable prepare')
    except BaseException as exc:
        errors.append(dict(type=type(exc).__name__, message=str(exc)))
    report = dict(schema=NAME+'.report', status='partial_failclosed' if errors else 'complete_model_only',
        prepared=prepared, prepare_sha256=prepare_sha, contract=contract(), rows=rows, errors=errors, calls=calls,
        model_before=before, model_after=after, device=device_info, known_roots=True, freshness=False,
        scientific_admission=False, P1_admission=False, physical_release=False)
    data = host.raw(report)
    with report_path.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    host.signal.fsync_dir(p)
    if errors: raise RuntimeError('partial fail-closed report: '+str(report_path)+' '+str(errors[0]))
    return dict(path=str(report_path), sha256=hashlib.sha256(data).hexdigest(), calls=calls, candidates=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--source', required=True); p.add_argument('--source-sha256', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('verify'); p.add_argument('--prepared', required=True); p.add_argument('--prepare-sha256', required=True)
    p = sub.add_parser('run'); p.add_argument('--prepared', required=True); p.add_argument('--prepare-sha256', required=True); p.add_argument('--output', required=True); p.add_argument('--authorization', required=True)
    a = parser.parse_args()
    if a.command == 'prepare': result = prepare(a.source, a.source_sha256, a.output)
    elif a.command == 'verify': verify(a.prepared, a.prepare_sha256); result = dict(verified=True, P1_admission=False)
    else: result = run(a.prepared, a.prepare_sha256, a.output, a.authorization)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
