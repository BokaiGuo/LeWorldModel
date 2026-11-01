"""Fixed saved prefix panel, nonlinear suffix continuation on HOST CUDA.

Post-hoc exploratory model-only experiment; no physics, training or gate change.
All 160 prefix occurrences are evaluated, including repeated candidates and arms.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_joint_suffix_fd_host_20261029 as source
from experiments.spectral_reachability import agent_p1_joint_vreach_saved_diff_20261029_v1 as saved
from experiments.spectral_reachability import agent_p1_joint_vreach_host_cuda_20261029_v1 as host
from experiments.spectral_reachability import p1_conditional_direct_screen_20261027_v1 as projection

NAME = 'p1_uphill_nonlinear_suffix_host_20261029'
UPDATES = 8
CAP = float(np.sqrt(5)*.75)


def proposal(logits, raw_grad, raw):
    x = np.asarray(logits, np.float64).reshape(10)
    g = np.asarray(raw_grad, np.float64).reshape(10)
    a = np.asarray(raw, np.float32).astype(np.float64).reshape(10)
    if not all(np.isfinite(z).all() for z in (x, g, a)):
        raise ValueError('nonfinite update')
    return projection.projected_logits(np.stack((x-g*(1-a*a), np.zeros(10))), CAP)[0]


def summarize(report, rows):
    if len(rows) != 160:
        raise ValueError('fixed candidate denominator')
    result = []
    for i, arm in enumerate(report['rows']):
        r = rows[20*i:20*(i+1)]
        if any(x['slot'] != arm['slot'] or x['arm'] != arm['arm'] for x in r):
            raise ValueError('arm roster')
        base = r[0]['best_regularized']
        for x in r:
            x['nonlinear_reach_better'] = x['best_regularized'] < base
            x['terminal_better'] = x['best_terminal_q'] < arm['result']['baseline_terminal_Q']
            x['exploratory_conjunction'] = x['stage_uphill'] and x['nonlinear_reach_better'] and x['terminal_better']
        result.extend(r)
    return dict(rows=result, summary=dict(candidates=len(result),
        stage_uphill=sum(x['stage_uphill'] for x in result),
        nonlinear_reach_better=sum(x['nonlinear_reach_better'] for x in result),
        terminal_better=sum(x['terminal_better'] for x in result),
        conjunction=sum(x['exploratory_conjunction'] for x in result)))


def run(output):
    report = saved.read_verified(source.REPORT, source.REPORT_SHA)
    receipt = saved.read_verified(source.RECEIPT, source.RECEIPT_SHA)
    if receipt.get('report_sha256') != source.REPORT_SHA:
        raise ValueError('receipt binding')
    p = saved.read_verified(report['prepared'], report['prepare_sha256'])
    saved.analyze(report, receipt, p)
    host.verify(report['prepared'], report['prepare_sha256'])
    output = Path(output)
    if not output.is_absolute() or output.parent != source.ROOT/'results' or output.resolve() != output or not output.name.startswith(NAME+'_') or output.exists():
        raise ValueError('exclusive output')
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    info = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    rows = []; calls = {'predict': 0, 'action_encoder': 0, 'reverse': 0}
    with torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
        model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
        before = host.signal.model_hash(model)
        if before != report['model_before']:
            raise ValueError('model hash')
        for module in model.modules():
            if str(Path(inspect.getfile(type(module))).resolve()) not in p['bindings']:
                raise ValueError('model class source unbound')
        def hook(name):
            def count(_module, _args): calls[name] += 1
            return count
        hooks = [model.predictor.register_forward_pre_hook(hook('predict')),
                 model.action_encoder.register_forward_pre_hook(hook('action_encoder'))]
        try:
            for arm_index, arm in enumerate(report['rows']):
                task = p['requests'][arm_index//2]['task']
                goal = torch.tensor(task['goal'], dtype=torch.float32, device=device)
                mean = torch.tensor(np.tile(task['action_mean'], 5), dtype=torch.float32, device=device)
                std = torch.tensor(np.tile(task['action_std'], 5), dtype=torch.float32, device=device)
                for c in arm['result']['candidates']:
                    z = torch.tensor(c['mid']['latent_history'], dtype=torch.float32, device=device)[None]
                    h = torch.tensor(c['mid']['action_history'], dtype=torch.float32, device=device)[None]
                    x = np.asarray(c['suffix_solve']['params'], np.float64)
                    evaluations = []
                    for iteration in range(UPDATES+1):
                        raw = np.tanh(np.asarray(x, np.float32)).reshape(5, 2)
                        action = torch.tensor(raw, dtype=torch.float32, device=device, requires_grad=iteration < UPDATES)
                        token = (action.reshape(10)-mean)/std
                        end = model.predict(z, model.action_encoder(torch.cat((h, token[None,None]), 1)))[:, -1][0]
                        loss = torch.mean((end-goal)**2)
                        endpoint = end.detach().cpu().numpy().astype(np.float64)
                        q = float(np.mean((endpoint-np.asarray(task['goal'], np.float64))**2))
                        if iteration == 0 and not np.array_equal(endpoint, np.asarray(c['end']['latent_history'][-1], np.float64)):
                            raise ValueError('saved incumbent replay mismatch')
                        grad = None
                        if iteration < UPDATES:
                            grad = torch.autograd.grad(loss, action)[0].detach().cpu().numpy()
                            calls['reverse'] += 1
                        evaluations.append(dict(iteration=iteration, logits=x.tolist(), raw=raw.tolist(), endpoint=endpoint.tolist(),
                            terminal_q=q, regularized=q+saved.DAMPING*float(x@x)/192,
                            raw_gradient=None if grad is None else grad.tolist()))
                        if grad is not None:
                            x = proposal(x, grad, raw)
                    best = min(evaluations, key=lambda e: (e['regularized'], e['iteration']))
                    rows.append(dict(slot=arm['slot'], arm=arm['arm'], start=c['start'], iteration=c['iteration'],
                        stage_uphill=c['stage_uphill'], best_iteration=best['iteration'], best_regularized=best['regularized'],
                        best_terminal_q=best['terminal_q'], evaluations=evaluations))
        finally:
            for handle in hooks: handle.remove()
        torch.cuda.synchronize(device)
        after = host.signal.model_hash(model)
        if before != after or calls != {'predict': 1440, 'action_encoder': 1440, 'reverse': 1280}:
            raise ValueError('model mutation or call denominator mismatch')
    host.F.check_bindings(p['bindings'])
    result = summarize(report, rows)
    result.update(schema=NAME, source_report_sha256=source.REPORT_SHA, source_receipt_sha256=source.RECEIPT_SHA,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), model_before=before, model_after=after,
        device=info, calls=calls, updates=UPDATES, action_penalty=saved.DAMPING, output_scope='known-root exploratory, no physical execution',
        scientific_admission=False, P1_admission=False, physical_release=False, training=False)
    return dict(path=str(output), sha256=saved.write(output, result), summary=result['summary'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    print(json.dumps(run(parser.parse_args().output)))


if __name__ == '__main__': main()
