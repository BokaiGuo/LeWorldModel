"""Fixed-panel two-packet nonlinear continuation; exploratory HOST CUDA only."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_uphill_nonlinear_suffix_host_20261029 as one

NAME = 'p1_regularized_suffix_host_20261030'
UPDATES = 8


def proposal(logits, raw_grad, raw):
    """Projected descent on terminal MSE plus the saved logits penalty."""
    x=np.asarray(logits,np.float64).reshape(10)
    g=np.asarray(raw_grad,np.float64).reshape(10)
    a=np.asarray(raw,np.float32).astype(np.float64).reshape(10)
    if not all(np.isfinite(z).all() for z in (x,g,a)):
        raise ValueError('nonfinite update')
    update=g*(1-a*a)+2*one.saved.DAMPING*x/192
    return one.projection.projected_logits(np.stack((x-update,np.zeros(10))),one.CAP)[0]


def advance(z, history, tokens, predict):
    """Preserve both state and action history, including the autograd graph."""
    import torch
    endpoints = []
    for token in tokens:
        outgoing = torch.cat((history, token[None, None]), 1)
        successor = predict(z, outgoing)[:, -1:]
        z = torch.cat((z[:, 1:], successor), 1)
        history = outgoing[:, 1:]
        endpoints.append(successor[0, 0])
    return endpoints


def summarize(source_report, rows):
    if len(rows) != 160:
        raise ValueError('fixed candidate denominator')
    for i, arm in enumerate(source_report['rows']):
        group = rows[i*20:(i+1)*20]
        if any((r['slot'], r['arm']) != (arm['slot'], arm['arm']) for r in group):
            raise ValueError('arm roster mismatch')
        baseline = group[0]['best_regularized']
        for row in group:
            row['better_than_same_horizon_winner'] = row['best_regularized'] < baseline
            row['uphill_and_better'] = row['stage_uphill'] and row['better_than_same_horizon_winner']
    return dict(candidates=len(rows), uphill=sum(r['stage_uphill'] for r in rows),
                better=sum(r['better_than_same_horizon_winner'] for r in rows),
                uphill_and_better=sum(r['uphill_and_better'] for r in rows))


def run(output):
    source, saved, host = one.source, one.saved, one.host
    report = saved.read_verified(source.REPORT, source.REPORT_SHA)
    receipt = saved.read_verified(source.RECEIPT, source.RECEIPT_SHA)
    if receipt.get('report_sha256') != source.REPORT_SHA:
        raise ValueError('receipt binding')
    prepared = saved.read_verified(report['prepared'], report['prepare_sha256'])
    saved.analyze(report, receipt, prepared)
    host.verify(report['prepared'], report['prepare_sha256'])
    output = Path(output)
    if not output.is_absolute() or output.parent != source.ROOT/'results' or output.resolve() != output or not output.name.startswith(NAME+'_') or output.exists():
        raise ValueError('exclusive output required')
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    device_info = require_cuda_allocation('cuda:0')
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    rows = []
    calls = dict(predict=0, action_encoder=0, reverse=0)
    with torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
        model = load_compatible_pretrained(prepared['policy'], device=device).eval().requires_grad_(False)
        before = host.signal.model_hash(model)
        if before != report['model_before']:
            raise ValueError('model hash')
        for module in model.modules():
            if str(Path(inspect.getfile(type(module))).resolve()) not in prepared['bindings']:
                raise ValueError('model class source unbound')
        def hook(name):
            def count(_module, _args): calls[name] += 1
            return count
        hooks = [model.predictor.register_forward_pre_hook(hook('predict')),
                 model.action_encoder.register_forward_pre_hook(hook('action_encoder'))]
        try:
            for arm_index, arm in enumerate(report['rows']):
                task = prepared['requests'][arm_index//2]['task']
                goal = torch.tensor(task['goal'], dtype=torch.float32, device=device)
                mean = torch.tensor(np.tile(task['action_mean'], 5), dtype=torch.float32, device=device)
                std = torch.tensor(np.tile(task['action_std'], 5), dtype=torch.float32, device=device)
                for c in arm['result']['candidates']:
                    z = torch.tensor(c['mid']['latent_history'], dtype=torch.float32, device=device)[None]
                    h = torch.tensor(c['mid']['action_history'], dtype=torch.float32, device=device)[None]
                    x = np.stack((np.asarray(c['suffix_solve']['params'], np.float64), np.zeros(10)))
                    evaluations = []
                    for iteration in range(UPDATES+1):
                        raw = np.tanh(x.astype(np.float32)).reshape(2, 5, 2)
                        action = torch.tensor(raw, dtype=torch.float32, device=device, requires_grad=iteration < UPDATES)
                        tokens = [(action[k].reshape(10)-mean)/std for k in range(2)]
                        ends = advance(z, h, tokens, lambda states, actions: model.predict(states, model.action_encoder(actions)))
                        if iteration == 0 and not np.array_equal(ends[0].detach().cpu().numpy().astype(np.float64), np.asarray(c['end']['latent_history'][-1], np.float64)):
                            raise ValueError('first packet saved replay mismatch')
                        loss = torch.mean((ends[-1]-goal)**2)
                        endpoint = ends[-1].detach().cpu().numpy().astype(np.float64)
                        q = float(np.mean((endpoint-np.asarray(task['goal'], np.float64))**2))
                        grad = None
                        if iteration < UPDATES:
                            grad = torch.autograd.grad(loss, action)[0].detach().cpu().numpy()
                            calls['reverse'] += 1
                        evaluations.append(dict(iteration=iteration, logits=x.tolist(), raw=raw.tolist(), endpoint=endpoint.tolist(), terminal_q=q,
                            regularized=q+saved.DAMPING*float(np.sum(x*x))/192, raw_gradient=None if grad is None else grad.tolist()))
                        if grad is not None:
                            x = np.stack([proposal(x[k], grad[k], raw[k]) for k in range(2)])
                    best = min(evaluations, key=lambda e: (e['regularized'], e['iteration']))
                    rows.append(dict(slot=arm['slot'], arm=arm['arm'], start=c['start'], iteration=c['iteration'], stage_uphill=c['stage_uphill'], best_iteration=best['iteration'], best_regularized=best['regularized'], evaluations=evaluations))
        finally:
            for handle in hooks: handle.remove()
        torch.cuda.synchronize(device)
        after = host.signal.model_hash(model)
        if before != after or calls != dict(predict=2880, action_encoder=2880, reverse=1280):
            raise ValueError('model mutation or call denominator mismatch')
    host.F.check_bindings(prepared['bindings'])
    summary = summarize(report, rows)
    result = dict(schema=NAME, source_report_sha256=source.REPORT_SHA, source_receipt_sha256=source.RECEIPT_SHA,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), device=device_info,
        model_before=before, model_after=after, calls=calls, rows=rows, summary=summary,
        scope='known-root post-hoc model exploration', training=False, physical_release=False, P1_admission=False)
    return dict(path=str(output), sha256=saved.write(output, result), summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    print('Exploring objective-consistent two-packet continuation on the complete fixed candidate panel; not physical evidence.', flush=True)
    print(json.dumps(run(parser.parse_args().output)))


if __name__ == '__main__':
    main()
