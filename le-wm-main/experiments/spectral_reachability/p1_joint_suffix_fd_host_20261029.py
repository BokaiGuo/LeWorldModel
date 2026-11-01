"""Host-CUDA directional finite differences at saved conditional H1 contexts.

Four fixed slots and four fixed winner-start iterations; no selection by outcomes.
Uses saved per-candidate suffix direction; no physics, training or P1 admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p1_joint_vreach_saved_diff_20261029_v1 as saved
from experiments.spectral_reachability import agent_p1_joint_vreach_host_cuda_20261029_v1 as host

NAME = 'p1_joint_suffix_fd_host_20261029'
ROOT = Path(__file__).resolve().parents[2]
REPORT_SHA = '25eb16c1e2515f98d4e8ace4fafbf1a8d5cbc74d79762421ba248387d1f272c5'
RECEIPT_SHA = '7b3e3f9f7940a38e8ee19bbc36a337a0317960f05aac5b5fe8de27f734805179'
REPORT = ROOT/'results/agent_p1_joint_vreach_host_cuda_20261029_v1_cuda_MAIN01/report.json'
RECEIPT = ROOT/'results/agent_p1_joint_vreach_host_cuda_20261029_v1_consume_MAIN01.json'
SCALES = (.001, .01, .1, 1.0)


def candidates(report, prepared):
    rows = []
    for slot in range(4):
        arm = report['rows'][2*slot]
        if arm['slot'] != slot or arm['arm'] != 'projected_ascent':
            raise ValueError('fixed slot and arm mismatch')
        for iteration in range(4):
            c = arm['result']['candidates'][iteration]
            if (c['start'], c['iteration']) != (0, iteration):
                raise ValueError('fixed winner start sequence')
            rows.append((slot, iteration, prepared['requests'][slot]['task'], c))
    return rows


def estimate(jacobian, direction, eps, plus, minus):
    j = np.asarray(jacobian, np.float64)
    d = np.asarray(direction, np.float32).astype(np.float64).reshape(10)
    p, m = np.asarray(plus, np.float64), np.asarray(minus, np.float64)
    if j.shape != (192, 10) or p.shape != (192,) or m.shape != (192,) or not all(np.isfinite(x).all() for x in (j,d,p,m)) or eps <= 0:
        raise ValueError('invalid directional comparison')
    reference = j@d
    fd = (p-m)/(2*eps)
    denom = float(np.linalg.norm(reference))
    error = float(np.linalg.norm(fd-reference))
    return dict(fd_l2=float(np.linalg.norm(fd)), reference_l2=denom, error_l2=error,
                relative_error=None if denom == 0 else error/denom)


def run(output):
    report = saved.read_verified(REPORT, REPORT_SHA)
    receipt = saved.read_verified(RECEIPT, RECEIPT_SHA)
    if receipt.get('report_sha256') != REPORT_SHA:
        raise ValueError('receipt binding')
    prep = saved.read_verified(report['prepared'], report['prepare_sha256'])
    saved.analyze(report, receipt, prep)
    host.verify(report['prepared'], report['prepare_sha256'])
    output = Path(output)
    if not output.is_absolute() or output.parent != ROOT/'results' or output.resolve() != output or not output.name.startswith(NAME+'_'):
        raise ValueError('exclusive results file required')
    if output.exists():
        raise ValueError('output exists')
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    device_info = require_cuda_allocation('cuda:0')
    device = torch.device('cuda:0')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    rows = []
    calls = 0
    with torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
        model = load_compatible_pretrained(prep['policy'], device=device).eval().requires_grad_(False)
        before = host.signal.model_hash(model)
        if before != report['model_before'] or before['original_state_dict'] != prep['expected_model_hash']:
            raise ValueError('model hash mismatch')
        def step(task, context, raw):
            nonlocal calls
            a = np.asarray(raw, np.float32).reshape(5, 2)
            z = torch.tensor(context['latent_history'], dtype=torch.float32, device=device)[None]
            h = torch.tensor(context['action_history'], dtype=torch.float32, device=device)[None]
            token = (torch.tensor(a, device=device).reshape(10) - torch.tensor(np.tile(task['action_mean'], 5), dtype=torch.float32, device=device)) / torch.tensor(np.tile(task['action_std'], 5), dtype=torch.float32, device=device)
            with torch.no_grad():
                calls += 1
                value = model.predict(z, model.action_encoder(torch.cat((h, token[None,None]), 1)))[:, -1]
            return value[0].detach().cpu().numpy().astype(np.float64)
        for slot, iteration, task, c in candidates(report, prep):
            direction = np.asarray(c['suffix'], np.float32).reshape(10)
            zero = step(task, c['mid'], np.zeros(10, np.float32))
            full = step(task, c['mid'], direction)
            nominal = np.asarray(c['signal']['nominal_context']['latent_history'][-1], np.float64)
            endpoint = np.asarray(c['end']['latent_history'][-1], np.float64)
            if not np.allclose(zero, nominal, rtol=1e-5, atol=1e-6) or not np.allclose(full, endpoint, rtol=1e-5, atol=1e-6):
                raise ValueError('saved same-action replay mismatch')
            scales = []
            for scale in SCALES:
                eps = float(np.float32(scale))
                raw = direction*np.float32(scale)
                plus, minus = step(task, c['mid'], raw), step(task, c['mid'], -raw)
                scales.append(dict(scale=scale, epsilon_float32=eps, plus=plus.tolist(), minus=minus.tolist(),
                                   **estimate(c['signal']['jacobian'], direction, eps, plus, minus)))
            rows.append(dict(slot=slot, iteration=iteration, nominal_max_abs=float(np.max(np.abs(zero-nominal))),
                             full_max_abs=float(np.max(np.abs(full-endpoint))), direction=direction.tolist(),
                             zero=zero.tolist(), full=full.tolist(), scales=scales))
        torch.cuda.synchronize(device)
        after = host.signal.model_hash(model)
        if before != after or calls != 160:
            raise ValueError('model mutation or call denominator mismatch')
    host.F.check_bindings(prep['bindings'])
    value = dict(schema=NAME, source_report_sha256=REPORT_SHA, source_receipt_sha256=RECEIPT_SHA,
                 code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), device=device_info,
                 model_before=before, model_after=after, model_calls=calls, rows=rows,
                 scope='known roots fixed panel, exploratory; no independent roots or physical evidence',
                 training=False, physical_release=False, P1_admission=False)
    return dict(path=str(output), sha256=saved.write(output, value), model_calls=calls, rows=len(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    print(json.dumps(run(p.parse_args().output)))


if __name__ == '__main__':
    main()
