"""Replay all eight fixed40 prefixes from their original contexts on host CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p1_fixed40_observed_stage_20261029 as observed
from experiments.spectral_reachability import agent_p1_teacher_library40_history_localization_20261028_v1 as history

ROOT = history.ROOT
OBSERVED = ROOT/'results/p1_fixed40_observed_stage_20261029_MAIN01.json'
OBSERVED_SHA = '065177a8f64ebb28ccdc70251d65222c60af9c86de2c628e0baa39accdb57bc6'


def score(prediction, goal):
    p, g = [np.asarray(x, dtype=np.float64) for x in (prediction, goal)]
    if p.shape != (192,) or g.shape != (192,) or not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError('finite 192-dimensional vectors required')
    return float(np.mean((p-g)**2))


def run(out):
    if not out.is_absolute() or out.parent != ROOT/'results' or out.exists():
        raise ValueError('exclusive absolute results file required')
    data = OBSERVED.read_bytes()
    if hashlib.sha256(data).hexdigest() != OBSERVED_SHA:
        raise ValueError('observed stage binding mismatch')
    obs = json.loads(data)
    rebuilt = observed.build()
    if rebuilt != obs:
        raise ValueError('observed source reconstruction mismatch')
    prepared = history.rgb.read(history.TEACHER_PREP, history.TEACHER_PREP_SHA)
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    device_info = require_cuda_allocation('cuda:0')
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    rows = []
    with torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
        model = load_compatible_pretrained(prepared['policy'], device=device).eval().requires_grad_(False)
        before = history.teacher.enc.tensor_hash(model)
        if before != prepared['expected_model_hash']:
            raise ValueError('frozen checkpoint mismatch')
        mean = torch.tensor(np.tile(history.teacher.kernel.ACTION_MEAN, 5), dtype=torch.float32, device=device)
        std = torch.tensor(np.tile(history.teacher.kernel.ACTION_STD, 5), dtype=torch.float32, device=device)
        with torch.no_grad():
            for i, anchor in enumerate(prepared['anchors']):
                context = anchor['root_context']
                raw = np.asarray(anchor['prefix_commands'], dtype='<f4')
                if raw.shape != (5, 2) or not np.isfinite(raw).all() or anchor['anchor_id'] != obs['rows'][i]['anchor_id']:
                    raise ValueError('prefix or anchor mismatch')
                z = torch.tensor(context['latent_history'], dtype=torch.float32, device=device)[None]
                h = torch.tensor(context['action_history'], dtype=torch.float32, device=device)[None]
                token = (torch.tensor(raw.reshape(10), device=device)-mean)/std
                pred = model.predict(z, model.action_encoder(torch.cat((h, token[None, None]), 1)))[:, -1][0].cpu().numpy()
                goal = anchor['goal']
                rows.append(dict(anchor_index=i, anchor_id=anchor['anchor_id'], task_id=anchor['task_id'],
                                 prefix_sha256=hashlib.sha256(raw.tobytes()).hexdigest(), predicted_latent=pred.tolist(),
                                 model_stage_Q=score(pred, goal), observed_stage=obs['rows'][i]['observed_stage']))
        torch.cuda.synchronize(device)
        after = history.teacher.enc.tensor_hash(model)
        if before != after or len(rows) != 8:
            raise ValueError('model mutation or incomplete roster')
    history.rgb.verify_bindings(rebuilt['source_bindings'])
    result = dict(schema='p1_fixed40_prefix_host_20261029', observed_sha256=OBSERVED_SHA, rows=rows,
                  source_bindings=rebuilt['source_bindings'], model_hash_before=before, model_hash_after=after,
                  device=device_info, model_calls=8, known_roots=2, physical_calls=0, P1_admission=False,
                  scope='exploratory known roots, actual stage is observed latent, not physical task success')
    payload = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with out.open('xb') as stream:
        stream.write(payload)
    print(json.dumps(dict(path=str(out), sha256=hashlib.sha256(payload).hexdigest(), model_calls=8)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print('Replay all fixed prefixes to separate model and observed stage cost; no physical calls.', flush=True)
    run(args.output)


if __name__ == '__main__':
    main()
