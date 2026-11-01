"""Compare conditional local reach at predicted and observed fixed40 successors."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_fixed40_prefix_consume_20261029 as prefix_consumer
from experiments.spectral_reachability import p1_fixed40_prefix_host_20261029 as prefix
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as compiler

ROOT = prefix.ROOT
PREFIX = ROOT/'results/p1_fixed40_prefix_host_20261029_MAIN01.json'
PREFIX_SHA = 'f41e10d13d1f7552193e88e1fcce0bbd426fba526fc1a99d36750624f0c589ad'


def metrics(jacobian, nominal, goal):
    j = np.asarray(jacobian, dtype=np.float64)
    nominal, goal = [np.asarray(x, dtype=np.float64) for x in (nominal, goal)]
    if j.shape != (192, 10) or nominal.shape != (192,) or goal.shape != (192,) or not all(np.isfinite(x).all() for x in (j, nominal, goal)):
        raise ValueError('bad conditional reach inputs')
    return compiler.bounded_ridge(j, goal-nominal)


def run(output):
    if not output.is_absolute() or output.parent != ROOT/'results' or output.exists():
        raise ValueError('exclusive absolute results path required')
    prefix_consumer.verify(PREFIX, PREFIX_SHA)
    pred = json.loads(PREFIX.read_bytes())
    history = prefix.history
    prepared = history.rgb.read(history.TEACHER_PREP, history.TEACHER_PREP_SHA)
    name, sha = history.rgb.INPUTS['compiler']
    compiled = history.rgb.read(ROOT/name, sha)
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    info = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    rows = []
    with torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
        model = load_compatible_pretrained(prepared['policy'], device=device).eval().requires_grad_(False)
        before = history.teacher.enc.tensor_hash(model)
        if before != prepared['expected_model_hash']:
            raise ValueError('checkpoint mismatch')
        transition = compiler.make_transition(model, device)
        for i, (anchor, source, saved) in enumerate(zip(prepared['anchors'], compiled['rows'], pred['rows'], strict=True)):
            observed = source['blind_input']['context']
            if anchor['anchor_id'] != saved['anchor_id']:
                raise ValueError('anchor mismatch')
            history_tensor = torch.tensor(observed['action_history'], dtype=torch.float32, device=device)[None]
            row = dict(anchor_index=i, anchor_id=anchor['anchor_id'], task_id=anchor['task_id'], cases={})
            for case in ('predicted', 'observed'):
                latent = np.asarray(observed['latent_history'], dtype=np.float32).copy()
                if case == 'predicted':
                    latent[-1] = np.asarray(saved['predicted_latent'], dtype=np.float32)
                emb = torch.tensor(latent, dtype=torch.float32, device=device)[None]
                zero = torch.zeros((5,2), dtype=torch.float32, device=device)
                def fn(raw):
                    return transition(emb, history_tensor, raw)[0][0,-1]
                nominal = fn(zero).detach().cpu().numpy()
                columns = []
                for k in range(10):
                    tangent = torch.zeros_like(zero); tangent.reshape(-1)[k] = 1
                    _, tangent_out = torch.autograd.functional.jvp(fn, zero, tangent, create_graph=False)
                    columns.append(tangent_out.detach().cpu().numpy())
                jacobian = np.stack(columns, axis=1)
                direction = torch.ones_like(zero)/np.sqrt(10.)
                eps = 0.001
                with torch.no_grad():
                    fd = ((fn(eps*direction)-fn(-eps*direction))/(2*eps)).cpu().numpy()
                analytic = jacobian @ direction.cpu().numpy().reshape(-1)
                err = float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(analytic),1e-12))
                row['cases'][case] = dict(nominal=nominal.tolist(), jacobian=jacobian.tolist(),
                    solve=metrics(jacobian, nominal, anchor['goal']), fd_directional_relative_error=err)
            rows.append(row)
        torch.cuda.synchronize(device)
        after = history.teacher.enc.tensor_hash(model)
        if before != after or len(rows) != 8:
            raise ValueError('mutated model or incomplete roster')
    history.rgb.verify_bindings(pred['source_bindings'])
    result = dict(schema='p1_fixed40_conditional_reach_host_20261029', prefix_sha256=PREFIX_SHA,
                  source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=rows, device=info, model_hash_before=before, model_hash_after=after,
                  fixed_contexts=16, jvp_calls=160, fd_pairs=16, physical_calls=0, P1_admission=False,
                  scope='known roots exploratory; model Jacobian at two successor contexts, not physical controllability')
    payload = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with output.open('xb') as stream:
        stream.write(payload)
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(payload).hexdigest(), contexts=16)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print('Compare same-prefix local model reach at predicted and actual RGB successor; no physical calls.', flush=True)
    run(args.output)


if __name__ == '__main__':
    main()
