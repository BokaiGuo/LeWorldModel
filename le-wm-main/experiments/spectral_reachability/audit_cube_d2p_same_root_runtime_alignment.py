"""Host-CUDA equivalence audit of same-root trainer and deployed P1 tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_cube_action_adapter_planning import _image_transform
from experiments.spectral_reachability import cube_d2p_training_source as source
from experiments.spectral_reachability import preflight_tworoom_d2p_p0 as runtime
from experiments.spectral_reachability.train_cube_d2p_same_root_ranking_repair import _Groups, _normalise, _rgb, _terminal, _sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--device', choices=('cuda',), default='cuda')
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1:
        raise ValueError('new alignment output and positive sample count required')
    manifest = json.loads((args.checkpoint / 'repair_manifest.json').read_text())
    if manifest.get('status') != 'trained_same_root_ranking_repair':
        raise ValueError('alignment audit requires a complete same-root repair')
    host = require_cuda_allocation(args.device)
    device = torch.device('cuda')
    receipt = source.validate_receipt(Path(manifest['receipt']))
    mean_np = np.asarray(receipt['calibration_action_mean'], np.float32)
    std_np = np.asarray(receipt['calibration_action_std'], np.float32)
    mean = torch.as_tensor(mean_np, device=device).view(1, 1, -1)
    std = torch.as_tensor(std_np, device=device).view(1, 1, -1)
    dataset = _Groups(Path(manifest['pool']['path']), split='validation')
    model = load_compatible_pretrained(args.checkpoint, device=device).eval()
    base = load_compatible_pretrained(Path(manifest['base_checkpoint']), device=device).eval()
    visual_same = all(torch.equal(value, base.state_dict()[name]) for name, value in model.state_dict().items() if name.startswith(('encoder.', 'projector.')))
    del base
    before = runtime._parameter_hash(model)
    transform = _image_transform()
    low = torch.full((source.RAW_ACTION_DIM,), -1.0, device=device)
    rows = []
    with torch.no_grad():
        for index in range(min(args.samples, len(dataset))):
            item = dataset[index]
            candidates = item['candidates'][:3]
            c = len(candidates)
            context = _rgb(item['context'][None], transform=transform, device=device)
            prefix = _normalise(item['prefix'][None], mean=mean, std=std, device=device)
            future = _normalise(candidates[None], mean=mean, std=std, device=device)
            actions = torch.cat((prefix[:, 1:].expand(c, -1, -1), future[0, :, None]), dim=1)
            predicted = _terminal(model, context.expand(c, -1, -1, -1, -1).contiguous(), actions)
            errors = []
            single_errors = []
            pixel_errors = []
            action_errors = []
            for j in range(c):
                frames = item['context'].permute(0, 2, 3, 1).numpy()
                runtime_pixels = runtime._frames_tensor(list(frames[1:]), transform=transform, device=device)
                runtime_actions = runtime._root_anchored_action_sequence(
                    runtime._normalize_raw_packets(item['prefix'].to(device), mean_np, std_np, action_block=5, action_dim=5),
                    runtime._normalize_raw_packets(candidates[j:j+1].to(device), mean_np, std_np, action_block=5, action_dim=5),
                )
                pixel_errors.append(float((context[:, 1:4] - runtime_pixels).abs().max()))
                action_errors.append(float((actions[j] - runtime_actions).abs().max()))
                single_prediction = _terminal(model, context.contiguous(), actions[j:j+1].contiguous())[0]
                deployed = runtime._terminal_prediction(model, list(frames[:3]), frames[3], item['prefix'].numpy(), candidates[j:j+1].to(device), mean=mean_np, std=std_np, action_block=5, action_dim=5, transform=transform, device=device, low=low, high=-low, future_packets_are_raw=True)
                errors.append(float((predicted[j] - deployed).abs().max()))
                single_errors.append(float((single_prediction - deployed).abs().max()))
            rows.append({'validation_root_seed': int(dataset.root_seeds[dataset.indices[index]]), 'candidate_terminal_max_abs_error': errors, 'singleton_terminal_max_abs_error': single_errors, 'pixel_max_abs_error': pixel_errors, 'action_max_abs_error': action_errors})
    maximum = max(v for row in rows for v in row['candidate_terminal_max_abs_error'])
    gates = {'cuda_only': True, 'trainer_vs_p1_max_abs_error_le_2e_5': maximum <= 2e-5, 'visual_encoder_and_projector_equal_base': visual_same, 'parameter_hash_unchanged': before == runtime._parameter_hash(model)}
    result = {'schema': 'cube_d2p_same_root_runtime_alignment.v1', 'status': 'pass' if all(gates.values()) else 'fail', 'cuda_host': host, 'checkpoint_weights_sha256': _sha256(args.checkpoint / 'weights.pt'), 'gates': gates, 'max_abs_error': maximum, 'rows': rows, 'claim_boundary': 'Training/runtime tensor equivalence only; no efficacy or P1 mechanism claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))
    if result['status'] != 'pass':
        raise SystemExit('same-root trainer/runtime alignment failed')


if __name__ == '__main__':
    main()
