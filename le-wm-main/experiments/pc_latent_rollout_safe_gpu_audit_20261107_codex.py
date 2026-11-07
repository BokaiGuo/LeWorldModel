"""GPU parity check on two exposed TwoRoom RGB inputs, not a control benchmark."""
import json
import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_latent_rollout_safe_20261107_codex import rollout_single_latent
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_tworoom_real_vs_imagined_feedback_20261107_codex import latent_rollout, prior

OUTPUT = Path("results/pc_latent_rollout_safe_gpu_audit_v4_20261107_codex.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = prior.source
    source.verify()
    _, model, _, transform = source.setup()
    inputs = source.b.q.old.base.source.block.prior.inputs.INPUT
    fingerprint_before = model_fingerprint(model)
    rng = np.random.default_rng(2718)
    rows, parallel_inputs = [], []
    with np.load(inputs, allow_pickle=False) as data:
        for root in (38, 39):
            with torch.no_grad():
                image = transform(data["current"][root])[None, None].cuda()
                initial = model.encode({"pixels": image})["emb"][0, 0].detach()
                target = transform(data["goal"][root])[None, None].cuda()
                goal = model.encode({"pixels": target})["emb"][0, 0].detach()
            for horizon in (1, 3, 5):
                actions = torch.tensor(rng.normal(0, 0.5, (8, horizon, 10)),
                                       device="cuda", dtype=torch.float32, requires_grad=True)
                reference = latent_rollout(model, initial, actions)
                gref = torch.autograd.grad(reference.square().sum(), actions)[0]
                repeated = latent_rollout(model, initial, actions)
                grepeated = torch.autograd.grad(repeated.square().sum(), actions)[0]
                candidate = rollout_single_latent(model, initial, actions)
                gcandidate = torch.autograd.grad(candidate.square().sum(), actions)[0]
                reference_goal = latent_rollout(model, initial, actions)
                grgoal = torch.autograd.grad((reference_goal[:, -1] - goal).square().sum(), actions)[0]
                candidate_goal = rollout_single_latent(model, initial, actions)
                gcgoal = torch.autograd.grad((candidate_goal[:, -1] - goal).square().sum(), actions)[0]
                torch.cuda.synchronize()
                output_error = float((candidate - reference).detach().abs().max())
                gradient_error = float((gcandidate - gref).detach().abs().max())
                goal_gradient_error = float((gcgoal - grgoal).abs().max())
                rows.append({"root": root, "horizon": horizon, "samples": 8,
                             "max_output_error": output_error, "max_action_gradient_error": gradient_error,
                             "max_goal_cost_gradient_error": goal_gradient_error,
                             "max_reference_gradient": float(gref.abs().max()),
                             "native_repeat_output_error": float((repeated - reference).detach().abs().max()),
                             "native_repeat_gradient_error": float((grepeated - gref).abs().max()),
                             "strict_parity_pass": max(output_error, gradient_error, goal_gradient_error) <= 1e-5})
                if horizon == 5:
                    parallel_inputs.append((initial, actions.detach()))
    def evaluate(item):
        with torch.inference_mode():
            return rollout_single_latent(model, *item)
    sequential = [evaluate(item) for item in parallel_inputs]
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(evaluate, parallel_inputs))
    torch.cuda.synchronize()
    parallel_error = max(float((x - y).abs().max()) for x, y in zip(sequential, parallel))
    fingerprint_after = model_fingerprint(model)
    if fingerprint_before != fingerprint_after:
        raise ValueError("model parameters changed")
    model_source = Path(inspect.getsourcefile(type(model)))
    OUTPUT.write_text(json.dumps({"track": "rgb_only_dev", "input_sha256": file_hash(inputs),
                                  "model_fingerprint": fingerprint_before, "rows": rows,
                                  "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
                                  "model_source": str(model_source), "model_source_sha256": file_hash(model_source),
                                  "parallel_inference_max_error": parallel_error,
                                  "strict_parity_pass": all(row["strict_parity_pass"] for row in rows)
                                                        and parallel_error <= 1e-5,
                                  "scope": "exposed TwoRoom images, synthetic normalized action packets; "
                                           "not control or independent evidence"}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
