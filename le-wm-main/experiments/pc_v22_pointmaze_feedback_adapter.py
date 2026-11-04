"""v22 and shared-bank CEM adapter for the existing PointMaze RGB harness.

The harness owns model encoding, actual prediction calls, observation
history, execution and persistence. No simulator state is accepted here.
"""
import numpy as np
import torch

from experiments.spectral_reachability import pc_global_reachability_v22 as compiler
from experiments.spectral_reachability import pc_response_temporal_search_v1 as temporal
from experiments.spectral_reachability import pc_timed_program_execution_v6 as timed

METHOD = "v22_shared_timed_feedback_development"
CAP = 1920


def decide(arm, root, step, target, predict, resource, carry=None):
    if arm not in ("response", "cem") or step not in (0, 1, 2) or root < 0:
        raise ValueError("invalid feedback decision")
    if resource.token_cap != CAP or resource.token_samples != 0:
        raise ValueError("expected a fresh matching resource meter")
    horizon = 15 - 5 * step
    if (carry is None) != (step == 0):
        raise ValueError("carry must match feedback step")
    nominal = tuple(tuple(tuple(float(value) for value in action) for action in item)
                    for item in temporal.NOMINALS)
    config = compiler.Config(action_dim=2, segment_length=5, segments=horizon // 5,
                             model_packet_length=5, nominals=nominal, token_cap=CAP,
                             seed=2030101000 + root * 100 + step * 10000)
    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else None
    output = timed.execute(predict, target, config, arm="v22" if arm == "response" else "multistart_cem",
                           carry=carry, baseline_seed=2030102000 + root * 100 + step,
                           synchronize=synchronize)
    if output["resource"]["prediction_tokens"] != resource.token_samples:
        raise RuntimeError("harness and shared planner prediction meters disagree")
    chain = np.asarray(output["result"]["selected"], np.float32)
    if chain.shape != (horizon, 2) or not np.isfinite(chain).all() or np.any(np.abs(chain) > 1):
        raise ValueError("invalid selected action chain")
    return {"selected": {"name": "v22_paired" if arm == "response" else "cem_paired",
                         "chain": chain.tolist(), "terminal_score": output["result"]["selected_score"]},
            "planner": output["result"], "calls": output["calls"], "resource": output["resource"]}
