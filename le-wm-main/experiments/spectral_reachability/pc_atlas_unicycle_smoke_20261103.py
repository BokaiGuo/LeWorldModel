"""Previously exposed analytical unicycle development smoke; no neural inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .d2p_atlas_search_v2 import AtlasConfig, AtlasSearch
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_exact_unicycle_development import TURN, evaluate, step
from .pc_exact_unicycle_strong_baseline import multistart_gradient


class UnicycleBackend:
    def rollout(self, context, actions):
        states = list(context["latent_history"])
        history = list(context["action_history"])
        state = np.asarray(states[-1], dtype=float)
        for action in actions:
            state = step(state, action)
            states = (states + [state.tolist()])[-2:]
            history = (history + [action.tolist()])[-1:]
        return {**context, "latent_history": states, "action_history": history}

    def jacobian(self, context, actions):
        result = np.zeros((3, actions.size))
        angle = context["latent_history"][-1][2]
        for i, (velocity, turn) in enumerate(actions):
            result = np.array([[1., 0., -velocity * np.sin(angle)],
                               [0., 1., velocity * np.cos(angle)], [0., 0., 1.]]) @ result
            result[:, 2*i:2*i+2] += [[np.cos(angle), 0.], [np.sin(angle), 0.], [0., 1.]]
            angle += turn
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = {"layout": "states_with_between_state_actions_v1", "context_length": 2,
            "latent_dim": 3, "action_dim": 2, "latent_history": [[0., 0., 0.]] * 2,
            "action_history": [[0., 0.]]}
    rows = []
    for i in range(8):
        angle = 2 * np.pi * i / 64 + .031
        goal = np.array([np.cos(angle), np.sin(angle)])
        compiler = LocalCompiler(UnicycleBackend(), goal=[*goal, 0.], sqrt_q=np.diag([1., 1., 0.]),
                                 action_scale=[1., 1.], low=[-1., -TURN], high=[1., TURN],
                                 config=CompilerConfig(packet_steps=1, spectral_modes=2,
                                                       max_rollout_operations=1000,
                                                       max_jacobian_operations=100))
        result = AtlasSearch(compiler, AtlasConfig(seed=42+i)).search(root)
        chosen = result["programs"][result["selected_index"]]
        atlas_error, _ = evaluate(np.asarray(chosen["actions"]), goal)
        gradient, ledger = multistart_gradient(goal)
        gradient_error, _ = evaluate(gradient, goal)
        rows.append({"index": i, "goal": goal.tolist(), "atlas_error": atlas_error,
                     "gradient_error": gradient_error, "gradient_ledger": ledger, "atlas": result})
    report = {"scope": "exposed development analytical smoke; not equal-budget or physical evidence",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "compiler_sha256": hashlib.sha256(Path(__file__).with_name("d2p_atlas_search_v2.py").read_bytes()).hexdigest(),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps({"roots": len(rows),
                      "atlas_mean_error": float(np.mean([r["atlas_error"] for r in rows])),
                      "gradient_mean_error": float(np.mean([r["gradient_error"] for r in rows])),
                      "qualified": sum(r["atlas"]["qualified_reconfiguration_index"] is not None for r in rows)}))


if __name__ == "__main__":
    main()
