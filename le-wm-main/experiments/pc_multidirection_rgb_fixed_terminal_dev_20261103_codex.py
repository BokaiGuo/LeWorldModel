"""Trace the frozen multidirection compiler on the existing RGB development bridge."""
import hashlib
import inspect
from pathlib import Path

import numpy as np

from experiments import pc_fixed_terminal_fair_feedback_dev_20261103_codex as runner
from experiments.spectral_reachability import pc_multidirection_feedback_v1 as feedback
from experiments.spectral_reachability import pc_multidirection_successor_compiler_v1 as core
from experiments.spectral_reachability import pc_response_diverse_compiler_v1 as response
from experiments.spectral_reachability import pc_cem_scale_floor_v1 as suffix

PROTOCOL = runner.ROOT.parent / "docs/PC_MULTIDIRECTION_RGB_FIXED_TERMINAL_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = runner.ROOT / "results/pc_multidirection_rgb_fixed_terminal_dev_20261103_codex"
ARMS = ("multidirection", "cem")
baseline_decide = runner.decide


def source_hashes():
    modules = (runner, core, feedback, response, suffix, core.prior, core.prior.policy)
    files = {str(Path(module.__file__).resolve()) for module in modules}
    bridge = Path(inspect.getfile(runner.ShadowRGBBridge)).resolve()
    files.update((str(Path(__file__).resolve()), str(bridge),
                  str(bridge.with_name("pc_pointmaze_shadow_rgb_bridge_worker_20261103_codex.py")),
                  str(Path(runner.optimize.__code__.co_filename).resolve())))
    return {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sorted(files)}


def decide(arm, root, step, target, predict, token_meter, carry=None):
    if arm not in ARMS or step not in (0, 1, 2):
        raise ValueError("invalid arm or decision")
    trace = []

    def traced(actions):
        requested = np.asarray(actions, dtype=float)
        if (requested.ndim != 3 or requested.shape[2] != 2 or requested.shape[1] not in (5, 10, 15)
                or not np.isfinite(requested).all() or np.any(np.abs(requested) > 1)):
            raise ValueError("invalid raw action proposal")
        before = token_meter.token_samples
        positions, scores = predict(requested)
        tokens = len(requested) * requested.shape[1] // 5
        if token_meter.token_samples - before != tokens:
            raise ValueError("prediction accounting mismatch")
        if not np.isfinite(positions).all() or not np.isfinite(scores).all():
            raise ValueError("nonfinite prediction")
        trace.append({"requested_actions": requested.tolist(),
                      "model_raw_float32_actions": requested.astype(np.float32).tolist(),
                      "positions": np.asarray(positions).tolist(), "scores": np.asarray(scores).tolist(),
                      "tokens": tokens})
        return positions, scores

    if arm == "cem":
        result = baseline_decide("cem", root, step, target, traced, token_meter, carry)
    elif step == 0:
        if carry is not None:
            raise ValueError("unexpected initial carry")
        result = core.decide(root, target, traced, token_meter)
    else:
        result = feedback.decide(root, step, target, traced, token_meter, carry)
    result["prediction_trace"] = trace
    result["actual_source_sha256"] = source_hashes()
    return result


def main():
    runner.PROTOCOL, runner.PREFIX, runner.ARMS, runner.decide = PROTOCOL, PREFIX, ARMS, decide
    runner.main()


if __name__ == "__main__":
    main()
