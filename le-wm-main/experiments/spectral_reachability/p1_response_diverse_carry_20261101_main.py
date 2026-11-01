"""Fixed multidirectional Jacobian initialization with first-feedback carry."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_first_feedback_carry_20261101_main as carry
from . import e3_cp10_goal_dls_20261031 as dls
from . import e3_cp10_dls_sourcefix_20261031 as sourcefix

NAME = "p1_response_diverse_carry_20261101_main"
DOC = Path(__file__).resolve().parents[3] / "docs/D2P_P1_RESPONSE_DIVERSE_PROTOCOL_20261101_MAIN.md"


def response_bank(jac, projection, goal, nominal):
    jac, projection = np.asarray(jac, np.float64), np.asarray(projection, np.float64)
    goal, nominal = np.asarray(goal, np.float64), np.asarray(nominal, np.float64)
    if (jac.shape != (192, 30) or projection.shape != (192, 192)
            or goal.shape != (192,) or nominal.shape != (192,)
            or not all(np.isfinite(x).all() for x in (jac, projection, goal, nominal))):
        raise ValueError("finite goal/Jacobian dimensions")
    matrix, residual = projection @ jac, projection @ (goal - nominal)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    if singular[0] == 0:
        return np.zeros((8, 3, 5, 2), np.float32), singular
    damping = .01 * singular[0] ** 2
    direction = right.T @ ((singular / (singular ** 2 + damping)) * (left.T @ residual))
    direction /= max(1., np.max(np.abs(direction)))
    plans = [np.zeros(30), direction]
    for index in range(3):
        vector = right[index].copy()
        # Canonical orientation; both signs are included regardless of task labels.
        if vector[np.argmax(abs(vector))] < 0:
            vector *= -1
        vector *= .5 / max(np.max(abs(vector)), 1e-30)
        plans.extend((vector, -vector))
    return np.asarray(plans, np.float32).reshape(8, 3, 5, 2), singular


def setup(root):
    experiment = carry.setup(root)
    runner, control = experiment.runner, carry.control
    dls.goal_bank = response_bank
    old_build, old_protocol, old_contract = runner.build, runner.protocol, control.contract_for
    source_sha = runner.sha(Path(__file__).resolve())

    def contract_for(task, model_sha, hardware_sha):
        policy, contract, config = old_contract(task, model_sha, hardware_sha)
        return policy, sourcefix.bind_initializer(contract, source_sha), config

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), DOC):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = old_protocol()
        value.update(structured_initializer_sha256=source_sha,
                     structured_initialization="zero_DLS_plus_minus_top3_response_directions",
                     spectral_peak_action=.5, selection_uses_physical_labels=False)
        return value

    control.contract_for = contract_for
    runner.build, runner.protocol = build, protocol
    runner.NAME = f"{NAME}_root{root}"
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple tag")
    exp = setup(args.root)
    prefix = exp.runner.ROOT / "results" / f"{NAME}_root{args.root}"
    prepared, output = Path(str(prefix) + f"_PREP_{args.tag}.json"), Path(str(prefix) + f"_RUN_{args.tag}")
    if prepared.exists() or output.exists():
        raise ValueError("exclusive output")
    receipt = exp.runner.prepare(str(prepared))
    print(json.dumps(receipt), flush=True)
    with exp.extra.gpu.locking.gpu_lock(exp.extra.gpu.LOCK):
        result = exp.runner.run(str(prepared), receipt["sha256"], str(output),
                               "MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "recorded_known_root_not_admitted" else 1)
