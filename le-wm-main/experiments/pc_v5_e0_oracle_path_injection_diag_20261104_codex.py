"""Privileged post-hoc injection into the actual V5 full-chain tracker."""
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_v5_e0_prospective_mechanism_dev_20261104_codex as source
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_target_reach_canonical_v5 as method
from experiments.spectral_reachability import pc_target_reach_feedback_v4 as v4
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PROTOCOL = source.ROOT.parent / "docs/PC_V5_E0_ORACLE_PATH_INJECTION_DIAG_PROTOCOL_20261104_CODEX.md"
WITNESSES = source.PREFIX.with_name(source.PREFIX.name + "_privileged_feasibility_diag.json")
OUTPUT = source.PREFIX.with_name(source.PREFIX.name + "_oracle_path_injection_diag.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    method_lines, start = inspect.getsourcelines(method.decide)
    matches = [start + i for i, line in enumerate(method_lines) if line.strip() == 'if best["chain"] is None:']
    if len(matches) != 1:
        raise ValueError("injection anchor ambiguous")
    anchor = matches[0]
    deps = seal.capture([Path(__file__), PROTOCOL, WITNESSES, source.PLANS, source.AUDIT,
                         Path(method.__file__), Path(v4.__file__), Path(core.__file__),
                         Path(source.exact.__file__), Path(source.exact.wall.__file__),
                         Path(source.exact.wall.physics.__file__)])
    plans = json.loads(source.PLANS.read_text())
    if json.loads(source.AUDIT.read_text())["plans_sha256"] != source.sha(source.PLANS):
        raise ValueError("source audit mismatch")
    witnesses = json.loads(WITNESSES.read_text())["rows"]
    rows = []
    print("Diagnosing V5 selection and prefix retention with privileged stored witnesses.", flush=True)
    for old, oracle in zip(plans["rows"], witnesses, strict=True):
        if old["root"] != oracle["root"] or old["task"] != oracle["task"]:
            raise ValueError("task mismatch")
        root, task = old["root"], old["task"]
        goal = np.asarray(task["goal"])
        witness = np.asarray(oracle["witnesses"][0]["float32_actions"])
        if not oracle["witnesses"][0]["successful"]:
            raise ValueError("missing successful first witness")

        def run(inject):
            meter, calls = Meter(1920), []

            def predict(actions):
                raw = np.asarray(actions, dtype=np.float32)

                def evaluate(batch):
                    positions = source.exact.endpoints(task, batch)
                    return positions, np.linalg.norm(positions-goal, axis=1)

                return meter.predict(raw, evaluate)

            def tracer(frame, event, arg):
                if (event == "line" and frame.f_code is method.decide.__code__
                        and frame.f_lineno == anchor):
                    if calls:
                        raise RuntimeError("multiple injection calls")
                    before = meter.token_samples
                    _, scores = frame.f_locals["tracked"](witness[None])
                    calls.append({"before_tokens": before, "after_tokens": meter.token_samples,
                                  "score": float(scores[0])})
                return tracer

            if inject:
                if sys.gettrace() is not None:
                    raise RuntimeError("existing trace hook")
                sys.settrace(tracer)
            try:
                result = method.decide("response", "response", root, 0, goal, predict, meter)
            finally:
                if inject:
                    sys.settrace(None)
            if inject and (len(calls) != 1 or calls[0]["after_tokens"] - calls[0]["before_tokens"] != 3):
                raise ValueError("injection cost mismatch")
            return result, meter.as_dict(), calls

        control, counts, _ = run(False)
        original = old["result"]
        for key in ("selected", "first_bank", "prefix_scores", "retained", "refinement_start",
                    "precomparison", "comparison_chain", "comparison_score", "abstained"):
            if control[key] != original[key]:
                raise ValueError(f"control mismatch root={root} key={key}")
        deterministic_counts = ("prediction_calls", "trajectory_samples", "prediction_token_samples",
                                "token_cap", "backward_calls")
        if any(counts[key] != old["counts"][key] for key in deterministic_counts):
            raise ValueError("control meter mismatch")
        injected, injected_counts, calls = run(True)
        dispatched = np.asarray(injected["selected"]["chain"], dtype=np.float32)
        endpoint = source.exact.endpoints(task, dispatched[None])[0]
        replay_score = float(np.linalg.norm(endpoint-goal))
        np.testing.assert_allclose(replay_score, injected["selected"]["terminal_score"], atol=1e-12, rtol=0)
        if injected_counts["prediction_token_samples"] > 1920:
            raise ValueError("budget overflow")

        def extra_predict(actions):
            raw = np.asarray(actions, dtype=np.float32)
            pos = source.exact.endpoints(task, raw)
            return pos, np.linalg.norm(pos-goal, axis=1)

        prefix = witness[:5]
        _, prefix_scores = extra_predict(prefix[None])
        structure = {"svd": 0, "fd_action_samples": 0, "svd_seconds": 0.0}
        _, response = v4.response(core.NOMINALS[0], prefix, goal, extra_predict, structure)
        augmented = original["successor_responses"] + [response]
        scores = original["prefix_scores"] + [float(prefix_scores[0])]
        retained = v4.retain(scores, augmented)
        idx = len(scores)-1
        rows.append({"root": root, "hook": calls[0], "control_score": original["selected"]["terminal_score"],
                     "refinement_start": injected["refinement_start"], "precomparison": injected["precomparison"],
                     "selected": injected["selected"], "abstained": injected["abstained"],
                     "dispatch_endpoint": endpoint.tolist(), "dispatch_score": replay_score,
                     "counts": injected_counts, "witness_prefix_index": idx, "retained_augmented": retained,
                     "prefix_score": float(prefix_scores[0]), "witness_response": response,
                     "prefix_retained": idx in retained})
        print(f"root={root} completed", flush=True)
    seal.verify(deps)
    source.write(OUTPUT, {"dependencies": deps, "anchor_line": anchor,
                          "scope": "privileged, post-hoc instrumentation; not planner performance", "rows": rows})


if __name__ == "__main__":
    main()
