"""Carry-conditioned response nominals on a fixed E0 development panel."""
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from experiments import pc_e0_response_waypoint_direction_dev_20261104_codex as prior
from experiments import pc_e0_v11_small_amplitude_dev_20261104_codex as small
from experiments import pc_e0_v12_remainder_dev_20261104_codex as remainder
from experiments import pc_e0_v13_global_proposals_dev_20261104_codex as previous
from experiments.pc_e0_wall_independent_branch_replay_dev_20261104_codex import independent
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_recursive_response_v12 as v12
from experiments.spectral_reachability import pc_recursive_response_v13 as v13
from experiments.spectral_reachability import pc_recursive_response_v14 as v14
from experiments.spectral_reachability import pc_recursive_response_v14_child as v14_child
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
OUT = ROOT / "results/pc_e0_v14_carry_nominal_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_E0_V14_CARRY_NOMINAL_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("v14_carry_nominal", "v13_global_response", "v12_response", "global_cem", "direct_cem")
CAP = 1920


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.31, .33), (-1., 1.35), (-.28, .24))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    old = (prior.tasks(), prior.previous_waypoint.tasks(), prior.prior.tasks(),
           prior.prior.previous.tasks(), small.tasks(), remainder.tasks(), previous.tasks())
    if any(task in panel for task in tasks() for panel in old):
        raise ValueError("task overlaps an explicitly checked panel")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(prior.__file__),
        Path(prior.v11.__file__), Path(v12.__file__), Path(v13.__file__), Path(remainder.__file__),
        Path(previous.__file__), Path(v14.__file__), Path(v14_child.__file__),
        Path(prior.core.__file__), Path(small.__file__),
        Path(prior.prior.previous.exact.__file__), Path(independent.__code__.co_filename)])
    OUT.mkdir()
    writer = prior.prior.previous.write
    writer(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "cap": CAP,
        "compiler_seed": "2029111000 + root", "cem_init_seed": "2029112000 + root*100 + restart",
        "cem_seed": "2029113000 + root*100 + restart",
        "global_cem_seed": "2029124000 + root", "dependencies": deps,
        "scope": "development only; broader historical independence unknown"})
    rows = []
    for root, task in enumerate(tasks()):
        goal = np.asarray(task["goal"])
        for arm in ARMS:
            meter, trace, error, result = Meter(CAP), [], None, None
            started = time.monotonic()

            def predict(actions):
                raw = np.asarray(actions, np.float32)
                def evaluate(batch):
                    position = prior.prior.previous.exact.endpoints(task, batch)
                    return position, np.linalg.norm(position - goal, axis=1)
                positions, scores = meter.predict(raw, evaluate)
                trace.append((raw.copy(), positions.copy(), scores.copy()))
                return positions, scores

            try:
                if arm in ("v14_carry_nominal", "v13_global_response", "v12_response", "global_cem"):
                    nominal = tuple(tuple(tuple(float(x) for x in point) for point in item)
                                    for item in prior.core.NOMINALS)
                    compiler = v12 if arm == "v12_response" else v14 if arm == "v14_carry_nominal" else v13
                    config = compiler.Config(action_dim=2, segment_length=5, segments=3,
                        model_packet_length=5, nominals=nominal, max_modes=1, token_cap=CAP,
                        seed=2029111000 + root)
                    if arm != "global_cem":
                        result = compiler.compile_program(predict, goal, config)
                        selected, score = result["selected"], result["selected_score"]
                        if result["resource"]["prediction_tokens"] != meter.token_samples:
                            raise ValueError("compiler and meter differ")
                    else:
                        bank = v13.proposal_bank(config)
                        _, scores = predict(bank)
                        choice = int(np.argmin(scores))
                        count = (CAP - meter.token_samples - 6) // 3
                        chain, value, generations = prior.core.cem(
                            bank[choice], [], count,
                            np.random.default_rng(2029124000 + root),
                            predict, float(scores[choice]), 15)
                        comparison = np.asarray(chain).copy()
                        comparison[:5] = 0
                        pair = np.stack((chain, comparison))
                        _, values = predict(pair)
                        final_choice = int(np.argmin(values))
                        selected, score = pair[final_choice], float(values[final_choice])
                        result = {"bank_choice": choice, "cem_score": value,
                                  "cem_samples": count, "generations": generations,
                                  "pair_scores": values.tolist(), "pair_choice": final_choice}
                else:
                    bases = [np.zeros((15, 2))]
                    for restart in (1, 2):
                        rng = np.random.default_rng(2029112000 + root * 100 + restart)
                        bases.append(np.clip(prior.core.interpolate(
                            rng.normal(0, .6, size=(1, 4, 2)), 15)[0], -1, 1))
                    _, scores = predict(np.asarray(bases))
                    choice = int(np.argmin(scores))
                    selected, score = bases[choice], float(scores[choice])
                    runs = []
                    for restart, count in enumerate((213, 212, 212)):
                        chain, value, generations = prior.core.cem(
                            bases[restart], [], count,
                            np.random.default_rng(2029113000 + root * 100 + restart),
                            predict, float(scores[restart]), 4)
                        runs.append({"restart": restart, "count": count, "score": value,
                                     "generations": generations})
                        if value < score:
                            selected, score = chain, value
                    result = {"selected": np.asarray(selected, np.float32).tolist(),
                              "score": score, "runs": runs}
            except Exception:
                error = traceback.format_exc()
                selected, score = None, None
            path = OUT / f"root{root:02d}_{arm}.npz"
            arrays = {f"{name}_{i}": value for i, triple in enumerate(trace)
                      for name, value in zip(("actions", "positions", "scores"), triple, strict=True)}
            with path.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
            rows.append({"root": root, "task": task, "arm": arm,
                "status": "failed" if error else "completed", "error": error, "result": result,
                "selected": np.asarray(selected, np.float32).tolist() if selected is not None else None,
                "score": float(score) if score is not None else None, "counts": meter.as_dict(),
                "wall_seconds": time.monotonic() - started, "trace_file": path.name,
                "trace_sha256": seal.digest(path), "batches": len(trace)})
            print(f"root={root} arm={arm} score={score} tokens={meter.token_samples} error={bool(error)}",
                  flush=True)
    seal.verify(deps)
    writer(OUT / "result.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit():
    if (OUT / "audit.json").exists():
        raise FileExistsError("audit exists")
    manifest = json.loads((OUT / "manifest.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    seal.verify(manifest["dependencies"])
    if result["manifest_sha256"] != seal.digest(OUT / "manifest.json") or manifest["tasks"] != tasks():
        raise ValueError("manifest mismatch")
    if len(result["rows"]) != len(tasks()) * len(ARMS):
        raise ValueError("missing result")
    counts = {arm: {"complete": 0, "success": 0, "scores": [], "tokens": []} for arm in ARMS}
    samples = 0
    for row, (root, arm) in zip(result["rows"], itertools.product(range(8), ARMS), strict=True):
        if (row["root"], row["arm"], row["task"]) != (root, arm, tasks()[root]):
            raise ValueError("result identity")
        path = OUT / row["trace_file"]
        if seal.digest(path) != row["trace_sha256"]:
            raise ValueError("trace hash")
        tokens, trajectories, full_scores, found = 0, 0, [], False
        with np.load(path, allow_pickle=False) as data:
            for i in range(row["batches"]):
                actions, positions, scores = (data[f"{name}_{i}"]
                                               for name in ("actions", "positions", "scores"))
                if actions.dtype != np.float32 or actions.shape[1] % 5 or np.max(np.abs(actions)) > 1:
                    raise ValueError("action contract")
                exact = prior.prior.previous.exact.endpoints(row["task"], actions)
                path_values, _ = independent(row["task"]["start"], actions)
                np.testing.assert_allclose(path_values[:, -1, :2], exact, atol=1e-12, rtol=0)
                np.testing.assert_allclose(positions, exact, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(exact - row["task"]["goal"], axis=1),
                                           atol=1e-12, rtol=0)
                tokens += len(actions) * actions.shape[1] // 5
                trajectories += len(actions)
                if actions.shape[1] == 15:
                    full_scores.extend(scores.tolist())
                    if row["selected"] is not None:
                        found |= bool(np.any(np.all(actions == np.asarray(row["selected"], np.float32),
                                                    axis=(1, 2)) & (np.abs(scores-row["score"]) < 1e-12)))
        if tokens > CAP or tokens != row["counts"]["prediction_token_samples"]:
            raise ValueError("token accounting")
        if trajectories != row["counts"]["trajectory_samples"]:
            raise ValueError("sample accounting")
        if row["status"] == "completed":
            if not found or row["score"] > min(full_scores) + 1e-12:
                raise ValueError("complete-chain incumbent discarded")
            counts[arm]["complete"] += 1
            counts[arm]["success"] += int(row["score"] < .05)
            counts[arm]["scores"].append(row["score"])
        counts[arm]["tokens"].append(tokens)
        samples += trajectories
    paired = []
    for root in range(8):
        new = next(r for r in result["rows"] if r["root"] == root and r["arm"] == "v13_global_response")
        conditioned = next(r for r in result["rows"] if r["root"] == root and r["arm"] == "v14_carry_nominal")
        control = next(r for r in result["rows"] if r["root"] == root and r["arm"] == "global_cem")
        if any(r["status"] != "completed" for r in (control, new, conditioned)):
            paired.append({"root": root, "status": "incomplete"})
            continue
        with np.load(OUT / control["trace_file"], allow_pickle=False) as control_trace, \
                np.load(OUT / new["trace_file"], allow_pickle=False) as new_trace, \
                np.load(OUT / conditioned["trace_file"], allow_pickle=False) as conditioned_trace:
            for name in ("actions", "positions", "scores"):
                np.testing.assert_array_equal(control_trace[f"{name}_0"], new_trace[f"{name}_0"])
                np.testing.assert_array_equal(control_trace[f"{name}_0"], conditioned_trace[f"{name}_0"])
            nominal = tuple(tuple(tuple(float(x) for x in point) for point in item)
                            for item in prior.core.NOMINALS)
            config = v13.Config(action_dim=2, segment_length=5, segments=3,
                model_packet_length=5, nominals=nominal, max_modes=1, token_cap=CAP,
                seed=2029111000 + root)
            np.testing.assert_array_equal(new_trace["actions_0"], v13.proposal_bank(config))
        chosen = np.asarray(conditioned["result"]["global_proposals"]["actions"][
            conditioned["result"]["global_proposals"]["choice"]], np.float32)
        changed_layers = 0
        for layer14, layer13 in zip(conditioned["result"]["child"]["layers"],
                                    new["result"]["child"]["layers"], strict=True):
            depth = layer14["depth"]
            if depth != layer13["depth"]:
                raise ValueError("layer mismatch")
            expected = np.clip(chosen[depth*5:(depth+1)*5], -.95, .95)
            nominees = [c for c in layer14["candidates"]
                        if c["path"][-1]["nominal"] == 1 and c["path"][-1]["mode"] is None]
            if not nominees:
                raise ValueError("carry nominal missing")
            for nominee in nominees:
                np.testing.assert_allclose(np.asarray(nominee["chain"], np.float32)[-5:], expected,
                                           atol=1e-7, rtol=0)
            changed_layers += int(layer14["candidates"] != layer13["candidates"])
        paired.append({"root": root, "status": "passed",
                       "changed_response_layers": changed_layers,
                       "score_difference_v14_minus_v13": conditioned["score"] - new["score"]})
    seal.verify(manifest["dependencies"])
    prior.prior.previous.write(OUT / "audit.json", {"status": "passed", "result_sha256": seal.digest(OUT / "result.json"),
        "counts": counts, "paired": paired, "trajectory_samples": samples,
        "scope": "same-backend plus independent collision implementation; not independent replication"})
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
