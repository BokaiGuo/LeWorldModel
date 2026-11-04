"""Sealed fixed-panel E0 development run and same-backend trace audit."""
import itertools
import json
from pathlib import Path
import sys

import numpy as np

from experiments import pc_v5_e0_prospective_mechanism_dev_20261104_codex as previous
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_recursive_response_v10 as v10
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_target_reach_action_dim_v6 as v6
from experiments.spectral_reachability import pc_target_reach_paired_abstention_v7 as v7
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = previous.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RECURSIVE_RESPONSE_V10_E0_DEV_PROTOCOL_20261104_CODEX.md"
OUT = ROOT / "results/pc_recursive_response_v10_e0_dev_20261104_codex"
ARMS = ("v10_mode1", "v10_mode2", "v7", "cem_smooth", "cem_action")


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.21, .23), (-.5, .6), (-.1, .1))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if len(tasks()) != 8 or any(task in previous.tasks() for task in tasks()):
        raise ValueError("bad development grid")
    dependencies = seal.capture([Path(__file__), PROTOCOL, Path(v10.__file__), Path(v7.__file__),
        Path(v6.__file__), Path(v6.v4.__file__), Path(core.__file__), Path(previous.exact.__file__),
        Path(previous.exact.wall.__file__), Path(previous.__file__)])
    OUT.mkdir()
    previous.write(OUT / "manifest.json", {"tasks": tasks(), "arms": ARMS, "dependencies": dependencies,
        "scope": "E0 development only, no independence or gate claim"})
    rows = []
    for root, task in enumerate(tasks()):
        goal = np.asarray(task["goal"])
        for arm in ARMS:
            meter, trace = Meter(1920), []

            def predict(actions):
                raw = np.asarray(actions, dtype=np.float32)
                def evaluate(batch):
                    pos = previous.exact.endpoints(task, batch)
                    return pos, np.linalg.norm(pos-goal, axis=1)
                positions, scores = meter.predict(raw, evaluate)
                trace.append((raw.copy(), positions.copy(), scores.copy()))
                return positions, scores

            if arm.startswith("v10"):
                config = v10.Config(action_dim=2, segment_length=5, segments=3,
                    nominals=tuple(tuple(tuple(float(v) for v in point) for point in nominal) for nominal in core.NOMINALS),
                    max_modes=int(arm[-1]), token_cap=1920, seed=2027113100 + root)
                result = v10.compile_program(predict, goal, config)
                selected, score = result["selected"], result["selected_score"]
                if result["resource"]["prediction_tokens"] != meter.token_samples:
                    raise ValueError("v10 meter discrepancy")
            elif arm == "v7":
                result = v7.decide("response", "response", root, 0, goal, predict, meter, core.NOMINALS)
                selected, score = result["selected"]["chain"], result["selected"]["terminal_score"]
            else:
                knots = 4 if arm == "cem_smooth" else 15
                bases = [np.zeros((15, 2))]
                for restart in (1, 2):
                    rng = np.random.default_rng(2027112900 + root*100 + restart)
                    bases.append(np.clip(core.interpolate(rng.normal(0, .6, size=(1, knots, 2)), 15)[0], -1, 1))
                _, initial = predict(np.asarray(bases))
                index = int(np.argmin(initial))
                selected, score, runs = bases[index], float(initial[index]), []
                for restart, count in enumerate((213, 212, 212)):
                    chain, value, generations = core.cem(bases[restart], [], count,
                        np.random.default_rng(2027112800 + root*100 + restart), predict, float(initial[restart]), knots)
                    runs.append({"restart": restart, "count": count, "score": value, "generations": generations})
                    if value < score:
                        selected, score = chain, value
                result = {"selected": np.asarray(selected, np.float32).tolist(), "score": score, "runs": runs}
            path = OUT / f"root{root:02d}_{arm}.npz"
            arrays = {f"{name}_{i}": value for i, triple in enumerate(trace)
                      for name, value in zip(("actions", "positions", "scores"), triple, strict=True)}
            with path.open("xb") as f:
                np.savez_compressed(f, **arrays)
            rows.append({"root": root, "task": task, "arm": arm, "result": result,
                         "selected": np.asarray(selected, np.float32).tolist(), "score": float(score),
                         "counts": meter.as_dict(), "trace_file": path.name, "trace_sha256": previous.sha(path),
                         "batches": len(trace)})
            print(f"root={root} {arm}: distance={score:.6f}, tokens={meter.token_samples}", flush=True)
    seal.verify(dependencies)
    previous.write(OUT / "result.json", {"manifest_sha256": previous.sha(OUT / "manifest.json"), "rows": rows})


def audit():
    destination = OUT / "audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((OUT / "manifest.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    seal.verify(manifest["dependencies"])
    if manifest["tasks"] != tasks() or result["manifest_sha256"] != previous.sha(OUT / "manifest.json"):
        raise ValueError("manifest mismatch")
    if len(result["rows"]) != len(tasks()) * len(ARMS):
        raise ValueError("missing rows")
    total = 0
    for row, (root, arm) in zip(result["rows"], itertools.product(range(8), ARMS), strict=True):
        if row["root"] != root or row["arm"] != arm or row["task"] != tasks()[root]:
            raise ValueError("row mismatch")
        path = OUT / row["trace_file"]
        if previous.sha(path) != row["trace_sha256"]:
            raise ValueError("trace hash mismatch")
        tokens, samples, all_full = 0, 0, []
        with np.load(path, allow_pickle=False) as data:
            batches = []
            for i in range(row["batches"]):
                raw, stored, values = (data[f"{name}_{i}"] for name in ("actions", "positions", "scores"))
                if raw.dtype != np.float32 or np.abs(raw).max() > 1:
                    raise ValueError("bad actions")
                pos = previous.exact.endpoints(row["task"], raw)
                score = np.linalg.norm(pos - np.asarray(row["task"]["goal"]), axis=1)
                np.testing.assert_allclose(pos, stored, atol=1e-12, rtol=0)
                np.testing.assert_allclose(score, values, atol=1e-12, rtol=0)
                tokens += len(raw) * (raw.shape[1] // 5)
                samples += len(raw)
                if raw.shape[1] == 15:
                    all_full.extend(score.tolist())
                batches.append((raw, pos, score))
            selected = np.asarray(row["selected"], np.float32)
            found = any(np.any(np.all(b[0] == selected, axis=(1, 2)) & (np.abs(b[2]-row["score"]) < 1e-12))
                        for b in batches if b[0].shape[1] == 15)
            if not found or row["score"] > min(all_full) + 1e-12:
                raise ValueError("selected not full-chain incumbent")
            if arm.startswith("v10"):
                item = row["result"]
                if not np.array_equal(batches[-1][0][int(item["abstained"])], selected):
                    raise ValueError("paired selection mismatch")
                cursor = 1
                for layer in item["layers"]:
                    for response in layer["responses"]:
                        pos = batches[cursor][1]
                        matrix = ((pos[1:11] - pos[11:21]) / .1).T
                        np.testing.assert_allclose(matrix, response["matrix"], atol=1e-12, rtol=0)
                        u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
                        contrib = singular**2/(singular**2+.1)*(u.T @ (np.asarray(row["task"]["goal"])-pos[0]))**2
                        modes = np.argsort(-contrib[:int(np.sum(singular > 1e-10))], kind="stable")[:int(arm[-1])]
                        np.testing.assert_array_equal(modes, response["selected_modes"])
                        cursor += 1
                    cand = layer["candidates"]
                    np.testing.assert_array_equal(batches[cursor][0], np.asarray([x["chain"] for x in cand], np.float32))
                    np.testing.assert_allclose(batches[cursor][2], [x["score"] for x in cand], atol=1e-12, rtol=0)
                    cursor += 1
        if tokens > 1920 or tokens != row["counts"]["prediction_token_samples"] or samples != row["counts"]["trajectory_samples"]:
            raise ValueError("resource mismatch")
        total += samples
    seal.verify(manifest["dependencies"])
    previous.write(destination, {"result_sha256": previous.sha(OUT / "result.json"), "plans": len(result["rows"]),
        "trajectory_samples": total, "scope": "same exact backend trace and selected-response audit"})
    print(f"audited {len(result['rows'])} plans and {total} trajectories")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
