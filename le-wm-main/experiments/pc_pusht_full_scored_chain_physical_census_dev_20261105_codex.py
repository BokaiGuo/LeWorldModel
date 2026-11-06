"""Post-hoc physical census of every distinct recorded full-horizon plan."""
import argparse
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments import pc_pusht_continuous_unprivileged_plan_dev_20261105_codex as old

ROOT, sha, save = old.ROOT, old.sha, old.save
OUT = ROOT / "results/pc_pusht_full_scored_chain_physical_census_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_FULL_SCORED_CHAIN_PHYSICAL_CENSUS_DEV_PROTOCOL_20261105_CODEX.md"
CASES = ((9, "response"), (9, "cem"), (11, "response"), (11, "cem"))


def extract(decision):
    unique, actions, low, high, count, first = {}, [], [], [], [], []
    total, short = 0, 0
    for ti, trace in enumerate(decision["evaluations"]):
        batch = np.asarray(trace["actions"], dtype=np.float32)
        scores = np.asarray(trace["scores"], dtype=np.float64)
        if batch.ndim != 3 or batch.shape[2] != 2 or scores.shape != (len(batch),) or \
                not np.isfinite(batch).all() or not np.isfinite(scores).all():
            raise ValueError("invalid trace")
        if batch.shape[1] != 15:
            short += len(batch)
            continue
        for j, (chain, score) in enumerate(zip(batch, scores, strict=True)):
            total += 1
            key = chain.tobytes()
            if key not in unique:
                unique[key] = len(actions)
                actions.append(chain.copy())
                low.append(float(score))
                high.append(float(score))
                count.append(1)
                first.append((ti, j))
            else:
                k = unique[key]
                low[k] = min(low[k], float(score))
                high[k] = max(high[k], float(score))
                count[k] += 1
    return (np.asarray(actions, dtype=np.float32), np.asarray(low, dtype=np.float64),
            np.asarray(high, dtype=np.float64), np.asarray(count, dtype=np.int32),
            np.asarray(first, dtype=np.int32), total, short)


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    old.verify_prepared()
    seal = json.loads((old.OUT / "plan_seal.json").read_text())
    report = json.loads((old.OUT / "report.json").read_text())
    if report["plan_seal_sha256"] != sha(old.OUT / "plan_seal.json"):
        raise ValueError("source plan seal mismatch")
    OUT.mkdir()
    manifest = {"status": "running", "bindings": {str(p): sha(p) for p in
                (Path(__file__), PROTOCOL, Path(old.__file__), old.OUT / "prepared.json",
                 old.OUT / "plan_seal.json", old.OUT / "report.json")}, "rows": []}
    try:
        for seq, arm in CASES:
            decision_path = old.OUT / f"root_{seq}_{arm}/output/decision.json"
            if sha(decision_path) != seal["decisions"][f"{seq}_{arm}"]:
                raise ValueError("decision changed")
            decision = json.loads(decision_path.read_text())
            actions, low, high, count, first, total, short = extract(decision)
            selected = np.asarray(decision["decision"]["selected"]["chain"], np.float32)
            selected_matches = np.flatnonzero(np.all(actions == selected, axis=(1, 2)))
            if len(selected_matches) != 1 or not np.isclose(
                    low[selected_matches[0]], decision["decision"]["selected"]["terminal_score"],
                    rtol=1e-5, atol=1e-7):
                raise ValueError("selected chain does not bind scored bank")
            path = OUT / f"root_{seq}_{arm}"
            path.mkdir()
            with (path / "bank.npz").open("xb") as f:
                np.savez_compressed(f, actions=actions, score_min=low, score_max=high, occurrences=count,
                                    first_indices=first, selected_index=selected_matches[0])
            manifest["rows"].append({"sequence": seq, "arm": arm, "bank_sha256": sha(path / "bank.npz"),
                                     "source_decision_sha256": sha(decision_path), "full_occurrences": total,
                                     "short_occurrences_not_evaluated": short, "unique_full": len(actions),
                                     "max_duplicate_score_range": float(np.max(high-low))})
        manifest["status"] = "completed"
    except Exception:
        manifest.update(status="failed", error=traceback.format_exc())
    save(OUT / "manifest.json", manifest)
    if manifest["status"] != "completed":
        raise RuntimeError(manifest["error"])


def verify():
    old.verify_prepared()
    m = json.loads((OUT / "manifest.json").read_text())
    if m["status"] != "completed" or [(x["sequence"], x["arm"]) for x in m["rows"]] != list(CASES):
        raise ValueError("manifest roster")
    if any(sha(Path(p)) != h for p, h in m["bindings"].items()):
        raise ValueError("dependency changed")
    for row in m["rows"]:
        seq, arm = row["sequence"], row["arm"]
        if sha(OUT / f"root_{seq}_{arm}/bank.npz") != row["bank_sha256"] or \
                sha(old.OUT / f"root_{seq}_{arm}/output/decision.json") != row["source_decision_sha256"]:
            raise ValueError("bank or decision changed")
    return m


def physical():
    m = verify()
    overlay = (ROOT / ".jepa-physics-site").resolve()
    if os.environ.get("PYTHONPATH") != str(overlay):
        raise ValueError("exact physics overlay required")
    import gym
    import pymunk
    import skimage
    if ((gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2")
            or any(not Path(z.__file__).resolve().is_relative_to(overlay) for z in (gym, pymunk, skimage))):
        raise ValueError("physics dependency mismatch")
    src = old.src
    base = src.visible.core.old.old.prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    if runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] != \
            "native_import_ready_physics_not_executed":
        raise ValueError("runtime preflight")
    spec = importlib.util.spec_from_file_location("census_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    data, velocity = [torch.load(src.DATA / name, map_location="cpu", weights_only=True).numpy()
                      for name in ("states.pth", "velocities.pth")]
    for row in m["rows"]:
        seq, arm = row["sequence"], row["arm"]
        path = OUT / f"root_{seq}_{arm}"
        if (path / "physical.json").exists():
            stored = json.loads((path / "physical.json").read_text())
            if stored["status"] != "completed" or stored["bank_sha256"] != sha(path / "bank.npz") or \
                    stored["arrays_sha256"] != sha(path / "physical.npz"):
                raise ValueError("previous case failed or changed")
            print(f"{seq} {arm}: verified completed case", flush=True)
            continue
        with np.load(path / "bank.npz", allow_pickle=False) as f:
            actions = f["actions"].copy()
        with np.load(src.OUT / f"root_{seq}/physical.npz", allow_pickle=False) as f:
            warm, target, history = (f[k].copy() for k in ("warm", "goal_state", "history_states"))
        with np.load(src.OUT / f"root_{seq}/input/context.npz", allow_pickle=False) as f:
            rgb = f["rgb_history"].copy()
        initial = np.concatenate((data[seq, 0], velocity[seq, 0])).astype(np.float64)
        final = np.full((len(actions), 7), np.nan, dtype=np.float32)
        outside = np.zeros((len(actions), 16), dtype=np.bool_)
        valid = np.zeros(len(actions), dtype=np.bool_)
        failures = []
        started = time.monotonic()
        for i, chain in enumerate(actions):
            env = None
            try:
                env = native.PushTEnv(**dict(base.CTOR, reset_to_state=initial.copy()))
                env.seed(0)
                obs, state = env.reset()
                np.testing.assert_array_equal(state, history[0])
                np.testing.assert_array_equal(obs["visual"], rgb[0])
                for t, u in enumerate(warm):
                    obs, _, done, info = env.step(u.astype(np.float64))
                    if done:
                        raise RuntimeError("warmup done")
                    if t in (4, 9):
                        h = 1 if t == 4 else 2
                        np.testing.assert_array_equal(info["state"], history[h])
                        np.testing.assert_array_equal(obs["visual"], rgb[h])
                outside[i, 0] = np.any((info["state"][:4] < 40) | (info["state"][:4] > 472))
                for t, u in enumerate(chain):
                    obs, _, done, info = env.step(u.astype(np.float64))
                    if done:
                        raise RuntimeError("chain done")
                    outside[i, t+1] = np.any((info["state"][:4] < 40) | (info["state"][:4] > 472))
                final[i] = info["state"]
                valid[i] = True
            except Exception:
                failures.append({"index": i, "error": traceback.format_exc()})
            finally:
                if env is not None:
                    env.close()
            if (i+1) % 100 == 0 or i+1 == len(actions):
                print(f"{seq} {arm}: {i+1}/{len(actions)}", flush=True)
        with (path / "physical.npz").open("xb") as f:
            np.savez_compressed(f, states=final, target=target, valid=valid, outside=outside)
        save(path / "physical.json", {"status": "completed" if not failures else "failed",
                                     "bank_sha256": sha(path / "bank.npz"), "arrays_sha256": sha(path / "physical.npz"),
                                     "env_sha256": sha(ROOT / runtime.ENV_REL), "seconds": time.monotonic()-started,
                                     "failures": failures})
        if failures:
            raise RuntimeError(f"{seq} {arm}: retained {len(failures)} failed chains")


def summarize():
    m = verify()
    rows = []
    for row in m["rows"]:
        seq, arm = row["sequence"], row["arm"]
        path = OUT / f"root_{seq}_{arm}"
        rec = json.loads((path / "physical.json").read_text())
        if rec["status"] != "completed" or rec["bank_sha256"] != sha(path / "bank.npz") or \
                rec["arrays_sha256"] != sha(path / "physical.npz"):
            raise ValueError("incomplete physical record")
        with np.load(path / "bank.npz", allow_pickle=False) as f:
            scores, selected = f["score_min"].copy(), int(f["selected_index"])
        with np.load(path / "physical.npz", allow_pickle=False) as f:
            states, target, valid, outside = (f[k].copy() for k in ("states", "target", "valid", "outside"))
        if not valid.all():
            raise ValueError("invalid candidates in completed case")
        joint = np.linalg.norm(states[:, :4]-target[:4], axis=1)
        obj = np.linalg.norm(states[:, 2:4]-target[2:4], axis=1)
        angle = np.abs((states[:, 4]-target[4]+np.pi) % (2*np.pi)-np.pi)
        success = (joint < 20) & (angle < np.pi/9)
        rank = np.argsort(scores, kind="stable")
        hits = np.flatnonzero(success[rank])
        rows.append({"sequence": seq, "arm": arm, "unique_full": len(scores),
                     "full_occurrences": row["full_occurrences"],
                     "short_occurrences_not_evaluated": row["short_occurrences_not_evaluated"],
                     "max_duplicate_score_range": row["max_duplicate_score_range"],
                     "success_count": int(success.sum()), "first_success_rank": int(hits[0]+1) if len(hits) else None,
                     "selected_index": selected, "selected_rank": int(np.flatnonzero(rank == selected)[0]+1),
                     "selected_success": bool(success[selected]), "selected_joint": float(joint[selected]),
                     "minimum_joint": float(joint.min()), "minimum_object": float(obj.min()),
                     "outside_any_count": int(np.sum(outside.any(axis=1))),
                     "physical_seconds": rec["seconds"], "physical_sha256": sha(path / "physical.npz")})
    save(OUT / "report.json", {"status": "completed", "manifest_sha256": sha(OUT / "manifest.json"), "rows": rows,
                                "scope": "posthoc full recorded plans, evaluation only"})
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "physical", "summarize"))
    phase = parser.parse_args().phase
    print(f"Post-hoc physical census: {phase}", flush=True)
    {"prepare": prepare, "physical": physical, "summarize": summarize}[phase]()
