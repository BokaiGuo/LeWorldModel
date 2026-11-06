"""Reconstruct all recorded chains and replay each success and final selection."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_pusht_full_scored_chain_physical_census_dev_20261105_codex as run


def main():
    manifest = run.verify()
    out, root, old, sha = run.OUT, run.ROOT, run.old, run.sha
    if (out / "audit.json").exists():
        raise FileExistsError(out / "audit.json")
    overlay = (root / ".jepa-physics-site").resolve()
    assert os.environ.get("PYTHONPATH") == str(overlay)
    import gym
    import pymunk
    import skimage
    assert (gym.__version__, pymunk.version, skimage.__version__) == ("0.23.1", "6.8.0", "0.25.2")
    assert all(Path(x.__file__).resolve().is_relative_to(overlay) for x in (gym, pymunk, skimage))
    src = old.src
    base = src.visible.core.old.old.prior.helpers(root)
    runtime, parent = base.load_runtime(root)
    assert runtime.probe(root, parent, root / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] == \
        "native_import_ready_physics_not_executed"
    spec = importlib.util.spec_from_file_location("census_independent_audit_native", root / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    states, velocities = [torch.load(src.DATA / name, map_location="cpu", weights_only=True).numpy()
                          for name in ("states.pth", "velocities.pth")]
    report = json.loads((out / "report.json").read_text())
    assert report["manifest_sha256"] == sha(out / "manifest.json")
    audit_rows = []
    for row, bankmeta in zip(report["rows"], manifest["rows"], strict=True):
        seq, arm = row["sequence"], row["arm"]
        assert (seq, arm) == (bankmeta["sequence"], bankmeta["arm"])
        path = out / f"root_{seq}_{arm}"
        decision = json.loads((old.OUT / f"root_{seq}_{arm}/output/decision.json").read_text())
        # Independent implementation of insertion-order float32 deduplication.
        lookup = {}
        for ti, trace in enumerate(decision["evaluations"]):
            for j, (chain, score) in enumerate(zip(trace["actions"], trace["scores"], strict=True)):
                a = np.asarray(chain, dtype=np.float32)
                if a.shape != (15, 2):
                    continue
                key = a.tobytes()
                if key not in lookup:
                    lookup[key] = [a, float(score), float(score), 1, (ti, j)]
                else:
                    x = lookup[key]
                    x[1], x[2], x[3] = min(x[1], score), max(x[2], score), x[3] + 1
        values = list(lookup.values())
        with np.load(path / "bank.npz", allow_pickle=False) as f:
            actions, scores, selected = f["actions"].copy(), f["score_min"].copy(), int(f["selected_index"])
            np.testing.assert_array_equal(actions, np.stack([x[0] for x in values]))
            np.testing.assert_array_equal(scores, [x[1] for x in values])
            np.testing.assert_array_equal(f["score_max"], [x[2] for x in values])
            np.testing.assert_array_equal(f["occurrences"], [x[3] for x in values])
            np.testing.assert_array_equal(f["first_indices"], [x[4] for x in values])
        selected_action = np.asarray(decision["decision"]["selected"]["chain"], dtype=np.float32)
        np.testing.assert_array_equal(actions[selected], selected_action)
        physical = json.loads((path / "physical.json").read_text())
        assert physical["status"] == "completed" and physical["bank_sha256"] == sha(path / "bank.npz")
        assert physical["arrays_sha256"] == row["physical_sha256"] == sha(path / "physical.npz")
        with np.load(path / "physical.npz", allow_pickle=False) as f:
            finals, target, valid, outside = (f[k].copy() for k in ("states", "target", "valid", "outside"))
        assert valid.all()
        with np.load(src.OUT / f"root_{seq}/physical.npz", allow_pickle=False) as f:
            warm, history = f["warm"].copy(), f["history_states"].copy()
            np.testing.assert_array_equal(target, f["goal_state"])
        with np.load(src.OUT / f"root_{seq}/input/context.npz", allow_pickle=False) as f:
            history_rgb = f["rgb_history"].copy()
        joint = np.linalg.norm(finals[:, :4]-target[:4], axis=1)
        obj = np.linalg.norm(finals[:, 2:4]-target[2:4], axis=1)
        angle = np.abs((finals[:, 4]-target[4]+np.pi) % (2*np.pi)-np.pi)
        success = (joint < 20) & (angle < np.pi/9)
        rank = np.argsort(scores, kind="stable")
        hits = np.flatnonzero(success[rank])
        assert row["success_count"] == int(success.sum())
        assert row["first_success_rank"] == (int(hits[0])+1 if len(hits) else None)
        assert row["selected_index"] == selected and row["selected_rank"] == int(np.flatnonzero(rank == selected)[0]+1)
        assert np.isclose(row["minimum_joint"], np.min(joint)) and np.isclose(row["minimum_object"], np.min(obj))
        assert row["outside_any_count"] == int(outside.any(axis=1).sum())
        check = sorted(set(np.flatnonzero(success).tolist() + [selected]))
        initial = np.concatenate((states[seq, 0], velocities[seq, 0])).astype(np.float64)
        for idx in check:
            env = native.PushTEnv(**dict(base.CTOR, reset_to_state=initial.copy()))
            try:
                env.seed(0)
                obs, state = env.reset()
                np.testing.assert_array_equal(state, history[0])
                np.testing.assert_array_equal(obs["visual"], history_rgb[0])
                for t, action in enumerate(warm):
                    obs, _, done, info = env.step(action.astype(np.float64))
                    assert not done
                    if t in (4, 9):
                        h = 1 if t == 4 else 2
                        np.testing.assert_array_equal(info["state"], history[h])
                        np.testing.assert_array_equal(obs["visual"], history_rgb[h])
                flags = [bool(np.any((info["state"][:4] < 40) | (info["state"][:4] > 472)))]
                for action in actions[idx]:
                    obs, _, done, info = env.step(action.astype(np.float64))
                    assert not done
                    flags.append(bool(np.any((info["state"][:4] < 40) | (info["state"][:4] > 472))))
                np.testing.assert_array_equal(info["state"], finals[idx])
                np.testing.assert_array_equal(flags, outside[idx])
            finally:
                env.close()
        audit_rows.append({"sequence": seq, "arm": arm, "full_unique_rebuilt": len(actions),
                           "successful_chains_replayed": int(success.sum()),
                           "selection_and_all_successes_replayed": len(check)})
        print(f"sequence {seq} {arm}: census arithmetic and {len(check)} selected/success chains checked", flush=True)
    with (out / "audit.json").open("x") as f:
        json.dump({"status": "passed", "auditor_sha256": sha(Path(__file__)), "report_sha256": sha(out / "report.json"),
                   "rows": audit_rows, "scope": "all record arithmetic; successful and selected chains replayed same backend"}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
