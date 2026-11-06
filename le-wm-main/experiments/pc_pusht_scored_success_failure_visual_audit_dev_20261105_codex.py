"""Physically labelled post-hoc subset for predictive-versus-encoded scoring."""
import argparse
import importlib.util
import json
import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments import pc_pusht_full_scored_chain_physical_census_dev_20261105_codex as census

ROOT, sha, save = census.ROOT, census.sha, census.save
OUT = ROOT / "results/pc_pusht_scored_success_failure_visual_audit_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_SCORED_SUCCESS_FAILURE_VISUAL_AUDIT_DEV_PROTOCOL_20261105_CODEX.md"


def chosen(scores, success):
    ranking = np.argsort(scores, kind="stable")
    yes, no = ranking[success[ranking]], ranking[~success[ranking]]
    if len(yes) > 64 or len(yes) + len(no) < 64:
        raise ValueError("invalid success/failure census for fixed 64 selection")
    return np.concatenate((yes, no[:64-len(yes)]))


def physical():
    if OUT.exists():
        raise FileExistsError(OUT)
    census.verify()
    census_audit = json.loads((census.OUT / "audit.json").read_text())
    if census_audit["status"] != "passed" or census_audit["report_sha256"] != sha(census.OUT / "report.json"):
        raise ValueError("census audit unavailable")
    overlay = (ROOT / ".jepa-physics-site").resolve()
    if os.environ.get("PYTHONPATH") != str(overlay):
        raise ValueError("exact physics overlay required")
    import gym
    import pymunk
    import skimage
    if ((gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2")
            or any(not Path(x.__file__).resolve().is_relative_to(overlay) for x in (gym, pymunk, skimage))):
        raise ValueError("physics dependency mismatch")
    old, src = census.old, census.old.src
    base = src.visible.core.old.old.prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    if runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] != \
            "native_import_ready_physics_not_executed":
        raise ValueError("runtime preflight")
    spec = importlib.util.spec_from_file_location("score_audit_subset_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    states, velocities = [torch.load(src.DATA / name, map_location="cpu", weights_only=True).numpy()
                          for name in ("states.pth", "velocities.pth")]
    deps = (Path(__file__), PROTOCOL, Path(census.__file__), census.OUT / "manifest.json",
            census.OUT / "report.json", census.OUT / "audit.json", src.WORKER, ROOT / runtime.ENV_REL)
    bindings = {str(p): sha(p) for p in deps}
    OUT.mkdir()
    result = {"status": "running", "bindings": bindings, "rows": []}
    try:
        for arm in ("response", "cem"):
            cp, op = census.OUT / f"root_11_{arm}", old.OUT / f"root_11_{arm}"
            path = OUT / f"root_11_{arm}"
            (path / "input").mkdir(parents=True)
            (path / "output").mkdir()
            with np.load(cp / "bank.npz", allow_pickle=False) as f:
                actions, scores, selected = f["actions"].copy(), f["score_min"].copy(), int(f["selected_index"])
            with np.load(cp / "physical.npz", allow_pickle=False) as f:
                final, target, valid = f["states"].copy(), f["target"].copy(), f["valid"].copy()
            if not valid.all():
                raise ValueError("census includes failed physical evaluations")
            joint = np.linalg.norm(final[:, :4]-target[:4], axis=1)
            angle = np.abs((final[:, 4]-target[4]+np.pi) % (2*np.pi)-np.pi)
            success = (joint < 20) & (angle < np.pi/9)
            ids = chosen(scores, success)
            with np.load(src.OUT / "root_11/physical.npz", allow_pickle=False) as f:
                warm, history = f["warm"].copy(), f["history_states"].copy()
                np.testing.assert_array_equal(f["goal_state"], target)
            with np.load(op / "input/context.npz", allow_pickle=False) as f:
                history_rgb, goal_rgb = f["rgb_history"].copy(), f["goal_rgb"][0].copy()
            initial = np.concatenate((states[11, 0], velocities[11, 0])).astype(np.float64)
            terminal = []
            for idx in ids:
                env = native.PushTEnv(**dict(base.CTOR, reset_to_state=initial.copy()))
                try:
                    env.seed(0)
                    obs, state = env.reset()
                    np.testing.assert_array_equal(state, history[0])
                    np.testing.assert_array_equal(obs["visual"], history_rgb[0])
                    for t, action in enumerate(warm):
                        obs, _, done, info = env.step(action.astype(np.float64))
                        if done:
                            raise RuntimeError("early warmup done")
                        if t in (4, 9):
                            h = 1 if t == 4 else 2
                            np.testing.assert_array_equal(info["state"], history[h])
                            np.testing.assert_array_equal(obs["visual"], history_rgb[h])
                    for action in actions[idx]:
                        obs, _, done, info = env.step(action.astype(np.float64))
                        if done:
                            raise RuntimeError("early candidate done")
                    np.testing.assert_array_equal(info["state"], final[idx])
                    terminal.append(obs["visual"].copy())
                finally:
                    env.close()
            for name in ("context.npz", "model_metadata.json"):
                shutil.copyfile(op / "input" / name, path / "input" / name)
            with (path / "input/bank.npy").open("xb") as f:
                np.save(f, actions[ids])
            with (path / "input/images.npz").open("xb") as f:
                np.savez_compressed(f, goal_rgb=goal_rgb, terminal_rgb=np.stack(terminal))
            with (path / "selection.npz").open("xb") as f:
                np.savez_compressed(f, source_indices=ids, success=success[ids], original_scores=scores[ids],
                                    selected_source_index=selected)
            result["rows"].append({"arm": arm, "selection_sha256": sha(path / "selection.npz"),
                                   "source_bank_sha256": sha(cp / "bank.npz"),
                                   "source_physical_sha256": sha(cp / "physical.npz"),
                                   "inputs": {p.name: sha(p) for p in sorted((path / "input").iterdir())}})
            print(f"{arm}: 64 post-hoc labelled chains physically replayed", flush=True)
        if any(sha(Path(p)) != h for p, h in bindings.items()):
            raise ValueError("dependency changed")
        result["status"] = "completed"
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    save(OUT / "physical.json", result)
    if result["status"] != "completed":
        raise RuntimeError(result["error"])


def verify():
    census.verify()
    result = json.loads((OUT / "physical.json").read_text())
    if result["status"] != "completed" or [r["arm"] for r in result["rows"]] != ["response", "cem"]:
        raise ValueError("incomplete source")
    if any(sha(Path(p)) != h for p, h in result["bindings"].items()):
        raise ValueError("dependency changed")
    for row in result["rows"]:
        arm = row["arm"]
        path, source = OUT / f"root_11_{arm}", census.OUT / f"root_11_{arm}"
        if row["selection_sha256"] != sha(path / "selection.npz") or \
                row["source_bank_sha256"] != sha(source / "bank.npz") or \
                row["source_physical_sha256"] != sha(source / "physical.npz") or any(
                    sha(path / "input" / name) != digest for name, digest in row["inputs"].items()):
            raise ValueError("input mismatch")


def neural():
    verify()
    for arm in ("response", "cem"):
        path = OUT / f"root_11_{arm}"
        census.old.src.visible.core.old.neural.launch(path, "experiments.pc_jepa_v22_contact_latent_error_worker_20261105_codex")
        print(f"{arm}: scored 64 post-hoc labelled chains", flush=True)


def summarize():
    verify()
    rows = []
    for arm in ("response", "cem"):
        path = OUT / f"root_11_{arm}"
        receipt = json.loads((path / "output/receipt.json").read_text())
        if receipt["status"] != "completed" or receipt["worker_sha256"] != sha(census.old.src.WORKER) or \
                receipt["latents_sha256"] != sha(path / "output/latents.npz"):
            raise ValueError("neural output mismatch")
        for key, name in (("context_sha256", "context.npz"), ("bank_sha256", "bank.npy"),
                          ("images_sha256", "images.npz"), ("metadata_sha256", "model_metadata.json")):
            if receipt[key] != sha(path / "input" / name):
                raise ValueError("neural input mismatch")
        with np.load(path / "output/latents.npz", allow_pickle=False) as f:
            pred, actual, goal = (f[k].astype(np.float64) for k in ("predicted", "actual", "goal"))
        with np.load(path / "selection.npz", allow_pickle=False) as f:
            indices, success, original = (f[k].copy() for k in ("source_indices", "success", "original_scores"))
            selected = int(f["selected_source_index"])
        p, a = ((pred-goal)**2).mean((1, 2)), ((actual-goal)**2).mean((1, 2))
        latent = ((pred-actual)**2).mean((1, 2))
        metrics = {}
        for name, values in (("original", original), ("predicted_rescore", p), ("actual_encoded", a)):
            rank = np.argsort(values, kind="stable")
            hits = np.flatnonzero(success[rank])
            yes, no = values[success], values[~success]
            metrics[name] = {"top_source_index": int(indices[rank[0]]), "top_success": bool(success[rank[0]]),
                             "first_success_rank": int(hits[0]+1) if len(hits) else None,
                             "success_failure_wins": int(np.sum(yes[:, None] < no[None, :])),
                             "success_failure_ties": int(np.sum(yes[:, None] == no[None, :])),
                             "success_failure_pairs": int(len(yes)*len(no))}
        selected_match = np.flatnonzero(indices == selected)
        if len(selected_match) != 1:
            raise ValueError("selected final chain missing")
        j = int(selected_match[0])
        rows.append({"arm": arm, "count": len(indices), "success_count": int(success.sum()),
                     "max_abs_prediction_rescore_diff": float(np.max(np.abs(p-original))),
                     "mean_prediction_latent_mse": float(latent.mean()),
                     "selected": {"source_index": selected, "success": bool(success[j]),
                                  "original": float(original[j]), "predicted_rescore": float(p[j]),
                                  "actual_encoded": float(a[j]), "latent_mse": float(latent[j])},
                     "successful": [{"source_index": int(indices[k]), "original": float(original[k]),
                                     "predicted_rescore": float(p[k]), "actual_encoded": float(a[k]),
                                     "latent_mse": float(latent[k])} for k in np.flatnonzero(success)],
                     "metrics": metrics})
    save(OUT / "report.json", {"status": "completed", "physical_sha256": sha(OUT / "physical.json"),
                               "rows": rows, "scope": "physical-label-conditioned posthoc scoring"})
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("physical", "neural", "summarize"))
    phase = parser.parse_args().phase
    print(f"Privileged successful-versus-failed scoring audit: {phase}", flush=True)
    {"physical": physical, "neural": neural, "summarize": summarize}[phase]()
