"""Evaluation-only static angle sensitivity on eight replayed v22 terminal states."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np

from experiments import pc_jepa_v22_unqueried_suffix_dev_20261105_codex as old
from experiments import pc_jepa_v12_contact_score_bank_neural_posthoc_20261104_codex as neural
from experiments.audit_pc_jepa_v22_unqueried_suffix_dev_20261105_codex import independent_order

ROOT, SOURCE, sha = old.ROOT, old.SOURCE, old.sha
BANK = ROOT / "results/pc_jepa_v22_privileged_contact_suffix_20261105_codex"
OUT = ROOT / "results/pc_jepa_v22_static_angle_v2_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_V22_STATIC_ANGLE_PROTOCOL_20261105_CODEX.md"
INDICES = (8, 9, 24, 25, 40, 41, 56, 57)
OFFSETS = (-0.8, -0.4, -0.2, -0.1, 0., 0.1, 0.2, 0.4)


def render():
    if OUT.exists():
        raise FileExistsError(OUT)
    if os.environ.get("PYTHONPATH") != str((ROOT / ".jepa-physics-site").resolve()):
        raise ValueError("physics overlay required")
    import gym
    import pymunk
    import skimage
    overlay = (ROOT / ".jepa-physics-site").resolve()
    if ((gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2")
            or any(not Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage))):
        raise ValueError("physics version or origin mismatch")
    audit = json.loads((BANK / "audit.json").read_text())
    physical = json.loads((BANK / "physical.json").read_text())
    if (audit["status"] != "passed" or audit["physical_sha256"] != sha(BANK / "physical.json")
            or physical["arrays_sha256"] != sha(BANK / "physical_arrays.npz")):
        raise ValueError("physical source audit mismatch")
    base = old.prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    if runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] != "native_import_ready_physics_not_executed":
        raise ValueError("runtime preflight")
    spec = importlib.util.spec_from_file_location("static_angle_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    goal_state = np.load(SOURCE / "goal_state_evaluation_only.npy", allow_pickle=False)
    goal_rgb = np.load(SOURCE / "goal_rgb.npy", allow_pickle=False)
    witness = np.load(SOURCE / "witness_actions_evaluation_only.npy", allow_pickle=False)
    with np.load(BANK / "bank.npz", allow_pickle=False) as f:
        actions = f["actions"].copy()
    with np.load(BANK / "physical_arrays.npz", allow_pickle=False) as f:
        saved_rgb, saved_states = f["terminal_rgb"].copy(), f["states"].copy()
    metadata = SOURCE / "response_0/input/model_metadata.json"
    paths = [PROTOCOL, Path(__file__), ROOT / runtime.ENV_REL, SOURCE / "manifest.json",
             SOURCE / "goal_state_evaluation_only.npy", SOURCE / "goal_rgb.npy",
             SOURCE / "witness_actions_evaluation_only.npy", BANK / "bank.npz", BANK / "physical.json",
             BANK / "physical_arrays.npz", BANK / "audit.json", metadata]
    bindings = {str(p): sha(p) for p in paths}
    OUT.mkdir()
    (OUT / "input").mkdir()
    (OUT / "output").mkdir()
    record = {"status": "running", "bindings": bindings, "rows": [],
              "indices": INDICES, "offsets": OFFSETS, "scope": "evaluation-only static rendering"}

    def image(env, no_contacts):
        cls, original = native.DrawOptions, native.DrawOptions.__init__
        if no_contacts:
            def omit(self, surface):
                original(self, surface)
                self.flags &= ~pymunk.SpaceDebugDrawOptions.DRAW_COLLISION_POINTS
            cls.__init__ = omit
        try:
            return env.render("rgb_array").copy()
        finally:
            cls.__init__ = original

    def replay(chain):
        env = native.PushTEnv(**base.CTOR)
        env.seed(manifest["seed"])
        obs, state = env.reset()
        for action in np.concatenate((np.asarray(manifest["warmup"]), chain)).astype(np.float64):
            obs, _, done, info = env.step(action)
            if done:
                env.close()
                raise ValueError("unexpected termination")
            state = info["state"]
        return env, obs, state

    try:
        env, obs, state = replay(witness)
        try:
            np.testing.assert_array_equal(obs["visual"], goal_rgb)
            np.testing.assert_array_equal(state, goal_state)
            reference = image(env, True)
            goal_overlay_diff = int(np.count_nonzero(image(env, False) != reference))
        finally:
            env.close()
        frames = []
        for index in INDICES:
            env, obs, state = replay(actions[index])
            try:
                np.testing.assert_array_equal(obs["visual"], saved_rgb[index])
                np.testing.assert_array_equal(state, saved_states[index])
                original = float(env.block.angle)
                agent, block = tuple(env.agent.position), tuple(env.block.position)
                overlay_diff = int(np.count_nonzero(image(env, False) != image(env, True)))
                for offset in OFFSETS:
                    env.block.angle = float(goal_state[4]) + offset
                    env.block.position = block
                    env.space.reindex_shapes_for_body(env.block)
                    if tuple(env.agent.position) != agent or tuple(env.block.position) != block:
                        raise ValueError("non-angle geometry changed")
                    frames.append(image(env, True))
                env.block.angle = original
                env.block.position = block
                env.space.reindex_shapes_for_body(env.block)
                np.testing.assert_array_equal(image(env, False), obs["visual"])
                record["rows"].append({"index": index, "state": state.tolist(),
                                       "native_no_contact_pixel_channel_difference": overlay_diff})
                print(f"Replayed index {index}; rendered 8 angle variants.", flush=True)
            finally:
                env.close()
        shutil.copyfile(metadata, OUT / "input/model_metadata.json")
        with (OUT / "input/images.npz").open("xb") as handle:
            np.savez_compressed(handle, goal_rgb=reference, terminal_rgb=np.stack(frames))
        if any(sha(Path(p)) != digest for p, digest in bindings.items()):
            raise ValueError("source changed during render")
        record.update(status="completed", images_sha256=sha(OUT / "input/images.npz"),
                      metadata_sha256=sha(OUT / "input/model_metadata.json"),
                      goal_native_no_contact_pixel_channel_difference=goal_overlay_diff)
    except Exception:
        record.update(status="failed", error=traceback.format_exc())
    with (OUT / "render.json").open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if record["status"] != "completed":
        raise RuntimeError(record["error"])


def score():
    record = json.loads((OUT / "render.json").read_text())
    if (record["status"] != "completed" or any(sha(Path(p)) != d for p, d in record["bindings"].items())
            or sha(OUT / "input/images.npz") != record["images_sha256"]
            or sha(OUT / "input/model_metadata.json") != record["metadata_sha256"]):
        raise ValueError("render binding mismatch")
    neural.launch(OUT, "experiments.pc_jepa_v11_same_bank_actual_encoding_worker_20261104_codex")


def summarize():
    render_record = json.loads((OUT / "render.json").read_text())
    result = json.loads((OUT / "output/encoded.json").read_text())
    if (result["image_sha256"] != sha(OUT / "input/images.npz")
            or result["metadata_sha256"] != sha(OUT / "input/model_metadata.json")
            or result["worker_sha256"] != sha(neural.ENCODE_WORKER)
            or any(sha(Path(p)) != d for p, d in render_record["bindings"].items())):
        raise ValueError("score binding mismatch")
    scores = np.asarray(result["scores"], np.float64).reshape(len(INDICES), len(OFFSETS))
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite scores")
    rows = [{"index": index, "scores": scores[i].tolist(),
             "angle_order": independent_order(scores[i], np.abs(OFFSETS))}
            for i, index in enumerate(INDICES)]
    report = {"status": "completed", "render_sha256": sha(OUT / "render.json"),
              "encoded_sha256": sha(OUT / "output/encoded.json"),
              "offsets": OFFSETS, "rows": rows, "scope": "one exposed root; static evaluation only"}
    with (OUT / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("render", "score", "summarize"))
    phase = parser.parse_args().phase
    print(f"Evaluation-only static angle diagnostic: {phase}", flush=True)
    {"render": render, "score": score, "summarize": summarize}[phase]()


if __name__ == "__main__":
    main()
