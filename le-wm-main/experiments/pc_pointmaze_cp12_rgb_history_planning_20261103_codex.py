"""Sealed cp12 RGB context, matched CEM decisions, execution, then evaluation."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from experiments import pc_causal_history_h10_suffix_20261103 as h
from experiments.pc_pointmaze_cp12_renderer_source_audit_20261103_codex import gl_info
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

ROOT = h.ROOT
SOURCE = ROOT / "results/pc_causal_mixed_rejected_physical_20261103_DEV01.json"
ARCHIVE = ROOT / "results/a0_pointmaze_archive_bank_alignment_20261102_DEV01.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CP12_REGENERATED_RGB_PAIRED_PLANNING_PROTOCOL_20261103_CODEX.md"
LOCK = ROOT.parent / "docs/PC_POINTMAZE_CP12_PAIRED_PLANNING_IMPLEMENTATION_LOCK_20261103_CODEX.md"
OUT = ROOT / "results/pc_pointmaze_cp12_rgb_history_paired_20261103_CODEX"
CONTEXT = OUT.with_name(OUT.name + "_context.json")
DECISIONS = OUT.with_name(OUT.name + "_decisions.json")
EXECUTION = OUT.with_name(OUT.name + "_execution.json")
EVALUATION = OUT.with_name(OUT.name + "_evaluation.json")
EXPECTED = {
    SOURCE: "07ada31f9a58c3472fd84778f749c2df6cfb14db6442531954e5e7003ab587f8",
    ARCHIVE: "93cdc7067ca623f1eb80439d1e41ff18b3be4468d8befdc3d9c6cecad26d3291",
    h.DEFAULT_CHECKPOINT: "a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc",
    h.DEFAULT_CONFIG: "c666b4251f72f56c08c69ab223cfe645625f39705f1c84285320edc98c205a66",
    h.CAMERA: h.CAMERA_SHA, h.OLD: h.OLD_SHA, h.OBSTACLE: h.OBSTACLE_SHA,
    PROTOCOL: "a6fdd0592d4818c5e0eb9d2499b464587d14acbc45e0432625d756b33f5516d3",
    LOCK: "e4ba8244fe5893bb95a36fd91f04d70add2cef91daf89fcee9c8648c5469b9fd",
}
ROOTS = (8343, 8345, 8349, 8351)


def sha(path):
    return h.sha256_file(path)


def files():
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"input changed {path}")
    source, obstacle, archive = (json.loads(p.read_text()) for p in (SOURCE, h.OBSTACLE, ARCHIVE))
    for path, digest in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                         (h.OBSTACLE.with_suffix(".npz"), obstacle["arrays_sha256"])):
        if sha(path) != digest:
            raise ValueError(f"arrays changed {path}")
    for name, field in (("actions.pth", "actions_sha256"), ("states.pth", "states_sha256"),
                        ("seq_lengths.pth", "sequence_lengths_sha256")):
        if sha(h.DEFAULT_DATASET / name) != archive["provenance"][field]:
            raise ValueError(f"archive changed {name}")
    if [r["root"] for r in source["rows"]] != list(ROOTS):
        raise ValueError("root roster")
    names = []
    for row in source["rows"]:
        expected = [(arm, candidate) for arm in ("raw", "spectral") for candidate in (2, 3, 4, 5)]
        if [(b["arm"], b["candidate"]) for b in row["branches"]] != expected:
            raise ValueError("branch roster")
        names += [f"r{row['root']}_a{('raw', 'spectral').index(arm)}_p{candidate}" for arm, candidate in expected]
    return source, archive, names


def deps():
    paths = list(EXPECTED) + [Path(__file__), SOURCE.with_suffix(".npz"), h.OBSTACLE.with_suffix(".npz"),
                              h.DEFAULT_DATASET / "actions.pth", h.DEFAULT_DATASET / "states.pth",
                              h.DEFAULT_DATASET / "seq_lengths.pth",
                              ROOT / "experiments/pointmaze_action_history_unroll_20261103.py",
                              ROOT / "experiments/pointmaze_native_rgb_bridge_20261102.py",
                              ROOT / "experiments/a1_pointmaze_rgb_causal_velocity_20261103.py",
                              ROOT / "experiments/pc_pointmaze_affine_feedback_cem_20261103.py",
                              ROOT / "external/jepa-wms/app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py"]
    return {str(p): sha(p) for p in paths}


def check_dependencies(mapping):
    if any(sha(Path(path)) != digest for path, digest in mapping.items()):
        raise ValueError("sealed dependency changed")


class Journal:
    def __init__(self, stage):
        path = OUT.with_name(OUT.name + f"_{stage}_events.jsonl")
        self.file = path.open("x")
        self.index = 0

    def record(self, event, **kw):
        row = {"index": self.index, "event": event, "time_ns": time.time_ns(), **kw}
        self.file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.file.flush()
        self.index += 1

    def close(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def replay(start, actions, journal, name, expected=None):
    journal.record("reset_attempt", branch=name)
    with h.NativeRGBBridge(start.astype(np.float64)) as env:
        journal.record("reset_returned", branch=name)
        frames = [env.initial_rgb]
        if expected is not None and not np.array_equal(frames[0], expected[0]):
            raise RuntimeError(f"prefix mismatch {name} frame 0")
        for i, action in enumerate(actions[:12]):
            journal.record("command_attempt", branch=name, step=i + 1)
            frame = env.step(action[None])
            journal.record("command_returned", branch=name, step=i + 1)
            if expected is not None and not np.array_equal(frame, expected[i+1]):
                raise RuntimeError(f"prefix mismatch {name} frame {i+1}")
            frames.append(frame)
    return np.stack(frames)


def cem(evaluate, prior, lower, upper, noise):
    if noise.shape != (3, 16, 10) or prior.shape != (10,) or not torch.isfinite(noise).all():
        raise ValueError("invalid CEM inputs")
    mean, std = prior.clone(), torch.full_like(prior, 1/3)
    trace = []
    for j in range(3):
        candidates = (mean + noise[j] * std).clamp(lower, upper)
        candidates[0] = mean.clamp(lower, upper)
        scores = evaluate(candidates)
        if scores.shape != (16,) or not torch.isfinite(scores).all():
            raise ValueError("invalid CEM score")
        elite = torch.argsort(scores, stable=True)[:4]
        chosen = candidates[elite]
        mean, std = chosen.mean(0), chosen.std(0, correction=1).clamp_min(1e-5)
        trace.append((candidates.clone(), scores.clone(), elite.clone(), mean.clone(), std.clone()))
    final = torch.stack((prior, mean)).clamp(lower, upper)
    score = evaluate(final)
    if score.shape != (2,) or not torch.isfinite(score).all():
        raise ValueError("invalid final scores")
    winner = int(score[1] < score[0])
    return final[winner], trace, final, score, winner


def collect():
    if CONTEXT.exists() or CONTEXT.with_suffix(".npz").exists():
        raise FileExistsError("context output exists")
    source, _, names = files()
    gl = gl_info()
    if gl["gl_vendor"] != "NVIDIA Corporation" or "NVIDIA" not in gl["gl_renderer"]:
        raise RuntimeError("NVIDIA renderer required")
    journal, arrays, rows = Journal("context"), {}, []
    started = time.monotonic()
    try:
        with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as data:
            for i, name in enumerate(names):
                actions, start = data[name + "_actions"], data["starts"][i]
                if actions.shape != (17, 2) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
                    raise ValueError("invalid source actions")
                frames = replay(start, actions, journal, name)
                arrays[name + "_frames"] = frames
                arrays[name + "_actions"] = actions[:12]
                rows.append({"name": name, "old_frame_exact": [bool(np.array_equal(a, b))
                             for a, b in zip(frames, data[name + "_history"][:13], strict=True)]})
                print(f"context {i+1}/32 {name}", flush=True)
        with CONTEXT.with_suffix(".npz").open("xb") as f:
            np.savez_compressed(f, **arrays)
        journal.record("completed")
        journal.close()
        write_json(CONTEXT, {"schema": "cp12_new_rgb_context.v1", "rows": rows, "gl": gl,
                             "arrays_sha256": sha(CONTEXT.with_suffix(".npz")), "dependencies": deps(),
                             "completed_prefix_commands": 384, "elapsed_seconds": time.monotonic()-started})
    finally:
        if not journal.file.closed:
            journal.close()


def load_sealed(path):
    doc = json.loads(path.read_text())
    if sha(path.with_suffix(".npz")) != doc["arrays_sha256"]:
        raise ValueError(f"sealed arrays changed: {path}")
    check_dependencies(doc["dependencies"])
    return doc


def plan():
    if DECISIONS.exists() or DECISIONS.with_suffix(".npz").exists():
        raise FileExistsError("decision output exists")
    source, archive, names = files()
    ctx = load_sealed(CONTEXT)
    if [r["name"] for r in ctx["rows"]] != names:
        raise ValueError("context roster drift")
    gl = gl_info()
    if gl["gl_vendor"] != "NVIDIA Corporation" or gl["gl_renderer"] != ctx["gl"]["gl_renderer"] or gl["gl_version"] != ctx["gl"]["gl_version"]:
        raise ValueError("renderer drift")
    journal, outputs, rows = Journal("plan"), {}, []
    started, count = time.monotonic(), {"one": 0, "three": 0, "parity_samples": 0, "encodes": 0, "batches": 0}
    try:
        with np.load(CONTEXT.with_suffix(".npz"), allow_pickle=False) as data, \
             np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
             h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            device = h.require_cuda()
            args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                      config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
            journal.record("model_load_attempt")
            model, dataset = h.load_model_and_dataset(args, device)
            before = h.parameter_hash(model)
            journal.record("model_load_returned", parameter_hash=before)
            if before != source["model_before"]:
                raise ValueError("model mismatch")
            mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
            if mean.cpu().tolist() != archive["provenance"]["action_mean"] or std.cpu().tolist() != archive["provenance"]["action_std"]:
                raise ValueError("normalizer mismatch")
            if not np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, rtol=0, atol=1e-7) or not np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, rtol=0, atol=1e-7):
                raise ValueError("proprio normalizer mismatch")
            lower, upper = ((-1 - mean)/std).repeat(5), ((1 - mean)/std).repeat(5)
            prior = ((-mean)/std).repeat(5)
            weight, _, _ = h.get_encoder()
            pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
            pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
            camera, old = (json.loads(p.read_text()) for p in (h.CAMERA, h.OLD))
            metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
            goal_rgb = obstacle["goal_rgb"]
            goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
            journal.record("encode_attempt", item="goal")
            target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
            count["encodes"] += 1
            journal.record("encode_returned", item="goal")
            for index, name in enumerate(names):
                frames, actions = data[name + "_frames"], data[name + "_actions"]
                if frames.shape != (13,224,224,3) or actions.shape != (12,2):
                    raise ValueError("context shape mismatch")
                enc = []
                states = []
                for t in (2, 7, 12):
                    state = h.causal_state(frames[t-1], frames[t], metadata, offset, .05)
                    journal.record("encode_attempt", item=name, t=t)
                    enc.append(h.encode_with_state(model, frames[t], state, device))
                    journal.record("encode_returned", item=name, t=t)
                    count["encodes"] += 1
                    states.append(state)
                hist = ((torch.as_tensor(actions[2:12], device=device) - mean) / std).reshape(2, 1, 10)
                torch.testing.assert_close(hist.reshape(10, 2)*std + mean, torch.as_tensor(actions[2:12], device=device), rtol=1e-5, atol=1e-6)
                context3 = {k: torch.cat([x[k] for x in enc], dim=1) for k in ("visual", "proprio")}
                context1 = {k: enc[-1][k] for k in ("visual", "proprio")}
                if index == 0:
                    for n in (16, 2):
                        future = prior.reshape(1,1,10).expand(1,n,10)
                        journal.record("parity_attempt", batch=n)
                        with torch.no_grad():
                            a = unroll_with_action_history(model, context1, future[:0], future, 1)
                            b = model.unroll(TensorDict(context1, batch_size=[1,1], device=device), future)
                        for k in ("visual", "proprio"):
                            torch.testing.assert_close(a[k], b[k], rtol=1e-5, atol=1e-6)
                        count["parity_samples"] += 2*n
                        journal.record("parity_returned", batch=n)
                gen = torch.Generator(device="cpu").manual_seed(2026110300 + index)
                noise = torch.randn((3,16,10), generator=gen, dtype=torch.float32).to(device)
                outputs[name + "_noise"] = noise.cpu().numpy()
                outputs[name + "_history_packets"] = hist.cpu().numpy()
                outputs[name + "_states"] = np.stack(states)
                entry = {"name": name, "seed": 2026110300+index, "arms": {}}
                for arm, context, history, window in (("one", context1, hist[:0], 1), ("three", context3, hist, 3)):
                    def evaluate(candidate):
                        n = candidate.shape[0]
                        future = candidate.unsqueeze(0)
                        past = history.expand(-1,n,-1)
                        journal.record("predict_attempt", branch=name, arm=arm, samples=n)
                        with torch.no_grad():
                            pred = unroll_with_action_history(model, context, past, future, window)["proprio"][-1]
                            result = h.projected_cost(pred.reshape(n,256,16), target, pinv, pos_std)
                        count[arm] += n
                        count["batches"] += 1
                        journal.record("predict_returned", branch=name, arm=arm, samples=n)
                        return result
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    chosen, trace, final, scores, winner = cem(evaluate, prior, lower, upper, noise)
                    torch.cuda.synchronize()
                    raw = (chosen.reshape(5,2)*std + mean).clamp(-1,1)
                    torch.testing.assert_close(((raw-mean)/std).reshape(10), chosen, rtol=1e-5, atol=1e-6)
                    outputs[f"{name}_{arm}_raw"] = raw.cpu().numpy()
                    outputs[f"{name}_{arm}_chosen"] = chosen.cpu().numpy()
                    outputs[f"{name}_{arm}_final"] = final.cpu().numpy()
                    outputs[f"{name}_{arm}_final_scores"] = scores.cpu().numpy()
                    for j, (c,s,e,m,v) in enumerate(trace):
                        for label, tensor in (("candidates",c),("scores",s),("elite",e),("mean",m),("std",v)):
                            outputs[f"{name}_{arm}_{j}_{label}"] = tensor.cpu().numpy()
                    entry["arms"][arm] = {"winner": winner, "scores": scores.cpu().tolist(),
                                           "optimizer_wall_seconds": time.monotonic()-t0}
                rows.append(entry)
                print(f"planned {index+1}/32 {name}", flush=True)
            if count != {"one": 1600, "three": 1600, "parity_samples": 36, "encodes": 97, "batches": 256}:
                raise RuntimeError(f"budget mismatch {count}")
            after = h.parameter_hash(model)
            if before != after:
                raise RuntimeError("model parameters changed")
        with DECISIONS.with_suffix(".npz").open("xb") as f:
            np.savez_compressed(f, **outputs)
        journal.record("completed")
        journal.close()
        write_json(DECISIONS, {"schema":"cp12_rgb_history_decisions.v1", "rows":rows, "counts":count,
                               "model_before":before, "model_after":after, "device":torch.cuda.get_device_name(device),
                               "context_sha256":sha(CONTEXT), "arrays_sha256":sha(DECISIONS.with_suffix(".npz")),
                               "dependencies":deps(), "elapsed_seconds":time.monotonic()-started})
    finally:
        if not journal.file.closed:
            journal.close()


def execute():
    if EXECUTION.exists() or EXECUTION.with_suffix(".npz").exists():
        raise FileExistsError("execution output exists")
    _, _, names = files()
    context, decisions = load_sealed(CONTEXT), load_sealed(DECISIONS)
    if decisions["context_sha256"] != sha(CONTEXT) or [r["name"] for r in decisions["rows"]] != names:
        raise ValueError("decision context/roster mismatch")
    gl = gl_info()
    if gl["gl_vendor"] != "NVIDIA Corporation" or gl["gl_renderer"] != context["gl"]["gl_renderer"] or gl["gl_version"] != context["gl"]["gl_version"]:
        raise ValueError("renderer drift")
    journal, outputs, rows, started = Journal("execute"), {}, [], time.monotonic()
    try:
        with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source, \
             np.load(CONTEXT.with_suffix(".npz"), allow_pickle=False) as ctx, \
             np.load(DECISIONS.with_suffix(".npz"), allow_pickle=False) as plans:
            for i, name in enumerate(names):
                actions, frames = ctx[name+"_actions"], ctx[name+"_frames"]
                if not np.array_equal(actions, source[name+"_actions"][:12]):
                    raise ValueError("action prefix drift")
                branch = {"name":name, "arms":{}}
                for arm in ("one","three"):
                    tail = plans[f"{name}_{arm}_raw"]
                    if tail.shape != (5,2) or not np.isfinite(tail).all() or np.any(np.abs(tail)>1):
                        raise ValueError("invalid saved tail")
                    journal.record("reset_attempt", branch=name, arm=arm)
                    with h.NativeRGBBridge(source["starts"][i]) as env:
                        journal.record("reset_returned", branch=name, arm=arm)
                        if not np.array_equal(env.initial_rgb, frames[0]):
                            raise RuntimeError("initial RGB mismatch")
                        for t, action in enumerate(actions):
                            journal.record("command_attempt",branch=name,arm=arm,step=t+1)
                            rgb = env.step(action[None])
                            journal.record("command_returned",branch=name,arm=arm,step=t+1)
                            if not np.array_equal(rgb, frames[t+1]):
                                raise RuntimeError(f"prefix RGB mismatch {name} {arm} {t+1}")
                        terminal = []
                        for t, action in enumerate(tail):
                            journal.record("command_attempt",branch=name,arm=arm,step=t+13)
                            terminal.append(env.step(action[None]))
                            journal.record("command_returned",branch=name,arm=arm,step=t+13)
                        if env.step_count != 17:
                            raise RuntimeError("clock mismatch")
                    outputs[f"{name}_{arm}_tail_frames"] = np.stack(terminal)
                    outputs[f"{name}_{arm}_actions"] = np.concatenate((actions,tail))
                    branch["arms"][arm] = {"prefix_match_frames":13}
                rows.append(branch)
                print(f"executed {i+1}/32 {name}",flush=True)
        with EXECUTION.with_suffix(".npz").open("xb") as f:
            np.savez_compressed(f,**outputs)
        journal.record("completed")
        journal.close()
        write_json(EXECUTION, {"schema":"cp12_rgb_history_execution.v1", "rows":rows,
                               "context_sha256":sha(CONTEXT), "decisions_sha256":sha(DECISIONS),
                               "arrays_sha256":sha(EXECUTION.with_suffix(".npz")), "dependencies":deps(),
                               "counts":{"reset":64,"prefix_commands":768,"tail_commands":320},
                               "elapsed_seconds":time.monotonic()-started})
    finally:
        if not journal.file.closed:
            journal.close()


def evaluate():
    if EVALUATION.exists() or EVALUATION.with_suffix(".npz").exists():
        raise FileExistsError("evaluation output exists")
    _, _, names = files()
    execution, decision, context = load_sealed(EXECUTION), load_sealed(DECISIONS), load_sealed(CONTEXT)
    if execution["decisions_sha256"] != sha(DECISIONS) or execution["context_sha256"] != sha(CONTEXT) or decision["context_sha256"] != sha(CONTEXT):
        raise ValueError("stage binding failed")
    journal, starts, sequences = Journal("evaluate"), [], []
    try:
        with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source, \
             np.load(EXECUTION.with_suffix(".npz"), allow_pickle=False) as output:
            for i, name in enumerate(names):
                for arm in ("one","three"):
                    starts.append(source["starts"][i])
                    sequences.append(output[f"{name}_{arm}_actions"])
        journal.record("endpoint_batch_attempt",trajectories=64,commands=1088)
        endpoints = h.physical.replay_position_endpoints(np.asarray(starts), np.asarray(sequences))
        journal.record("endpoint_batch_returned",trajectories=64,commands=1088)
        distance = np.linalg.norm(endpoints[:,:2] - h.GOAL[:2],axis=1).reshape(32,2)
        rows = [{"name":name, "one_distance":float(distance[i,0]), "three_distance":float(distance[i,1]),
                 "one_success":bool(distance[i,0] < .5), "three_success":bool(distance[i,1] < .5)}
                for i,name in enumerate(names)]
        with EVALUATION.with_suffix(".npz").open("xb") as f:
            np.savez_compressed(f,endpoints=endpoints,distance=distance)
        journal.record("completed")
        journal.close()
        write_json(EVALUATION, {"schema":"cp12_rgb_history_evaluation.v1","rows":rows,
                               "execution_sha256":sha(EXECUTION),"arrays_sha256":sha(EVALUATION.with_suffix(".npz")),
                               "dependencies":deps(),"position_contract":h.physical.position_backend_receipt(),
                               "counts":{"position_replay_trajectories":64,"position_replay_commands":1088},
                               "scope":"exposed development, position replay evaluation only"})
    finally:
        if not journal.file.closed:
            journal.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect","plan","execute","evaluate"))
    getattr(__import__(__name__,fromlist=["dummy"]), parser.parse_args().stage)()


if __name__ == "__main__":
    main()
