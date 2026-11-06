"""Paired v23/shared-CEM PointMaze live RGB integration on a declared root."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_unified_rgb_live_integration_20261103_codex as bridge
from experiments import pc_durable_rgb_dispatch as journal
from experiments import pc_prediction_journal_v1 as prediction_journal
from experiments import pc_v23_pointmaze_feedback_adapter as feedback
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

old, h, ROOT = bridge.old, bridge.h, bridge.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V23_POINTMAZE_LIVE_FEEDBACK_CONTRACT_20261104_CODEX.md"
PREFIX = None
MANIFEST = None
ARMS = ("response", "cem")


def paths(arm):
    stem = PREFIX.with_name(PREFIX.name + "_" + arm)
    return ([stem.with_name(stem.name + f"_decision{i}.json") for i in range(3)],
            stem.with_name(stem.name + "_images.npz"),
            stem.with_name(stem.name + "_execution.json"))


def dependencies(contract_path):
    prior_replay = json.loads(bridge.REPLAY.read_text())
    for path, digest in prior_replay["dependencies"].items():
        if old.prior.sha(Path(path)) != digest:
            raise ValueError("frozen model replay dependency drift")
    files = (Path(__file__), PROTOCOL, contract_path, Path(bridge.__file__), bridge.REPLAY,
             Path(feedback.__file__), Path(journal.__file__), Path(prediction_journal.__file__),
             Path(feedback.compiler.__file__),
             Path(feedback.compiler.core.__file__), Path(feedback.timed.__file__),
             Path(feedback.timed.pc_global_shared_cem_v23.__file__),
             Path(feedback.timed.pc_global_shared_cem_v23.cem.__file__),
             Path(feedback.temporal.__file__), Path(feedback.timed.__file__),
             Path(unroll_with_action_history.__code__.co_filename))
    return {**prior_replay["dependencies"],
            **{str(p.resolve()): old.prior.sha(p) for p in files}}


def durable_predictor(directory, meter, raw_predict, binding, cap):
    """Persist each attempted prediction before invoking the metered model."""
    directory = Path(directory)
    recorded = prediction_journal.PredictionJournal(
        directory, lambda actions: meter.predict(actions, raw_predict),
        action_dim=2, packet_length=5, token_cap=cap)
    journal.write_json(directory / "binding.json", binding)
    return recorded


def validate_contract(contract, expected_goal):
    start = np.asarray(contract["start"], dtype=float)
    root_id = contract["root_id"]
    name = contract["output_name"]
    if (start.shape != (2,) or not np.isfinite(start).all()
            or np.array_equal(start, [0.85, 0.85])
            or not isinstance(root_id, int) or isinstance(root_id, bool) or root_id < 0
            or not isinstance(name, str) or not name.startswith("pc_v23_")
            or Path(name).name != name or name in (".", "..")
            or contract.get("scope") != "development"
            or contract.get("goal_sha256") != expected_goal):
        raise ValueError("invalid task contract; old stopped root is not permitted")
    return start, root_id, name


def main():
    global PREFIX, MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-contract", required=True, type=Path)
    args_cli = parser.parse_args()
    contract_path = args_cli.task_contract.resolve(strict=True)
    contract = json.loads(contract_path.read_text())
    start, root_id, name = validate_contract(
        contract, json.loads(old.DECISIONS.read_text())["goal_sha256"])
    PREFIX = ROOT / "results" / name
    MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    outputs = [MANIFEST] + [path for arm in ARMS for group in paths(arm)
                            for path in (group if isinstance(group, list) else [group])]
    outputs += [PREFIX.with_name(PREFIX.name + f"_{arm}_journal") for arm in ARMS]
    if any(p.exists() for p in outputs):
        raise FileExistsError("prior or incomplete RGB diagnostic")
    hashes = dependencies(contract_path)
    source = json.loads(old.DECISIONS.read_text())
    sha = old.prior.sha
    journal.write_json(MANIFEST, {"dependencies": hashes, "root": root_id, "setup_start": start.tolist(), "arms": ARMS,
                              "method": feedback.METHOD, "scope": "exposed-goal RGB development diagnostic",
                              "independence": "unknown", "physical_success": "not evaluated"})
    print("Two arms; one exposed root; fixed terminal and real RGB feedback.", flush=True)
    started = time.monotonic()
    camera, calibration = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    with np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        model_hash = h.parameter_hash(model)
        if model_hash != source["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
            raise ValueError("checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if (not np.array_equal(mean.cpu().numpy(), source["action_mean"])
                or not np.array_equal(std.cpu().numpy(), source["action_std"])
                or not bool(torch.isfinite(std).all() and (std > 0).all())):
            raise ValueError("normalizer mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        if old.digest_image(goal) != source["goal_sha256"]:
            raise ValueError("goal mismatch")
        with torch.no_grad():
            goal_state, _ = h.estimate_state(goal, metadata, offset)
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
            goal_position = ((target.mean(0) @ pinv.T)[:2] * pos_std).cpu().numpy()
            torch.cuda.synchronize()
        warm_hashes = None
        for arm in ARMS:
            decisions, images, receipt_path = paths(arm)
            journal_dir = PREFIX.with_name(PREFIX.name + f"_{arm}_journal")
            journal_dir.mkdir(exist_ok=False)
            with bridge.runner.ShadowRGBBridge(np.asarray([*start, 0., 0.])) as env:
                journal.write_frame(journal_dir / "initial_rgb.npy", env.initial_rgb)
                warm, warm_times = journal.dispatch(env, old.WARMUP, journal_dir / "warmup")
                if journal.inspect(journal_dir / "warmup")["returned"] != len(old.WARMUP):
                    raise ValueError("incomplete warmup journal")
                frames = [env.initial_rgb.copy(), *warm]
                current_hashes = [old.digest_image(frame) for frame in frames]
                if warm_hashes is not None and warm_hashes != current_hashes:
                    raise ValueError("paired warmup RGB mismatch")
                warm_hashes = current_hashes
                actions, carry, rows = old.WARMUP.tolist(), None, []
                for step in range(3):
                    ticks = bridge.runner.previous.prior.context_ticks(len(frames))
                    encoded_at = time.monotonic()
                    with torch.no_grad():
                        entries = [h.encode_with_state(model, frames[t],
                                   h.causal_state(frames[t-1], frames[t], metadata, offset, .05), device)
                                   for t in ticks]
                        torch.cuda.synchronize()
                    ctx = {key: torch.cat([entry[key] for entry in entries], dim=1) for key in ("visual", "proprio")}
                    history_actions = np.asarray(actions[-10:], dtype=float)
                    history = ((torch.as_tensor(history_actions, dtype=torch.float32, device=device) - mean)
                               / std).reshape(2, 1, 10)
                    encoding_seconds = time.monotonic() - encoded_at
                    binding = {"root": root_id, "arm": arm, "step": step, "ticks": ticks,
                               "rgb_sha256": [old.digest_image(frames[t]) for t in ticks],
                               "previous_rgb_sha256": [old.digest_image(frames[t-1]) for t in ticks],
                               "history_actions": history_actions.tolist(), "goal_sha256": old.digest_image(goal),
                               "goal_position": goal_position.tolist(), "model_hash": model_hash,
                               "checkpoint_sha256": sha(h.DEFAULT_CHECKPOINT),
                               "action_mean": mean.cpu().tolist(), "action_std": std.cpu().tolist(), "carry": carry}
                    meter = bridge.runner.base.Meter(feedback.CAP)
                    trace = []

                    def raw(batch):
                        count, length, _ = batch.shape
                        packets = ((torch.as_tensor(batch, dtype=torch.float32, device=device)-mean)/std)
                        packets = packets.reshape(count, length//5, 10).transpose(0, 1)
                        with torch.no_grad():
                            out = unroll_with_action_history(model, ctx, history.expand(2, count, 10),
                                                             packets, 3)["proprio"][-1]
                            features = out.reshape(count, 256, 16)
                            pos = (features.mean(1) @ pinv.T)[:, :2] * pos_std
                            score = h.projected_cost(features, target, pinv, pos_std)
                            torch.cuda.synchronize()
                        return pos.cpu().numpy(), score.cpu().numpy()

                    recorded_predict = durable_predictor(
                        journal_dir / f"prediction_step{step}", meter, raw, binding, feedback.CAP)

                    def predict(chains):
                        requested = np.asarray(chains, dtype=float)
                        converted = requested.astype(np.float32)
                        positions, scores = recorded_predict(converted)
                        trace.append({"requested": requested.tolist(), "raw": converted.tolist(),
                                      "positions": positions.tolist(), "scores": scores.tolist(),
                                      "tokens": len(converted) * converted.shape[1] // 5})
                        return positions, scores

                    planning_at = time.monotonic()
                    result = feedback.decide(arm, root_id, step, goal_position, predict, meter, carry)
                    if (recorded_predict.tokens != meter.token_samples
                            or recorded_predict.calls != meter.calls):
                        raise ValueError("prediction journal and meter disagree")
                    selected = np.asarray(result["selected"]["chain"], dtype=float)
                    dispatched = selected[:5].astype(np.float32).tolist()
                    if selected.shape != (15 - 5 * step, 2) or meter.token_samples > feedback.CAP:
                        raise ValueError("invalid plan")
                    decision = {"binding": binding, "context_id": bridge.digest_context(binding),
                                "result": result, "trace": trace, "counts": meter.as_dict(),
                                "dispatch_float32": dispatched, "encoding_seconds": encoding_seconds,
                                "planning_seconds": time.monotonic() - planning_at,
                                "manifest_sha256": sha(MANIFEST), "full_cost_certified": False}
                    journal.write_json(decisions[step], decision)
                    sealed_ns = time.monotonic_ns()
                    observed, times = journal.dispatch(env, dispatched, journal_dir / f"step{step}")
                    if journal.inspect(journal_dir / f"step{step}")["returned"] != len(dispatched):
                        raise ValueError("incomplete dispatch journal")
                    if not times or times[0][0] < sealed_ns:
                        raise ValueError("dispatch preceded seal")
                    frames.extend(observed)
                    actions.extend(dispatched)
                    rows.append({"step": step, "decision_sha256": sha(decisions[step]), "sealed_ns": sealed_ns,
                                 "actions": dispatched, "times_ns": times,
                                 "rgb_sha256": [old.digest_image(frame) for frame in observed]})
                    carry = selected[5:].tolist() if step < 2 else None
                    print(f"{arm} step={step} tokens={meter.token_samples} sealed and executed", flush=True)
                with images.open("xb") as stream:
                    np.savez_compressed(stream, rgb=np.stack(frames))
                if len(frames) != 31 or h.parameter_hash(model) != model_hash:
                    raise ValueError("frame count or model parameters changed")
                receipt = {"root": root_id, "arm": arm, "rows": rows, "warmup_times_ns": warm_times,
                           "executed_actions": actions[15:], "images_sha256": sha(images),
                           "manifest_sha256": sha(MANIFEST), "model_hash": model_hash,
                           "encoding_calls": 9, "shared_goal_encoding_calls": 1,
                           "elapsed_seconds_since_start": time.monotonic() - started,
                           "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                                       "torch": torch.__version__, "cuda": torch.version.cuda,
                                       "gpu": torch.cuda.get_device_name(device)},
                           "full_cost_certified": False}
                journal.write_json(receipt_path, receipt)
        if hashes != dependencies(contract_path):
            raise ValueError("dependency drift")
    print("Live RGB diagnostic sealed.", flush=True)


if __name__ == "__main__":
    main()
