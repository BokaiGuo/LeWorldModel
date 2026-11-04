"""Three-decision PushT RGB feedback with v4 and two-start CEM."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_pusht_saved_context_adapter_20261104_codex as saved
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import audit_pusht_d2p_packet_contract as phys
from experiments.spectral_reachability import pusht_d2p_packet_contract as packets
from experiments.spectral_reachability import pusht_rgb_observability_guard as guardlib
from experiments.spectral_reachability import pc_response_temporal_two_start_cem_v1 as baseline
from experiments.spectral_reachability import run_pusht_sampling_comparison_20261027_v2 as metrics
from experiments.spectral_reachability import run_pusht_d2p_p1_physical_calibration as pose
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT, HOST = saved.ROOT, saved.HOST
PROTOCOL = ROOT.parent / "docs/PC_V4_PUSHT_RGB_FEEDBACK_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v4_pusht_rgb_feedback_dev_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
PHYS_REPORT = ROOT / "results/pusht_d2p_packet_contract_20261026_physical_v1/report.json"
PHYS_PROTOCOL = PHYS_REPORT.with_name("protocol.json")
ARMS = ("response", "smooth_cem")


def path(arm, name):
    return PREFIX.with_name(PREFIX.name + f"_{arm}_{name}")


def save_npz(file, **items):
    with file.open("xb") as stream:
        np.savez_compressed(stream, **items)
        stream.flush()
        os.fsync(stream.fileno())


class Guarded:
    def __init__(self, env, root_id, stream):
        self.env, self.unwrapped, self.stream = env, env.unwrapped, stream
        self.guard = guardlib.RgbObservabilityGuard(root_id=root_id, agent_rgb=(65, 105, 225))
        self.count = 0
        self.guard.observe(phys.render_without_advancing(env), primitive_step=0)

    def render(self):
        return self.env.render()

    def step(self, command):
        command = self.guard.dispatch(command)
        self.stream.write(json.dumps({"kind": "attempt", "step": self.count + 1,
                                      "command": command.tolist()}, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        transition = self.env.step(command)
        self.count += 1
        rgb = phys.render_without_advancing(self.env)
        state = packets.state_of(self.env)
        event = {"kind": "returned", "step": self.count, "state": state.tolist(),
                 "rgb_sha256": phys._array_hash(rgb), "terminated": bool(transition[2]),
                 "truncated": bool(transition[3])}
        self.stream.write(json.dumps(event, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        if not transition[2] and not transition[3]:
            self.guard.observe(rgb, primitive_step=self.count)
        return transition


def main():
    outputs = [MANIFEST, RESULT] + [path(a, "execution.json") for a in ARMS] + [
        path(a, f"decision{i}.json") for a in ARMS for i in range(3)] + [
        path(a, f"context{i}.npz") for a in ARMS for i in range(4)] + [path(a, "events.jsonl") for a in ARMS]
    if any(p.exists() for p in outputs):
        raise FileExistsError("existing or partial output")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != HOST.UUID:
        raise ValueError("cooperative GPU UUID required")
    prior_manifest, prior_result = json.loads(saved.MANIFEST.read_text()), json.loads(saved.RESULT.read_text())
    prior_audit = json.loads(saved.PREFIX.with_name(saved.PREFIX.name + "_audit.json").read_text())
    seal.verify(prior_manifest["seal"])
    if prior_result["manifest_sha256"] != HOST.bridge.sha_file(saved.MANIFEST) or prior_audit["result_sha256"] != HOST.bridge.sha_file(saved.RESULT):
        raise ValueError("saved-context diagnostic binding mismatch")
    receipt = json.loads(PHYS_REPORT.read_text())
    packets.validate_receipt(receipt)
    if receipt["protocol_sha256"] != HOST.bridge.sha_file(PHYS_PROTOCOL):
        raise ValueError("physical protocol mismatch")
    rowlist, data_info = phys.select_rows(phys.DEFAULT_DATA, (0,))
    source, bindings, config = saved.checked.validate_inputs(saved.SOURCE, saved.POLICY)
    deps = seal.capture([Path(__file__), PROTOCOL, PHYS_REPORT, PHYS_PROTOCOL, saved.MANIFEST,
        saved.RESULT, saved.PREFIX.with_name(saved.PREFIX.name + "_audit.json"), Path(baseline.__file__),
        Path(phys.__file__), Path(packets.__file__), Path(guardlib.__file__), Path(metrics.__file__),
        Path(pose.__file__), *saved.checked.sources_before_allocate(source, config), *[Path(x) for x in bindings]])
    HOST.write_new(MANIFEST, {"seal": deps, "data_selection": data_info, "arms": ARMS,
        "scope": "exposed episode zero; diagnostic goal; training provenance unknown"})
    print("PushT RGB feedback: two independent arms, episode 0, fixed 15-step terminal.", flush=True)
    row, sha = rowlist[0], HOST.bridge.sha_file
    mean = np.tile(np.asarray(saved.p0.PUBLIC_ACTION_MEAN, np.float32), 5)
    std = np.tile(np.asarray(saved.p0.PUBLIC_ACTION_STD, np.float32), 5)
    summaries = []
    with HOST.lock.gpu_lock(HOST.LOCK):
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if HOST.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != HOST.canonical_gpu_uuid(HOST.UUID):
            raise ValueError("unexpected GPU")
        device = torch.device("cuda:0")
        model = saved.load_compatible_pretrained(saved.POLICY, device="cuda").eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        before = saved.checked.tensor_state_hash(model)
        lower = torch.as_tensor((-1 - mean) / std, device=device)
        upper = torch.as_tensor((1 - mean) / std, device=device)
        transform = saved.p0.image_transform()
        for arm in ARMS:
            env = phys.official_env_factory()
            status, failure, decisions, outcome = "completed", None, [], []
            guarded, goal_check = None, None
            try:
                _, info = env.reset(seed=0, options={"state": row["states"][0].copy(),
                                                     "goal_state": phys.DIAGNOSTIC_GOAL.copy()})
                goal_check = phys._goal_contract(env, info, row, phys.official_env_factory)
                goal_rgb = np.asarray(info["goal"], np.uint8).copy()
                with path(arm, "events.jsonl").open("x", encoding="utf-8") as journal:
                    guarded = Guarded(env, phys._array_hash(row["states"][0]), journal)
                    history = packets.BoundaryHistory(guarded, root_state=row["states"][0],
                                                      goal=phys.DIAGNOSTIC_GOAL)
                    for packet in row["actions"][:15].reshape(3, 5, 2):
                        history.execute(packet)
                    if history.step != 15:
                        raise ValueError("warmup mismatch")
                    with torch.no_grad():
                        goal = saved.p0._goal_latent(model, goal_rgb, transform=transform, device=device)

                    def snapshot(step):
                        frames = np.stack(history.frames[-3:])
                        commands = history.packets[-3:].reshape(15, 2).astype(np.float32)
                        state = packets.state_of(env)
                        save_npz(path(arm, f"context{step}.npz"), frames=frames, history=commands, goal=goal_rgb)
                        outcome.append({"state": state.tolist(),
                            "official": metrics.official_checked(env, state, phys.DIAGNOSTIC_GOAL),
                            "pose": pose._pose_label(state, phys.DIAGNOSTIC_GOAL)})
                        return frames, commands

                    carry = None
                    for step in range(3):
                        frames, commands = snapshot(step)
                        with torch.no_grad():
                            images = saved.p0._frames_tensor(list(frames), transform=transform, device=device)
                            encoded = model.encode({"pixels": images})["emb"]
                            hist = saved.live.adapter.raw_to_packets(commands[None], mean, std, device=device)
                            chart = saved.live.adapter.goal_chart(encoded, goal, 2027111800 + step)
                            target = (goal @ chart).cpu().numpy()
                        meter, trace = Meter(1920), []

                        def predict(actions):
                            requested = np.asarray(actions, dtype=float)
                            raw = requested.astype(np.float32)
                            def evaluate(batch):
                                future = saved.live.adapter.raw_to_packets(batch, mean, std, device=device)
                                with torch.no_grad():
                                    terminal = saved.live.adapter.cached_terminal(model, encoded, hist, future, lower, upper)
                                    position, score = terminal @ chart, (terminal - goal).square().mean(dim=1)
                                    torch.cuda.synchronize()
                                return position.cpu().numpy(), score.cpu().numpy()
                            positions, scores = meter.predict(raw, evaluate)
                            trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                                          "positions": positions.tolist(), "scores": scores.tolist(),
                                          "tokens": len(raw) * raw.shape[1] // 5})
                            return positions, scores

                        start = time.monotonic()
                        method = saved.method if arm == "response" else baseline
                        result = method.decide(arm, 0, step, target, predict, meter, carry)
                        selected = np.asarray(result["selected"]["chain"], np.float32)
                        decision = path(arm, f"decision{step}.json")
                        HOST.write_new(decision, {"manifest_sha256": sha(MANIFEST),
                            "context_sha256": sha(path(arm, f"context{step}.npz")), "step": step,
                            "carry_in": carry, "chart": chart.cpu().tolist(), "target": target.tolist(),
                            "result": result, "trace": trace, "counts": meter.as_dict(),
                            "planning_seconds": time.monotonic()-start, "dispatch": selected[:5].tolist()})
                        sealed_ns = time.monotonic_ns()
                        history.execute(selected[:5])
                        decisions.append({"step": step, "decision_sha256": sha(decision), "sealed_ns": sealed_ns,
                            "executed_ns": time.monotonic_ns(), "dispatch": selected[:5].tolist()})
                        carry = selected[5:].tolist() if step < 2 else None
                        print(f"{arm} step={step} tokens={meter.token_samples}", flush=True)
                    snapshot(3)
                    guard_receipt = guarded.guard.receipt()
            except Exception as exc:
                status, failure = "stopped", {"type": type(exc).__name__, "message": str(exc)}
                guard_receipt = guarded.guard.receipt() if guarded is not None else None
                print(f"{arm} stopped: {failure}", flush=True)
            finally:
                env.close()
            HOST.write_new(path(arm, "execution.json"), {"manifest_sha256": sha(MANIFEST), "arm": arm,
                "status": status, "failure": failure, "decisions": decisions, "outcomes": outcome,
                "goal_check": goal_check, "guard": guard_receipt,
                "events_sha256": sha(path(arm, "events.jsonl")) if path(arm, "events.jsonl").exists() else None})
            summaries.append({"arm": arm, "status": status, "failure": failure,
                              "execution_sha256": sha(path(arm, "execution.json"))})
        after = saved.checked.tensor_state_hash(model)
        if before != after:
            raise ValueError("model drift")
    seal.verify(deps)
    HOST.write_new(RESULT, {"manifest_sha256": sha(MANIFEST), "rows": summaries, "model_before": before,
        "model_after": after, "cuda_allocation": allocation, "scope": "exposed root development only"})


if __name__ == "__main__":
    main()
