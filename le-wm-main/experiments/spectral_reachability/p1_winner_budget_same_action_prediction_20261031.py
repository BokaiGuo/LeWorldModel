"""Post-hoc known-root one-packet model/actual RGB prediction preparation.

No physical execution. This is not a P1, equal-budget or generalization test.
The run command requires explicit host CUDA authorization and never uses CPU
as a substitute for neural inference.
"""
import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from experiments.spectral_reachability import codex_four_arm_common_rgb_launcher_20261030 as prior
from experiments.spectral_reachability import p1_four_arm_winner_budget_cuda_20261031 as winner

NAME = "p1_winner_budget_same_action_prediction_20261031"
ROOT = prior.ROOT
RUN = ROOT / "results/p1_four_arm_winner_budget_cuda_20261031_RUN01"
PREP = ROOT / "results/p1_four_arm_winner_budget_cuda_20261031_prepare_RUN01.json"
REPORT_SHA = "96c6976a2cdb9f808ef69fe7f798a8874e58209f694862b26f909fdceb92dcf0"
PREP_SHA = "da6632fa6628b2c67bc1be95bde55813b716004ea0ed818f67b528bc0ed9ba7c"
MODEL_SHA = "4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546"
ARMS = ("structured", "native", "random", "immediate_progress")
BOUNDARIES = (0, 5, 10, 15)
AUTH = "MAIN_AUTHORIZES_FOUR_ARM_FIXED_ACTION_PREDICTION_CUDA_ONLY"
need = prior.require


def sha(path):
    return prior.sha(Path(path))


def f32(value, shape):
    a = np.asarray(value)
    need(a.shape == shape and np.issubdtype(a.dtype, np.number) and np.isfinite(a).all()
         and np.array_equal(a, a.astype(np.float32)), "literal finite f32")
    return a.astype(np.float32)


def png(receipt):
    p = Path(receipt["path"])
    need(p.parent == RUN and p == p.resolve() and p.is_file() and not p.is_symlink()
         and sha(p) == receipt["file_sha256"], "pinned actual PNG")
    with Image.open(p) as image:
        need(image.format == "PNG" and image.mode == "RGB" and image.size == (224, 224), "RGB ABI")
        pixels = np.asarray(image).copy()
    need(hashlib.sha256(pixels.tobytes()).hexdigest() == receipt["pixel_sha256"], "RGB pixel identity")
    return pixels


def saved_inputs():
    """Rebuild each original node task and bind its exact executed packet."""
    need(sha(RUN / "report.json") == REPORT_SHA, "fixed report hash")
    winner.configure()
    p = prior.verify(str(PREP), PREP_SHA)
    r = prior.signal.read(RUN / "report.json", 2*1024**2)
    need(r["status"] == "recorded_known_root_not_admitted" and not r["errors"]
         and r["prepare_sha256"] == PREP_SHA and r["model_before"] == r["model_after"]
         and r["model_before"]["original_state_dict"] == MODEL_SHA, "report/model identity")
    journal = RUN / "journal.jsonl"
    need(sha(journal) == r["journal_sha256"], "journal hash")
    events = [json.loads(line) for line in journal.read_bytes().splitlines()]
    previous = None
    for index, e in enumerate(events):
        need(e["index"] == index and e["previous_sha256"] == previous
             and e["event_sha256"] == prior.old.source.digest({k: v for k, v in e.items() if k != "event_sha256"}),
             "journal chain")
        previous = e["event_sha256"]
    need(previous == r["journal_chain_tail"] and events[-1]["kind"] == "run_finished", "journal terminal")
    need([row["arm"] for row in r["rows"]] == list(ARMS), "four fixed arms")
    base = p["source"]["tasks"][0]
    result = []
    for row in r["rows"]:
        arm, eid = row["arm"], row["episode_id"]
        need(eid == "s0_" + arm and row["status"] == "completed20" and row["control_returned"] == 20,
             "complete same-episode trajectory")
        actions = f32(row["executed_raw"], (20, 2))
        need((np.abs(actions) <= 1).all(), "action box")
        returned = [e for e in events if e["kind"] == "control_return" and e["episode_id"] == eid]
        warm = [e for e in events if e["kind"] == "warmup_return" and e["episode_id"] == eid]
        need([e["step"] for e in warm] == list(range(1, 11))
             and all(not e["terminated"] and not e["truncated"] for e in warm), "warmup history")
        warm_actions = f32([e["command"] for e in warm], (10, 2))
        need([e["step"] for e in returned] == list(range(1, 21))
             and f32([e["command"] for e in returned], (20, 2)).tobytes() == actions.tobytes()
             and all(not e["terminated"] and not e["truncated"] for e in returned), "physical action binding")
        cps = row["checkpoints"]
        need([c["step"] for c in cps] == [0, 5, 10, 15, 20], "checkpoint schedule")
        for cp in cps:
            png(cp)
        context = deepcopy(base["root_context"])
        initial_history = None
        for i, boundary in enumerate(BOUNDARIES):
            stem = f"{eid}_b{boundary:02}"
            seal = prior.signal.read(RUN / (stem + "_seal.json"), 4*1024**2)
            encpath = RUN / (stem + "_encoding.json")
            enc = prior.signal.read(encpath, 4*1024**2)
            identity, obs = seal["identity"], seal["identity"]["observation"]
            need(identity["episode_id"] == eid and identity["arm"] == arm and identity["boundary"] == boundary
                 and obs["png_sha256"] == cps[i]["file_sha256"] and obs["pixel_sha256"] == cps[i]["pixel_sha256"]
                 and obs["encoding_sha256"] == prior.old.source.digest(enc), "boundary observation identity")
            response = enc["response"]
            ld = response["latent"]
            lp = Path(ld["path"])
            need(enc["actual_step"] == boundary and response["device_type"] == "cuda"
                 and response["model_hash_before"] == response["model_hash_after"] == MODEL_SHA
                 and response["model_unchanged"] is True and lp.parent == RUN / (stem + "_encode")
                 and lp == lp.resolve() and not lp.is_symlink() and sha(lp) == ld["file_sha256"], "encoded identity")
            z = np.load(lp, allow_pickle=False)
            need(z.shape == (192,) and z.dtype == np.float32 and np.isfinite(z).all()
                 and hashlib.sha256(z.tobytes()).hexdigest() == ld["array_sha256"]
                 and prior.old.packet_digest(z.tolist()) == obs["observed_latent_sha256"], "saved latent")
            history = f32(identity["raw10"], (10, 2))
            if boundary == 0:
                need(history.tobytes() == warm_actions.tobytes(), "actual warm history")
                initial_history = history.copy()
                context["latent_history"][-1] = z.tolist()
                context["action_history"] = prior.rgb.moving.normalized(base, history)
            else:
                expected_history = np.concatenate((initial_history, actions[:boundary]))[-10:]
                need(history.tobytes() == expected_history.tobytes(), "own actual raw history")
                context = prior.rgb.moving.roll(context, z, history, base)
            task = prior.p3.local_task(base, context, identity)
            need(prior.digest(task) == seal["task_sha256"], "exact original node context")
            receipt = seal["receipt"]
            recorded = r["nodes"][eid][i]
            need(recorded["boundary"] == boundary and recorded["status"] == "sealed"
                 and recorded["receipt"] == receipt, "report-bound node receipt")
            encoded_events = [e for e in events if e["kind"] == "boundary_encoded"
                              and e["episode_id"] == eid and e["actual_step"] == boundary]
            need(len(encoded_events) == 1 and encoded_events[0]["index"] == obs["encoded_index"]
                 and encoded_events[0]["encoding_sha256"] == obs["encoding_sha256"], "journal-bound encoding")
            commands = f32(receipt["commands"], (5, 2))
            need(commands.tobytes() == actions[boundary:boundary+5].tobytes(), "selected executed packet")
            trace = Path(receipt["trace"])
            need(trace.parent == RUN / (stem + "_winner") and sha(trace) == receipt["trace_sha256"],
                 "selected candidate trace")
            result.append(dict(arm=arm, boundary=boundary, task=task, actions=commands,
                               target=cps[i+1], source=cps[i], task_sha256=seal["task_sha256"],
                               seal_sha256=sha(RUN / (stem+"_seal.json"))))
    need(len(result) == 16 and sha(RUN / "report.json") == REPORT_SHA, "all nodes stable")
    return p, r, result


def predict_one(model, torch, device, node, emit=lambda *_: None):
    """One actual raw packet, using the same history and normalizer as planning."""
    t = node["task"]
    states = torch.tensor(t["root_context"]["latent_history"], dtype=torch.float32, device=device)[None]
    history = torch.tensor(t["root_context"]["action_history"], dtype=torch.float32, device=device)[None]
    mean = torch.tensor(t["action_mean"], dtype=torch.float32, device=device)
    std = torch.tensor(t["action_std"], dtype=torch.float32, device=device)
    need(bool((std > 0).all().item()), "positive normalizer")
    action = torch.tensor(node["actions"], dtype=torch.float32, device=device)
    token = ((action-mean)/std).reshape(1, 1, 10)
    emit("action_encoder_intent")
    encoded = model.action_encoder(torch.cat((history, token), dim=1))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    emit("action_encoder_return")
    emit("predict_intent")
    predicted = model.predict(states, encoded)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    emit("predict_return")
    need(tuple(predicted.shape) == (1, 3, 192) and bool(torch.isfinite(predicted).all().item()), "prediction ABI")
    return predicted[0, -1].detach().cpu().numpy().copy()


def errors(pred, actual, goal):
    pred, actual, goal = [f32(x, (192,)).astype(np.float64) for x in (pred, actual, goal)]
    residual = pred - actual
    actual_goal = actual - goal
    return dict(residual=residual.tolist(), mse=float(np.mean(residual**2)),
                l2=float(np.linalg.norm(residual)),
                predicted_goal_mse=float(np.mean((pred-goal)**2)),
                actual_goal_mse=float(np.mean(actual_goal**2)),
                goal_mse_cross_term=float(2*np.mean(actual_goal*residual)))


def run(output, authorization):
    need(authorization == AUTH, "explicit HOST CUDA authorization")
    started = time.monotonic()
    code_sha = sha(Path(__file__))
    prepared, report, nodes = saved_inputs()
    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT / "results" and
         output.name.startswith(NAME + "_") and not os.path.lexists(output), "exclusive output")
    need(shutil.disk_usage(output.parent).free >= 216*1024**2, "output disk floor and reserve")
    output.mkdir()
    prior.signal.fsync_dir(output.parent)
    journal_path = output / "journal.jsonl"
    def emit(stage, **data):
        line = json.dumps(dict(stage=stage, seconds=time.monotonic()-started, **data),
                          sort_keys=True, allow_nan=False).encode()+b"\n"
        with journal_path.open("ab") as stream:
            need(stream.write(line) == len(line), "journal write")
            stream.flush()
            os.fsync(stream.fileno())
    emit("start", report_sha256=REPORT_SHA, prepare_sha256=PREP_SHA, nodes=16,
         inherited_cost="unknown", gpu_time="unknown", flops="unknown")
    try:
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from experiments import checkpoint_compat, gpu_preflight
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as visual
        host = gpu_preflight.require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        def limits():
            need(time.monotonic()-started <= 1800 and
                 torch.cuda.memory_allocated(device) <= 3*1024**3 and
                 torch.cuda.memory_reserved(device) <= 4*1024**3, "wall/CUDA cap")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        emit("model_load_intent")
        model = checkpoint_compat.load_compatible_pretrained(prepared["source"]["policy"], device=device).eval().requires_grad_(False)
        before = prior.signal.model_hash(model)
        need(before["original_state_dict"] == MODEL_SHA and before == report["model_before"]
             and all(not x.training for x in model.modules())
             and all(p.device == device and not p.requires_grad for p in model.parameters())
             and all(b.device == device for b in model.buffers()), "frozen CUDA model identity")
        need(all(str(Path(inspect.getfile(type(x))).resolve()) in prepared["sources"]
                 for x in model.modules()), "model classes source closure")
        limits()
        emit("model_load_return", model_hash=before, host=host)
        results = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            transform = visual.image_transform()
            for n in nodes:
                limits()
                pred = predict_one(model, torch, device, n,
                    lambda stage: emit(stage, arm=n["arm"], boundary=n["boundary"]))
                limits()
                emit("target_encode_intent", arm=n["arm"], step=n["boundary"]+5)
                pixels = visual._image_tensor(png(n["target"]), transform=transform, device=device)
                encoded = model.encode({"pixels": pixels})["emb"]
                need(tuple(encoded.shape) == (1, 1, 192) and bool(torch.isfinite(encoded).all().item()), "RGB encoding ABI")
                actual = encoded[0, 0].detach().cpu().numpy().copy()
                torch.cuda.synchronize(device)
                limits()
                emit("target_encode_return", arm=n["arm"], step=n["boundary"]+5)
                results.append(dict(arm=n["arm"], boundary=n["boundary"], target_step=n["boundary"]+5,
                    task_sha256=n["task_sha256"], seal_sha256=n["seal_sha256"],
                    action_float32_hex=n["actions"].tobytes().hex(),
                    source_png_sha256=n["source"]["file_sha256"], target_png_sha256=n["target"]["file_sha256"],
                    predicted=pred.tolist(), actual_rgb_latent=actual.tolist(),
                    **errors(pred, actual, n["task"]["goal"])))
        need(prior.signal.model_hash(model) == before and sha(RUN/"report.json") == REPORT_SHA
             and sha(Path(__file__)) == code_sha,
             "post-run model and report identity")
        saved_inputs()
        out = dict(schema=NAME+".result.v1", status="exploratory_complete", records=results,
                   counts=dict(action_encoder=16,predict=16,encoder=16),
                   analysis_source_sha256=code_sha, model_hash=before,
                   end_to_end_cost="unknown", P1_admission=False, fairness_proven=False,
                   independent_confirmation=False)
        path = output/"result.json"
        with path.open("xb") as stream:
            stream.write(json.dumps(out, sort_keys=True, allow_nan=False).encode())
            stream.flush(); os.fsync(stream.fileno())
        prior.signal.fsync_dir(output)
        emit("complete", result_sha256=sha(path))
        return {"status":out["status"],"result_sha256":sha(path),"P1_admission":False}
    except BaseException as exc:
        try:
            emit("failure", type=type(exc).__name__, message=str(exc)[:1000])
        except BaseException as secondary:
            exc.add_note("journal failure: "+repr(secondary))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    p = sub.add_parser("run")
    p.add_argument("--output", required=True)
    p.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        _, _, nodes = saved_inputs()
        result = dict(nodes=len(nodes), target_steps=[5,10,15,20], neural_execution=False, P1_admission=False)
    else:
        result = run(args.output, args.authorization)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
