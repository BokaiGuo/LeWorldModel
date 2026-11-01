"""Known-root fixed-action open, last-frame-only and full-feedback CUDA replay."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_four_arm_common_rgb_root1_20261030_main as runner
from . import agent_p4_moving_multiboundary_20261028_v1 as moving
from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as gpu
from . import preflight_pusht_d2p_p0 as p0

ROOT = runner.ROOT
RUN = ROOT / "results/p1_hybrid_root1_20261031_RUN02"
PREP = ROOT / "results/p1_hybrid_root1_20261031_prepare_v2.json"
REPORT_SHA = "48db2f48c87c786d8a835a099c4a0ac009ad52e396ca768f4416bea6888ee3a4"
PREP_SHA = "2a11d5329b2333e414cd72540397257f53f617a82feeca9e1e997306d7ce9204"
CHECK = ROOT / "results/p1_hybrid_root1_saved_20261031_RUN02.json"
CHECK_SHA = "c2ae7caee38b0103e6964476e32cb534e3a2bbe52fce67127b2a0b8f8799dd06"
NAME = "p1_hybrid_context_ablation_20261031"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    output = Path(output)
    require(output.is_absolute() and output.parent == ROOT / "results"
            and output.name.startswith(NAME + "_") and not output.exists()
            and not output.is_symlink(), "exclusive output")
    for path, expected in ((PREP, PREP_SHA), (RUN / "report.json", REPORT_SHA), (CHECK, CHECK_SHA)):
        require(runner.sha(path) == expected, "pinned source")
    prepared, report, check = read(PREP), read(RUN / "report.json"), read(CHECK)
    require(check["nodes"] == 4 and check["report_sha256"] == REPORT_SHA
            and report["status"] == "recorded_known_root_not_admitted"
            and report["rows"][0]["status"] == "completed20", "completed sealed source")
    row = report["rows"][0]
    actions = np.asarray(row["executed_raw"], np.float32).reshape(4, 5, 2)
    source = prepared["source"]
    base = source["tasks"][1]
    context = deepcopy(base["root_context"])
    inputs, sources = [], {str(PREP): PREP_SHA, str(RUN / "report.json"): REPORT_SHA, str(CHECK): CHECK_SHA}
    for i, boundary in enumerate((0, 5, 10, 15)):
        seal_path = RUN / f"s1_structured_b{boundary:02}_seal.json"
        encoding_path = RUN / f"s1_structured_b{boundary:02}_encoding.json"
        trace_path = RUN / f"s1_structured_b{boundary:02}_winner/candidates.jsonl"
        seal, encoding = read(seal_path), read(encoding_path)
        receipt = report["nodes"]["s1_structured"][i]["receipt"]
        require(seal["receipt"] == receipt and runner.sha(trace_path) == receipt["trace_sha256"]
                and encoding["response"]["request_sha256"] == encoding["request"]["request_sha256"],
                "node and encoding identity")
        latent_info = encoding["response"]["latent"]
        latent_path = Path(latent_info["path"])
        require(runner.sha(latent_path) == latent_info["file_sha256"], "latent file hash")
        z = np.load(latent_path, allow_pickle=False)
        require(z.shape == (192,) and z.dtype == np.float32 and np.isfinite(z).all()
                and hashlib.sha256(z.tobytes()).hexdigest() == latent_info["array_sha256"], "latent data")
        raw10 = np.asarray(seal["identity"]["raw10"], np.float32)
        if i:
            require(np.array_equal(raw10[-5:], actions[i-1]), "executed previous packet")
            context = moving.roll(context, z, raw10, base)
        else:
            context["latent_history"][-1] = z.tolist()
            context["action_history"] = moving.normalized(base, raw10)
        task = runner.p3.local_task(base, context, seal["identity"])
        require(runner.digest(task) == seal["task_sha256"], "reconstructed planner task")
        require(np.array_equal(np.asarray(receipt["commands"], np.float32), actions[i]), "dispatched packet")
        lines = [json.loads(x) for x in trace_path.read_bytes().splitlines()]
        plans = [x["payload"]["raw"] for x in lines if x["kind"] == "optimized_candidates"]
        require(len(plans) == 1, "one candidate bank")
        plan = np.asarray(plans[0], np.float32)[receipt["selected_index"]]
        require(plan.shape == (3, 5, 2) and np.array_equal(plan[0], actions[i]), "selected plan")
        inputs.append(dict(context=deepcopy(context), goal=base["goal"], action=actions[i],
                           plan=plan, score=receipt["scores"][receipt["selected_index"]], actual=z))
        for p in (seal_path, encoding_path, trace_path, latent_path):
            sources[str(p)] = runner.sha(p)
    for step in (15, 20):
        path = RUN / f"s1_structured_plan_{step:02}.png"
        checkpoint = next(c for c in row["checkpoints"] if c["step"] == step)
        require(runner.sha(path) == checkpoint["file_sha256"], "checkpoint png")
        sources[str(path)] = runner.sha(path)
    sources[str(Path(__file__).resolve())] = runner.sha(Path(__file__).resolve())
    protocol = ROOT.parent / "docs/D2P_P1_HYBRID_CONTEXT_ABLATION_PROTOCOL_20261031.md"
    sources[str(protocol)] = runner.sha(protocol)

    with gpu.locking.gpu_lock(gpu.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(source["policy"], device=device).eval().requires_grad_(False)
        before = runner.signal.model_hash(model)
        require(before["original_state_dict"] == source["expected_model_hash"], "checkpoint hash")
        transform = p0.image_transform()
        def image_latent(step):
            with Image.open(RUN / f"s1_structured_plan_{step:02}.png") as image:
                pixels = np.asarray(image.convert("RGB")).copy()
            checkpoint = next(c for c in row["checkpoints"] if c["step"] == step)
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == checkpoint["pixel_sha256"], "image pixels")
            return model.encode({"pixels": p0._image_tensor(pixels, transform=transform, device=device)})["emb"][0, 0]
        def tensor(x):
            return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
        def predict(states, history, raw):
            token = ((raw - mean) / std).reshape(1, 1, 10)
            all_actions = torch.cat((history, token), dim=1)
            output = model.predict(states, model.action_encoder(all_actions))
            return torch.cat((states[:, 1:], output[:, -1:]), dim=1), all_actions[:, 1:]
        results = []
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            mean, std = tensor(base["action_mean"]), tensor(base["action_std"])
            actual15, actual20 = image_latent(15), image_latent(20)
            require(torch.allclose(actual15, tensor(inputs[3]["actual"]), atol=1e-6, rtol=1e-6),
                    "15 RGB encoding mismatch")
            open_states = tensor(inputs[0]["context"]["latent_history"])[None]
            open_history = tensor(inputs[0]["context"]["action_history"])[None]
            goal = tensor(base["goal"])
            for i, item in enumerate(inputs):
                states = tensor(item["context"]["latent_history"])[None]
                hist = tensor(item["context"]["action_history"])[None]
                plan_states, plan_hist = states, hist
                for packet in item["plan"]:
                    plan_states, plan_hist = predict(plan_states, plan_hist, tensor(packet))
                score = float((plan_states[0, -1] - goal).square().mean())
                require(abs(score - item["score"]) <= 1e-5, "selected plan score mismatch")
                require(torch.equal(open_history, hist), "actual and open action histories differ")
                mixed_states = torch.cat((open_states[:, :-1], tensor(item["actual"])[None, None]), dim=1)
                last_only_states, _ = predict(mixed_states, open_history, tensor(item["action"]))
                observed_states, _ = predict(states, hist, tensor(item["action"]))
                open_states, open_history = predict(open_states, open_history, tensor(item["action"]))
                actual = tensor(inputs[i+1]["actual"]) if i < 3 else actual20
                pred, feedback, last_only = open_states[0, -1], observed_states[0, -1], last_only_states[0, -1]
                results.append(dict(start=i*5, end=(i+1)*5, replay_score=score,
                    open_rgb_mse=float((pred-actual).square().mean()),
                    last_only_rgb_mse=float((last_only-actual).square().mean()),
                    feedback_rgb_mse=float((feedback-actual).square().mean()),
                    open_goal_mse=float((pred-goal).square().mean()),
                    last_only_goal_mse=float((last_only-goal).square().mean()),
                    feedback_goal_mse=float((feedback-goal).square().mean()),
                    actual_goal_mse=float((actual-goal).square().mean()),
                    open_latent=pred.cpu().tolist(), last_only_latent=last_only.cpu().tolist(),
                    feedback_latent=feedback.cpu().tolist(),
                    actual_latent=actual.cpu().tolist()))
        torch.cuda.synchronize(device)
        after = runner.signal.model_hash(model)
        require(before == after and all(runner.sha(Path(p)) == h for p, h in sources.items()), "source/model drift")
        result = dict(schema=NAME, status="completed_known_root_diagnostic", host=host, rows=results,
                      goal=base["goal"], model_before=before, model_after=after, sources=sources,
                      counts=dict(predict=24, action_encoder=24, encode=2),
                      independent_confirmation=False, P1_admission=False, equal_budget_comparison=False)
        payload = json.dumps(result, sort_keys=True, allow_nan=False, indent=2).encode() + b"\n"
        with output.open("xb") as f:
            require(f.write(payload) == len(payload), "write")
    print(json.dumps({"output": str(output), "sha256": runner.sha(output), "rows": len(results)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Compare same actions with open, last-frame-only, and full actual context on host CUDA.", flush=True)
    run(args.output)
