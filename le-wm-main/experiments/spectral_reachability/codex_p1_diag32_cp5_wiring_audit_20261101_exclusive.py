"""Saved-input audit plus fixed, separately charged CUDA numerical spot checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from . import codex_p1_diag32_cp5_live_verify_20261101_exclusive as saved

live, ROOT = saved.live, saved.ROOT
NAME = "codex_p1_diag32_cp5_wiring_audit_20261101_exclusive"
require, sha = live.require, live.sha


def inputs():
    verification = saved.verify()
    value = json.loads(live.PRESEAL.read_bytes())
    report = json.loads((saved.DIRECTORY / "report.json").read_bytes())
    task, _, _, _ = live.fixed.blind.load(0)
    _, physical_data, expected_root, _, _, _, _ = live.physical.inputs()
    ref = physical_data["references"][0]
    events = saved.analysis.journal((saved.DIRECTORY / "journal.jsonl").read_bytes(), report["journal_chain_tail"])
    rows = []
    for row in report["rows"]:
        role = row["role"]
        root = row["observations"][0]
        require(all(root[k] == expected_root[k] for k in ("state", "snapshot", "pixel_sha256", "goal", "goal_rgb_sha256"))
                and np.array_equal(np.asarray(root["goal"]), ref["goal"]), "root/physical goal")
        require(live.engine.digest(task["goal"]) == value["goal_sha256"] == live.engine.digest(
            json.loads(live.physical.SOURCE_PREP.read_bytes())["source"]["tasks"][0]["goal"]), "model goal")
        prefix = np.asarray(row["executed_raw"][:5], np.float32)
        require(np.array_equal(prefix, np.asarray(value["prefixes"][role], np.float32)), "actual prefix")
        cp5 = row["cp5"]
        identity = {"role": role, "boundary": 5, "pixel_sha256": cp5["receipt"]["pixel_sha256"],
                    "prefix": prefix.tolist(), "preseal_sha256": live.PRESEAL_SHA}
        rebuilt = live.fixed.feedback_task(task, value["prefixes"][role], prefix,
                                          np.asarray(value["raw_warmup10"], np.float32)[5:],
                                          cp5["latent"], identity)
        require(live.engine.digest(rebuilt) == cp5["task_sha256"] and rebuilt["goal"] == task["goal"]
                and rebuilt["root_context"]["latent_history"][:2] == task["root_context"]["latent_history"][1:]
                and rebuilt["root_context"]["latent_history"][2] == cp5["latent"]
                and rebuilt["root_context"]["action_history"] == value["cp5_action_histories_expected"][role],
                "cp5 task/goal/history")
        # Check exact action/score/gradient dispatch schedule, not only aggregate counts.
        neural = [e for e in events if e.get("role") == role and e["kind"] in
                  ("neural_attempt", "neural_return", "candidate_score")]
        require(neural[0]["kind"] == "neural_attempt" and neural[0]["resource"] == "cp5_rgb_encode"
                and neural[1]["kind"] == "neural_return" and neural[1]["resource"] == "cp5_rgb_encode", "encoding")
        pos = 2
        scores = []
        for c in range(8):
            for i in range(33):
                for packet in range(2):
                    for kind in ("action_encoder", "predict"):
                        for status in ("neural_attempt", "neural_return"):
                            e = neural[pos]
                            require(e["kind"] == status and e["resource"] == kind and
                                    e["context"] == {"candidate": c, "iteration": i, "packet": packet},
                                    "forward dispatch order")
                            pos += 1
                e = neural[pos]
                require(e["kind"] == "candidate_score" and e["candidate"] == c and e["iteration"] == i,
                        "score order")
                scores.append({k: e[k] for k in ("candidate", "iteration", "q2", "actions")})
                pos += 1
                if i < 32:
                    for status in ("neural_attempt", "neural_return"):
                        e = neural[pos]
                        require(e["kind"] == status and e["resource"] == "reverse" and
                                e["context"] == {"candidate": c, "iteration": i}, "reverse order")
                        pos += 1
        require(pos == len(neural), "extra model events")
        initial = [scores[c * 33] for c in range(8)]
        bank = np.asarray(value["initial_suffix_bank"], np.float32)
        require(all(np.array_equal(np.asarray(e["actions"], np.float32), bank[c]) for c, e in enumerate(initial)),
                "same initial suffix bank")
        chosen = min(scores, key=lambda x: (x["q2"], x["candidate"], x["iteration"]))
        require(chosen == row["suffix"], "selected score/actions")
        image = Path(cp5["receipt"]["path"])
        rows.append({"role": role, "task": rebuilt, "latent": cp5["latent"], "png": str(image),
                     "png_sha256": cp5["receipt"]["file_sha256"], "pixel_sha256": cp5["receipt"]["pixel_sha256"],
                     "first": scores[0], "first_next": scores[1], "winner": chosen})
    return verification, value, rows


def contract():
    _, value, rows = inputs()
    bindings = {str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
                str(Path(saved.__file__).resolve()): sha(Path(saved.__file__).resolve()),
                str(saved.CONTRACT): sha(saved.CONTRACT), str(saved.DIRECTORY / "report.json"): saved.REPORT_SHA,
                str(saved.DIRECTORY / "journal.jsonl"): sha(saved.DIRECTORY / "journal.jsonl")}
    bindings.update({r["png"]: r["png_sha256"] for r in rows})
    return {"schema": NAME + ".protocol.v1", "source_preseal_sha256": live.PRESEAL_SHA,
            "source_report_sha256": saved.REPORT_SHA, "roles": value["roles"],
            "checks_per_role": "reencode actual cp5 RGB; recompute candidate 0 iteration 0 score and gradient update; recompute chosen score",
            "limits": {"encode": 3, "action_encoder": 12, "predict": 12, "reverse": 3, "wall_seconds": 600},
            "numerical_tolerance": {"latent_max_abs": 1e-5, "score_abs": 1e-5, "updated_action_max_abs": 1e-5},
            "physical_calls": 0, "development_audit_not_confirmation": True, "bindings": bindings}


def run(prep, expected, output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    require(sha(prep) == expected and json.loads(prep.read_bytes()) == contract(), "audit contract")
    _, value, rows = inputs()
    output.mkdir()
    started = time.monotonic()
    ledger = output / "events.jsonl"
    counts = dict.fromkeys(("encode", "action_encoder", "predict", "reverse"), 0)
    returns = dict(counts)
    def emit(event):
        payload = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
        require(time.monotonic() - started < 600 and (ledger.stat().st_size if ledger.exists() else 0)
                + len(payload) <= 1024 * 1024, "audit limit")
        with ledger.open("ab") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    def call(kind, role, fn):
        cap = {"encode": 3, "action_encoder": 12, "predict": 12, "reverse": 3}[kind]
        require(counts[kind] < cap, "audit call cap")
        counts[kind] += 1
        emit({"event": "attempt", "kind": kind, "role": role})
        result = fn()
        returns[kind] += 1
        emit({"event": "return", "kind": kind, "role": role})
        return result
    with live.physical.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        host = require_cuda_allocation("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        model = load_compatible_pretrained(value["model_policy"], device="cuda:0").eval().requires_grad_(False)
        before = live.engine.signal.model_hash(model)
        require(before["original_state_dict"] == value["model_sha256"], "model")
        results = []
        with torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            for row in rows:
                role, task = row["role"], row["task"]
                require(sha(row["png"]) == row["png_sha256"], "frame drift")
                with Image.open(row["png"]) as image:
                    pixels = np.asarray(image).copy()
                require(hashlib.sha256(pixels.tobytes()).hexdigest() == row["pixel_sha256"], "pixels")
                tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device="cuda:0")
                pixels_tensor = live.p0._image_tensor(pixels, transform=live.p0.image_transform(), device="cuda:0")
                with torch.inference_mode():
                    latent = call("encode", role, lambda: model.encode({"pixels": pixels_tensor}))["emb"][0, 0]
                latent_error = float(torch.max(torch.abs(latent - tensor(row["latent"]))).item())
                states = tensor(task["root_context"]["latent_history"])[None]
                history = tensor(task["root_context"]["action_history"])[None]
                goal, mean, std = tensor(task["goal"])[None], tensor(task["action_mean"]), tensor(task["action_std"])
                def score(action):
                    z, past = states, history
                    for packet in action:
                        context = torch.cat((past, ((packet - mean) / std).reshape(1, 1, 10)), 1)
                        encoded = call("action_encoder", role, lambda: model.action_encoder(context))
                        pred = call("predict", role, lambda: model.predict(z, encoded))
                        z = torch.cat((z, pred[:, -1:]), 1)[:, -3:]
                        past = context[:, -2:]
                    return (z[:, -1] - goal).square().mean()
                action = tensor(row["first"]["actions"]).clone().requires_grad_(True)
                first_score = score(action)
                gradient = call("reverse", role, lambda: torch.autograd.grad(first_score, action)[0])
                next_action = (action - .05 * gradient).clamp(-1, 1).detach()
                step_error = float(torch.max(torch.abs(next_action - tensor(row["first_next"]["actions"]))).item())
                with torch.no_grad():
                    winner_score = score(tensor(row["winner"]["actions"]))
                first_error = abs(float(first_score.item()) - row["first"]["q2"])
                winner_error = abs(float(winner_score.item()) - row["winner"]["q2"])
                result = {"role": role, "latent_max_abs_error": latent_error, "first_q2_abs_error": first_error,
                          "winner_q2_abs_error": winner_error, "first_update_max_abs_error": step_error}
                emit({"event": "numerical_check", **result})
                results.append(result)
                require(max(latent_error, first_error, winner_error, step_error) <= 1e-5, "numerical mismatch")
        torch.cuda.synchronize()
        after = live.engine.signal.model_hash(model)
        require(before == after and counts == returns == {"encode": 3, "action_encoder": 12, "predict": 12, "reverse": 3}
                and sha(prep) == expected and json.loads(prep.read_bytes()) == contract(), "final integrity")
        return live.fixed.blind.write_new(output / "report.json",
            {"schema": NAME + ".report.v1", "status": "passed_spot_checks", "checks": results,
             "counts": counts, "returns": returns, "model_before": before, "model_after": after, "host": host,
             "protocol_sha256": expected, "events_sha256": sha(ledger), "physical_calls": 0,
             "limitations": "numerical spot checks, not all 264 gradients; known-root post-outcome audit"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prep = ROOT / "results" / (NAME + "_PREP_" + args.tag + ".json")
    output = ROOT / "results" / (NAME + "_RUN_" + args.tag)
    require(not os.path.lexists(prep) and not os.path.lexists(output), "exclusive output")
    digest = live.fixed.blind.write_new(prep, contract())
    print(json.dumps({"protocol": str(prep), "sha256": digest}), flush=True)
    print(json.dumps({"report_sha256": run(prep, digest, output), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
