"""Compare the fixed model rollout and executed terminal RGB on host CUDA."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_adam_checkpoint_physical_pair_20261031 as pair
from . import p1_adam_checkpoint_curve_20261031 as curve

NAME = "p1_adam_terminal_rgb_cuda_20261031"
ROOT = pair.ROOT
PAIR = ROOT / "results/p1_adam_checkpoint_physical_pair_20261031_RUN01.json"
PAIR_SHA = "6b1d9fd80a4465a1d9e0cdb6ded0092773ec11b1d19f04217a385e0fc1378051"
PROTOCOL = ROOT.parent / "docs/D2P_P1_ADAM_TERMINAL_RGB_PROTOCOL_20261031_MAIN.md"
RUNS = {
    5: (ROOT / "results/p1_adam_checkpoint_physical_5_20261031_run_RUN01",
        "8fcfe45c1a4b226fb180e591c603af6db553415df6e71a9202d6fad10c9f5dad"),
    10: (ROOT / "results/p1_adam_checkpoint_physical_10_20261031_run_RUN01",
         "8894d549f6e6a96e790626b7025f754547e8b43319b4f9636be63cae21a2ba4f"),
    20: (ROOT / "results/p1_uphill_multistart_physical_20261030_main_run_RUN01",
         "623f23bb61d06ae105be4c0b34be6dabee1c7775b1fd5591e3791e53995a669c"),
}
SEALED = {
    5: ("p1_adam_checkpoint_physical_5_20261031_prepare_RUN01.json", "eaec5459a4aa5755941ef2749fb0f6e4264600eb3c33f379b5266860a8b938af",
        "p1_adam_checkpoint_physical_5_20261031_consume_RUN01.json", "7b413749a1a792bab575a6574e1027f237bc38f376cbe61de5bc01599595dd2f"),
    10: ("p1_adam_checkpoint_physical_10_20261031_prepare_RUN01.json", "5be37dd41e7edfb228390a8924ecfaf9688ea0e54cd93386a3d042aaf43b9202",
         "p1_adam_checkpoint_physical_10_20261031_consume_RUN01.json", "21158d681fe62d9a42cbfd046ea5b7080403a410b50e733ffd7d78093e4ecedc"),
    20: ("p1_uphill_multistart_physical_20261030_main_prepare_RUN01.json", "c58e71cf4a5f42d4ab5668993e95869d64718a3b3d5ed37172816dc980e82ff8",
         "p1_uphill_multistart_physical_20261030_main_consume_RUN01.json", "575ee720d8282d349ae06f8f06e27e4caa592e65b228637722e9e4ca6183d870"),
}
need = pair.pairing.need


def verify_binding(i, raw, observed, shard, prepared, consumed, rec):
    need(all(prepared[k] == observed[k] == consumed[k]
             for k in ("index", "role", "condition", "root_slot")), "sealed identity")
    need(i == prepared["index"] and shard["status"] == "recorded_not_consumed",
         "sealed shard")
    need(bytes.fromhex(prepared["raw10_float32_hex"]) == raw.tobytes()
         and hashlib.sha256(raw.tobytes()).hexdigest() == prepared["raw10_sha256"],
         "sealed dispatched action")
    need(rec["status"] == consumed["status"] == observed["budgets_status"], "sealed status")
    if consumed["completed_endpoint"] is not None:
        need(rec["responses"][-1]["state"] == consumed["completed_endpoint"], "sealed endpoint")


def collect():
    sha = pair.physical.sha
    combined = pair.pairing.read(PAIR, PAIR_SHA)
    plan = pair.pairing.read(pair.physical.PLAN, pair.physical.PLAN_SHA)
    bundle = pair.pairing.read(pair.physical.prior.planner.bundle.OUT,
                               pair.physical.prior.planner.BUNDLE_SHA)
    need(len(combined["rows"]) == len(plan["rows"]) == len(bundle["rows"]) == 24, "roster")
    sources = {str(PAIR): PAIR_SHA, str(pair.physical.PLAN): pair.physical.PLAN_SHA,
               str(pair.physical.prior.planner.bundle.OUT): pair.physical.prior.planner.BUNDLE_SHA,
               str(Path(__file__).resolve()): sha(Path(__file__).resolve()), str(PROTOCOL): sha(PROTOCOL)}
    entries, stopped = [], []
    for step, (directory, report_sha) in RUNS.items():
        prep_name, prep_sha, consume_name, consume_sha = SEALED[step]
        prep_path, consume_path = ROOT / "results" / prep_name, ROOT / "results" / consume_name
        prepared = pair.pairing.read(prep_path, prep_sha)
        consumed = pair.pairing.read(consume_path, consume_sha)
        need(len(prepared["roster"]) == len(consumed["rows"]) == 24, "sealed roster")
        sources[str(prep_path)], sources[str(consume_path)] = prep_sha, consume_sha
        run = pair.pairing.read(directory / "run_report.json", report_sha)
        need(run["status"] == "recorded_not_consumed" and len(run["shards"]) == 24, "run roster")
        sources[str(directory / "run_report.json")] = report_sha
        for i, (row, proposal, observed, shard) in enumerate(zip(
                bundle["rows"], plan["rows"], combined["rows"], run["shards"], strict=True)):
            need(row["index"] == proposal["index"] == observed["index"] == shard["index"] == i
                 and row["role"] == observed["role"] == proposal["role"]
                 and row["logical_pair"] == observed["logical_pair"] == proposal["logical_pair"],
                 "row identity")
            checkpoint = proposal["checkpoints"][str(step)]
            chosen = checkpoint["winner"]
            raw = np.asarray(checkpoint["candidate_actions"][chosen], dtype="<f4")
            need(raw.shape == (2, 5, 2) and np.isfinite(raw).all()
                 and hashlib.sha256(raw.tobytes()).hexdigest() == checkpoint["candidate_sha256"][chosen]
                 and checkpoint["scores"][chosen] == observed["budgets"][str(step)]["model_best_score"],
                 "plan action and score")
            report_path = directory / f"row_{i:02d}" / "report.json"
            report = pair.pairing.read(report_path, shard["report_sha256"])
            sources[str(report_path)] = shard["report_sha256"]
            rec = report["row"]
            outcome = observed["budgets"][str(step)]
            verify_binding(i, raw, {**observed, "budgets_status": outcome["status"]},
                           shard, prepared["roster"][i], consumed["rows"][i], rec)
            need(rec["status"] == outcome["status"], "status binding")
            if outcome["endpoint"] is None:
                need(outcome["status"] == "native_stopped", "missing endpoint")
                stopped.append({"index": i, "step": step, "status": outcome["status"]})
                continue
            need(outcome["status"] == "probe_completed" and len(rec["responses"]) == 10,
                 "full endpoint")
            frame = rec["trajectory"][-1]
            need(frame["phase"] == "probe" and frame["step"] == 10
                 and frame["condition"] == observed["condition"]
                 and frame["overlay_index"] == (1 if observed["condition"] == "on" else 0)
                 and frame["state"] == rec["responses"][-1]["state"], "terminal frame binding")
            png = directory / f"row_{i:02d}" / frame["png"]
            need(png.name == frame["png"] and sha(png) == frame["file_sha256"], "frame file")
            with Image.open(png) as image:
                pixels = np.asarray(image).copy()
            need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
                 and hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"], "frame pixels")
            sources[str(png)] = frame["file_sha256"]
            entries.append({"index": i, "step": step, "row": row, "action": raw,
                            "rgb": pixels, "score": checkpoint["scores"][chosen],
                            "pixel_sha256": frame["pixel_sha256"],
                            "action_sha256": checkpoint["candidate_sha256"][chosen],
                            "physical_endpoint": outcome["endpoint"]})
    need(len(entries) + len(stopped) == 72, "complete denominator")
    return entries, stopped, sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    curve.planner.configure()
    base, prior = curve.planner.adam.prior.base, curve.planner.adam.prior
    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(),
         "exclusive output")
    entries, stopped, sources = collect()
    for path, digest in base.bindings().items():
        need(path not in sources or sources[path] == digest, "source collision")
        sources[path] = digest
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "checkpoint")
        transform = p0.image_transform()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for entry in entries:
                row = entry["row"]
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp_rgb = np.asarray(image).copy()
                def encode(im):
                    return model.encode({"pixels": p0._image_tensor(
                        im, transform=transform, device=device)})["emb"][0, 0]
                cp, actual = encode(cp_rgb), encode(entry["rgb"])
                action = torch.as_tensor(entry["action"][None], dtype=torch.float32, device=device)
                pred, goal = prior.fixed.replay(model, row, cp, action)
                predicted_score = float((pred-goal).square().mean())
                actual_score = float((actual-goal).square().mean())
                error = float((pred.reshape(-1)-actual.reshape(-1)).square().mean())
                need(np.isfinite([predicted_score, actual_score, error]).all()
                     and abs(predicted_score-entry["score"]) <= 1e-5, "replayed score")
                rows.append({"index": entry["index"], "step": entry["step"],
                             "role": row["role"], "logical_pair": row["logical_pair"],
                             "action_sha256": entry["action_sha256"], "pixel_sha256": entry["pixel_sha256"],
                             "predicted_goal_mse": predicted_score, "actual_rgb_goal_mse": actual_score,
                             "prediction_rgb_mse": error, "predicted_latent": pred.reshape(-1).cpu().tolist(),
                             "actual_latent": actual.reshape(-1).cpu().tolist(),
                             "physical_endpoint": entry["physical_endpoint"]})
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        need(before == after and all(base.sha(path) == digest for path, digest in sources.items()), "drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "stopped": stopped,
                  "sources": sources, "host": host, "model_before": before, "model_after": after,
                  "calls_code_path": {"encode": 2*len(rows), "predict": 2*len(rows),
                                      "action_encoder": 2*len(rows)},
                  "P1_admission": False, "independent_confirmation": False,
                  "limitations": "Known roots and repeated trajectories; no post-stop endpoint imputation."}
        digest = base.write(output, result)
    print(json.dumps({"rows": len(rows), "stopped": len(stopped), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Replay fixed completed actions and encode their real terminal RGB on host CUDA.", flush=True)
    run(parser.parse_args().output)
