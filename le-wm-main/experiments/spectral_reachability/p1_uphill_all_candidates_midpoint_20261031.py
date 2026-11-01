"""Diagnose fixed candidate predictions against real five-step RGB observations."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_uphill_multistart_physical_20261030_main as original
from . import p1_uphill_nonwinner_physical_20261031 as nonwinner
from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base

NAME = "p1_uphill_all_candidates_midpoint_20261031"
ROOT = base.ROOT
PROTOCOL = ROOT.parent / "docs/D2P_P1_ALL_CANDIDATE_MIDPOINT_PROTOCOL_20261031_MAIN.md"
REPORTS = [
    ("p1_uphill_multistart_physical_20261030_main", "623f23bb61d06ae105be4c0b34be6dabee1c7775b1fd5591e3791e53995a669c",
     "575ee720d8282d349ae06f8f06e27e4caa592e65b228637722e9e4ca6183d870"),
    ("p1_uphill_nonwinner_physical_20261031", "cdbb63c78e0866a333ba104e317935a526c5b5313ff78bc6aa3d435e836b6f68",
     "ec31143edc8d479c3b47aa038818ff20171e5e971473eb4a3b0165792a686b80"),
    ("p1_uphill_nonwinner_physical_20261031", "602d387dc464e636ea43f2b6688773cd9d5016d7bca5a3c9833797a2d2ad95e6",
     "60e3adaef66bb320d3f24ba2b82722900f48945d6968503f2a55ab69b6e3e9d7"),
    ("p1_uphill_nonwinner_physical_20261031", "5ca09b9cd58328688e9037556026da7c5d0eddd89daed3425ec2124ed5423f3f",
     "910ad07157f6986cf72721927365365dedc1165d829ff013ce650a95a0322e36"),
]
PREP_SHAS = ["5ce3e38a1c86945383a92eb28d735aa95a0f043dda6709b875b9d244fa38aad2",
             "45aea30301f95f79ca99527c8233e83f84b70fd02eb2f8c5df281b77445c4de1",
             "e02c138e6cfe40230ddecce9cf052259d67085ffad5a1f9f06ee783ead175d84"]


def read(path, digest):
    if base.sha(path) != digest:
        raise ValueError("source hash: " + str(path))
    return json.loads(Path(path).read_bytes())


def inputs():
    original.planner.configure()
    bundle = read(original.planner.bundle.OUT, original.planner.BUNDLE_SHA)
    plan = read(original.PLAN, original.PLAN_SHA)
    base_prepare_path = ROOT / "results/p1_uphill_multistart_physical_20261030_main_prepare_RUN01.json"
    base_prepare_sha = "c58e71cf4a5f42d4ab5668993e95869d64718a3b3d5ed37172816dc980e82ff8"
    base_prepare = read(base_prepare_path, base_prepare_sha)
    sources = {str(original.planner.bundle.OUT): original.planner.BUNDLE_SHA,
               str(original.PLAN): original.PLAN_SHA,
               str(base_prepare_path): base_prepare_sha,
               str(Path(__file__).resolve()): base.sha(Path(__file__).resolve()),
               str(PROTOCOL): base.sha(PROTOCOL)}
    entries = []
    for kind, (name, report_hash, consume_hash) in enumerate(REPORTS):
        suffix = "" if kind == 0 else f"_r{kind-1}"
        run = ROOT / f"results/{name}_run{suffix}_RUN01"
        consume = ROOT / f"results/{name}_consume{suffix}_RUN01.json"
        report = read(run / "run_report.json", report_hash)
        validated = read(consume, consume_hash)
        prep = base_prepare if kind == 0 else read(
            ROOT / f"results/{name}_prepare_r{kind-1}_RUN01.json", PREP_SHAS[kind-1])
        sources[str(run / "run_report.json")] = report_hash
        sources[str(consume)] = consume_hash
        if kind:
            sources[str(ROOT / f"results/{name}_prepare_r{kind-1}_RUN01.json")] = PREP_SHAS[kind-1]
        if report["status"] != "recorded_not_consumed" or len(report["shards"]) != 24 or len(validated["rows"]) != 24:
            raise ValueError("incomplete physical panel")
        for i, (row, item, checked, shard) in enumerate(zip(
                bundle["rows"], prep["roster"], validated["rows"], report["shards"], strict=True)):
            if row["index"] != i or item["index"] != i or checked["index"] != i or shard["index"] != i \
                    or checked["returned_suffix"] < 5 or shard["status"] != "recorded_not_consumed":
                raise ValueError("trajectory roster")
            raw = np.frombuffer(bytes.fromhex(item["raw10_float32_hex"]), dtype="<f4").reshape(2, 5, 2)
            if hashlib.sha256(raw.tobytes()).hexdigest() != item["raw10_sha256"]:
                raise ValueError("action hash")
            report_path = run / f"row_{i:02d}/report.json"
            physical = read(report_path, shard["report_sha256"])["row"]
            frames = [f for f in physical["trajectory"] if f["phase"] == "probe" and f["step"] == 5
                      and f["condition"] == physical["condition"]
                      and f["overlay_index"] == (1 if physical["condition"] == "on" else 0)]
            if len(frames) != 1 or frames[0]["state"] != physical["responses"][4]["state"]:
                raise ValueError("midpoint state")
            frame = frames[0]
            path = report_path.parent / frame["png"]
            if path.name != frame["png"] or base.sha(path) != frame["file_sha256"]:
                raise ValueError("midpoint file")
            with Image.open(path) as image:
                midpoint = np.asarray(image).copy()
            if midpoint.shape != (224, 224, 3) or midpoint.dtype != np.uint8 \
                    or hashlib.sha256(midpoint.tobytes()).hexdigest() != frame["pixel_sha256"]:
                raise ValueError("midpoint pixels")
            if kind == 0:
                candidate = plan["rows"][i]["selected_index"]
                score = plan["rows"][i]["candidate_scores"][candidate]
            else:
                candidate = item["candidate_index"]
                score = plan["rows"][i]["candidate_scores"][candidate]
            entries.append({"index": i, "kind": kind, "candidate": candidate, "blind": row,
                            "action": raw.copy(), "score": score, "midpoint": midpoint,
                            "midpoint_pixel_sha256": frame["pixel_sha256"],
                            "success": checked["last_accepted_state_metrics"]["native_success"],
                            "status": checked["status"], "returned_suffix": checked["returned_suffix"]})
            sources[str(report_path)] = shard["report_sha256"]
            sources[str(path)] = frame["file_sha256"]
    if len(entries) != 96:
        raise ValueError("complete candidate set")
    return entries, sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    if not output.is_absolute() or output.parent != ROOT / "results" \
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink():
        raise ValueError("exclusive output")
    entries, sources = inputs()
    sources.update(base.bindings())
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        transform = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for entry in entries:
                row = entry["blind"]
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp_pixels = np.asarray(image).copy()
                def encode(pixels):
                    return model.encode({"pixels": p0._image_tensor(
                        pixels, transform=transform, device=device)})["emb"][0, 0]
                cp, actual = encode(cp_pixels), encode(entry["midpoint"])
                states, history, goal, mean, std = base.bridge.cp5.context(row, cp, device=device)
                raw = torch.as_tensor(entry["action"][None], dtype=torch.float32, device=device)
                ctx1 = torch.cat((history, ((raw[:, 0] - mean) / std).reshape(1, 1, 10)), dim=1)
                pred5 = model.predict(states, model.action_encoder(ctx1))[:, -1:]
                ctx2 = torch.cat((ctx1[:, -2:], ((raw[:, 1] - mean) / std).reshape(1, 1, 10)), dim=1)
                enc2 = model.action_encoder(ctx2)
                open10 = model.predict(torch.cat((states, pred5), dim=1)[:, -3:], enc2)[0, -1]
                refreshed10 = model.predict(torch.cat((states, actual[None, None]), dim=1)[:, -3:], enc2)[0, -1]
                mse = lambda a, b: float((a-b).square().mean())
                open_score = mse(open10, goal[0])
                if abs(open_score - entry["score"]) > 1e-5:
                    raise ValueError(f"score mismatch {entry['index']} {entry['kind']}")
                rows.append({"index": entry["index"], "kind": entry["kind"], "candidate": entry["candidate"],
                             "role": row["role"], "logical_pair": row["logical_pair"],
                             "midpoint_pixel_sha256": entry["midpoint_pixel_sha256"],
                             "prediction_midpoint_mse": mse(pred5[0, 0], actual),
                             "predicted_midpoint_goal_mse": mse(pred5[0, 0], goal[0]),
                             "actual_midpoint_goal_mse": mse(actual, goal[0]),
                             "open_terminal_goal_mse": open_score,
                             "refreshed_terminal_goal_mse": mse(refreshed10, goal[0]),
                             "success": entry["success"], "status": entry["status"],
                             "returned_suffix": entry["returned_suffix"]})
        torch.cuda.synchronize(device)
        if before != base.model_hash(model) or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("source drift")
        digest = base.write(output, {"schema": NAME, "status": "complete", "rows": rows, "sources": sources,
                                     "host": host, "model_before": before, "model_after": before,
                                     "P1_admission": False, "independent_confirmation": False,
                                     "limitations": "Known roots; observed midpoint is an offline oracle; "
                                                    "early stops have no fixed-horizon actual endpoint."})
    print(json.dumps({"output": str(output), "sha256": digest, "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Compare all fixed candidate predictions with real midpoint RGB on CUDA.", flush=True)
    run(parser.parse_args().output)
