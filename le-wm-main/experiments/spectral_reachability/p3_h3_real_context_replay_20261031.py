"""Replay all completed H3 physical pairs from their real RGB history on CUDA."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p3_h3_model_physical_direction_20261031 as joined
from . import p3_raw_chart_nonlinear_20261030_main as nonlinear
from . import preflight_pusht_d2p_p0 as p0

ROOT, base = joined.ROOT, nonlinear.base
NAME = "p3_h3_real_context_replay_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P3_H3_REAL_CONTEXT_REPLAY_PROTOCOL_20261031.md"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def collect():
    paths = {str(ROOT / "results" / fn): digest for fn, digest in joined.SOURCES.values()}
    need(all(base.sha(Path(p)) == digest for p, digest in paths.items()), "fixed input hashes")
    prepared = read(ROOT / "results" / joined.SOURCES["prepare"][0])
    model = read(ROOT / "results" / joined.SOURCES["model"][0])
    panel = read(ROOT / "results" / joined.SOURCES["physical"][0])
    analysis = joined.analyze(prepared, model, panel)
    need(analysis["complete_pairs"] == 13, "complete unique pairs")
    roster = nonlinear.load_roster()
    tasks = {(index, row["index"], slot["policy"], slot["root"]): task
             for index, slot, task, horizon, row in roster if horizon == 3}
    model_rows = {(r["decision_index"], r["candidate"], r["policy"], r["root"]): r
                  for r in model["rows"] if r["horizon"] == 3}
    need(len(tasks) == len(model_rows) == 36, "task roster")
    for p, digest in panel["sources"].items():
        need(base.sha(Path(p)) == digest, "panel source hash")
        paths[p] = digest
    paths.update({str(PROTOCOL): base.sha(PROTOCOL), str(Path(__file__).resolve()): base.sha(Path(__file__)),
                  str(Path(joined.__file__).resolve()): base.sha(Path(joined.__file__)),
                  str(nonlinear.JOURNAL): nonlinear.JOURNAL_SHA})
    receipts = {}
    for p in panel["sources"]:
        if "_pair_20261031_BATCH" in p:
            receipt = read(Path(p))
            need(receipt["batch_index"] not in receipts, "unique batch receipt")
            receipts[receipt["batch_index"]] = receipt
    need(len(receipts) == 12, "twelve batch receipts")
    out, excluded = [], []
    for group in analysis["groups"]:
        for item in group["duplicates"]:
            logical = item["logical_pair"]
            if item["physical_new_minus_old"] is None:
                excluded.append({"logical_pair": logical, "old_status": item["old_status"],
                                 "new_status": item["new_status"]})
                continue
            batch, pair = divmod(logical, 3)
            receipt = receipts[batch]
            report_hash = receipt.get("source_report_sha256", receipt.get("report_sha256"))
            directory = ROOT / f"results/p3_raw_h3_physical_run_20261031_BATCH{batch:02d}_RUN01"
            report_path = directory / "report.json"
            need(base.sha(report_path) == report_hash, "physical report hash")
            paths[str(report_path)] = report_hash
            report = read(report_path)
            for offset, variant in enumerate(("old", "new")):
                row, plan = report["rows"][2 * pair + offset], prepared["plans"][2 * logical + offset]
                need(all(row["selected_plan"].get(k) == v for k, v in plan.items())
                     and row["selected_plan"].get("arm") == row["arm"] and row["own_returned"] == 20
                     and row["status"] == "horizon_completed", "verified physical row")
                key = (plan["decision_index"], plan["candidate"], plan["policy"], plan["root_slot"])
                mr, task = model_rows[key], tasks[key]
                actions = np.asarray(plan["commands"], dtype="<f4")
                need(actions.shape == (20, 2) and
                     actions[15:].tobytes() == np.asarray(mr[variant + "_action"], dtype="<f4").tobytes(),
                     "actual model suffix")
                frames = []
                for step in (5, 10, 15, 20):
                    matches = [c for c in row["checkpoints"] if c["phase"] == "plan" and c["step"] == step]
                    need(len(matches) == 1, "checkpoint roster")
                    cp = matches[0]
                    image_path = directory / cp["png"]
                    need(image_path.parent == directory and not image_path.is_symlink()
                         and base.sha(image_path) == cp["file_sha256"], "checkpoint file")
                    with Image.open(image_path) as image:
                        pixels = np.asarray(image.convert("RGB")).copy()
                    need(pixels.shape == (224, 224, 3)
                         and hashlib.sha256(pixels.tobytes()).hexdigest() == cp["pixel_sha256"],
                         "checkpoint pixels")
                    paths[str(image_path)] = cp["file_sha256"]
                    frames.append(pixels)
                out.append({"logical_pair": logical, "variant": variant, "actions": actions,
                            "frames": frames, "task": task, "model": mr, "key": key})
    need(len(out) == 52 and len(excluded) == 10, "fixed complete and censored roster")
    need(all(base.sha(Path(p)) == digest for p, digest in paths.items()), "input drift")
    return out, excluded, paths


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT / "results" and output.name.startswith(NAME + "_")
         and not output.exists() and not output.is_symlink(), "exclusive output")
    entries, excluded, sources = collect()
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
        need(before == base.MODEL_HASH, "frozen checkpoint")
        transform = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for item in entries:
                tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                embeddings = [model.encode({"pixels": p0._image_tensor(image, transform=transform,
                                                                       device=device)})["emb"][0, 0]
                              for image in item["frames"]]
                states = torch.stack(embeddings[:3]).unsqueeze(0)
                actions = tensor(item["actions"])
                mean, std = tensor(item["task"]["action_mean"]), tensor(item["task"]["action_std"])
                need(states.shape == (1, 3, 192) and mean.shape == std.shape == (2,)
                     and bool((std > 0).all()), "context ABI")
                tokens = ((actions[5:20] - mean) / std).reshape(1, 3, 10)
                pred = model.predict(states, model.action_encoder(tokens))[0, -1]
                goal = tensor(item["task"]["goal"])
                projection = tensor(item["task"]["projection"])
                actual = embeddings[3]
                opened = tensor(item["model"][item["variant"] + "_terminal"])
                cost = lambda z: float(((projection @ (goal - z)).square().mean()).item())
                rows.append({"logical_pair": item["logical_pair"], "variant": item["variant"],
                             "decision_index": item["key"][0], "candidate": item["key"][1],
                             "policy": item["key"][2], "root_slot": item["key"][3],
                             "open_goal_cost": cost(opened), "refreshed_goal_cost": cost(pred),
                             "actual_goal_cost": cost(actual),
                             "open_terminal_mse": float((opened - actual).square().mean().item()),
                             "refreshed_terminal_mse": float((pred - actual).square().mean().item()),
                             "open_latent": opened.cpu().tolist(), "refreshed_latent": pred.cpu().tolist(),
                             "actual_latent": actual.cpu().tolist(), "goal_latent": goal.cpu().tolist()})
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        need(before == after and all(base.sha(Path(p)) == d for p, d in sources.items()), "model/source drift")
        result = {"schema": NAME, "rows": rows, "excluded": excluded, "host": host, "sources": sources,
                  "model_before": before, "model_after": after,
                  "calls_code_path": {"encode": 4 * len(rows), "predict": len(rows),
                                      "action_encoder": len(rows)},
                  "P1_admission": False, "P3_admission": False, "independent_confirmation": False,
                  "limitations": "Known roots, post-hoc real history; duplicates not independent; no replanning."}
        digest = base.write(output, result)
    print(json.dumps({"rows": len(rows), "excluded_pairs": len(excluded), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    print("Replay all fully observed H3 actions using real cp5/cp10/cp15 RGB history on CUDA.", flush=True)
    run(arguments.output)
