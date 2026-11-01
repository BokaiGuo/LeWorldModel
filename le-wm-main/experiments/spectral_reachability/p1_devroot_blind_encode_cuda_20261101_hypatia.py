"""Fixed dev-only blind RGB encoder; no sourcepack, planner, task export or training."""
import argparse
import json
import os
from pathlib import Path

from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as moving

low = moving.low
ROOT = moving.ROOT
NAME = "p1_devroot_blind_encode_cuda_20261101_hypatia"
BLIND = ROOT / "results/p1_devroot_blind_payload_20261101_hypatia_LOCAL01"
SLOTS = [
    (BLIND / "dev_slot0_blind_inputs.npz", "81458d05a6f5a24883df99f525abcfbc7278856bad67ad90b35a74949aabc742"),
    (BLIND / "dev_slot1_blind_inputs.npz", "13c0d44ea9d9f82ad6d77b0d0bb109585ae64e8a48ac0c3f93f4b25c82702196"),
]
PREP = ROOT / ("results/" + NAME + "_PREP01.json")
OUT = ROOT / ("results/" + NAME + "_RUN01")


def build():
    rows = []
    for index, (path, sha) in enumerate(SLOTS):
        packet = moving.load_blind(path, sha)
        rows.append({"slot": index, "path": str(path), "sha256": sha,
                     "pixel_sha256": [low.digest(a.tobytes()) for a in
                                      [*packet["rgb_history"], packet["goal_rgb"]]],
                     "raw_sha256": low.digest(packet["raw_action_history"].tobytes())})
    norm = low.native_metadata()
    low.require(norm == moving.normalization(), "native normalizer")
    paths = moving.source_files() + [Path(__file__).resolve(),
                                      ROOT / ("tests/test_" + NAME + ".py")]
    bindings = {str(Path(f).resolve()): low.sha(f) for f in paths}
    bindings.update(low.MODELS)
    bindings.update({str(path): sha for path, sha in SLOTS})
    for path, sha in bindings.items():
        low.require(low.sha(path) == sha, "binding mismatch: " + path)
    return {"schema": NAME + ".prepare.v1", "slots": rows, "bindings": bindings,
            "normalization": norm, "model_tensor_sha256": low.MODEL_TENSOR_SHA,
            "contract": moving.contract(), "purpose": "development encoding only",
            "planner_authorized": False, "P1_admission": False, "freshness": "unknown",
            "training_disjoint": "unknown", "task_export": False}


def prepare():
    raw = low.canonical(build())
    with PREP.open("xb") as f:
        low.require(f.write(raw) == len(raw), "short prepare")
        f.flush()
        os.fsync(f.fileno())
    return low.sha(PREP)


def verify(expected):
    low.require(low.sha(PREP) == expected, "external prepare SHA")
    data = low.read(PREP)
    low.require(data == build(), "prepare/input/model/source drift")
    return data


def run(expected):
    plan = verify(expected)
    store = low.Store(OUT)
    ledger = low.Ledger(2, store.event)
    errors, rows = [], [{"slot": i, "status": "not_started", "images": []} for i in (0, 1)]
    model = None
    before = after = None
    try:
        store.json("binding.json", {"prepare_sha256": expected, "purpose": "development only"})
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(low.POLICY), device=device).eval().requires_grad_(False)
        low.frozen(model)
        before = low.tensor_hash(model)
        low.require(before == low.MODEL_TENSOR_SHA, "frozen tensor SHA")
        transform = p0.image_transform()
        ledger.sync = lambda: torch.cuda.synchronize(device)
        ledger.no_grad = lambda: not torch.is_grad_enabled()
        with sdpa_kernel(SDPBackend.MATH), low.guard(model, torch, ledger, errors):
            for row, slot in zip(rows, plan["slots"], strict=True):
                row["status"] = "attempted"
                packet = moving.load_blind(slot["path"], slot["sha256"])
                for offset, image in enumerate([*packet["rgb_history"], packet["goal_rgb"]]):
                    index = 4 * slot["slot"] + offset
                    low.require(low.digest(image.tobytes()) == slot["pixel_sha256"][offset], "pixel binding")
                    ledger.image = index
                    store.event({"kind": "image_attempt", "index": index, "role": moving.ROLES[offset]})
                    with torch.inference_mode():
                        x = p0._image_tensor(image.copy(), transform=transform, device=device)
                        low.require(x.device.type == "cuda" and tuple(x.shape) == (1, 1, 3, 224, 224),
                                    "native RGB CUDA input")
                        z = model.encode({"pixels": x})["emb"]
                        low.require(z.device.type == "cuda" and z.dtype == torch.float32
                                    and tuple(z.shape) == (1, 1, 192), "native projected CLS")
                        value = z[0, 0].detach().cpu().numpy().copy()
                    low.native(value)
                    row["images"].append(store.latent(index, value))
                    store.event({"kind": "image_saved", "index": index, "sha256": row["images"][-1]["sha256"]})
                    ledger.image = None
                row["status"] = "encoded_dev_only"
    except BaseException as exc:
        errors.insert(0, low.error("execution", exc))
        try:
            store.event({"kind": "execution_error", "error": errors[0]})
        except BaseException as secondary:
            errors.append(low.error("error_journal", secondary))
        host = locals().get("host")
    finally:
        ledger.image = None
        for row in rows:
            if row["status"] == "attempted":
                row["status"] = "failed_partial"
        if model is not None:
            try:
                low.frozen(model)
                after = low.tensor_hash(model)
            except BaseException as exc:
                errors.append(low.error("model_final", exc))
        try:
            verify(expected)
        except BaseException as exc:
            errors.append(low.error("input_final", exc))
        counts = ledger.counts()
        complete = (not errors and not ledger.blocked and before == after == low.MODEL_TENSOR_SHA
                    and counts == {k: {"attempted": 8, "completed": 8} for k in ("encoder", "projector")}
                    and all(row["status"] == "encoded_dev_only" for row in rows))
        report = {"schema": NAME + ".report.v1", "status": "dev_encoding_complete" if complete else "partial_or_failed",
                  "rows": rows, "counts": counts, "errors": errors, "model_before": before, "model_after": after,
                  "host": host, "planner_authorized": False, "P1_admission": False,
                  "freshness": "unknown", "training_disjoint": "unknown", "task_export": False}
        store.json("report.json", report, final=True)
    low.require(complete, "partial dev encoding retained")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "verify", "run"])
    parser.add_argument("--prepare-sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare())
    elif args.command == "verify":
        print(json.dumps({"verified": bool(verify(args.prepare_sha256)), "neural_calls": 0}))
    else:
        print(json.dumps(run(args.prepare_sha256)))


if __name__ == "__main__":
    main()
