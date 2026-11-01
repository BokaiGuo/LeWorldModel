"""Host-CUDA encode of the sealed second development root."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np

from . import p1_queue_goal_blind_cuda_20261101_codex as prior
from . import p1_queue_goal_blind_adapter_dev02_20261101_codex as adapter

b, locking = prior.b, prior.locking
ROOT = b.ROOT
NAME = "p1_queue_goal_blind_cuda_dev02_20261101_codex"
MANIFEST = ROOT / "results/p1_queue_goal_blind_adapter_dev02_20261101_codex_DEV02.json"
MANIFEST_SHA = "e98066fc33a12336c77791957bfdbfea94f3a538bcc706a21f5b9a8340a72c1e"
BLIND_SHA = "17da3ad46233bd048601c32353487b182e15ce0db1a0fff36c6738ab023e2fa7"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"


def build():
    b.require(b.sha(MANIFEST) == MANIFEST_SHA, "manifest")
    manifest = b.read(MANIFEST)
    arrays, receipt_sha, audit = adapter.source()
    packet = b.load_blind(manifest["blind_path"], BLIND_SHA)
    b.require(
        manifest["blind_sha256"] == BLIND_SHA
        and manifest["seed"] == audit["seed"] == 301000002
        and receipt_sha == manifest["source_receipt_sha256"]
        and all(np.array_equal(packet[key], arrays[key]) for key in arrays),
        "development root identity",
    )
    pixels = [b.digest(value.tobytes()) for value in [*packet["rgb_history"], packet["goal_rgb"]]]
    action = b.digest(packet["raw_action_history"].tobytes())
    slot = {
        "order": 0,
        "source_slot": 0,
        "blind_npz": manifest["blind_path"],
        "blind_sha256": BLIND_SHA,
        "pixel_sha256": pixels,
        "raw_action_sha256": action,
    }
    files = set(b.source_files()) | {
        Path(__file__),
        Path(adapter.__file__),
        Path(prior.__file__),
        Path(locking.__file__),
    }
    bindings = {str(path.resolve()): b.sha(path) for path in files}
    bindings.update(b.MODELS)
    bindings[str(MANIFEST)] = MANIFEST_SHA
    bindings[manifest["blind_path"]] = BLIND_SHA
    return {
        "schema": NAME + ".prepare.v1",
        "slots": [slot],
        "normalization": b.native_metadata(),
        "model_bindings": b.MODELS,
        "policy": str(b.POLICY),
        "expected_model_hash": b.MODEL_TENSOR_SHA,
        "arguments": {"whitelist_sha": MANIFEST_SHA},
        "bindings_sha256": bindings,
        "source_receipt_sha256": receipt_sha,
        "seed": manifest["seed"],
        "contract": b.contract(1),
        "classification": "development_holdout02_only",
        "historical_coverage": "unknown",
        "training_membership": "unknown",
        "independent_confirmation": False,
        "P1_admission": False,
    }


def prepare(output):
    value = build()
    output = Path(output).absolute()
    b.require(not output.exists(), "exclusive prepare")
    with output.open("xb") as stream:
        stream.write(b.canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(output), "sha256": b.sha(output), "seed": value["seed"]}


def verify_prepare(path, expected):
    b.require(b.sha(path) == expected and b.read(path) == build(), "prepare closure")
    return b.read(path)


def export_task(prepared, expected, packet, values):
    export = b.task_export(prepared, expected, prepared["slots"][0], packet, values)
    export["source_admission"] = "development holdout02 receipt verified; no independent admission"
    export["new_independent_confirmation"] = False
    export["P1_admission"] = False
    export["development_seed"] = prepared["seed"]
    b.task_interfaces(export)
    return export


def run(prepared, expected, output):
    with locking.gpu_lock(LOCK):
        plan = verify_prepare(prepared, expected)
        store = b.Store(output)
        ledger = b.Ledger(1, store.event)
        errors, rows = [], []
        model = None
        before = after = host = None
        source_ok = False
        store.json("binding.json", {"prepared": str(Path(prepared).absolute()), "prepare_sha256": expected})
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            import torch
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0

            host = require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = False
            model = load_compatible_pretrained(plan["policy"], device=device).eval().requires_grad_(False)
            b.frozen(model)
            before = b.tensor_hash(model)
            b.require(before == plan["expected_model_hash"], "model identity")
            transform = p0.image_transform()
            ledger.sync = lambda: torch.cuda.synchronize(device)
            ledger.no_grad = lambda: not torch.is_grad_enabled()
            packet = b.load_blind(plan["slots"][0]["blind_npz"], BLIND_SHA)
            values = []
            with b.guard(model, torch, ledger, errors):
                for index, rgb in enumerate([*packet["rgb_history"], packet["goal_rgb"]]):
                    ledger.image = index
                    role = ("observed0", "observed5", "observed10", "goal")[index]
                    store.event({"kind": "image_attempt", "index": index, "role": role,
                                 "pixel_sha256": plan["slots"][0]["pixel_sha256"][index]})
                    with torch.inference_mode():
                        tensor = p0._image_tensor(rgb.copy(), transform=transform, device=device)
                        latent = model.encode({"pixels": tensor})["emb"][0, 0].detach().cpu().numpy().copy()
                    b.require(
                        latent.shape == (192,) and latent.dtype == np.float32,
                        "native CUDA latent",
                    )
                    saved = store.latent(index, latent)
                    rows.append({"role": role, "pixel_sha256": plan["slots"][0]["pixel_sha256"][index],
                                 "latent": saved})
                    values.append(latent)
                    store.event({"kind": "image_saved", "index": index, "latent": saved})
                    ledger.image = None
            store.json("task.json.gz", export_task(plan, expected, packet, values))
        except BaseException as exc:
            errors.insert(0, b.error("execution", exc))
        finally:
            ledger.image = None
            if model is not None:
                try:
                    b.frozen(model)
                    after = b.tensor_hash(model)
                except BaseException as exc:
                    errors.append(b.error("final_model", exc))
            try:
                verify_prepare(prepared, expected)
                source_ok = True
            except BaseException as exc:
                errors.append(b.error("final_source", exc))
            counts = ledger.counts()
            complete = (
                not errors
                and not ledger.blocked
                and source_ok
                and len(rows) == 4
                and before == after == b.MODEL_TENSOR_SHA
                and counts == {key: {"attempted": 4, "completed": 4} for key in counts}
            )
            hashes = {path.name: b.sha(path) for path in store.output.iterdir() if path.is_file()}
            report = {
                "schema": NAME + ".report.v1",
                "status": "complete" if complete else "partial_or_failed",
                "prepare_sha256": expected,
                "rows": rows,
                "resources": counts,
                "errors": errors,
                "ledger": deepcopy(ledger.entries),
                "blocked": ledger.blocked,
                "artifact_sha256": hashes,
                "source_verified": source_ok,
                "model_hash_before": before,
                "model_hash_after": after,
                "host": host,
                "seed": plan["seed"],
                "classification": "development_holdout02_only",
                "historical_coverage": "unknown",
                "training_membership": "unknown",
                "independent_confirmation": False,
                "P1_admission": False,
                "predict_calls": 0,
                "action_encoder_calls": 0,
                "backward_calls": 0,
                "physics_calls": 0,
            }
            store.json("report.json", report, final=True)
        b.require(complete, "partial result retained: " + str(errors))
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha256")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare(args.output) if args.command == "prepare" else run(
        args.prepared, args.prepare_sha256, args.output
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
