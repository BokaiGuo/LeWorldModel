"""CUDA encode one sealed reused development seed; never confirmation."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np

from . import p1_constructive_blind_encode_cuda_20261027_v1 as b
from . import p1_queue_goal_blind_adapter_20261101_codex as adapter
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking

ROOT = b.ROOT
NAME = "p1_queue_goal_blind_cuda_20261101_codex"
MANIFEST = ROOT / "results/p1_queue_goal_blind_adapter_20261101_codex_DEV01.json"
MANIFEST_SHA = "8680fb0d553e700d51246b65663aa66f3dae14fc56c3c0fa4cf5df390b763fa2"
BLIND_SHA = "9b56d6d1689ec27617207348a8428598d487289ab61b2d9b9593585df718dfeb"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"


def build():
    b.require(b.sha(MANIFEST) == MANIFEST_SHA, "frozen adapter manifest")
    manifest = b.read(MANIFEST)
    b.require(manifest["blind_sha256"] == BLIND_SHA and manifest["seed"] == 301000001
              and manifest["independent_confirmation"] is False and manifest["P1_admission"] is False,
              "fixed development identity")
    arrays, receipt_sha, audit = adapter.source()
    b.require(receipt_sha == manifest["source_receipt_sha256"] and audit["seed"] == manifest["seed"],
              "receipt identity")
    packet = b.load_blind(manifest["blind_path"], BLIND_SHA)
    b.require(all(np.array_equal(packet[k], arrays[k]) for k in arrays), "receipt/blind exact equality")
    pixels = [b.digest(x.tobytes()) for x in [*packet["rgb_history"], packet["goal_rgb"]]]
    action = b.digest(packet["raw_action_history"].tobytes())
    b.require(pixels == manifest["pixel_sha256"] and action == manifest["action_sha256"], "manifest identity")
    slot = dict(order=0, source_slot=0, blind_npz=manifest["blind_path"], blind_sha256=BLIND_SHA,
                pixel_sha256=pixels, raw_action_sha256=action)
    files = set(b.source_files()) | {Path(__file__), Path(adapter.__file__),
                                    Path(locking.__file__), Path(adapter.saved.__file__),
                                    ROOT / "tests/test_p1_queue_goal_blind_cuda_20261101_codex.py",
                                    ROOT / "docs/D2P_P1_QUEUE_GOAL_BLIND_CUDA_20261101_CODEX.md"}
    bindings = {str(p.resolve()): b.sha(p) for p in files}
    bindings.update(b.MODELS)
    bindings[str(MANIFEST)] = MANIFEST_SHA
    bindings[manifest["blind_path"]] = BLIND_SHA
    return dict(schema=NAME + ".prepare.v1", slots=[slot], normalization=b.native_metadata(),
                model_bindings=b.MODELS, policy=str(b.POLICY), expected_model_hash=b.MODEL_TENSOR_SHA,
                arguments=dict(whitelist_sha=MANIFEST_SHA), bindings_sha256=bindings,
                source_receipt_sha256=receipt_sha, seed=manifest["seed"],
                contract=b.contract(1), classification="reused_development_only",
                historical_coverage="unknown", training_membership="unknown",
                independent_confirmation=False, P1_admission=False)


def prepare(output):
    p = build()
    output = Path(output).absolute()
    b.require(not output.exists(), "exclusive prepare")
    with output.open("xb") as f:
        f.write(b.canonical(p))
        f.flush()
        os.fsync(f.fileno())
    return dict(path=str(output), sha256=b.sha(output), seed=p["seed"])


def verify_prepare(path, expected):
    b.require(b.sha(path) == expected, "prepare external hash")
    p = b.read(path)
    b.require(p == build(), "prepare/source/model closure")
    return p


def export_task(p, expected, packet, values):
    export = b.task_export(p, expected, p["slots"][0], packet, values)
    # The old exporter labels its inputs as MAIN attested. This independent
    # runner instead validates the actual saved queue chain itself.
    export["source_admission"] = "development receipt chain verified; no independent admission"
    export["new_independent_confirmation"] = False
    export["P1_admission"] = False
    export["development_seed"] = p["seed"]
    b.task_interfaces(export)
    return export


def run(prepared, expected, output):
    with locking.gpu_lock(LOCK):
        p = verify_prepare(prepared, expected)
        store = b.Store(output)
        ledger = b.Ledger(1, store.event)
        errors, rows = [], []
        model = None
        before = after = host = None
        source_ok = False
        store.json("binding.json", dict(prepared=str(Path(prepared).absolute()), prepare_sha256=expected))
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
            model = load_compatible_pretrained(p["policy"], device=device).eval().requires_grad_(False)
            b.frozen(model)
            before = b.tensor_hash(model)
            b.require(before == p["expected_model_hash"], "model tensor identity")
            transform = p0.image_transform()
            ledger.sync = lambda: torch.cuda.synchronize(device)
            ledger.no_grad = lambda: not torch.is_grad_enabled()
            packet = b.load_blind(p["slots"][0]["blind_npz"], BLIND_SHA)
            values = []
            with b.guard(model, torch, ledger, errors):
                for i, rgb in enumerate([*packet["rgb_history"], packet["goal_rgb"]]):
                    ledger.image = i
                    role = ("observed0", "observed5", "observed10", "goal")[i]
                    store.event(dict(kind="image_attempt", index=i, role=role,
                                     pixel_sha256=p["slots"][0]["pixel_sha256"][i]))
                    with torch.inference_mode():
                        x = p0._image_tensor(rgb.copy(), transform=transform, device=device)
                        b.require(x.device.type == "cuda" and tuple(x.shape) == (1, 1, 3, 224, 224),
                                  "native CUDA input")
                        z = model.encode({"pixels": x})["emb"]
                        b.require(z.device.type == "cuda" and z.dtype == torch.float32
                                  and tuple(z.shape) == (1, 1, 192), "native CUDA output")
                        value = z[0, 0].detach().cpu().numpy().copy()
                    b.require([(e["resource"], e["completed"]) for e in ledger.entries if e["image"] == i]
                              == [("encoder", True), ("projector", True)], "actual dispatch")
                    saved = store.latent(i, value)
                    rows.append(dict(role=role, pixel_sha256=p["slots"][0]["pixel_sha256"][i], latent=saved))
                    values.append(value)
                    store.event(dict(kind="image_saved", index=i, latent=saved))
                    ledger.image = None
            export = export_task(p, expected, packet, values)
            store.json("task.json.gz", export)
        except BaseException as exc:
            errors.insert(0, b.error("execution", exc))
            try:
                store.event(dict(kind="execution_error", error=errors[0]))
            except BaseException as secondary:
                errors.append(b.error("journal", secondary))
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
            complete = (not errors and not ledger.blocked and source_ok and len(rows) == 4
                        and before == after == b.MODEL_TENSOR_SHA
                        and counts == {k: dict(attempted=4, completed=4) for k in counts})
            hashes = {f.name: b.sha(f) for f in store.output.iterdir() if f.is_file()}
            result = dict(schema=NAME + ".report.v1", status="complete" if complete else "partial_or_failed",
                          prepare_sha256=expected, rows=rows, resources=counts, errors=errors,
                          ledger=deepcopy(ledger.entries), blocked=ledger.blocked,
                          artifact_sha256=hashes, source_verified=source_ok,
                          model_hash_before=before, model_hash_after=after, host=host,
                          seed=p["seed"], classification="reused_development_only",
                          historical_coverage="unknown", training_membership="unknown",
                          independent_confirmation=False, P1_admission=False,
                          predict_calls=0, action_encoder_calls=0, backward_calls=0, physics_calls=0)
            store.json("report.json", result, final=True)
        b.require(complete, "partial result retained: " + str(errors))
        return result


def verify_saved(prepared, expected, output):
    p = verify_prepare(prepared, expected)
    directory = Path(output).resolve(strict=True)
    report = b.read(directory / "report.json")
    b.require(report["schema"] == NAME + ".report.v1" and report["status"] == "complete"
              and report["prepare_sha256"] == expected and report["errors"] == []
              and report["source_verified"] is True and report["independent_confirmation"] is False
              and report["P1_admission"] is False, "report contract")
    b.require(set(report["artifact_sha256"]) == {"binding.json", "image.events.jsonl", "task.json.gz",
              *{f"image_{i:02d}.native192.npy" for i in range(4)}}
              and all(report[k] == 0 for k in
                      ("predict_calls", "action_encoder_calls", "backward_calls", "physics_calls")),
              "artifact roster and zero other calls")
    for name, sha in report["artifact_sha256"].items():
        b.require(Path(name).name == name and b.sha(directory / name) == sha, "artifact hash")
    b.require(report["model_hash_before"] == report["model_hash_after"] == b.MODEL_TENSOR_SHA
              and report["resources"] == {k: dict(attempted=4, completed=4) for k in ("encoder", "projector")}
              and report["blocked"] == [], "model and resources")
    expected_entries = [(i, k) for i in range(4) for k in ("encoder", "projector")]
    b.require([(e["image"], e["resource"]) for e in report["ledger"]] == expected_entries
              and all(e["completed"] is True for e in report["ledger"]), "ordered ledger")
    events = [json.loads(line) for line in (directory / "image.events.jsonl").read_text().splitlines()]
    module_events = [e for e in events if e["kind"] in ("module_attempt", "module_return")]
    b.require(len(module_events) == 16 and all(
        module_events[2*i]["kind"] == "module_attempt" and
        module_events[2*i+1]["kind"] == "module_return" and
        module_events[2*i]["entry"]["index"] == module_events[2*i+1]["entry"]["index"] == i and
        module_events[2*i]["entry"]["completed"] is False and
        module_events[2*i+1]["entry"] == report["ledger"][i] for i in range(8)), "ledger journal")
    packet = b.load_blind(p["slots"][0]["blind_npz"], BLIND_SHA)
    values = []
    b.require(len(report["rows"]) == 4, "four real images")
    for i, row in enumerate(report["rows"]):
        b.require(row["role"] == ("observed0", "observed5", "observed10", "goal")[i]
                  and row["pixel_sha256"] == p["slots"][0]["pixel_sha256"][i], "image role")
        latent = directory / f"image_{i:02d}.native192.npy"
        b.require(row["latent"]["sha256"] == b.sha(latent), "latent hash")
        with latent.open("rb") as f:
            z = np.load(f, allow_pickle=False)
        b.native(z)
        values.append(z)
    export = b.read(directory / "task.json.gz")
    b.require(export == export_task(p, expected, packet, values), "actual task recomputation")
    return dict(status="saved_cuda_artifacts_verified", report_sha256=b.sha(directory / "report.json"),
                seed=p["seed"], image_encoder_calls=4, projector_calls=4,
                independent_confirmation=False, P1_admission=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha256")
    parser.add_argument("--output", required=True)
    a = parser.parse_args()
    result = prepare(a.output) if a.command == "prepare" else (
        run(a.prepared, a.prepare_sha256, a.output) if a.command == "run"
        else verify_saved(a.prepared, a.prepare_sha256, a.output))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
