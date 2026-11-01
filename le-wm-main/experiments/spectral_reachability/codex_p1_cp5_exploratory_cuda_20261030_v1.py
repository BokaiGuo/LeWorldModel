"""Known-panel 24-row cp5 CUDA planning only; no physical execution.

Only the fixed blind input bundle is read as task data. There is no OS file
isolation: trusted code/runtime can access other host files. Runtime source
census is observational, not a complete dependency closure.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

from . import codex_p1_cp5_cuda_replay_20261030_v1 as bridge
from . import codex_p1_cp5_durable_budget_20261030_v1 as encbudget
from . import codex_p1_cp5_unified_consumer_20261030_v1 as consumer
from . import codex_p1_cp5_24_driver_20261030_v1 as driver24
from . import codex_four_arm_durable_20261030 as durable
from . import codex_four_arm_full_cost_20261030 as spec
from . import codex_four_arm_leaf_suffix_20261030 as leafmod
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking

ROOT = Path(__file__).resolve().parents[2]
NAME = "codex_p1_cp5_exploratory_cuda_20261030_v1"
BUNDLE = ROOT / "results/codex_p1_three_prefix_blind_inputs_20261030_v1.json"
BUNDLE_SHA = "070cb2fccffc79a1cf4a9480396f6b2d5eec4ff3b72928c8704fda8a23095cc8"
MODEL = Path("/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm")
MODEL_FILES = {"weights.pt": "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607",
               "config.json": "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"}
MODEL_HASH = {"original_state_dict": "4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546",
              "all_buffers": "ca7a531449da81ea2dee6eb79716f4bf840803b9df4dd046b8a591e8f7b8bd78"}
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
CONFIG = {"steps": 2, "learning_rate": 0.05, "history_size": 3}
WALL = 1800
ENC_CAPS = dict.fromkeys(encbudget.AXES, 0)
ENC_CAPS.update(encoder=1, projector=1)


def sha(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing or symlink input: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    raw = spec.canonical(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()


def bindings():
    modules = (bridge, bridge.cp5, bridge.cp5.blind, bridge.cp5.optimizer, bridge.replay,
               encbudget, consumer, consumer.optimizer, driver24, durable, spec, leafmod, locking)
    paths = {Path(m.__file__).resolve() for m in modules}
    paths.add(Path(__file__).resolve())
    paths.update(MODEL / name for name in MODEL_FILES)
    result = {str(p): sha(p) for p in sorted(paths)}
    if any(result[str(MODEL/name)] != expected for name, expected in MODEL_FILES.items()):
        raise ValueError("checkpoint file mismatch")
    return result


def make_contract(row, source_hashes):
    mean = bridge.torch.tensor(row["action_mean"], dtype=bridge.torch.float32).tolist()
    std = bridge.torch.tensor(row["action_std"], dtype=bridge.torch.float32).tolist()
    policy = {"schema": bridge.replay.POLICY_SCHEMA, "candidate_ids": ["optimized"],
              "horizon": 2, "history_size": 3, "metric": "terminal_latent_mean_squared_error",
              "tie": "first_declared_candidate", "replay_batch": 1,
              "action_chart": "raw_packet_5x2_box_minus1_to1", "mean": mean, "std": std}
    identity = {k: spec.digest({"field": k, "row": row["index"], "bundle": BUNDLE_SHA}) for k in spec.IDENTITY}
    identity.update(model=MODEL_HASH["original_state_dict"],
                    optimizer=source_hashes[str(Path(bridge.cp5.optimizer.__file__).resolve())],
                    optimizer_config=spec.digest(CONFIG),
                    meter=source_hashes[str(Path(durable.__file__).resolve())])
    arms = {arm: {"initializer_sha256": spec.digest({"constant": "zero_raw_2x5x2"}),
                  "optimizer_sha256": identity["optimizer"],
                  "optimizer_config_sha256": identity["optimizer_config"]} for arm in spec.ARMS}
    caps = dict.fromkeys(spec.AXES, 0)
    caps.update(action_encode_calls=6, predict_calls=6, predict_samples=6,
                backward_calls=2, vjp_rows=2, replay_calls=1, selection_evaluations=1)
    contract = {"schema": spec.NAME + ".contract.v1", "identity": identity, "arms": arms,
                "scope": "standalone_per_root_per_arm_including_attributable_setup",
                "phase_policy_sha256": spec.digest(policy),
                "levels": [{"id": "fixed", "caps": caps},
                           {"id": "unused", "caps": {k: v+1 for k, v in caps.items()}}]}
    return policy, contract


def model_hash(model):
    state, buffers = hashlib.sha256(), hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous()
        state.update(name.encode()); state.update(str(a.dtype).encode())
        state.update(str(tuple(a.shape)).encode()); state.update(a.numpy().tobytes())
    for name, value in model.named_buffers():
        a = value.detach().cpu().contiguous()
        buffers.update(json.dumps([name, str(a.dtype), list(a.shape)], sort_keys=True,
                                  separators=(",", ":"), allow_nan=False).encode())
        buffers.update(a.numpy().tobytes())
    return {"original_state_dict": state.hexdigest(), "all_buffers": buffers.hexdigest()}


def census():
    files = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            p = Path(path).resolve()
            if p.is_file() and "/results/" not in str(p):
                files.add(p)
    return {str(p): sha(p) for p in sorted(files)}


def run(output):
    output = Path(output)
    if (not output.is_absolute() or output.parent != ROOT/"results" or
            not output.name.startswith(NAME+"_") or os.path.lexists(output)):
        raise ValueError("exclusive output under results required")
    if shutil.disk_usage(output.parent).free < 512*1024**2:
        raise ValueError("disk reserve")
    if sha(BUNDLE) != BUNDLE_SHA:
        raise ValueError("blind bundle mismatch")
    bundle = json.loads(BUNDLE.read_bytes())
    sources = bindings()
    # Validate all rows before allocating a GPU or creating an attempt.
    for i, row in enumerate(bundle["rows"]):
        bridge.cp5.blind.validate_row(row)
        if row["index"] != i:
            raise ValueError("row order")
    if len(bundle["rows"]) != 24:
        raise ValueError("24-row denominator")
    with locking.gpu_lock(LOCK):
        output.mkdir(mode=0o700)
        started = time.monotonic()
        manifest = {"schema": NAME+".manifest", "bundle_sha256": BUNDLE_SHA,
                    "checkpoint": MODEL_FILES, "expected_model": MODEL_HASH, "sources": sources,
                    "optimizer": CONFIG, "encoder_caps": ENC_CAPS, "wall_seconds": WALL,
                    "rows": 24, "no_training": True, "physical_execution": False,
                    "os_isolation_certified": False, "dependency_closure_complete": False,
                    "cooperative_lock_not_global": True, "P1_admission": False}
        manifest_sha = write(output/"manifest.json", manifest)
        rows, error, host, before, after = [], None, None, None, None
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            import torch
            from torch.nn.attention import SDPBackend, sdpa_kernel
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
            host = require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            model = load_compatible_pretrained(str(MODEL), device=device).eval().requires_grad_(False)
            before = model_hash(model)
            if before != MODEL_HASH:
                raise ValueError("frozen model hash mismatch")
            classes = {str(Path(inspect.getfile(type(m))).resolve()) for m in model.modules()}
            write(output/"model_classes.json", {p: sha(p) for p in sorted(classes)})
            transform = p0.image_transform()
            def persist(event):
                index = event["index"]
                suffix = {"row_start": "start", "row_complete": "result", "row_failure": "failure"}[event["kind"]]
                write(output/f"row_{index:02d}_{suffix}.json", event)
            def run_row(row):
                if time.monotonic()-started >= WALL or torch.cuda.memory_reserved(device) > 6*1024**3:
                    raise ValueError("global wall or CUDA memory cap")
                index = row["index"]
                folder = output/f"row_{index:02d}"
                folder.mkdir()
                eb = encbudget.Budget(folder/"encoder.jsonl", ENC_CAPS, WALL)
                edigest = None
                try:
                    latent = bridge.encode_cp5(row, model, eb, transform, device)
                finally:
                    edigest = eb.close()
                policy, contract = make_contract(row, sources)
                digest = spec.digest(contract)
                meter = durable.Driver(folder/"planning", contract, digest, "structured", "fixed", f"row_{index:02d}")
                try:
                    leaf = leafmod.LeafAdapter(model, meter, contract, digest)
                    result = bridge.plan_and_replay(row, latent, leaf, CONFIG, policy, contract)
                    torch.cuda.synchronize(device)
                finally:
                    receipt = meter.finish()
                account = consumer.consume(folder/"encoder.jsonl", edigest, ENC_CAPS, WALL,
                                           receipt["journal"], receipt["sha256"], digest,
                                           "structured", "fixed", f"row_{index:02d}")
                if not account["complete"]:
                    raise ValueError("ledger not complete")
                value = {"index": index, "actions": result["actions"].cpu().tolist(),
                         "score": result["scores"][0], "ledger_receipt": account}
                rows.append(value)
                return value
            with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                driver24.drive(bundle, run_row, persist)
            after = model_hash(model)
            if before != after or sources != bindings() or sha(BUNDLE) != BUNDLE_SHA:
                raise ValueError("source, model, or input drift")
            write(output/"runtime_census.json", census())
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:1500]}
        report = {"schema": NAME+".report", "status": "complete" if error is None and len(rows)==24 else "partial",
                  "error": error, "manifest_sha256": manifest_sha, "host": host, "rows_completed": len(rows),
                  "model_before": before, "model_after": after, "elapsed_seconds": time.monotonic()-started,
                  "physical_execution": False, "isolation_certified": False, "fairness_proven": False,
                  "P1_admission": False, "dependency_closure_complete": False}
        write(output/"report.json", report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args().output))
