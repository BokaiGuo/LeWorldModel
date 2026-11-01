"""Exclusive host call-site adapters and saved-RGB CUDA visual integration probe."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

from . import p1_visual_sink_guard_20261101_leibniz as visual_guard
from . import p1_total_guard_host_bridge_20261101_leibniz as bridge
from . import codex_p1_blind_uphill_bank_20261101_exclusive as blind
from . import codex_four_arm_rgb_physical_20261030 as rgb
from . import codex_p1_full_cem_host_20261101_exclusive as cem
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking

g, visual = visual_guard.total, visual_guard.visual
ROOT = blind.ROOT
NAME = "p1_visual_guard_host_20261101_leibniz"
HERE = Path(__file__).resolve()
DOC = ROOT.parent / "docs" / (NAME + ".md")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
AUTH = "AUTHORIZE_P1_SAVED_VISUAL_PROBE_CUDA"
SREPORT = ROOT / "results/codex_p1_common_structured_cp10_host_20261101_exclusive_RUN_CODEX01/report.json"
CREPORT = ROOT / "results/p1_common_budget_cem_host_20261101_leibniz_root0_RUN_LEIBNIZ01/report.json"
SHAS = ("6c49744a051ab327ca91c22f0aa5c19261eacff51c472fdf0b1cd4915d80f4cd",
        "28d1df63a0aa883256229d1a7fb5a9d2b78e96086f79b29ef9205703dee8a6c2")
ENCODING = SREPORT.parent / "s0_structured_b00_encoding.json"
LATENT = SREPORT.parent / "s0_structured_b00_encode/native192.npy"


class MovingProxy:
    """Local per-execution redirect; never mutate the sealed moving module."""
    def __init__(self, guard):
        self.guard = guard

    def __getattr__(self, name):
        if name == "encode_boundary":
            return lambda *args: visual_guard.encode_boundary(self.guard, *args)
        return getattr(rgb.moving, name)


def structured_episode(guard):
    """Clone the actual structured episode with a private moving global."""
    env = dict(rgb.episode.__globals__)
    env["moving"] = MovingProxy(guard)
    return FunctionType(rgb.episode.__code__, env, "guarded_structured_episode",
                        rgb.episode.__defaults__, rgb.episode.__closure__)


def cem_runner(guard, runner):
    """Private runner.rgb.moving redirect used by the actual CEM execute_one."""
    copied_rgb = SimpleNamespace(**{**vars(runner.rgb), "moving": MovingProxy(guard)})
    return SimpleNamespace(**{**vars(runner), "rgb": copied_rgb})


def require_full_host_ready():
    bridge.require_host_ready()


def prepare_value():
    task, _, source_files, policy = blind.load(0)
    prepared = json.loads((ROOT / "results" / blind.ITEMS[0][0]).read_bytes())
    reports = []
    for path, sha in zip((SREPORT, CREPORT), SHAS, strict=True):
        g.require(blind.host.sha(path) == sha, "pinned report")
        report = json.loads(path.read_bytes())
        root = [x for x in report["rows"][0]["checkpoints"] if x["step"] == 0]
        g.require(len(root) == 1 and blind.host.sha(Path(root[0]["path"])) == root[0]["file_sha256"],
                  "root image binding")
        reports.append((report, root[0]))
    g.require(reports[0][0]["cp10_goal_binding"] == reports[1][0]["cp10_goal_binding"]
              and reports[0][0]["model_before"] == reports[1][0]["model_before"]
              and reports[0][1]["pixel_sha256"] == reports[1][1]["pixel_sha256"], "paired root/model/RGB")
    encoded = json.loads(ENCODING.read_bytes())
    g.require(blind.host.sha(LATENT) == blind.ITEMS[0][-1]
              and encoded["response"]["latent"]["file_sha256"] == blind.ITEMS[0][-1], "prior root encoding")
    sources = dict(prepared["sources"])
    sources.update(source_files)
    for path in (HERE, DOC, TEST, Path(visual_guard.__file__).resolve(), Path(g.__file__).resolve(),
                 Path(bridge.__file__).resolve(), Path(visual_guard.encoding.__file__).resolve(),
                 Path(visual_guard.moving.__file__).resolve(), Path(visual.__file__).resolve(),
                 Path(rgb.__file__).resolve(), Path(cem.__file__).resolve(),
                 SREPORT, CREPORT, ENCODING, LATENT, *(Path(x[1]["path"]) for x in reports)):
        h = blind.host.sha(path)
        g.require(str(path) not in sources or sources[str(path)] == h, "source conflict")
        sources[str(path)] = h
    g.check_sources({"source_sha256": sources})
    return {"schema": NAME + ".prepare.v1", "sources": sources, "policy": policy,
            "model_hash": reports[0][0]["model_before"], "root_binding": reports[0][0]["cp10_goal_binding"],
            "task_sha256": blind.host.digest(task), "root_images": {a: r[1] for a, r in zip(("structured", "cem"), reports)},
            "request_template": encoded["request"], "expected_latent_path": str(LATENT),
            "scope": "saved_rgb_visual_handler_only_not_full_host", "physical_execution": False,
            "P1_admission": False}


def contract(value, arm):
    caps = dict.fromkeys(g.AXES, 0)
    caps.update(model_load=1, source_encode_calls=1, source_encode_samples=1)
    h = value["root_binding"]
    identity = {"root": h["root_sha256"], "task": value["task_sha256"], "goal": h["goal_latent_sha256"],
                "model": value["model_hash"]["original_state_dict"], "action_normalizer": "0" * 64,
                "hardware": g.digest({"device": "host_cuda_0"}), "method": g.digest(arm),
                "controller": g.digest("saved_rgb_probe_no_controller")}
    return {"schema": g.NAME + ".contract.v1", "run_id": NAME + "_" + arm, "arm": arm, "caps": caps,
            "source_sha256": value["sources"], "identity": identity, "hook_modules": g.HOOKS,
            "shared_initialization": "full_standalone_charge_each_arm",
            "scope": "per_root_per_arm_start_before_load"}


class Store(visual_guard.moving.old.Store):
    """Existing durable visual Store with exclusively named new output."""
    def __init__(self, path):
        path = Path(path)
        g.require(path.is_absolute() and path.parent == ROOT / "results" and path.name.startswith(NAME + "_RUN_")
                  and not path.exists() and not path.is_symlink(), "exclusive output")
        path.mkdir()
        self.path, self.previous, self.index = path, None, 0


def run_probe(prepared, expected, output):
    from experiments import checkpoint_compat, gpu_preflight
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    prepared = Path(prepared)
    g.require(prepared.is_absolute() and prepared.parent == ROOT / "results"
              and blind.host.sha(prepared) == expected, "prepare hash")
    value = json.loads(prepared.read_bytes())
    g.require(value == prepare_value(), "prepare/source drift")
    g.validate_pair(contract(value, "structured"), contract(value, "cem"))
    with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        store = Store(output)
        rows = []
        for arm in ("structured", "cem"):
            c = contract(value, arm)
            guard = g.Guard(store.path / (arm + "_guard.jsonl"), c, g.digest(c))
            failure, host, before, after, response, error_max = None, None, None, None, None, None
            try:
                os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
                host = gpu_preflight.require_cuda_allocation("cuda:0")
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                device = torch.device("cuda:0")
                with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                    model = bridge.guarded_load(guard, lambda: checkpoint_compat.load_compatible_pretrained(
                        value["policy"], device=device).eval().requires_grad_(False))
                    before = blind.host.signal.model_hash(model)
                    g.require(before == value["model_hash"], "model hash")
                    image = value["root_images"][arm]
                    data = Path(image["path"]).read_bytes()
                    name = f"{arm}_probe_plan_05.png"
                    g.require(store.write(name, data) == image["file_sha256"], "saved RGB bytes")
                    request = deepcopy(value["request_template"])
                    request.update(episode_id=arm + "_probe", png={"path": str(store.path / name),
                                   "file_sha256": image["file_sha256"], "pixel_sha256": image["pixel_sha256"]})
                    request["request_sha256"] = visual.source.digest({k: v for k, v in request.items()
                                                                       if k != "request_sha256"})
                    meter = visual.Meter(store)
                    meter.episode, meter.region = arm + "_probe", "encoding"
                    handler = visual.Handler(None)
                    def execute():
                        sink = visual.Sink(store, meter, arm + "_probe", "encode")
                        result = handler.encode_observation(request, sink, model=model, device=device,
                                                            check_limits=lambda: None)
                        sink.complete()
                        return result
                    response = visual_guard.guarded_boundary(guard, meter, store, execute)
                    latent = store.load_array(response["latent"])
                    expected_latent = np.load(value["expected_latent_path"], allow_pickle=False)
                    error_max = float(np.max(np.abs(latent - expected_latent)))
                    g.require(np.allclose(latent, expected_latent, rtol=1e-5, atol=1e-6), "saved root latent mismatch")
                    after = blind.host.signal.model_hash(model)
                    g.require(before == after, "model drift")
            except BaseException as exc:
                guard.failed = True
                failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            receipt = guard.close()
            verified = g.verify_journal(receipt["journal"], receipt["sha256"], g.digest(c))
            rows.append({"arm": arm, "failure": failure, "host": host, "model_before": before, "model_after": after,
                         "rgb_sha256": value["root_images"][arm]["pixel_sha256"], "response": response,
                         "root_latent_max_abs_error": error_max, "guard": receipt, "verified": verified})
            if failure:
                break
        result = {"schema": NAME + ".probe.v1", "status": "visual_probe_complete" if len(rows) == 2
                  and all(r["failure"] is None for r in rows) else "partial", "rows": rows,
                  "prepared": str(prepared), "prepare_sha256": expected, "full_host_executed": False,
                  "physical_execution": False, "coverage_verified": False, "fairness_proven": False,
                  "full_cost_certified": False, "P1_admission": False, "missing": bridge.gate()["missing"]}
        sha = blind.write_new(store.path / "report.json", result)
        return {"status": result["status"], "report_sha256": sha, "output": str(store.path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-prepare")
    s = sub.add_parser("prepare")
    s.add_argument("--output", required=True)
    s = sub.add_parser("probe")
    for key in ("prepared", "sha256", "output", "authorization"):
        s.add_argument("--" + key, required=True)
    s.add_argument("--authorize-host-cuda", action="store_true", required=True)
    sub.add_parser("full-host")
    args = p.parse_args()
    if args.command == "full-host":
        require_full_host_ready()
    if args.command == "dry-prepare":
        value = prepare_value()
        g.validate_pair(contract(value, "structured"), contract(value, "cem"))
        result = {"status": "dry_prepared_saved_rgb_only", "digest": g.digest(value), "full_host_ready": False}
    elif args.command == "prepare":
        path = Path(args.output)
        g.require(path.is_absolute() and path.parent == ROOT / "results"
                  and path.name.startswith(NAME + "_PREP_") and path.suffix == ".json", "prepare path")
        result = {"path": str(path), "sha256": blind.write_new(path, prepare_value())}
    else:
        g.require(args.authorization == AUTH, "explicit CUDA authorization")
        result = run_probe(args.prepared, args.sha256, args.output)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "probe" and result["status"] != "visual_probe_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
