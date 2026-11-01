"""Known-root four-arm engineering CUDA entry; never launches physics or training."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.spectral_reachability import agent_p1_joint_vreach_host_cuda_20261029_v1 as source
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import codex_four_arm_cuda_abi_probe_20261030 as prior
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_final_replay_20261030 as replay
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_raw_candidates_20261030 as variant

NAME = "codex_four_arm_host_cuda_20261030"
ROOT = Path(__file__).resolve().parents[2]
AUTH = "MAIN_AUTHORIZES_FOUR_ARM_ENGINEERING_CUDA_ONLY"
TRACE_LIMIT = 16 * 1024 * 1024
DISK_FLOOR = 256 * 1024 * 1024
WALL_NS = 600 * 10**9
ALLOC = 4 * 1024**3
RESERVED = 6 * 1024**3
WEIGHTS_SHA = "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"
CONFIG_SHA = "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resource_caps(*, dim, horizon, population, iterations):
    evals = population * (iterations + 1)
    prediction = evals * horizon + max(horizon, population)
    caps = dict.fromkeys(spec.AXES, 0)
    caps.update(action_encode_calls=prediction, predict_calls=prediction, predict_samples=prediction,
                backward_calls=dim, vjp_rows=dim, jacobian_calls=1, solve_calls=1,
                selection_evaluations=evals+population, replay_calls=evals+population)
    return caps


class Guard:
    """Cooperative wall/memory check; no preemption of an active operation."""
    def __init__(self, wall_ns, clock, memory, allocated_cap, reserved_cap):
        self.wall, self.clock, self.memory = wall_ns, clock, memory
        self.allocated, self.reserved = allocated_cap, reserved_cap
        self.start = self.last = clock()

    def check(self):
        now = self.clock()
        spec.require(type(now) is int and now >= self.last, "clock regression")
        self.last = now
        spec.require(now - self.start < self.wall, "wall cap")
        allocated, reserved = self.memory()
        spec.require(allocated <= self.allocated and reserved <= self.reserved, "CUDA memory cap")


def execute_arm(leaf, arm, states, history, goal, mean, std, config, policy, contract, *, emit):
    """Same engineering variant as run_arm, with full pre-score candidate trace."""
    try:
        spec.require(spec.digest(contract) == leaf.contract_sha and spec.digest(config) == leaf.config_sha
                     and spec.digest(policy) == contract["phase_policy_sha256"], "contract binding")
        with leaf.driver.path.open("rb") as stream:
            header = json.loads(stream.readline(1024 * 1024))["payload"]
        spec.require(arm in spec.ARMS and header["arm"] == arm and callable(emit), "arm/trace")
        spec.require(set(config) == {"schema", "iterations", "elite", "seed", "initial_std", "std_floor", "ridge"}
                     and config["schema"] == variant.CONFIG_SCHEMA, "configuration")
        n, h, horizon = len(policy["candidate_ids"]), policy["history_size"], policy["horizon"]
        spec.require(type(n) is int and n >= 2 and type(h) is int and h > 0 and type(horizon) is int
                     and horizon > 0 and type(config["iterations"]) is int and config["iterations"] > 0
                     and type(config["elite"]) is int and 0 < config["elite"] <= n
                     and type(config["seed"]) is int and 0 <= config["seed"] < 2**63, "schedule")
        for field in ("initial_std", "std_floor", "ridge"):
            spec.require(type(config[field]) in (int, float) and math.isfinite(config[field])
                         and config[field] > 0, "optimizer values")
        spec.require(config["std_floor"] <= config["initial_std"], "std floor")
        for tensor in (states, history, goal, mean, std):
            leaf._tensor(tensor)
            spec.require(tensor.is_floating_point() and leafmod._finite(tensor)
                         and tensor.device == states.device and tensor.dtype == states.dtype, "input tensor")
        spec.require(states.ndim == 3 and states.shape[:2] == (1, h) and history.shape == (1, h-1, 10)
                     and goal.shape == (1, states.shape[2]) and mean.shape == std.shape == (2,)
                     and bool((std > 0).all().item()), "task shapes")
        spec.require(torch.equal(mean, torch.tensor(policy["mean"], device=mean.device, dtype=mean.dtype))
                     and torch.equal(std, torch.tensor(policy["std"], device=std.device, dtype=std.dtype)),
                     "normalizer mismatch")
        gen = torch.Generator(device="cpu").manual_seed(config["seed"])
        shape = (1, horizon, 5, 2)
        center = torch.zeros(shape, dtype=states.dtype, device=states.device)
        scale = torch.full_like(center, config["initial_std"])

        def save_tensor(t):
            return t.detach().cpu().tolist()

        def sample(count):
            noise = torch.randn((count, horizon, 5, 2), generator=gen, dtype=states.dtype, device="cpu")
            return (center + scale * noise.to(states.device)).clamp(-1, 1)

        def evaluate(samples):
            return replay.replay_and_select(leaf, states, history, samples, goal, mean, std, policy, contract)

        if arm == "random":
            center = sample(1).detach()
        elif arm == "immediate_progress":
            trials = sample(n)
            emit("screening_candidates", {"raw": save_tensor(trials)})
            scores = []
            with torch.no_grad():
                for i in range(n):
                    leaf._dispatch("screening", "replay", {"replay_calls": 1}, lambda: None)
                    predicted = variant._normal(leaf, "screening", states, history, trials[i:i+1, :1], mean, std)
                    def score():
                        result = float((predicted-goal).square().mean().item())
                        spec.require(math.isfinite(result), "nonfinite score")
                        return result
                    score_value = leaf._dispatch("screening", "selection", {"selection_evaluations": 1}, score)
                    scores.append(score_value)
                    emit("screening_score", {"index": i, "score": score_value})
            idx = min(range(n), key=lambda i: (scores[i], i))
            emit("screening_selection", {"index": idx})
            center = trials[idx:idx+1]
        elif arm == "structured":
            leaf._dispatch("jacobian", "jacobian", {"jacobian_calls": 1}, lambda: None)
            raw = center.detach().requires_grad_(True)
            predicted = variant._normal(leaf, "jacobian", states.detach(), history.detach(), raw, mean, std)
            dim = predicted.shape[1]
            rows = []
            for i in range(dim):
                row = leaf._dispatch("jacobian", "backward", {"backward_calls": 1, "vjp_rows": 1},
                    lambda i=i: torch.autograd.grad(predicted[0, i], raw, retain_graph=i < dim-1)[0])
                spec.require(leafmod._finite(row), "nonfinite Jacobian")
                rows.append(row.reshape(-1))
            jac = torch.stack(rows)
            target = (goal.detach()-predicted.detach()).reshape(-1)
            emit("structured_signal", {"nominal": save_tensor(predicted), "jacobian": save_tensor(jac)})
            def solve():
                eye = torch.eye(jac.shape[1], dtype=jac.dtype, device=jac.device)
                return torch.linalg.solve(jac.T @ jac + config["ridge"]*eye, jac.T @ target)
            correction = leaf._dispatch("initialization", "solve", {"solve_calls": 1}, solve)
            spec.require(leafmod._finite(correction), "nonfinite structured correction")
            center = correction.reshape(shape).clamp(-1, 1).detach()
        emit("initial_center", {"raw": save_tensor(center), "arm": arm})
        for iteration in range(config["iterations"]):
            samples = sample(n)
            emit("generation_candidates", {"iteration": iteration, "raw": save_tensor(samples)})
            measured = evaluate(samples)
            elite = sorted(range(n), key=lambda i: (measured["scores"][i], i))[:config["elite"]]
            emit("generation_scores", {"iteration": iteration, "scores": measured["scores"], "elite": elite})
            chosen = samples[elite]
            center = chosen.mean(dim=0, keepdim=True)
            scale = chosen.std(dim=0, unbiased=False, keepdim=True).clamp_min(config["std_floor"])
        emit("final_replay_candidates", {"raw": save_tensor(samples)})
        final = evaluate(samples)
        index = final["selected_index"]
        emit("final_selection", {"scores": final["scores"], "index": index,
                                 "id": final["selected_id"], "raw": save_tensor(final["actions"][index:index+1])})
        return {"selected_raw": final["actions"][index:index+1].detach().clone(),
                "final_scores": final["scores"], "selected_index": index}
    except BaseException:
        leaf._abort()
        raise


def build():
    """Deterministic known-slot prepare; no model load, CUDA or physics."""
    p = source.verify(str(prior.PREP), prior.PREP_SHA)
    task = p["requests"][0]["task"]
    spec.require(len(task["goal"]) == 192 and len(task["root_context"]["latent_history"]) == 3
                 and len(task["root_context"]["action_history"]) == 2, "known task ABI")
    weights = Path(p["policy"]) / "weights.pt"
    model_config = Path(p["policy"]) / "config.json"
    spec.require(sha(weights) == WEIGHTS_SHA and sha(model_config) == CONFIG_SHA, "checkpoint file mismatch")
    config = {"schema": variant.CONFIG_SCHEMA, "iterations": 2, "elite": 1, "seed": 20261030,
              "initial_std": 0.4, "std_floor": 0.05, "ridge": 0.1}
    policy = {"schema": replay.POLICY_SCHEMA, "candidate_ids": ["candidate_0", "candidate_1", "candidate_2"],
              "horizon": 3, "history_size": 3, "metric": "terminal_latent_mean_squared_error",
              "tie": "first_declared_candidate", "replay_batch": 1,
              "action_chart": "raw_packet_5x2_box_minus1_to1",
              "mean": task["action_mean"], "std": task["action_std"]}
    from experiments import checkpoint_compat, gpu_preflight
    site_model = ROOT / ".venv/lib/python3.12/site-packages/stable_worldmodel/wm/lewm/lewm.py"
    modules = (sys.modules[__name__], spec, durable, consumer, leafmod, replay, variant, transport,
               source, prior, checkpoint_compat, gpu_preflight)
    paths = sorted({str(Path(m.__file__).resolve()) for m in modules} | {str(site_model.resolve())})
    sources = {path: sha(path) for path in paths}
    identity = {"root": task["binding"]["root_sha256"], "task": spec.digest(task),
                "model": p["expected_model_hash"], "action_domain": task["binding"]["action_domain_sha256"],
                "history": spec.digest(task["root_context"]), "goal": spec.digest(task["goal"]),
                "optimizer": sha(__file__), "optimizer_config": spec.digest(config),
                "meter": sha(durable.__file__), "hardware": spec.digest({"device_policy": "host_cuda_0"})}
    arms = {arm: {"initializer_sha256": spec.digest({"source_sha256": sha(__file__), "arm": arm}),
                  "optimizer_sha256": identity["optimizer"],
                  "optimizer_config_sha256": identity["optimizer_config"]} for arm in spec.ARMS}
    caps = resource_caps(dim=192, horizon=3, population=3, iterations=2)
    contract = {"schema": spec.NAME + ".contract.v1", "identity": identity, "arms": arms,
                "scope": "standalone_per_root_per_arm_including_attributable_setup",
                "phase_policy_sha256": spec.digest(policy),
                "levels": [{"id": "engineering", "caps": caps},
                           {"id": "unused_larger", "caps": {k: v+1 for k, v in caps.items()}}]}
    spec.preflight(contract, spec.digest(contract))
    return {"schema": NAME + ".prepare.v1", "known_request_slot": 0, "source_prepare_sha256": prior.PREP_SHA,
            "task": task, "checkpoint_directory": p["policy"], "checkpoint_weights_sha256": WEIGHTS_SHA,
            "checkpoint_config_sha256": CONFIG_SHA, "expected_model_state_sha256": p["expected_model_hash"],
            "config": config, "policy": policy, "contract": contract, "contract_sha256": spec.digest(contract),
            "sources": sources, "arms": list(spec.ARMS), "wall_ns": WALL_NS, "allocated_cap": ALLOC,
            "reserved_cap": RESERVED, "trace_bytes_per_arm": TRACE_LIMIT, "disk_floor": DISK_FLOOR,
            "labels": {"structured": "local_ridge_proxy_not_D2P_spectral_bank",
                       "native": "zero_center_CEM_variant_not_native_baseline",
                       "random": "random_center_CEM_variant",
                       "immediate_progress": "one_packet_screened_CEM_variant"},
            "fairness_proven": False, "physical_execution_authorized": False, "P1_admission": False}


def _write_new(path, payload):
    raw = spec.canonical(payload) + b"\n"
    with Path(path).open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short write")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(raw).hexdigest()


def _output_path(path, suffix):
    path = Path(path)
    spec.require(path.is_absolute() and path.parent == ROOT / "results" and
                 path.name.startswith(NAME + "_") and path.name.endswith(suffix)
                 and not path.exists() and not path.is_symlink() and path.parent.resolve() == path.parent,
                 "exclusive output path")
    return path


def prepare(path):
    path = _output_path(path, ".json")
    result = build()
    spec.require(shutil.disk_usage(path.parent).free >= DISK_FLOOR + 16*1024**2, "disk floor")
    return {"prepared": str(path), "sha256": _write_new(path, result)}


def read_prepare(path, expected):
    path = Path(path)
    spec.require(path.is_absolute() and path.parent == ROOT / "results" and path.is_file()
                 and not path.is_symlink() and spec.is_hash(expected) and path.stat().st_size < 8*1024**2,
                 "prepared path and external SHA")
    data = path.read_bytes()
    spec.require(hashlib.sha256(data).hexdigest() == expected, "prepared SHA")
    value = json.loads(data, object_pairs_hook=consumer.unique)
    spec.require(value == build(), "prepared source/inputs drift")
    return value


def run_cuda(output, prepared, authorization):
    spec.require(authorization == AUTH, "explicit CUDA authorization required")
    output = _output_path(output, "")
    spec.require(prepared == build(), "prepared source/inputs drift")
    spec.require(shutil.disk_usage(output.parent).free >= DISK_FLOOR + 4*(TRACE_LIMIT+durable.MAX_BYTES),
                 "disk floor and output allowance")
    output.mkdir()
    report = {"schema": NAME + ".report.v1", "status": "failed_or_partial",
              "prepared_canonical_sha256": spec.digest(prepared), "arms": [], "error": None,
              "arm_status": {arm: "not_started" for arm in spec.ARMS},
              "fairness_proven": False, "P1_admission": False, "physical_execution_authorized": False}
    started = time.monotonic_ns()
    try:
        info = require_cuda_allocation("cuda:0")
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        def memory():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        guard = Guard(WALL_NS, time.monotonic_ns, memory, ALLOC, RESERVED)
        task, policy, config = prepared["task"], prepared["policy"], prepared["config"]
        contract, contract_sha = prepared["contract"], prepared["contract_sha256"]
        with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            guard.check()
            model = load_compatible_pretrained(prepared["checkpoint_directory"], device=torch.device("cuda:0"))
            model.eval().requires_grad_(False)
            before = source.signal.model_hash(model)
            spec.require(before["original_state_dict"] == prepared["expected_model_state_sha256"], "model hash")
            guard.check()
            tensor = lambda x: torch.tensor(x, dtype=torch.float32, device="cuda:0")
            states = tensor(task["root_context"]["latent_history"])[None]
            history = tensor(task["root_context"]["action_history"])[None]
            goal, mean, std = tensor(task["goal"])[None], tensor(task["action_mean"]), tensor(task["action_std"])
            for arm in spec.ARMS:
                guard.check()
                arm_dir = output / arm
                arm_dir.mkdir()
                report["arm_status"][arm] = "failed_or_partial"
                trace = transport.Journal(arm_dir / "candidates.jsonl", TRACE_LIMIT)
                driver = durable.Driver(arm_dir / "ledger", contract, contract_sha, arm, "engineering", arm)
                result = None
                failure = None
                try:
                    adapter = leafmod.LeafAdapter(model, driver, contract, contract_sha)
                    original_emit = driver._emit
                    def checked_emit(kind, payload):
                        if kind == "intent":
                            try:
                                guard.check()
                            except BaseException:
                                original_emit("invalid", {})
                                raise
                        return original_emit(kind, payload)
                    driver._emit = checked_emit
                    def record(kind, payload):
                        guard.check()
                        spec.require(shutil.disk_usage(output).free >= DISK_FLOOR, "disk floor")
                        trace.emit(kind, payload)
                    trace.emit("header", {"schema": NAME + ".trace.v1", "contract_sha256": contract_sha,
                                          "arm": arm, "label": prepared["labels"][arm]})
                    result = execute_arm(adapter, arm, states, history, goal, mean, std, config, policy, contract,
                                         emit=record)
                    guard.check()
                    spec.require(source.signal.model_hash(model) == before, "model drift")
                    record("terminal", {"status": "complete"})
                except BaseException as exc:
                    failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
                    if not driver.failed and not driver.closed:
                        try:
                            driver.dispatch("setup", "__abort__", {}, lambda: None)
                        except BaseException:
                            pass
                    try:
                        trace.emit("terminal", {"status": "failed_or_partial", "error": failure})
                    except BaseException:
                        pass
                finally:
                    trace.close()
                    receipt = driver.finish()
                account = consumer.consume(receipt["journal"], receipt["sha256"], contract_sha,
                                           arm, "engineering", arm)
                entry = {"arm": arm, "status": "failed_or_partial" if failure else "complete",
                         "trace": str(arm_dir / "candidates.jsonl"),
                         "trace_sha256": sha(arm_dir / "candidates.jsonl"), "ledger": receipt,
                         "account": account, "error": failure}
                if result is not None and failure is None:
                    entry.update(selected_index=result["selected_index"], final_scores=result["final_scores"],
                                 selected_raw=result["selected_raw"].cpu().tolist())
                report["arms"].append(entry)
                report["arm_status"][arm] = entry["status"]
                if failure is not None:
                    break
            report["model_before"] = before
            report["model_after"] = source.signal.model_hash(model)
            spec.require(report["model_before"] == report["model_after"], "model changed")
            spec.require(prepared == build(), "source or inputs changed")
            guard.check()
            report["device"] = info
            report["peak_allocated"], report["peak_reserved"] = memory()
            if len(report["arms"]) == len(spec.ARMS) and all(a["status"] == "complete" for a in report["arms"]):
                report["status"] = "complete_engineering_not_admitted"
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    report["elapsed_ns"] = time.monotonic_ns() - started
    report_sha = _write_new(output / "report.json", report)
    return {"status": report["status"], "report_sha256": report_sha, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--output", required=True)
    b = sub.add_parser("run")
    b.add_argument("--prepared", required=True)
    b.add_argument("--sha256", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    else:
        result = run_cuda(args.output, read_prepare(args.prepared, args.sha256), args.authorization)
    print(json.dumps(result, sort_keys=True))
    return 0 if args.command == "prepare" or result["status"] == "complete_engineering_not_admitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
