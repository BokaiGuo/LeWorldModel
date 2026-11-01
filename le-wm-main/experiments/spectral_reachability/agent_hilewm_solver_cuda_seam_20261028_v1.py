"""MAIN-only CUDA analytic CEM integration; no neural model or CPU fallback."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

TAG = "agent_hilewm_solver_cuda_seam_20261028_v1"
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
TEST = ROOT / "tests" / ("test_" + TAG + ".py")
DOC = ROOT / "innerdocs" / (TAG + ".md")
BRIDGE = SOURCE.with_name("agent_hilewm_solver_batch_bridge_20261028_v1.py")
BRIDGE_SHA = "a651d5f0710d18bea1f9b2caaf3283286816ca775dd32dcdc619f11d219c448b"
PACKAGE = ROOT / ".venv/lib/python3.12/site-packages/stable_worldmodel"
CAP = 8 * 1024 * 1024
KEYS = ("candidates", "costs", "topk_vals", "topk_inds", "topk_candidates",
        "mean", "var", "prev_mean", "prev_var")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    path = Path(path)
    require(path.stat().st_size <= CAP, "JSON cap")
    return json.loads(path.read_bytes())


def protocol():
    return {"kind": "non_neural_analytic_installed_CEM_CUDA_seam", "batches": [5, 2],
            "chunk": 2, "samples": 8, "iterations": 3, "topk": 3, "horizon": 3,
            "dim": 2, "dtype": "float32", "var_scale": 1.0, "seed_base": 1729,
            "initial_action": "full_horizon_zero_CUDA_no_actor", "cost": "sum_H_D((a-z_goal)^2)",
            "cost_calls_cap": 12, "candidate_rows_cap": 168,
            "predict_API_calls": 0, "predictor_samples": 0, "AD": 0, "physics": 0,
            "output_cap_bytes": CAP, "floor_bytes": 200 * 1024 * 1024,
            "wall_seconds": 120, "saved_float_rtol": 8 * float(np.finfo(np.float32).eps),
            "saved_float_atol": 8 * float(np.finfo(np.float32).eps),
            "scientific_admission": False, "official_model_result": False}


def sources():
    files = [SOURCE, TEST, DOC, BRIDGE]
    files += [PACKAGE / p for p in (
        "__init__.py", "protocols.py", "solver/__init__.py", "solver/cem.py",
        "solver/utils.py", "solver/solver.py", "solver/callbacks/__init__.py",
        "solver/callbacks/common.py", "solver/callbacks/cem.py", "solver/callbacks/gd.py")]
    return {str(p): digest(p.read_bytes()) for p in files}


def bridge_module():
    require(digest(BRIDGE.read_bytes()) == BRIDGE_SHA, "frozen bridge SHA")
    name = "owned_hilewm_frozen_bridge_for_cuda_seam"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BRIDGE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def prepare(out):
    b = bridge_module()
    m = {"schema": TAG + ".prepare", "protocol": protocol(), "sources": sources(),
         "owner_evidence": b.source_evidence()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    path = out / "manifest.json"
    with path.open("xb") as handle:
        handle.write(encode(m))
    return path


def verify(manifest):
    m = read_json(manifest)
    require(m == {"schema": TAG + ".prepare", "protocol": protocol(), "sources": sources(),
                  "owner_evidence": bridge_module().source_evidence()}, "prepare/source/protocol drift")
    return m


def load_runtime():
    """Only run_cuda calls this, after explicit MAIN authorization + verify."""
    import torch
    require(torch.cuda.is_available(), "HOST CUDA required; CPU fallback forbidden")
    from stable_worldmodel.solver.cem import CEMSolver
    from stable_worldmodel.solver import utils
    from gymnasium.spaces import Box
    require(Path(inspect.getfile(CEMSolver)).resolve() == (PACKAGE / "solver/cem.py").resolve(),
            "installed CEM origin")
    require(Path(utils.__file__).resolve() == (PACKAGE / "solver/utils.py").resolve(), "utils origin")
    return torch, CEMSolver, Box


def cuda_abi(torch, device):
    """Exact tensor bytes witness; host copies only for serialization, not costs."""
    b = bridge_module()
    def accepts(tensor):
        return torch.is_tensor(tensor) and tensor.device == device and tensor.dtype == torch.float32
    def fingerprint(tensor):
        require(accepts(tensor), "CUDA float32 tensor required")
        array = tensor.detach().contiguous().cpu().numpy()
        return {**b.numpy_fingerprint(array), "device": str(tensor.device),
                "stride": list(tensor.stride()), "requires_grad": tensor.requires_grad}
    return b.TensorABI(accepts, fingerprint, "actual_torch_CUDA_float32")


def dump_array(out, name, array, inventory):
    require(name not in inventory and Path(name).suffix == ".npy", "exclusive array name")
    require(not array.dtype.hasobject and np.isfinite(array).all(), "finite nonobject array")
    path = Path(out) / name
    require(path.parent == Path(out), "flat array names only")
    # Conservative .npy header allowance before writing, actual bound afterward.
    used = sum(x["file_bytes"] for x in inventory.values())
    require(used + array.nbytes + 512 <= CAP - 1024 * 1024, "before_write array cap")
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    raw = path.read_bytes()
    inventory[name] = {"file_sha256": digest(raw), "file_bytes": len(raw),
                       "data_sha256": digest(array.tobytes(order="C")),
                       "dtype": array.dtype.str, "shape": list(array.shape)}
    return name


def check_deadline(start):
    require(time.monotonic() - start <= protocol()["wall_seconds"], "cooperative wall cap")


def run_case(torch, CEMSolver, Box, batch, out, report, case, start):
    p = protocol()
    device = torch.device("cuda", torch.cuda.current_device())
    abi = cuda_abi(torch, device)
    b = bridge_module()
    ids = torch.arange(batch, dtype=torch.float32, device=device)
    z_init = torch.stack((ids, ids + 0.5), dim=1)
    z_goal = torch.stack(((ids + 1) / 8, -(ids + 1) / 16), dim=1)
    info = {"planner_level": "high", "z_init": z_init, "z_goal": z_goal}
    sentinel = torch.zeros((), device=device, dtype=torch.float32)
    case.update(events=[], cost_attempts=[], resets=0, starts=0, ends=0)
    pending = {}
    current = {"chunk": -1}

    def save(name, tensor):
        return dump_array(out, f"B{batch}_{name}.npy", tensor.detach().cpu().numpy().copy(), report["arrays"])

    class Analytic:
        def parameters(self):
            return iter((sentinel,))  # dtype-only sentinel, no learned parameters

        def get_cost(self, cost_info, candidates):
            check_deadline(start)
            index = len(case["cost_attempts"])
            offset = current["chunk"] * p["chunk"]
            size = min(p["chunk"], batch-offset)
            attempt = {"index": index, "chunk": current["chunk"], "returned": False,
                       "candidate_rows": size * p["samples"]}
            case["cost_attempts"].append(attempt)  # before validation/dispatch; never refunded
            require(cost_info["planner_level"] == "high", "complete high metadata")
            require(candidates.is_cuda and not candidates.requires_grad, "CUDA inference candidates")
            for key in ("z_init", "z_goal"):
                value = cost_info[key]
                expected = info[key][offset:offset+size, None, :].expand(size, p["samples"], 2)
                require(value.device == device and value.dtype == torch.float32
                        and tuple(value.shape) == (size, p["samples"], 2)
                        and value.stride(1) == 0 and torch.equal(value, expected),
                        "actual torch expand/row lineage")
                attempt[key] = save(f"cost{index}_{key}", value)
            cost = (candidates - cost_info["z_goal"][:, :, None, :]).square().sum(dim=(2, 3))
            require(cost.is_cuda and not cost.requires_grad, "analytic cost CUDA no AD")
            attempt["cost"] = save(f"cost{index}_value", cost)
            attempt["expand_sample_stride_zero"] = True
            attempt["returned"] = True
            pending["attempt"] = attempt
            return cost

    class Recorder:
        output_key = TAG

        def __init__(self):
            self.history = []

        def reset(self):
            case["resets"] += 1

        def start_batch(self):
            current["chunk"] += 1
            case["starts"] += 1

        def end_solve(self):
            case["ends"] += 1

        def __call__(self, **state):
            check_deadline(start)
            attempt = pending.pop("attempt")
            event = {"step": state["step"], "chunk": current["chunk"],
                     "attempt": attempt["index"], "arrays": {}}
            case["events"].append(event)
            for key in KEYS:
                require(state[key].device == device and not state[key].requires_grad, "callback CUDA no AD")
                event["arrays"][key] = save(f"event{len(case['events'])-1}_{key}", state[key])
            vals, inds = torch.topk(state["costs"], k=p["topk"], dim=1, largest=False)
            rows = torch.arange(len(vals), device=device)[:, None].expand(-1, p["topk"])
            elites = state["candidates"][rows, inds]
            checks = {"topk_vals": torch.equal(vals, state["topk_vals"]),
                      "topk_inds": torch.equal(inds, state["topk_inds"]),
                      "gather": torch.equal(elites, state["topk_candidates"]),
                      "mean": torch.equal(elites.mean(dim=1), state["mean"]),
                      "sample_std": torch.equal(elites.std(dim=1), state["var"]),
                      "mean_candidate": torch.equal(state["candidates"][:, 0], state["prev_mean"])}
            event["cuda_exact_checks"] = checks
            require(all(checks.values()), "native CEM CUDA update checks")
            self.history.append({"chunk": current["chunk"], "step": state["step"]})

    facade = b.HighBatchBridge.bind_high_model(
        info, Analytic(), tensor_batch=batch, chunk_size=p["chunk"], num_samples=p["samples"],
        abi=abi, max_cost_calls=((batch+1)//2)*p["iterations"],
        max_candidate_rows=batch*p["samples"]*p["iterations"])
    callback = Recorder()
    solver = CEMSolver(model=facade, batch_size=p["chunk"], num_samples=p["samples"],
                       var_scale=p["var_scale"], n_steps=p["iterations"], topk=p["topk"],
                       device=device, seed=p["seed_base"]+batch, callbacks=[callback])
    config = SimpleNamespace(horizon=p["horizon"], action_block=1)
    solver.configure(action_space=Box(-1.0, 1.0, shape=(batch, p["dim"]), dtype=np.float32),
                     n_envs=batch, config=config)
    initial = torch.zeros((batch, p["horizon"], p["dim"]), device=device, dtype=torch.float32)
    try:
        with torch.inference_mode():
            outputs = solver.solve(dict(facade.solver_info), init_action=initial)
        torch.cuda.synchronize()
        case["outputs"] = {key: save("output_"+key, outputs[key] if key == "actions" else outputs[key][0])
                           for key in ("actions", "mean", "var")}
        case["outputs"]["costs"] = dump_array(out, f"B{batch}_output_costs.npy",
                                                    np.asarray(outputs["costs"], dtype=np.float32), report["arrays"])
        case["callback_output"] = outputs["callbacks"][TAG]
    finally:
        case["bridge_receipt"] = facade.receipt()


def run_cuda(manifest, out, *, main_authorized=False):
    require(main_authorized is True, "explicit MAIN authorization required")
    verify(manifest)
    start = time.monotonic()
    torch, solver, box = load_runtime()  # no output or CEM invocation if CUDA unavailable
    torch.cuda.synchronize()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    import shutil
    require(shutil.disk_usage(out).free >= protocol()["floor_bytes"], "disk floor")
    report = {"schema": TAG + ".report", "manifest_sha256": digest(Path(manifest).read_bytes()),
              "protocol": protocol(), "runtime": {"torch": str(torch.__version__),
              "cuda": str(torch.version.cuda), "device": torch.cuda.get_device_name()},
              "cases": [], "arrays": {}, "errors": [], "completed": False}
    try:
        for batch in protocol()["batches"]:
            check_deadline(start)
            case = {"batch": batch}
            report["cases"].append(case)
            run_case(torch, solver, box, batch, out, report, case, start)
            check_deadline(start)
        report["completed"] = True
    except Exception as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    report["elapsed_seconds"] = time.monotonic()-start
    raw = encode(report)
    require(len(raw) + sum(v["file_bytes"] for v in report["arrays"].values()) <= CAP, "output cap")
    with (out / "report.json").open("xb") as handle:
        handle.write(raw)
    return report


def close_float(actual, expected, message):
    p = protocol()
    require(np.isfinite(actual).all() and np.isfinite(expected).all()
            and np.allclose(actual, expected, rtol=p["saved_float_rtol"], atol=p["saved_float_atol"]), message)


def check_fingerprint(fp, value, *, expanded=False):
    require(fp["shape"] == list(value.shape) and fp["dtype"] == value.dtype.str
            and fp["sha256"] == digest(value.tobytes(order="C")) and fp["bytes"] == value.nbytes
            and fp["requires_grad"] is False and fp["device"].startswith("cuda:")
            and len(fp["stride"]) == value.ndim, "bridge tensor byte/device join")
    if expanded:
        require(fp["stride"][1] == 0, "expanded candidate stride")


def validate_case(case, arrays):
    """RAM saved recomputation, no torch/foreign solver import."""
    p = protocol()
    batch = case["batch"]
    require(batch in p["batches"], "batch identity")
    chunks = (batch+1)//2
    count = chunks*p["iterations"]
    require(len(case["events"]) == len(case["cost_attempts"]) == count, "complete cost/iteration denominator")
    require((case["resets"], case["starts"], case["ends"]) == (1, chunks, 1), "callback lifecycle")
    receipt = case["bridge_receipt"]
    require(receipt["backend"] == "actual_torch_CUDA_float32" and receipt["tensor_batch"] == batch
            and receipt["cost_attempts"] == receipt["cost_returns"] == count
            and receipt["cost_failures"] == 0
            and receipt["candidate_rows_attempted"] == batch*p["samples"]*p["iterations"], "bridge counters")
    require(len(receipt["calls"]) == count and all(c["returned"] and c["metadata"] == "high"
            for c in receipt["calls"]), "full bridge call records")
    witness = receipt["input_witness"]
    require(witness["original_keys"] == ["planner_level", "z_init", "z_goal"]
            and witness["original_metadata"] == {"planner_level": "high"}
            and witness["solver_keys"] == ["z_init", "z_goal"]
            and witness["original_tensors"] == witness["solver_tensors"], "input dual witness")
    ids_all = np.arange(batch, dtype=np.float32)
    for key, value in (("z_init", np.stack((ids_all, ids_all+0.5), axis=1)),
                       ("z_goal", np.stack(((ids_all+1)/8, -(ids_all+1)/16), axis=1))):
        check_fingerprint(witness["original_tensors"][key], value)
    history = []
    final_mean, final_var, final_cost = [], [], []
    for index, (event, attempt) in enumerate(zip(case["events"], case["cost_attempts"])):
        chunk, step = divmod(index, p["iterations"])
        n = min(2, batch-2*chunk)
        require((event["chunk"], event["step"], event["attempt"]) == (chunk, step, index), "iteration order")
        require(attempt["index"] == index and attempt["chunk"] == chunk and attempt["returned"]
                and attempt["candidate_rows"] == n*p["samples"] and attempt["expand_sample_stride_zero"], "cost attempt")
        require(set(event["arrays"]) == set(KEYS) and set(event["cuda_exact_checks"]) ==
                {"topk_vals", "topk_inds", "gather", "mean", "sample_std", "mean_candidate"}
                and all(v is True for v in event["cuda_exact_checks"].values()), "exact CUDA checks")
        x = {k: arrays[v] for k, v in event["arrays"].items()}
        shapes = {"candidates": (n, 8, 3, 2), "costs": (n, 8), "topk_vals": (n, 3),
                  "topk_inds": (n, 3), "topk_candidates": (n, 3, 3, 2),
                  **{k: (n, 3, 2) for k in ("mean", "var", "prev_mean", "prev_var")}}
        require(all(v.shape == shapes[k] and v.dtype == (np.int64 if k == "topk_inds" else np.float32)
                    for k, v in x.items()), "callback state ABI")
        c = x["candidates"]
        require(c.shape == (n, p["samples"], p["horizon"], p["dim"]) and c.dtype == np.float32, "candidates ABI")
        ids = np.arange(2*chunk, 2*chunk+n, dtype=np.float32)
        zi = np.stack((ids, ids+0.5), axis=1)
        zg = np.stack(((ids+1)/8, -(ids+1)/16), axis=1)
        for key, expected in (("z_init", zi), ("z_goal", zg)):
            require(np.array_equal(arrays[attempt[key]], np.repeat(expected[:, None], p["samples"], axis=1)), "source row identity")
        call = receipt["calls"][index]
        require(call["attempt"] == index+1 and call["batch"] == n
                and call["candidate_rows"] == n*p["samples"], "bridge per-attempt join")
        check_fingerprint(call["candidates"], c)
        check_fingerprint(call["cost"], x["costs"])
        for key in ("z_init", "z_goal"):
            check_fingerprint(call["expanded"][key], arrays[attempt[key]], expanded=True)
        close_float(x["costs"], ((c.astype(np.float64)-zg[:, None, None, :])**2).sum(axis=(2, 3)), "analytic cost")
        require(np.array_equal(x["costs"], arrays[attempt["cost"]]), "cost callback identity")
        inds = x["topk_inds"]
        require(inds.shape == (n, p["topk"]) and inds.dtype == np.int64
                and ((inds >= 0) & (inds < p["samples"])).all()
                and all(len(set(row.tolist())) == p["topk"] for row in inds), "elite indices")
        rows = np.arange(n)[:, None]
        elites = c[rows, inds]
        require(np.array_equal(elites, x["topk_candidates"])
                and np.array_equal(x["costs"][rows, inds], x["topk_vals"]), "elite gather")
        require(np.array_equal(np.sort(x["costs"], axis=1)[:, :p["topk"]], x["topk_vals"]), "topk smallest")
        close_float(x["mean"], elites.astype(np.float64).mean(axis=1), "mean update")
        close_float(x["var"], elites.astype(np.float64).std(axis=1, ddof=1), "sample std update")
        if step == 0:
            require(np.array_equal(x["prev_mean"], np.zeros((n, 3, 2), dtype=np.float32))
                    and np.array_equal(x["prev_var"], np.ones((n, 3, 2), dtype=np.float32)), "initial distribution")
        else:
            prev = case["events"][index-1]["arrays"]
            require(np.array_equal(x["prev_mean"], arrays[prev["mean"]])
                    and np.array_equal(x["prev_var"], arrays[prev["var"]]), "update chain")
        require(np.array_equal(c[:, 0], x["prev_mean"]), "first candidate mean")
        history.append({"chunk": chunk, "step": step})
        if step == p["iterations"]-1:
            final_mean.append(x["mean"])
            final_var.append(x["var"])
            final_cost.append(x["topk_vals"].astype(np.float64).mean(axis=1))
    require(case["callback_output"] == history, "full callback output")
    require(set(case["outputs"]) == {"actions", "mean", "var", "costs"}, "output keys")
    for key, expected in (("actions", np.concatenate(final_mean)), ("mean", np.concatenate(final_mean)),
                          ("var", np.concatenate(final_var))):
        require(np.array_equal(arrays[case["outputs"][key]], expected), "native output " + key)
    close_float(arrays[case["outputs"]["costs"]], np.concatenate(final_cost), "native elite logging cost")


def consume(manifest, report_path):
    verify(manifest)
    report_path = Path(report_path)
    r = read_json(report_path)
    require(r["schema"] == TAG + ".report" and r["manifest_sha256"] == digest(Path(manifest).read_bytes())
            and r["protocol"] == protocol() and r["completed"] is True and r["errors"] == [], "complete bound report")
    require(0 <= r["elapsed_seconds"] <= protocol()["wall_seconds"], "saved wall cap")
    require([c["batch"] for c in r["cases"]] == protocol()["batches"], "all fixed cases")
    require(sum(x["file_bytes"] for x in r["arrays"].values()) + report_path.stat().st_size <= CAP, "saved cap")
    arrays = {}
    for name, spec in r["arrays"].items():
        path = report_path.parent / name
        require(path.parent == report_path.parent and path.suffix == ".npy" and not path.is_symlink(), "array path")
        require(path.stat().st_size == spec["file_bytes"] <= CAP, "array file size")
        raw = path.read_bytes()
        require(digest(raw) == spec["file_sha256"], "array file SHA")
        value = np.load(path, allow_pickle=False)
        require(list(value.shape) == spec["shape"] and value.dtype.str == spec["dtype"]
                and not value.dtype.hasobject and np.isfinite(value).all()
                and digest(value.tobytes(order="C")) == spec["data_sha256"], "array payload")
        arrays[name] = value
    used = set()
    for case in r["cases"]:
        used.update(case["outputs"].values())
        for event in case["events"]:
            used.update(event["arrays"].values())
        for attempt in case["cost_attempts"]:
            used.update(attempt[k] for k in ("z_init", "z_goal", "cost"))
        validate_case(case, arrays)
    require(used == set(arrays), "no extra/omitted arrays")
    return {"schema": TAG + ".consume", "status": "all2_analytic_CUDA_CEM_cases_saved_verified",
            "report_sha256": digest(report_path.read_bytes()), "manifest_sha256": r["manifest_sha256"],
            "cost_attempts": 12, "cost_returns": 12, "candidate_rows_attempted": 168,
            "predict_API_calls": 0, "predictor_samples": 0, "AD": 0, "physics": 0,
            "scientific_admission": False, "official_model_result": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run", "consume"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--main-authorized", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.out is not None, "--out required")
        result = {"manifest": str(prepare(args.out))}
    else:
        require(args.manifest is not None, "--manifest required")
        if args.command == "verify":
            verify(args.manifest)
            result = {"verified": True}
        elif args.command == "run":
            require(args.out is not None, "--out required")
            result = run_cuda(args.manifest, args.out, main_authorized=args.main_authorized)
        else:
            require(args.report is not None, "--report required")
            result = consume(args.manifest, args.report)
            if args.out is not None:
                with args.out.open("xb") as handle:
                    handle.write(encode(result))
    print(json.dumps(result, indent=2))
    if args.command == "run" and not result["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
