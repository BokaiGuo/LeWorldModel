"""RAM schema/control tests. CUDA-labeled fixture fields are NOT execution evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/spectral_reachability/agent_hilewm_solver_cuda_seam_20261028_v1.py"
spec = importlib.util.spec_from_file_location("owned_hilewm_cuda_seam", SOURCE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def fp(a, expanded=False):
    return {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": m.digest(a.tobytes()),
            "bytes": a.nbytes, "device": "cuda:0", "requires_grad": False,
            "stride": [2, 0, 1] if expanded else [s//a.itemsize for s in a.strides]}


def build_case(batch, out, report):
    """Own NumPy fixture for exact producer layout; no installed solver calls."""
    rng = np.random.default_rng(1729+batch)
    arrays = {}
    p = m.protocol()
    def save(name, a):
        name = f"B{batch}_{name}.npy"
        a = np.array(a, copy=True)
        arrays[name] = a
        return m.dump_array(out, name, a, report["arrays"])
    ids = np.arange(batch, dtype=np.float32)
    original = {"z_init": np.stack((ids, ids+.5), axis=1),
                "z_goal": np.stack(((ids+1)/8, -(ids+1)/16), axis=1)}
    witness = {"original_keys": ["planner_level", "z_init", "z_goal"],
               "original_metadata": {"planner_level": "high"}, "solver_keys": ["z_init", "z_goal"],
               "original_tensors": {k: fp(v) for k, v in original.items()},
               "solver_tensors": {k: fp(v) for k, v in original.items()}}
    case = {"batch": batch, "resets": 1, "starts": (batch+1)//2, "ends": 1,
            "events": [], "cost_attempts": [], "callback_output": []}
    calls, means, stds, costs = [], [], [], []
    for chunk, start in enumerate(range(0, batch, 2)):
        n = min(2, batch-start)
        mean = np.zeros((n, 3, 2), dtype=np.float32)
        std = np.ones_like(mean)
        expanded = {k: np.repeat(v[start:start+n, None], 8, axis=1) for k, v in original.items()}
        for step in range(3):
            index = len(case["events"])
            candidates = rng.normal(size=(n, 8, 3, 2)).astype(np.float32)*std[:, None]+mean[:, None]
            candidates[:, 0] = mean
            cost = ((candidates-expanded["z_goal"][:, :, None])**2).sum(axis=(2, 3))
            inds = np.argsort(cost, axis=1)[:, :3].astype(np.int64)
            rows = np.arange(n)[:, None]
            elites = candidates[rows, inds]
            vals = cost[rows, inds]
            new_mean, new_std = elites.mean(axis=1), elites.std(axis=1, ddof=1)
            state = dict(candidates=candidates, costs=cost, topk_vals=vals, topk_inds=inds,
                         topk_candidates=elites, mean=new_mean, var=new_std, prev_mean=mean, prev_var=std)
            case["events"].append({"step": step, "chunk": chunk, "attempt": index,
                "arrays": {k: save(f"event{index}_{k}", v) for k, v in state.items()},
                "cuda_exact_checks": {k: True for k in
                    ("topk_vals", "topk_inds", "gather", "mean", "sample_std", "mean_candidate")}})
            case["cost_attempts"].append({"index": index, "chunk": chunk, "returned": True,
                "candidate_rows": n*8, "expand_sample_stride_zero": True,
                **{k: save(f"cost{index}_{k}", v) for k, v in expanded.items()},
                "cost": save(f"cost{index}_value", cost)})
            case["callback_output"].append({"chunk": chunk, "step": step})
            calls.append({"attempt": index+1, "batch": n, "candidate_rows": n*8, "metadata": "high",
                          "expanded": {k: fp(v, True) for k, v in expanded.items()},
                          "candidates": fp(candidates), "cost": fp(cost), "returned": True})
            mean, std = new_mean, new_std
        means.append(mean)
        stds.append(std)
        costs.append(vals.mean(axis=1))
    case["bridge_receipt"] = {"backend": "actual_torch_CUDA_float32", "tensor_batch": batch,
        "input_witness": witness, "cost_attempts": len(calls), "cost_returns": len(calls),
        "cost_failures": 0, "candidate_rows_attempted": batch*8*3, "calls": calls}
    case["outputs"] = {k: save("output_"+k, v) for k, v in
                       (("actions", np.concatenate(means)), ("mean", np.concatenate(means)),
                        ("var", np.concatenate(stds)), ("costs", np.concatenate(costs)))}
    return case, arrays


@pytest.fixture
def saved(tmp_path):
    manifest = m.prepare(tmp_path / "prepare")
    out = tmp_path / "RAM_ONLY_not_real_CUDA"
    out.mkdir()
    report = {"schema": m.TAG+".report", "manifest_sha256": m.digest(manifest.read_bytes()),
              "protocol": m.protocol(), "runtime": {"fixture": "RAM schema only"},
              "cases": [], "arrays": {}, "errors": [], "completed": True, "elapsed_seconds": .1}
    all_arrays = {}
    for batch in (5, 2):
        case, arrays = build_case(batch, out, report)
        report["cases"].append(case)
        all_arrays.update(arrays)
    path = out / "report.json"
    path.write_bytes(m.encode(report))
    return manifest, path, report, all_arrays


def test_fixed_design_no_neural_or_physics():
    p = m.protocol()
    assert p["batches"] == [5, 2] and p["chunk"] == 2
    assert p["cost_calls_cap"] == 12 and p["candidate_rows_cap"] == 168
    assert p["predict_API_calls"] == p["predictor_samples"] == p["AD"] == p["physics"] == 0


def test_real_prepare_verify(tmp_path):
    path = m.prepare(tmp_path / "prepare")
    assert m.verify(path)["owner_evidence"]["derived_total_envs"] == 4
    with pytest.raises(FileExistsError):
        m.prepare(path.parent)


@pytest.mark.parametrize("key", ["protocol", "sources", "owner_evidence"])
def test_prepare_mutation(tmp_path, key):
    path = m.prepare(tmp_path / "prepare")
    data = m.read_json(path)
    data[key] = {}
    path.write_bytes(m.encode(data))
    with pytest.raises(ValueError, match="drift"):
        m.verify(path)


def test_unauthorized_does_not_import_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "load_runtime", lambda: pytest.fail("runtime import"))
    with pytest.raises(ValueError, match="authorization"):
        m.run_cuda(tmp_path/"not_read", tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_cuda_refusal_no_output_or_cpu_fallback(tmp_path, monkeypatch):
    manifest = m.prepare(tmp_path/"prepare")
    def refuse():
        raise ValueError("HOST CUDA required; CPU fallback forbidden")
    monkeypatch.setattr(m, "load_runtime", refuse)
    with pytest.raises(ValueError, match="CPU fallback forbidden"):
        m.run_cuda(manifest, tmp_path/"out", main_authorized=True)
    assert not (tmp_path/"out").exists()


def test_saved_full_schema_and_all_denominators(saved):
    manifest, path, report, arrays = saved
    result = m.consume(manifest, path)
    assert result["cost_attempts"] == result["cost_returns"] == 12
    assert result["candidate_rows_attempted"] == 168
    assert not result["scientific_admission"] and not result["official_model_result"]
    assert [len(c["events"]) for c in report["cases"]] == [9, 3]
    assert report["cases"][0]["events"][6]["arrays"]["candidates"] in arrays


@pytest.mark.parametrize("mutation", ["missing_event", "order", "metadata", "std", "chain", "candidate0",
    "gather", "indices", "rows", "expanded", "flags", "output", "callbacks", "fingerprint", "input", "stride"])
def test_case_mutations_rejected(saved, mutation):
    _, _, report, arrays = saved
    case = deepcopy(report["cases"][0])
    arrays = {k: v.copy() for k, v in arrays.items()}
    first = case["events"][0]
    if mutation == "missing_event":
        case["events"].pop()
    elif mutation == "order":
        first["step"] = 1
    elif mutation == "metadata":
        case["bridge_receipt"]["calls"][0]["metadata"] = "h"
    elif mutation == "std":
        arrays[first["arrays"]["var"]] += .1
    elif mutation == "chain":
        arrays[case["events"][1]["arrays"]["prev_mean"]] += .1
    elif mutation == "candidate0":
        arrays[first["arrays"]["candidates"]][:, 0] += .1
    elif mutation == "gather":
        arrays[first["arrays"]["topk_candidates"]] += .1
    elif mutation == "indices":
        arrays[first["arrays"]["topk_inds"]][:] = 99
    elif mutation == "rows":
        case["cost_attempts"][0]["candidate_rows"] -= 1
    elif mutation == "expanded":
        arrays[case["cost_attempts"][0]["z_init"]] += 1
    elif mutation == "flags":
        first["cuda_exact_checks"]["sample_std"] = False
    elif mutation == "output":
        arrays[case["outputs"]["actions"]] += .1
    elif mutation == "callbacks":
        case["starts"] = 2
    elif mutation == "fingerprint":
        case["bridge_receipt"]["calls"][0]["cost"]["sha256"] = "0"*64
    elif mutation == "input":
        case["bridge_receipt"]["input_witness"]["original_metadata"] = {"planner_level": "h"}
    else:
        case["bridge_receipt"]["calls"][0]["expanded"]["z_init"]["stride"][1] = 2
    with pytest.raises(ValueError):
        m.validate_case(case, arrays)


@pytest.mark.parametrize("mutation", ["partial", "errors", "one_case", "binding", "wall", "protocol", "traversal", "extra"])
def test_report_mutations(saved, mutation):
    manifest, path, report, arrays = saved
    if mutation == "partial":
        report["completed"] = False
    elif mutation == "errors":
        report["errors"] = ["failure"]
    elif mutation == "one_case":
        report["cases"].pop()
    elif mutation == "binding":
        report["manifest_sha256"] = "0"*64
    elif mutation == "wall":
        report["elapsed_seconds"] = 121
    elif mutation == "protocol":
        report["protocol"]["chunk"] = 3
    elif mutation == "traversal":
        name = next(iter(report["arrays"]))
        report["arrays"]["../"+name] = report["arrays"].pop(name)
    else:
        m.dump_array(path.parent, "unused.npy", np.zeros(1), report["arrays"])
    path.write_bytes(m.encode(report))
    with pytest.raises(ValueError):
        m.consume(manifest, path)


def test_file_tamper_and_caps(saved, tmp_path):
    manifest, path, report, _ = saved
    name = next(iter(report["arrays"]))
    file = path.parent/name
    raw = file.read_bytes()
    file.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    with pytest.raises(ValueError, match="SHA"):
        m.consume(manifest, path)
    with pytest.raises(ValueError, match="cap"):
        m.dump_array(tmp_path, "big.npy", np.zeros(m.CAP, dtype=np.uint8), {})
    with pytest.raises(ValueError, match="flat"):
        m.dump_array(tmp_path, "../outside.npy", np.zeros(1), {})


def test_partial_run_control_flow_retains_failure(tmp_path, monkeypatch):
    manifest = m.prepare(tmp_path/"prepare")
    cuda = SimpleNamespace(synchronize=lambda: None, get_device_name=lambda: "RAM stub NOT CUDA")
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="stub"), __version__="stub")
    monkeypatch.setattr(m, "load_runtime", lambda: (torch, object(), object()))
    def failed(torch, solver, box, batch, out, report, case, start):
        case["cost_attempts"] = [{"returned": False, "candidate_rows": 16}]
        raise RuntimeError("owned non-neural control-path failure")
    monkeypatch.setattr(m, "run_case", failed)
    out = tmp_path/"failed"
    report = m.run_cuda(manifest, out, main_authorized=True)
    assert report["completed"] is False
    assert len(report["cases"]) == 1
    assert report["cases"][0]["cost_attempts"][0]["candidate_rows"] == 16
    assert m.read_json(out/"report.json")["errors"][0]["type"] == "RuntimeError"
    with pytest.raises(ValueError):
        m.consume(manifest, out/"report.json")


def test_cli_actual_prepare_verify(tmp_path):
    for args in (["prepare", "--out", str(tmp_path/"cli")],
                 ["verify", "--manifest", str(tmp_path/"cli/manifest.json")]):
        result = subprocess.run([sys.executable, "-B", str(SOURCE), *args], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)


def test_doc_is_delivered():
    assert m.DOC.is_file()
