"""Owned NumPy callback seam only: never import/execute stable_worldmodel/torch."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/spectral_reachability/agent_hilewm_solver_batch_bridge_20261028_v1.py"
spec = importlib.util.spec_from_file_location("owned_hilewm_bridge", SOURCE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def original(batch=5):
    z = np.arange(batch * 2, dtype=np.float32).reshape(batch, 2)
    return {"planner_level": "high", "z_init": z, "z_goal": z + 10}


def bridge(info=None, cost=None, **kwargs):
    if info is None:
        info = original()
    config = dict(tensor_batch=5, chunk_size=2, num_samples=3,
                  native_cost=cost or (lambda info, a: np.sum(a * a, axis=(2, 3))),
                  abi=m.NUMPY_ABI, max_cost_calls=3, max_candidate_rows=15)
    config.update(kwargs)
    return m.HighBatchBridge(info, **config)


def expanded(b, start=0, stop=2):
    info = {k: np.repeat(v[start:stop, None], b.samples, axis=1)
            for k, v in b.solver_info.items()}
    return info, np.zeros((stop-start, b.samples, 2, 1), dtype=np.float32)


def test_actual_source_static_evidence():
    r = m.source_evidence()
    assert r["derived_total_envs"] == 4
    assert r["count_line"] == 131 and r["slice_line"] == 163
    assert r["high_info_line"] == 792
    assert not r["foreign_code_executed"]


@pytest.mark.parametrize("owner", ["cem", "policy", "utils"])
def test_source_mutation_rejected(tmp_path, owner):
    paths = {"cem": m.CEM, "policy": m.POLICY, "utils": m.UTILS}
    bad = tmp_path / "mutated.txt"
    bad.write_bytes(paths[owner].read_bytes() + b"\n# changed\n")
    paths[owner] = bad
    with pytest.raises(ValueError, match="source SHA"):
        m.source_evidence(**paths)


@pytest.mark.parametrize("batch", [1, 2, 4, 5, 6])
def test_complete_bug_reproduction_and_safe_rows(batch):
    r = m.reproduce(batch, 1, 3)
    assert r["legacy_total_envs"] == 4
    assert [x["planner_level"] for x in r["legacy"]] == list("high")
    assert [x["planner_level"] for x in r["reorder_only"]] != ["high"] * batch
    assert sum(x["actual_tensor_batch"] for x in r["legacy"]) == min(4, batch)
    assert [x["row_ids"][0] for x in r["bridge"]] == list(range(batch))
    assert all(x["planner_level"] == "high" for x in r["bridge"])
    assert r["receipt"]["candidate_rows_attempted"] == batch * 3
    assert not r["receipt"]["official_unadapted_reproduction"]


@pytest.mark.parametrize("chunk", [1, 2, 4, 8])
def test_tail_and_candidate_values_no_cost_rewrite(chunk):
    source = original()
    before = {k: v.copy() for k, v in source.items() if isinstance(v, np.ndarray)}
    seen = []
    def native(info, a):
        assert list(info) == ["planner_level", "z_init", "z_goal"]
        assert info["planner_level"] == "high"
        seen.extend(info["z_init"][:, 0, 0].tolist())
        return np.sum(a, axis=(2, 3)) + info["z_goal"][:, :, 0]
    b = bridge(source, native, chunk_size=chunk, max_cost_calls=5)
    assert b.solver_info["z_init"] is source["z_init"]
    assert b.original_info["z_goal"] is source["z_goal"]
    def dispatch(info, n):
        a = np.arange(n*3*2, dtype=np.float32).reshape(n, 3, 2, 1)
        result = b.get_cost(info, a)
        np.testing.assert_array_equal(result, a.sum(axis=(2, 3)) + info["z_goal"][:, :, 0])
    m.numpy_batch_seam(b.solver_info, chunk_size=chunk, num_samples=3, callback=dispatch)
    assert seen == [0, 2, 4, 6, 8]
    assert b.attempts == (5 + chunk-1)//chunk and b.returned == b.attempts
    assert b.candidate_rows == 15
    for k in before:
        np.testing.assert_array_equal(before[k], source[k])
    with pytest.raises(TypeError):
        b.solver_info["planner_level"] = "high"


@pytest.mark.parametrize("change", ["batch", "empty", "scalar", "goalshape", "metadata", "extra", "object"])
def test_failclosed_before_callback(change):
    info = original()
    if change == "batch":
        info["z_init"] = info["z_init"][:4]
    elif change == "empty":
        info["z_init"] = info["z_init"][:0]
    elif change == "scalar":
        info["z_init"] = np.array(1)
    elif change == "goalshape":
        info["z_goal"] = np.zeros((5, 3))
    elif change == "metadata":
        info["planner_level"] = "h"
    elif change == "extra":
        info["other"] = np.zeros(5)
    else:
        info["z_goal"] = np.zeros((5, 2), dtype=object)
    def forbidden(*args):
        pytest.fail("callback dispatched")
    with pytest.raises(ValueError):
        bridge(info, forbidden)


@pytest.mark.parametrize("field,value", [("tensor_batch", True), ("tensor_batch", 0),
                                           ("chunk_size", -1), ("num_samples", 0),
                                           ("max_cost_calls", 0), ("max_candidate_rows", False)])
def test_invalid_caps(field, value):
    with pytest.raises(ValueError):
        bridge(**{field: value})


@pytest.mark.parametrize("bad", ["metadata", "shape", "samples", "rank", "chunk"])
def test_bad_expanded_input_no_dispatch(bad):
    b = bridge()
    info, a = expanded(b)
    if bad == "metadata":
        info["planner_level"] = "h"
    elif bad == "shape":
        info["z_goal"] = info["z_goal"][:, :1]
    elif bad == "samples":
        a = a[:, :1]
    elif bad == "rank":
        a = a[:, :, :, 0]
    else:
        info, a = expanded(b, 0, 3)
    with pytest.raises(ValueError):
        b.get_cost(info, a)
    assert b.attempts == 0


def test_attempts_not_refunded_on_failure_and_cap_before_dispatch():
    seen = []
    def fail(info, a):
        seen.append(1)
        raise RuntimeError("owned callback failure")
    b = bridge(cost=fail, max_cost_calls=1)
    info, a = expanded(b)
    with pytest.raises(RuntimeError):
        b.get_cost(info, a)
    with pytest.raises(ValueError, match="before_dispatch"):
        b.get_cost(info, a)
    assert seen == [1] and b.failed == 1 and b.candidate_rows == 6
    assert b.receipt()["cost_returns"] == 0


def test_row_cap_before_dispatch():
    b = bridge(max_candidate_rows=5)
    with pytest.raises(ValueError, match="before_dispatch"):
        b.get_cost(*expanded(b))
    assert b.attempts == 0 and not b.calls


@pytest.mark.parametrize("mutation", ["original", "expanded", "candidate", "mapping", "output"])
def test_mutations_and_wrong_output_rejected(mutation):
    source = original()
    def cost(info, a):
        if mutation == "original":
            source["z_init"][0, 0] += 1
        elif mutation == "expanded":
            info["z_init"][0, 0, 0] += 1
        elif mutation == "candidate":
            a[0, 0, 0, 0] += 1
        elif mutation == "mapping":
            info["planner_level"] = "low"
        return np.zeros((2, 1) if mutation == "output" else (2, 3))
    b = bridge(source, cost)
    with pytest.raises(ValueError):
        b.get_cost(*expanded(b))
    assert b.failed == 1 and b.returned == 0


def test_pre_dispatch_mutation_and_receipt_copy():
    b = bridge()
    receipt = b.receipt()
    receipt["input_witness"]["original_tensors"].clear()
    b.assert_inputs_unchanged()
    b.original_info["z_goal"][0, 0] += 1
    with pytest.raises(ValueError, match="mutation"):
        b.get_cost(*expanded(b))
    assert b.attempts == 0


def test_parameters_explicit_no_actor_or_implicit_delegation():
    b = bridge()
    with pytest.raises(ValueError, match="model.parameters"):
        b.parameters()
    sentinel = object()
    b = bridge(model_parameters=lambda: iter([sentinel]))
    assert next(b.parameters()) is sentinel
    assert not hasattr(b, "get_action")
    assert not hasattr(b, "run_cuda")


def test_explicit_owned_model_binding_rejects_actor_without_calling_it():
    class Native:
        def get_cost(self, info, a):
            assert info["planner_level"] == "high"
            return a.sum(axis=(2, 3))

        def parameters(self):
            return iter(["owned parameter sentinel"])

    kwargs = dict(tensor_batch=5, chunk_size=2, num_samples=3, abi=m.NUMPY_ABI,
                  max_cost_calls=3, max_candidate_rows=15)
    native = Native()
    b = m.HighBatchBridge.bind_high_model(original(), native, **kwargs)
    assert next(b.parameters()) == "owned parameter sentinel"
    b.get_cost(*expanded(b))
    native.get_action = lambda *a: pytest.fail("actor must not execute")
    with pytest.raises(ValueError, match="actor interface"):
        m.HighBatchBridge.bind_high_model(original(), native, **kwargs)


@pytest.mark.parametrize("order", [("z_goal", "planner_level", "z_init"),
                                   ("planner_level", "z_init", "z_goal")])
def test_original_order_float64_and_cost_identity(order):
    raw = original()
    raw = {k: v.astype(np.float64) if isinstance(v, np.ndarray) else v for k, v in raw.items()}
    info = {k: raw[k] for k in order}
    expected = np.arange(6, dtype=np.float64).reshape(2, 3)
    def cost(cost_info, a):
        assert cost_info["z_init"].dtype == np.float64
        assert cost_info["z_goal"].dtype == np.float64
        return expected
    b = bridge(info, cost)
    assert list(b.original_info) == list(order)
    assert b.get_cost(*expanded(b)) is expected


def test_cli_real_paths_static_check_and_reproduction():
    r = subprocess.run([sys.executable, "-B", str(SOURCE), "reproduce", "--batch", "5", "--chunk", "2"],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    output = json.loads(r.stdout)
    assert output["source_evidence"]["cem"]["path"] == str(m.CEM)
    assert output["reproduction"]["receipt"]["cost_returns"] == 3
    assert output["reproduction"]["receipt"]["predict_API_calls"] is None


def test_doc_exists():
    assert (ROOT / "innerdocs/agent_hilewm_solver_batch_bridge_20261028_v1.md").is_file()
