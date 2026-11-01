"""Explicit high-only CEM batch seam; no foreign imports, loader or solver execution."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

TAG = "agent_hilewm_solver_batch_bridge_20261028_v1"
ROOT = Path(__file__).resolve().parents[2]
CEM = ROOT / ".venv/lib/python3.12/site-packages/stable_worldmodel/solver/cem.py"
POLICY = ROOT / "results/p1_hilewm_text_contract_20261027_v1/texts/package__code__h_le_wm__planning__policies.py.source.txt"
UTILS = CEM.parent / "utils.py"
OWNER_SHAS = {
    "cem": "a6662e7f1f2c6ecaf0a8527255d531aa05ed583339aabdf2798d0206382ef51b",
    "policy": "d6e7d476b45e47c5015494775b4d96727b8ebb8c801f719a100ba32878f288d1",
    "utils": "d30c6225a9c4c83782132180a94c94188b07e0381d2c938c658fe5a49aed48e1",
}
FIELDS = {"planner_level", "z_init", "z_goal"}
TENSORS = ("z_init", "z_goal")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def positive(value, name):
    require(type(value) is int and value > 0, name)
    return value


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_evidence(cem=CEM, policy=POLICY, utils=UTILS):
    """Parse only. No import, compile, exec, pickle, or foreign method binding."""
    out = {}
    trees = {}
    for name, path in (("cem", cem), ("policy", policy), ("utils", utils)):
        raw = Path(path).read_bytes()
        require(sha(raw) == OWNER_SHAS[name], name + " source SHA")
        trees[name] = ast.parse(raw.decode())
        out[name] = {"path": str(Path(path).resolve()), "sha256": sha(raw)}
    solve = next(n for n in ast.walk(trees["cem"])
                 if isinstance(n, ast.FunctionDef) and n.name == "solve")
    expected = ast.dump(ast.parse("total_envs = len(next(iter(info_dict.values())))").body[0])
    counts = [n for n in ast.walk(solve) if isinstance(n, ast.Assign) and ast.dump(n) == expected]
    slices = [n for n in ast.walk(solve) if isinstance(n, ast.Assign)
              and ast.dump(n) == ast.dump(ast.parse("v_batch = v[start_idx:end_idx]").body[0])]
    require(len(counts) == len(slices) == 1, "CEM count/slice ABI")
    plan = next(n for n in ast.walk(trees["policy"])
                if isinstance(n, ast.FunctionDef) and n.name == "_plan_high")
    infos = [n.value for n in ast.walk(plan) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "high_info" for t in n.targets)]
    require(len(infos) == 1 and isinstance(infos[0], ast.Dict), "high_info ABI")
    keys = [ast.literal_eval(k) for k in infos[0].keys]
    require(keys == ["planner_level", "z_init", "z_goal"]
            and ast.literal_eval(infos[0].values[0]) == "high", "high_info order/value")
    out.update(count_line=counts[0].lineno, slice_line=slices[0].lineno,
               high_info_line=infos[0].lineno, first_value="high", derived_total_envs=4,
               foreign_code_executed=False)
    return out


def numpy_fingerprint(value):
    require(type(value) is np.ndarray, "RAM ndarray required")
    require(not value.dtype.hasobject, "object dtype forbidden")
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": sha(value.tobytes(order="C")), "bytes": value.nbytes}


@dataclass(frozen=True)
class TensorABI:
    """Caller-owned, audited tensor test/digest; never converts dtype/device/data.

    NumPy ABI below is RAM evidence only. Future CUDA integration must supply
    torch.is_tensor and an exact byte/dtype/shape digest under MAIN authority.
    """
    accepts: object
    fingerprint: object
    label: str


NUMPY_ABI = TensorABI(lambda x: type(x) is np.ndarray, numpy_fingerprint, "numpy_RAM_only")


def tensor_shape(abi, value):
    require(abi.accepts(value), "undeclared tensor type")
    shape = tuple(value.shape)
    require(all(type(d) is int and d > 0 for d in shape), "positive tensor dimensions required")
    return shape


class HighBatchBridge:
    """New explicit integration contract, NOT a patched official policy/solver.

    solver_info contains only original tensor objects. get_cost is the solver's
    fixed high-only facade. Its native_cost must be the unchanged high model's
    get_cost. Solver construction/math and output selection stay outside here.
    Single-threaded; callback ABI/fingerprints are trusted integration code.
    """

    @classmethod
    def bind_high_model(cls, original_info, native_model, **contract):
        """Future explicit model binding, never used by the RAM CLI.

        Reject actor-capable models rather than hiding get_action from CEM's
        prepare_init_action runtime protocol. No model method is called here.
        """
        require(not hasattr(native_model, "get_action"), "actor interface requires a new contract")
        require(callable(getattr(native_model, "get_cost", None))
                and callable(getattr(native_model, "parameters", None)), "native model API")
        return cls(original_info, native_cost=native_model.get_cost,
                   model_parameters=native_model.parameters, **contract)

    def __init__(self, original_info, *, tensor_batch, chunk_size, num_samples,
                 native_cost, abi, max_cost_calls, max_candidate_rows,
                 model_parameters=None):
        self.batch = positive(tensor_batch, "tensor_batch")
        self.chunk = positive(chunk_size, "chunk_size")
        self.samples = positive(num_samples, "num_samples")
        self.max_calls = positive(max_cost_calls, "max_cost_calls")
        self.max_rows = positive(max_candidate_rows, "max_candidate_rows")
        require(set(original_info) == FIELDS, "exact high_info fields required")
        require(type(original_info["planner_level"]) is str
                and original_info["planner_level"] == "high", "fixed high metadata")
        require(callable(native_cost), "native_cost callback required")
        self.abi = abi
        a, b = (tensor_shape(abi, original_info[k]) for k in TENSORS)
        require(len(a) >= 2 and a == b and a[0] == self.batch, "explicit tensor batch/shape mismatch")
        self.tail = a[1:]
        self.original_info = MappingProxyType(dict(original_info))
        self.solver_info = MappingProxyType({k: original_info[k] for k in TENSORS})
        self._native_cost = native_cost
        self._model_parameters = model_parameters
        self.calls = []
        self.attempts = self.returned = self.failed = self.candidate_rows = 0
        self._initial = self.input_witness()

    def parameters(self):
        # CEM reads next(model.parameters()).dtype. Never silently use a default
        # because our facade hid the original model's parameter iterator.
        require(callable(self._model_parameters), "MAIN must bind native model.parameters")
        return self._model_parameters()

    def input_witness(self):
        return {"original_keys": list(self.original_info),
                "original_metadata": {"planner_level": self.original_info["planner_level"]},
                "original_tensors": {k: self.abi.fingerprint(self.original_info[k]) for k in TENSORS},
                "solver_keys": list(self.solver_info),
                "solver_tensors": {k: self.abi.fingerprint(self.solver_info[k]) for k in TENSORS}}

    def assert_inputs_unchanged(self):
        require(self.input_witness() == self._initial, "original/solver tensor mutation")

    def get_cost(self, solver_expanded_info, action_candidates):
        self.assert_inputs_unchanged()
        require(set(solver_expanded_info) == set(TENSORS), "solver metadata/extra field forbidden")
        c = tensor_shape(self.abi, action_candidates)
        require(len(c) == 4 and c[0] <= min(self.chunk, self.batch)
                and c[1] == self.samples, "candidate B,S,H,D ABI")
        for key in TENSORS:
            require(tensor_shape(self.abi, solver_expanded_info[key]) == (c[0], c[1], *self.tail),
                    "expanded tensor B,S,latent ABI")
        rows = c[0] * c[1]
        require(self.attempts < self.max_calls and self.candidate_rows + rows <= self.max_rows,
                "before_dispatch cost cap")
        # Scalar is reintroduced AFTER CEM batching, never sliced or expanded.
        cost_info = {"planner_level": "high", **solver_expanded_info}
        tensor_before = {k: self.abi.fingerprint(v) for k, v in solver_expanded_info.items()}
        candidate_before = self.abi.fingerprint(action_candidates)
        event = {"attempt": self.attempts + 1, "batch": c[0], "candidate_rows": rows,
                 "metadata": "high", "expanded": tensor_before,
                 "candidates": candidate_before, "returned": False}
        self.calls.append(event)
        self.attempts += 1
        self.candidate_rows += rows  # failures never refund; NOT predictor/API count
        try:
            cost = self._native_cost(cost_info, action_candidates)
            require(tensor_shape(self.abi, cost) == c[:2], "cost B,S ABI")
            require(set(cost_info) == FIELDS and cost_info["planner_level"] == "high"
                    and all(cost_info[k] is solver_expanded_info[k] for k in TENSORS),
                    "cost callback changed mapping")
            require({k: self.abi.fingerprint(v) for k, v in solver_expanded_info.items()} == tensor_before
                    and self.abi.fingerprint(action_candidates) == candidate_before,
                    "cost callback mutated inputs")
            self.assert_inputs_unchanged()
            event.update(returned=True, cost=self.abi.fingerprint(cost))
            self.returned += 1
            return cost
        except BaseException as exc:
            self.failed += 1
            event["error_type"] = type(exc).__name__
            raise

    def receipt(self):
        return deepcopy({"schema": TAG, "backend": self.abi.label, "tensor_batch": self.batch,
                "input_witness": self._initial, "cost_attempts": self.attempts,
                "cost_returns": self.returned, "cost_failures": self.failed,
                "candidate_rows_attempted": self.candidate_rows, "calls": self.calls,
                "predict_API_calls": None, "predictor_samples": None,
                "scientific_admission": False, "official_unadapted_reproduction": False})


def numpy_batch_seam(info, *, chunk_size, num_samples, callback):
    """Owned reproduction of CEM count + slice + NumPy expansion ONLY.

    No sampling/topk/distribution updates and no foreign code execution.
    Same first-value len and unconditional slice as attested solve lines.
    Callback accepts expanded info and declared current_bs; can expose mismatch.
    """
    positive(chunk_size, "chunk_size")
    positive(num_samples, "num_samples")
    total_envs = len(next(iter(info.values())))
    results = []
    for start_idx in range(0, total_envs, chunk_size):
        end_idx = min(start_idx + chunk_size, total_envs)
        current_bs = end_idx - start_idx
        expanded_infos = {}
        for k, v in info.items():
            v_batch = v[start_idx:end_idx]
            if isinstance(v, np.ndarray):
                v_batch = np.repeat(v_batch[:, None, ...], num_samples, axis=1)
            expanded_infos[k] = v_batch
        results.append(callback(expanded_infos, current_bs))
    return results


def reproduce(batch=2, chunk=1, samples=3):
    positive(batch, "batch")
    z = np.arange(batch, dtype=np.float32)[:, None]
    original = {"planner_level": "high", "z_init": z, "z_goal": z.copy()}
    def observe(info, current_bs):
        return {"declared_batch": current_bs, "actual_tensor_batch": len(info["z_init"]),
                "planner_level": info.get("planner_level"),
                "row_ids": info["z_init"][:, 0, 0].tolist()}
    legacy = numpy_batch_seam(original, chunk_size=chunk, num_samples=samples, callback=observe)
    reordered = {"z_init": z, "z_goal": original["z_goal"], "planner_level": "high"}
    reorder_only = numpy_batch_seam(reordered, chunk_size=chunk, num_samples=samples, callback=observe)
    def cost(info, candidates):
        require(info["planner_level"] == "high", "metadata routing")
        return np.sum(candidates ** 2, axis=(2, 3))
    bridge = HighBatchBridge(original, tensor_batch=batch, chunk_size=chunk, num_samples=samples,
                             native_cost=cost, abi=NUMPY_ABI, max_cost_calls=batch,
                             max_candidate_rows=batch * samples)
    safe = []
    def dispatch(info, current_bs):
        candidates = np.zeros((current_bs, samples, 2, 1), dtype=np.float32)
        bridge.get_cost(info, candidates)
        safe.append(observe({"planner_level": "high", **info}, current_bs))
    numpy_batch_seam(bridge.solver_info, chunk_size=chunk, num_samples=samples, callback=dispatch)
    return {"schema": TAG, "mode": "owned_NumPy_batch_reproduction_not_CEM_optimization",
            "actual_batch": batch, "legacy_total_envs": 4, "legacy": legacy,
            "reorder_only": reorder_only, "bridge": safe, "receipt": bridge.receipt()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "reproduce"))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    evidence = source_evidence()
    result = {"source_evidence": evidence}
    if args.command == "reproduce":
        result["reproduction"] = reproduce(args.batch, args.chunk, args.samples)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
