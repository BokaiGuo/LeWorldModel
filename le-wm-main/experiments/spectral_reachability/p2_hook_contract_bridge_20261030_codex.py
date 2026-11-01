"""Opt-in bridge from the public scoped Ledger.call API to P2 contract caps.

Only hooked increments are guarded. This is not a runner, complete instrumentation,
standalone accounting, a durable journal, or a contract comparison receipt.
"""
import copy
import json
from pathlib import Path

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as contract_api

NAME = "p2_hook_contract_bridge_20261030_codex"
ROOT = Path(__file__).resolve().parents[2]
HOST = ROOT / "results/p2_h15_controlled_host_20261029_v1_cuda_MAIN01"
HOST_SHA = "188fe5552b269db22739a77042a318d1f60d1c1c69cddcf4c5aead20501fe6ce"

# Names describe identical units, not a conversion between API calls and work.
MAPPING = {
    "predict_samples": "predict_samples",
    "vjp_directions": "VJP_directions",
    "array_svd_calls": "array_SVD_calls",
    "solve_calls": "solve_calls",
    "replay_calls": "replay_calls",
}
UNMAPPED = tuple(k for k in contract_api.AXES if k not in MAPPING)


def projected_caps(contract, contract_sha, original_caps):
    """Tighten only directly mappable pre-dispatch caps; never loosen the old caps."""
    contract_api.validate_contract(contract)
    core.require(contract_api.hash_string(contract_sha) and
                 contract_api.sha(contract_api.canonical(contract)) == contract_sha,
                 "committed canonical contract digest")
    caps = core.counters(original_caps)
    for dst, src in MAPPING.items():
        caps[src] = min(caps[src], contract["axes"][dst]["cap"])
    return caps


class HookBridge:
    """Ledger-compatible opt-in seam for a NEW caller, not monkeypatching.

    The supplied sink must be called before dispatch by core.Ledger. The bridge
    cannot ensure the sink is durable, nor that all work passes through it.
    """
    def __init__(self, contract, contract_sha, original_caps, *, sink, clock=None):
        core.require(callable(sink), "explicit event sink required")
        self.contract = copy.deepcopy(contract)
        self.contract_sha = contract_sha
        caps = projected_caps(self.contract, contract_sha, original_caps)
        options = {"sink": sink}
        if clock is not None:
            core.require(callable(clock), "clock must be callable")
            options["clock"] = clock
        self._ledger = core.Ledger(caps, **options)

    @property
    def scope(self):
        return copy.deepcopy(self._ledger.scope)

    @scope.setter
    def scope(self, value):
        core.require(type(value) is dict and bool(value), "explicit nonempty scope")
        self._ledger.scope = copy.deepcopy(value)

    @property
    def attempted(self):
        return copy.deepcopy(self._ledger.attempted)

    @property
    def completed(self):
        return copy.deepcopy(self._ledger.completed)

    def call(self, resource, fn, *, shape=None, directions=None):
        return self._ledger.call(resource, fn, shape=shape, directions=directions)

    def receipt(self):
        """Raw native hook ledger, preserving original public API."""
        return self._ledger.receipt()

    def diagnostic(self):
        receipt = self.receipt()
        return summarize(receipt, self.contract_sha)


def summarize(receipt, contract_sha=None):
    """Recount original events; counters are observed hooks, not full coverage."""
    core.audit_ledger(receipt)
    observed = {}
    for axis, src in MAPPING.items():
        observed[axis] = {"attempted": receipt["attempted"][src],
                          "completed": receipt["completed"][src],
                          "hook_cap": receipt["caps"][src],
                          "coverage": "unknown"}
    unknown = {axis: None for axis in UNMAPPED}
    return {"schema": NAME + ".diagnostic.v1", "contract_sha256": contract_sha,
            "native_receipt_semantic_sha256": core.digest(receipt),
            "observed_hook_counts": observed, "unmapped_axes": unknown,
            "hook_wall_ns": receipt["wall_ns"],
            "hook_wall_semantics": "ledger lifetime only; not whole standalone planning wall",
            "contract_receipt_ready": False, "predispatch_full_vector_guard": False,
            "coverage_verified": False, "durability_verified": False,
            "P1_admission": False, "P2_admission": False}


def saved_h15_diagnostic():
    """Read pinned original host and child reports, then recount their hook events."""
    host = contract_api.read_bound(HOST / "report.json", HOST_SHA)
    core.require(host["status"] == "recorded_not_admitted" and len(host["slots"]) == 2,
                 "expected saved host")
    rows = []
    for i, slot in enumerate(host["slots"]):
        core.require(slot["root"] == i and slot["status"] == "complete", "slot roster")
        child = contract_api.read_bound(HOST / ("root" + str(i)) / "report.json", slot["report_sha256"])
        rows.append({"root_slot": i, "child_sha256": slot["report_sha256"],
                     "diagnostic": summarize(child["resources"])})
    return {"schema": NAME + ".saved_h15.v1", "host_sha256": HOST_SHA, "rows": rows,
            "source_boundary": "pinned host/child reports and independently recounted embedded ledger; "
                               "no full upstream source or original journal revalidation",
            "new_neural_calls": 0, "new_physical_calls": 0, "P1_admission": False, "P2_admission": False}


if __name__ == "__main__":
    print(json.dumps(saved_h15_diagnostic(), sort_keys=True, allow_nan=False))
