"""Coordinated known-root exploratory P2 planning. No physical execution."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time

from experiments.spectral_reachability import codex_p2_bank_bind_before_compare_20261030_v1 as bank
from experiments.spectral_reachability import codex_p2_bank_bound_saved_consumer_20261030_v1 as saved
from experiments.spectral_reachability import codex_p2_host_physical_bridge_20261030_v1 as bridge

frontier = bank.frontier
need = bank.need
NAME = "codex_p2_knownroot_host_cuda_20261030_v1"
CAPS = dict(predict=168, action_encoder=168, jacobian=41, vjp_directions=7872, svd=42, solve=41)
ACK = "COORDINATED_KNOWN_ROOT_EXPLORATION_NO_GLOBAL_DEDUP"


def inventory(protocol):
    modules = (sys.modules[__name__], bank, saved, bridge, bank.gate_module, frontier)
    paths = set(protocol["source_paths"]) | set(protocol["bindings"])
    paths.update(str(Path(m.__file__).resolve()) for m in modules)
    return {str(Path(p).resolve()): frontier.file_sha(Path(p).resolve()) for p in sorted(paths)}


def prepare(prepared_dir, prepared_sha256, output):
    directory, protocol, prepared = frontier.load_prepared(prepared_dir, prepared_sha256)
    need(prepared_sha256 == frontier.digest(prepared), "externally pinned prepared digest required")
    cases = protocol["context"]["cases"]
    need(tuple(c["root_sha256"] for c in cases) == frontier.ROOTS and
         [c["root_index"] for c in cases] == list(range(4)), "fixed ordered root census")
    sources = inventory(protocol)
    for path, sha in protocol["bindings"].items():
        need(sources[str(Path(path).resolve())] == sha, "inherited binding drift")
    out = Path(output)
    need(out.is_absolute() and out.parent.resolve() == out.parent and out.parent.is_dir()
         and not os.path.lexists(out), "new absolute prepare path")
    need(shutil.disk_usage(out.parent).free >= 300 * 1024**2, "disk reserve")
    out.mkdir(mode=0o700)
    manifest = dict(schema=NAME + ".prepare", status="prepared_not_executed",
                    upstream_prepared=str(directory), upstream_prepared_sha256=prepared_sha256,
                    protocol_sha256=frontier.digest(protocol), roots=list(frontier.ROOTS),
                    arms=list(frontier.ARMS), caps=CAPS, source_bindings=sources,
                    duplicate_control="cooperative_only_no_global_claim", known_roots=True,
                    independent_confirmation=False, equal_total_compute=False,
                    P1_admission=False, P2_admission=False, physical_authorized=False)
    bank._write(out / "prepare.json", manifest)
    return frontier.file_sha(out / "prepare.json")


def validate_prepare(directory, expected_sha):
    directory = Path(directory)
    need(directory.is_absolute() and directory.is_dir() and not directory.is_symlink(), "prepare directory")
    path = directory / "prepare.json"
    need(frontier.file_sha(path) == expected_sha, "external prepare file SHA mismatch")
    m = frontier.read(path)
    need(m["schema"] == NAME + ".prepare" and m["status"] == "prepared_not_executed" and
         m["roots"] == list(frontier.ROOTS) and m["arms"] == list(frontier.ARMS) and
         m["caps"] == CAPS and m["duplicate_control"] == "cooperative_only_no_global_claim" and
         m["known_roots"] is True and m["independent_confirmation"] is False and
         m["equal_total_compute"] is False and m["P1_admission"] is False and
         m["P2_admission"] is False and m["physical_authorized"] is False, "prepare contract")
    _, protocol, prepared = frontier.load_prepared(m["upstream_prepared"], m["upstream_prepared_sha256"])
    need(frontier.digest(protocol) == m["protocol_sha256"], "protocol drift")
    bridge.check_sources(m["source_bindings"])
    need(inventory(protocol) == m["source_bindings"], "source inventory drift")
    return m, protocol, prepared


def _run(*, manifest, protocol, prepared, output, backend):
    """Internal orchestration seam; CLI always supplies the actual CUDA backend."""
    out = Path(output)
    need(out.is_absolute() and out.parent.resolve() == out.parent and out.parent.is_dir()
         and not os.path.lexists(out), "new absolute run path")
    need(shutil.disk_usage(out.parent).free >= 300 * 1024**2, "disk reserve")
    out.mkdir(mode=0o700)  # only this path is claimed, not other entry points
    state = dict(schema=NAME + ".run", status="partial_or_failed", roots=[], 
                 prepare=deepcopy(manifest), duplicate_control="cooperative_only_no_global_claim",
                 independent_confirmation=False, equal_total_compute=False,
                 physical_authorized=False, P1_admission=False, P2_admission=False)
    start = time.monotonic()
    try:
        bank._write(out / "started.json", state)
        compiled_rows = frontier.read(protocol["context"]["compilation"])["rows"]
        index = {r["root_sha256"]: r["compiled"] for r in compiled_rows}
        need(len(compiled_rows) == 4 and set(index) == set(frontier.ROOTS), "compiled root census")
        cases = protocol["context"]["cases"]
        need(tuple(c["root_sha256"] for c in cases) == frontier.ROOTS and
             [c["root_index"] for c in cases] == list(range(4)), "ordered root census")
        backend.start(protocol)
        state["host"] = backend.host
        state["model_state_before"] = backend.model_hash()
        for case in cases:
            root = case["root_sha256"]
            backend.check_frozen()
            root_out = out / ("root_" + str(case["root_index"]))
            bank.execute_native(model=backend.model, device=backend.device, ready=case,
                                compiled=index[root], protocol=protocol, prepared=prepared,
                                caps=CAPS, output=root_out)
            report_sha = frontier.file_sha(root_out / "report.json")
            saved.consume(directory=root_out, expected_report_sha256=report_sha, ready=case,
                          compiled=index[root], protocol=protocol, prepared=prepared,
                          source_bindings=manifest["source_bindings"])
            need(backend.model_hash() == state["model_state_before"], "model mutated")
            backend.check_frozen()
            state["roots"].append(dict(root_sha256=root, root_index=case["root_index"],
                                       directory=str(root_out), report_sha256=report_sha))
            bank._write(out / ("root_" + str(case["root_index"]) + "_receipt.json"), state["roots"][-1])
        bridge.check_sources(manifest["source_bindings"])
        need(inventory(protocol) == manifest["source_bindings"], "source drift after execution")
        state["model_state_after"] = backend.model_hash()
        state["status"] = "exploratory_complete"
    except BaseException as exc:
        state["error"] = dict(type=type(exc).__name__, message=str(exc)[:1500])
        raise
    finally:
        state["wall_seconds"] = time.monotonic() - start
        # Failure to finalize is a failure, never a successful run.
        bank._write(out / "run_report.json", state)
    return state


class HostCuda:
    def start(self, protocol):
        import torch
        from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
        from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
        self.torch, self.checked = torch, checked
        self.host = base.require_cuda_allocation("cuda")
        need(torch.cuda.is_available(), "CUDA unavailable")
        self.device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model = base.load_compatible_pretrained(Path(protocol["context"]["policy"]),
                                                    device=self.device).eval().requires_grad_(False)
        self.check_frozen()

    def check_frozen(self):
        params = list(self.model.parameters())
        need(bool(params) and all(not m.training for m in self.model.modules()) and
             all(not p.requires_grad and p.device == self.device for p in params) and
             all(b.device == self.device for b in self.model.buffers()), "frozen CUDA model required")

    def model_hash(self):
        self.torch.cuda.synchronize(self.device)
        return self.checked.tensor_state_hash(self.model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute-cuda", action="store_true")
    parser.add_argument("--upstream-prepared")
    parser.add_argument("--upstream-prepared-sha256")
    parser.add_argument("--prepared")
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ack")
    a = parser.parse_args()
    if a.prepare:
        need(a.upstream_prepared and a.upstream_prepared_sha256, "upstream and digest required")
        result = {"prepare_file_sha256": prepare(a.upstream_prepared, a.upstream_prepared_sha256, a.output)}
    else:
        need(a.ack == ACK and a.prepared and a.prepared_sha256, "host coordination acknowledgement and prepare SHA required")
        m, p, s = validate_prepare(a.prepared, a.prepared_sha256)
        result = _run(manifest=m, protocol=p, prepared=s, output=a.output, backend=HostCuda())
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
