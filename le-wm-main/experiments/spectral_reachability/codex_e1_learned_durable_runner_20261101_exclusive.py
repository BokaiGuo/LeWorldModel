"""New-run durable instrumentation for the existing Hi learned runner.

Run as a module; journal and runner output must be new paths. This records
model-call counts and operation boundaries, not time, FLOPs, GPU usage,
monetary cost, or historical receipts.
The underlying runner remains MAIN-authorized and CUDA-only.
"""
import argparse
from contextlib import contextmanager
import json

from experiments.spectral_reachability.codex_e1_proposal_durable_events_20261101 import ProposalJournal

TRACE_MODELS = frozenset({"encoder", "projector", "action_encoder",
                          "low_predictor", "low_pred_proj"})
HIGH_MODELS = ("macro_to_condition", "high_predictor", "high_pred_proj")


@contextmanager
def instrument(runner, journal):
    """Temporarily instrument the actual runner, restoring it even on failure."""
    original_call = runner.Trace.call
    original_search = runner.high_search

    def traced_call(trace, kind, args, fn):
        with journal.attempt("trace." + kind):
            if kind in TRACE_MODELS:
                with journal.attempt("model_call", details={
                    "model": kind, "measurement": "call_count",
                    "source": "trace_call", "scope": "run_observed",
                }):
                    return original_call(trace, kind, args, fn)
            return original_call(trace, kind, args, fn)

    def traced_search(t, CEM, Box, root, z, goal, trace, row, index):
        with journal.attempt("high_search", details={"row_index": index}):
            handles = []
            pending = {name: [] for name in HIGH_MODELS}

            def pre(name):
                def hook(module, args):
                    attempt = journal.append({"type": "attempt", "operation": "model_call",
                        "details": {"model": name, "measurement": "call_count",
                                    "source": "forward_hook", "scope": "run_observed",
                                    "row_index": index}})
                    pending[name].append(attempt)
                return hook

            def post(name):
                def hook(module, args, output):
                    journal.append({"type": "return", "attempt_index": pending[name].pop()})
                return hook

            try:
                for name in HIGH_MODELS:
                    module = getattr(root, name)
                    handles.append(module.register_forward_pre_hook(pre(name)))
                    handles.append(module.register_forward_hook(post(name)))
                return original_search(t, CEM, Box, root, z, goal, trace, row, index)
            finally:
                for handle in handles:
                    handle.remove()

    runner.Trace.call = traced_call
    runner.high_search = traced_search
    try:
        yield
    finally:
        runner.Trace.call = original_call
        runner.high_search = original_search


def run(runner, *, manifest, manifest_sha256, output, journal_path, run_id, arm,
        main_authorized):
    if main_authorized is not True:
        raise ValueError("explicit MAIN authorization required")
    with ProposalJournal(journal_path, run_id=run_id, arm=arm,
                         proposal_kind="hilewm_high_first") as journal:
        journal.append({"type": "cost_scope", "scope": "shared_unallocated",
                        "allocation": None, "amount": None,
                        "reason": "shared costs are not measured or allocated by this runner",
                        "historical_costs": "unknown"})
        with instrument(runner, journal):
            with journal.attempt("run_cuda", details={
                "manifest": manifest, "manifest_sha256": manifest_sha256,
                "output": output,
            }):
                return runner.run_cuda(manifest, manifest_sha256, output,
                                       main_authorized=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "manifest-sha256", "output", "journal", "run-id", "arm"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--main-authorized", action="store_true")
    args = parser.parse_args(argv)
    if not args.main_authorized:
        parser.error("--main-authorized is required")
    from experiments.spectral_reachability import agent_hilewm_rgb_high_low_20261028_v1 as runner
    result = run(runner, manifest=args.manifest, manifest_sha256=args.manifest_sha256,
                 output=args.output, journal_path=args.journal, run_id=args.run_id,
                 arm=args.arm, main_authorized=True)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
