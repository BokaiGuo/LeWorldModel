"""Known-root development replay: plan guard failures remain local and invalid."""
import argparse
import json
import os
from pathlib import Path
import time

from . import p3_all30_cpu_physical_20261102_codexexclusive as old

ROOT, moving = old.ROOT, old.moving
NAME = "p3_all30_localguard_cpu_20261102_codexexclusive"
OLD_PREP = ROOT / "results/p3_all30_cpu_physical_20261102_codexexclusive_PREP_01.json"
OLD_SHA = "375da1e30a17cde98fb60879d0649c42a8e5f283ec2d0c56c702efdc27e4aadb"
AUTH = "P3_ALL30_LOCALGUARD_KNOWN_ROOT_CPU"
GUARD_ERROR = "RGB/canvas/state/pure-render guard; failed image retained"


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def destination(value, prefix, suffix=""):
    path = Path(value)
    require(path.is_absolute() and path.parent == ROOT / "results" and
            path.name.startswith(NAME + prefix) and path.name.endswith(suffix) and
            not os.path.lexists(path), "exclusive output")
    return path


def material():
    previous, data = old.verify(OLD_PREP, OLD_SHA)
    p = dict(previous)
    p["schema"] = NAME + ".prepare.v1"
    p["contract"] = dict(previous["contract"])
    p["contract"].update(
        scope="all 30 fixed candidates on the same two previously opened development roots",
        failure_rule="authenticated plan capture guard failure stops and invalidates only that candidate; "
                     "all other errors stop globally",
        candidate_local_guard_requirements="initial root authenticated; plan capture; rejected guard; "
                                           "render preserved dynamics; no secondary errors",
        denominator_rule="all 30 listed; failed candidates invalid, no imputation; "
                         "report completed, terminal, guard failed, global failed, and unstarted",
        no_retries=True, no_gpu=True, no_replan=True, independent_confirmation=False)
    p["previous_prepare_sha256"] = OLD_SHA
    p["bindings"] = dict(previous["bindings"])
    for path in (OLD_PREP, Path(__file__)):
        p["bindings"][str(path.resolve())] = moving.sha(path)
    for path, digest in p["bindings"].items():
        require(moving.sha(path) == digest, "source seal: " + path)
    return p, data


def verify(path, sha):
    require(moving.sha(path) == sha, "external prepare SHA")
    saved = moving.read(path)
    current, data = material()
    require(saved == current, "reconstructed preparation")
    return saved, data


class Store(old.Store):
    def __init__(self, path):
        super().__init__(path)
        self.latest_guard = None

    def emit(self, kind, **fields):
        super().emit(kind, **fields)
        if kind == "guard_receipt":
            self.latest_guard = dict(fields)


def local_guard(row, guard, branch):
    error = row["first_error"] or {}
    return (row["status"] == "failed" and row["initial_exact"] is True
            and not row["secondary_errors"] and error.get("stage") == "branch"
            and error.get("type") == "ValueError" and error.get("message") == GUARD_ERROR
            and guard is not None and guard.get("branch") == branch
            and guard.get("phase") == "plan" and guard.get("accepted") is False
            and guard.get("render_preserved_dynamics") is True
            and guard.get("step") == row["own_returned"])


def execute(p, data, store, factory):
    require(len(p["plans"]) == 30 and len(data["references"]) == 2, "fixed roster")
    rows = [dict(root_slot=x["root_slot"], arm=x["arm"], selection_digest=x["selection_digest"],
                 candidate_index=x["candidate_index"], status="not_started", first_error=None,
                 secondary_errors=[], warmup_returned=0, own_attempted=0, own_returned=0,
                 initial_exact=False, full_horizon=False, native_success=False,
                 terminated=False, truncated=False, PNG=0, checkpoints=[]) for x in p["plans"]]
    ops = moving.Ops(p, store)
    errors = []
    store.emit("run_started", prepare=p)
    try:
        for i, (plan, row) in enumerate(zip(p["plans"], rows, strict=True)):
            env, store.latest_guard = None, None
            try:
                store.check(old.FINAL)
                env = factory()
                moving.moving.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224
                        and env.unwrapped.render_mode == "rgb_array", "native render contract")
                row["status"] = "running"
                with moving.native_hooks(env, ops, row):
                    moving.branch(env, data["references"][plan["root_slot"]], data["plan"],
                                  plan, row, ops, store, i)
            except Exception as exc:
                moving.retain(row, exc, "setup_or_branch")
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as exc:
                        moving.retain(row, exc, "close")
            if local_guard(row, store.latest_guard, i):
                row["status"] = "guard_failed_invalid"
            store.emit("branch_closed", branch=i, receipt=row)
            if row["status"] not in ("horizon_completed", "native_success", "native_stopped",
                                     "guard_failed_invalid"):
                errors.append({"stage": "branch", "branch": i, "message": "global stop"})
                break
    except Exception as exc:
        errors.append({"stage": "execution", "type": type(exc).__name__, "message": str(exc)})
    try:
        for path, digest in p["bindings"].items():
            require(moving.sha(path) == digest, "final source drift: " + path)
    except Exception as exc:
        errors.append({"stage": "final_bindings", "type": type(exc).__name__, "message": str(exc)})
    started = [r for r in rows if r["status"] != "not_started"]
    if not errors:
        try:
            c = ops.counts
            require(len(started) == 30 and all(v["attempted"] == v["completed"] for v in c.values())
                    and c["reset"]["completed"] == 60 and c["set_state"]["completed"] == 120
                    and c["goal_render"]["completed"] == 60 and c["goal_bind"]["completed"] == 30
                    and c["public_step"]["completed"] == sum(r["warmup_returned"] + r["own_returned"] for r in rows)
                    and c["physics_step"]["completed"] == 120 + 10*c["public_step"]["completed"]
                    and c["explicit_render"]["completed"] == sum(r["PNG"] for r in rows), "resource census")
        except Exception as exc:
            errors.append({"stage": "final_census", "type": type(exc).__name__, "message": str(exc)})
    status = "all_candidates_processed" if not errors and len(started) == 30 else "partial_or_failed"
    counts = {key: sum(r["status"] == key for r in rows) for key in
              ("horizon_completed", "native_success", "native_stopped", "guard_failed_invalid",
               "failed", "not_started")}
    store.emit("run_finished", status=status, counts=ops.counts, candidate_counts=counts, errors=errors)
    report = {"schema": NAME + ".report.v1", "status": status, "rows": rows, "errors": errors,
              "contract": p["contract"], "candidate_counts": counts, "fixed_denominator": 30,
              "valid_endpoint_count": counts["horizon_completed"] + counts["native_success"] + counts["native_stopped"],
              "native_successes_valid": sum(r["native_success"] for r in rows
                                            if r["status"] in ("horizon_completed", "native_success", "native_stopped")),
              "counts": ops.counts, "journal_sha256": moving.sha(store.path / "journal.jsonl"),
              "journal_chain_tail": store.previous, "wall_seconds": time.monotonic()-ops.started,
              "P3_admission": False, "independent_outcome": False, "equal_total_compute": False,
              "new_neural_calls": 0}
    store.put("report.json", report, final=True)
    return report


def run(path, sha, output, authority):
    require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU authorization")
    p, data = verify(path, sha)
    store = Store(destination(output, "_RUN_"))
    store.put("execution_binding.json", {"prepared": str(Path(path).resolve()), "prepare_sha256": sha})
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                        with_target=False, render_action=False)
    return execute(p, data, store, factory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--authority")
    a = parser.parse_args()
    if a.action == "prepare":
        p, _ = material()
        path = destination(a.output, "_PREP_", ".json")
        print(json.dumps({"prepared": str(path), "sha256": old.audit.selected.host.host.write_new(path, p)}))
    else:
        r = run(a.prepared, a.sha256, a.output, a.authority)
        print(json.dumps({"status": r["status"], "candidate_counts": r["candidate_counts"],
                          "errors": r["errors"]}))


if __name__ == "__main__":
    main()
