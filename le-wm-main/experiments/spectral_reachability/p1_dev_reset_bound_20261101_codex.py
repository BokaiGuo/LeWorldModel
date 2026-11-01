"""Development-only, directly journaled native PushT random-reset producer."""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

from . import p1_future_root_queue_20261101_codex as queue
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking

NAME = "p1_dev_reset_bound_20261101_codex"
ROOT = queue.ROOT
SEEDS = [301000001, 301000002]
ENV_SHA = "c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
require = queue.require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependencies():
    # Imports are part of metadata preflight. No model or environment is created.
    modules = [importlib.import_module(name) for name in
               ("stable_worldmodel.envs.pusht.env", "stable_worldmodel.envs.utils",
                "stable_worldmodel.spaces", "gymnasium.core")]
    paths = [Path(__file__).resolve(), Path(queue.__file__).resolve(), Path(locking.__file__).resolve()]
    for module in modules:
        paths.append(Path(inspect.getsourcefile(module)).resolve())
    source_hashes = {str(path): sha(path) for path in paths}
    env = modules[0]
    require(sha(inspect.getsourcefile(env.PushT)) == ENV_SHA, "native PushT source drift")
    return source_hashes


def contract():
    queue.reject_known_seed_hits(SEEDS)
    return dict(schema=NAME + ".prepare.v1", seeds=SEEDS, source_sha256=dependencies(),
                history_sha256=queue.HISTORY_SHA, quarantine_sha256=queue.QUARANTINE_SHA,
                environment=dict(class_name="stable_worldmodel.envs.pusht.env.PushT",
                                 api="direct native class, no gym.make wrappers",
                                 kwargs=dict(render_mode="rgb_array", resolution=224, with_target=False)),
                max_reset_calls=len(SEEDS), public_steps=0, placement_resets=0, neural_calls=0,
                history="not_collected", training_membership="unknown", historical_coverage="unknown",
                freshness="unknown", independent_confirmation=False, P1_admission=False,
                status="development_only_no_formal_confirmation")


def emit(path, rows, kind, **fields):
    body = dict(index=len(rows), previous=rows[-1]["sha256"] if rows else None,
                kind=kind, **fields)
    event = dict(body, sha256=queue.digest(body))
    data = queue.canonical(event) + b"\n"
    with path.open("ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    rows.append(event)
    return event


def snapshot(body):
    import numpy as np
    state = np.asarray(body._get_obs(), dtype=np.float64)
    require(state.shape == (7,) and np.isfinite(state).all(), "state")
    bodies = {}
    for name in ("agent", "block"):
        b = getattr(body, name)
        bodies[name] = dict(position=list(map(float, b.position)), angle=float(b.angle),
                            velocity=list(map(float, b.velocity)), angular_velocity=float(b.angular_velocity))
    return dict(state=state.tolist(), bodies=bodies)


def run(prepared, expected_sha, output, factory=None):
    """Every reset in this producer is directly dispatched between durable events."""
    require(sha(prepared) == expected_sha, "external prepared file SHA")
    plan = queue.read_json(prepared)
    require(plan == contract(), "prepared source/contract drift")
    output = Path(output)
    output.mkdir(exist_ok=False)
    ledger = output / "journal.jsonl"
    rows, reports = [], []
    emit(ledger, rows, "run_started", prepare_sha256=expected_sha, planned=len(SEEDS))
    env = None
    errors = []
    try:
        module = importlib.import_module("stable_worldmodel.envs.pusht.env")
        cls = module.PushT
        require(sha(inspect.getsourcefile(cls)) == ENV_SHA, "runtime environment source")
        # Injectable factory is used solely by in-process tests; CLI cannot supply it.
        env = cls(**plan["environment"]["kwargs"]) if factory is None else factory()
        require(type(env) is cls if factory is None else True, "exact native class")
        for index, seed in enumerate(plan["seeds"]):
            item = dict(index=index, seed=seed, status="not_returned")
            reports.append(item)
            emit(ledger, rows, "reset_attempt", index_in_roster=index, seed=seed)
            try:
                obs, info = env.reset(seed=seed)
                import numpy as np
                state = np.asarray(obs["state"], dtype=np.float64)
                goal = np.asarray(info["goal_state"], dtype=np.float64)
                require(state.shape == goal.shape == (7,) and np.isfinite(state).all()
                        and np.isfinite(goal).all(), "native reset observation")
                snap = snapshot(env)
                require(np.array_equal(state, np.asarray(snap["state"])), "state identity")
                receipt = dict(state=state.tolist(), goal=goal.tolist(), snapshot=snap,
                               state_sha256=queue.digest(state.tolist()),
                               goal_sha256=queue.digest(goal.tolist()), snapshot_sha256=queue.digest(snap),
                               history_sha256=None, history_status="not_collected")
                item.update(status="returned", receipt=receipt)
                emit(ledger, rows, "reset_return", index_in_roster=index, seed=seed, receipt=receipt)
            except Exception as exc:
                item.update(status="failed", error_class=type(exc).__name__, message=str(exc)[:500])
                emit(ledger, rows, "reset_failed", index_in_roster=index, seed=seed,
                     error_class=type(exc).__name__, message=str(exc)[:500])
                # Failed reset may leave the environment in an unknown state: stop,
                # preserve not-started slots and never silently retry.
                break
        require(plan == contract(), "post-run source/contract drift")
    except Exception as exc:
        errors.append(dict(type=type(exc).__name__, message=str(exc)[:500]))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                errors.append(dict(type=type(exc).__name__, message=str(exc)[:500]))
        for index in range(len(reports), len(plan["seeds"])):
            reports.append(dict(index=index, seed=plan["seeds"][index], status="not_started"))
        emit(ledger, rows, "run_finished", errors=errors)
        report = dict(schema=NAME + ".report.v1", prepared_sha256=expected_sha, rows=reports,
                      errors=errors, journal_sha256=sha(ledger), journal_tail=rows[-1]["sha256"],
                      environment_scope="direct native PushT random reset only",
                      source_closure="bounded Python files; native libraries and full transitive closure not attested",
                      training_membership="unknown", historical_coverage="unknown", freshness="unknown",
                      P1_admission=False, independent_confirmation=False, neural_calls=0)
        queue.write_new(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--prepared", required=True)
    execute.add_argument("--sha256", required=True)
    execute.add_argument("--output", required=True)
    execute.add_argument("--authorize-development-reset", required=True, action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(queue.write_new(args.output, contract()))
        return
    with locking.gpu_lock(LOCK):
        result = run(args.prepared, args.sha256, args.output)
    print(json.dumps({"rows": [r["status"] for r in result["rows"]], "errors": result["errors"]}))
    if result["errors"] or any(r["status"] != "returned" for r in result["rows"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
