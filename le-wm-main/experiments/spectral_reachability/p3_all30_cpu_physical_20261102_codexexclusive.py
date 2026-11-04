"""Fixed 30 known-root candidate CPU replay, with fail-closed source and root guards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from . import p3_paired_rank_physical_audit_20261102_codexexclusive as audit
from . import p3_venv_winner_cpu_physical_20261102_codexexclusive as four

ROOT = audit.ROOT
moving = four.moving
NAME = "p3_all30_cpu_physical_20261102_codexexclusive"
INPUT = ROOT / "results/p3_paired_rank_physical_audit_20261102_codexexclusive_02.json"
INPUT_SHA = "e5d9f05c6dabaa7b9e1a910a26a7f328cbb40349fa99f5a07e1d3278cf95e43b"
CAP = 64 * 1024**2
FLOOR = 200 * 1024**2
FINAL = 2 * 1024**2
AUTH = "P3_ALL30_KNOWN_ROOT_CPU_REPLAY"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def destination(value, prefix, suffix=""):
    path = Path(value)
    require(path.is_absolute() and path.parent == ROOT / "results" and
            path.name.startswith(NAME + prefix) and path.name.endswith(suffix) and
            not os.path.lexists(path), "exclusive output")
    return path


def material():
    require(moving.sha(INPUT) == INPUT_SHA, "all-candidate audit SHA")
    saved = moving.read(INPUT)
    require(saved == audit.build(), "reconstructed candidate audit")
    previous, data = four.material()
    items = saved["next_input"]["actions"]
    require([(x["root_slot"], x["index"]) for x in items] ==
            [(r, i) for r in (0, 1) for i in range(15)], "fixed thirty candidate order")
    plans = []
    for item in items:
        raw = audit.action(item["commands"])
        require(hashlib.sha256(raw.tobytes()).hexdigest() == item["raw_float32_sha256"], "raw action SHA")
        payload = {"arm": "all_candidates", "root_slot": item["root_slot"],
                   "candidate_index": item["index"], "commands": item["commands"],
                   "raw_float32_sha256": item["raw_float32_sha256"]}
        plans.append(dict(payload, selection_digest=moving.source.digest(payload)))
    contract = moving.low.contract(30, 1800)
    contract.update(warmup_raw=previous["contract"]["warmup_raw"], output_cap_bytes=CAP,
                    final_reserve_bytes=FINAL, scope="all 30 fixed candidates on 2 already-opened known roots",
                    no_gpu=True, no_replan=True, no_retries=True,
                    failure_rule="any setup, guard, binding, storage or execution error stops all further dispatch",
                    native_terminal_rule="success or truncation stops only this branch; next fixed branch continues",
                    admission=False)
    bindings = dict(previous["bindings"])
    for path in (INPUT, Path(__file__), Path(audit.__file__)):
        bindings[str(path.resolve())] = moving.sha(path)
    require(bindings[str(INPUT)] == INPUT_SHA, "input seal")
    for path, digest in bindings.items():
        require(moving.sha(path) == digest, "binding drift: " + path)
    prep = {"schema": NAME + ".prepare.v1", "plans": plans, "contract": contract,
            "bindings": bindings, "sourcepack_sha256": previous["sourcepack_sha256"],
            "source_prepare_sha256": previous["source_prepare_sha256"], "candidate_input_sha256": INPUT_SHA,
            "P3_admission": False, "independent_outcome": False, "equal_total_compute": False}
    return prep, data


class Store(moving.Store):
    def __init__(self, path):
        self.path = Path(path)
        require(not os.path.lexists(self.path) and shutil.disk_usage(self.path.parent).free >= FLOOR + CAP,
                "exclusive directory and disk reserve")
        self.path.mkdir(mode=0o700)
        self.previous, self.index = None, 0

    def check(self, size=4096):
        used = sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
        require(used + size <= CAP - FINAL and shutil.disk_usage(self.path).free >= FLOOR + size,
                "output cap/disk floor")

    def write(self, name, value, append=False, final=False):
        path = self.path / name
        require(Path(name).name == name and name not in ("", ".", "..") and not path.is_symlink(),
                "flat nonsymlink output")
        used = sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
        require(used + len(value) <= CAP - (0 if final else FINAL) and
                shutil.disk_usage(self.path).free >= FLOOR + len(value), "output cap/disk floor")
        with path.open("ab" if append else "xb") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        fd = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return moving.sha(path)


def verify(path, expected):
    require(moving.sha(path) == expected, "external prepare SHA")
    saved = moving.read(path)
    rebuilt, data = material()
    require(saved == rebuilt, "prepare reconstruction")
    return saved, data


def execute(p, data, store, factory):
    require(len(p["plans"]) == 30 and len(data["references"]) == 2, "roster")
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
            env = None
            try:
                store.check(FINAL)
                env = factory()
                moving.moving.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224
                        and env.unwrapped.render_mode == "rgb_array", "native rendering contract")
                row["status"] = "running"
                with moving.native_hooks(env, ops, row):
                    moving.branch(env, data["references"][plan["root_slot"]], data["plan"], plan,
                                  row, ops, store, i)
            except Exception as exc:
                moving.retain(row, exc, "setup_or_branch")
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as exc:
                        moving.retain(row, exc, "close")
                store.emit("branch_closed", branch=i, receipt=row)
            if row["status"] not in ("horizon_completed", "native_success", "native_stopped"):
                errors.append({"stage": "branch", "branch": i, "message": "stop after failed branch"})
                break
    except Exception as exc:
        errors.append({"stage": "execution", "type": type(exc).__name__, "message": str(exc)})
    try:
        for path, digest in p["bindings"].items():
            require(moving.sha(path) == digest, "final source drift: " + path)
    except Exception as exc:
        errors.append({"stage": "final_bindings", "type": type(exc).__name__, "message": str(exc)})
    attempted = [r for r in rows if r["status"] != "not_started"]
    if not errors:
        try:
            c = ops.counts
            require(len(attempted) == 30 and all(v["attempted"] == v["completed"] for v in c.values())
                    and c["reset"]["completed"] == 60 and c["set_state"]["completed"] == 120
                    and c["goal_render"]["completed"] == 60 and c["goal_bind"]["completed"] == 30
                    and c["public_step"]["completed"] == sum(r["warmup_returned"] + r["own_returned"] for r in rows)
                    and c["physics_step"]["completed"] == 120 + 10*c["public_step"]["completed"]
                    and c["explicit_render"]["completed"] == sum(r["PNG"] for r in rows), "resource census")
        except Exception as exc:
            errors.append({"stage": "final_census", "type": type(exc).__name__, "message": str(exc)})
    status = "all_30_recorded" if not errors and len(attempted) == 30 else "partial_or_failed"
    store.emit("run_finished", status=status, counts=ops.counts, errors=errors)
    report = {"schema": NAME + ".report.v1", "status": status, "rows": rows, "errors": errors,
              "contract": p["contract"], "counts": ops.counts,
              "journal_sha256": moving.sha(store.path / "journal.jsonl"), "journal_chain_tail": store.previous,
              "wall_seconds": time.monotonic() - ops.started, "P3_admission": False,
              "independent_outcome": False, "equal_total_compute": False, "new_neural_calls": 0}
    store.put("report.json", report, final=True)
    return report


def run(prepared, sha, out, authority):
    require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU authorization")
    p, data = verify(prepared, sha)
    store = Store(destination(out, "_RUN_"))
    store.put("execution_binding.json", {"prepared": str(Path(prepared).resolve()), "prepare_sha256": sha})
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
    args = parser.parse_args()
    if args.action == "prepare":
        p, _ = material()
        path = destination(args.output, "_PREP_", ".json")
        print(json.dumps({"prepared": str(path), "sha256": audit.selected.host.host.write_new(path, p)}))
    else:
        report = run(args.prepared, args.sha256, args.output, args.authority)
        print(json.dumps({"status": report["status"], "errors": report["errors"],
                          "statuses": [r["status"] for r in report["rows"]]}))


if __name__ == "__main__":
    main()
