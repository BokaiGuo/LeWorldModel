"""Six-branch H3 physical lifecycle; caller authenticates and owns its store."""
from copy import deepcopy
import hashlib
import time

from . import p3_raw_h3_branch_20261031 as branch
from . import p3_raw_h3_physical_prepare_20261031 as prepared

host, require = prepared.h1.host, prepared.require
kernel = host.kernel
NAME = "p3_raw_h3_physical_batch_20261031"


def contract():
    return {"schema": NAME + ".contract", "branches": 6, "warmup_steps": 10,
            "requested_steps": 20, "wall_seconds": 900,
            "caps": {"environment_construct": 6, "environment_close": 6,
                     "reset": 12, "public_step": 180, "physics_step": 1824,
                     "set_state": 24, "goal_render": 12, "goal_bind": 6,
                     "explicit_render": 192},
            "physical_authorized": False, "scientific_pass": False}


def validate_plans(plans):
    require(type(plans) is list and len(plans) == 6, "six plans")
    for old, new in zip(plans[::2], plans[1::2], strict=True):
        require(old["variant"] == "old" and new["variant"] == "new"
                and all(old[k] == new[k] for k in ("root_slot", "root_sha256", "task_id",
                                                    "decision_index", "policy", "candidate",
                                                    "prefix_float32_sha256")), "pair identity")
        a, b = branch.raw20(old["commands"]), branch.raw20(new["commands"])
        require(a[:15].tobytes() == b[:15].tobytes(), "pair prefix bytes")
        for plan, actions in ((old, a), (new, b)):
            require(hashlib.sha256(actions.tobytes()).hexdigest() == plan["raw_float32_sha256"]
                    and hashlib.sha256(actions[:15].tobytes()).hexdigest()
                    == plan["prefix_float32_sha256"]
                    and kernel.source.digest({k: v for k, v in plan.items()
                                              if k != "selection_digest"}) == plan["selection_digest"],
                    "plan digest")
    return True


def execute(p, data, store, factory):
    """Execute a complete fixed six-row batch; failures remain recorded."""
    require(p["schema"] == NAME + ".prepare" and p["contract"] == contract(), "batch contract")
    validate_plans(p["plans"])
    require(len(data["references"]) == 2, "two source roots")
    rows = []
    for i, plan in enumerate(p["plans"]):
        arm = f"d{plan['decision_index']}_c{plan['candidate']}_{plan['variant']}"
        selected = {**deepcopy(plan), "arm": arm}
        rows.append(dict(branch_index=i, root_slot=plan["root_slot"], arm=arm,
                         selection_digest=plan["selection_digest"], selected_plan=selected,
                         status="not_started", first_error=None, secondary_errors=[],
                         warmup_returned=0, own_attempted=0, own_returned=0,
                         initial_exact=False, full_horizon=False, native_success=False,
                         terminated=False, truncated=False, endpoint_scope="not_started",
                         PNG=0, checkpoints=[], constructor_internal_work="not_attempted"))
    ops = kernel.Ops(p, store)
    errors = []
    try:
        store.emit("run_started", prepare=p)
        for i, row in enumerate(rows):
            plan = row["selected_plan"]
            env, holder = None, []
            ops.where = dict(branch=i, root_slot=plan["root_slot"], arm=plan["arm"],
                             selection_digest=plan["selection_digest"])

            def construct():
                row["constructor_internal_work"] = "unknown"
                value = factory()
                holder.append(value)
                return value
            try:
                env = ops.call("environment_construct", construct)
                kernel.moving.validate_env(env.unwrapped)
                require(all(getattr(env.unwrapped, k, None) is None for k in
                            ("space", "render_buffer", "latest_action")), "pristine constructor")
                require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224
                        and env.unwrapped.render_mode == "rgb_array", "native RGB interface")
                row["constructor_internal_work"] = "sealed_native_pristine_no_physics"
                row["status"] = "running"
                with kernel.native_hooks(env, ops, row):
                    branch.branch(env, data["references"][plan["root_slot"]], data["plan"],
                                  plan, row, ops, store, i)
            except BaseException as exc:
                kernel.retain(row, exc, "setup_or_branch")
            finally:
                if env is None and holder:
                    env = holder[0]
                if env is not None:
                    host.helper.close_env(env, row, ops, store)
                if row["status"] == "failed":
                    row["endpoint_scope"] = "failed_partial"
                store.emit("branch_closed", receipt=row)
    except BaseException as exc:
        errors.append({"stage": "execution", "type": type(exc).__name__, "message": str(exc)})
    try:
        for path, digest in p["bindings"].items():
            require(host.sha(path) == digest, "final source drift")
    except BaseException as exc:
        errors.append({"stage": "source", "type": type(exc).__name__, "message": str(exc)})
    complete = not errors and all(r["status"] in ("horizon_completed", "native_success", "native_stopped")
                                  and r["first_error"] is None and not r["secondary_errors"] for r in rows)
    if complete:
        try:
            counts = ops.counts
            steps = sum(r["warmup_returned"] + r["own_returned"] for r in rows)
            require(all(v["attempted"] == v["completed"] for v in counts.values())
                    and counts["environment_construct"]["completed"] == counts["environment_close"]["completed"] == 6
                    and counts["reset"]["completed"] == 12 and counts["set_state"]["completed"] == 24
                    and counts["goal_render"]["completed"] == 12 and counts["goal_bind"]["completed"] == 6
                    and counts["public_step"]["completed"] == steps
                    and counts["physics_step"]["completed"] == 24 + 10 * steps
                    and counts["explicit_render"]["completed"] == sum(r["PNG"] for r in rows),
                    "actual resource census")
        except BaseException as exc:
            errors.append({"stage": "resource_census", "type": type(exc).__name__, "message": str(exc)})
            complete = False
    result = {"schema": NAME + ".report", "status": "recorded_not_admitted" if complete else "partial_or_failed",
              "rows": rows, "errors": errors, "counts": ops.counts, "contract": p["contract"],
              "elapsed_seconds": time.monotonic() - ops.started, "fixed_denominator": 6,
              "new_neural_calls": 0, "scientific_pass": False}
    try:
        store.emit("run_finished", status=result["status"], counts=ops.counts, errors=errors)
        result["journal_sha256"] = host.sha(store.path / "journal.jsonl")
        result["journal_chain_tail"] = store.previous
    except BaseException as exc:
        result["status"] = "partial_or_failed"
        result["errors"].append({"stage": "journal", "type": type(exc).__name__, "message": str(exc)})
    store.put("report.json", result, final=True)
    return result
