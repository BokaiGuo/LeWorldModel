"""Durable per-feedback-node bridge for the exploratory spare-budget variant."""
from . import p1_goal_conditioned_spectral_20261031 as control
from . import codex_four_arm_common_suffix_control_20261030 as base
from . import codex_four_arm_full_cost_20261030 as spec
from . import codex_four_arm_durable_20261030 as durable
from . import codex_four_arm_leaf_suffix_20261030 as leafmod
from . import codex_four_arm_consumer_20261030 as consumer
from . import agent_p3_c3_k17_physical_transport_20261029_v1 as journal

TRACE_LIMIT = 16 * 1024 ** 2
BOUNDARIES = (0, 5, 10, 15)


def plan_node(task, arm, boundary, store, runtime):
    spec.require(arm in spec.ARMS and type(boundary) is int and boundary in BOUNDARIES, "arm/boundary")
    policy, contract, config = control.contract_for(task, runtime["model_sha256"], runtime["hardware_sha256"])
    contract_sha = spec.digest(contract)
    name = "s0_" + arm + f"_b{boundary:02}_winner"
    ns = store.namespace(name)
    trace = journal.Journal(ns / "candidates.jsonl", TRACE_LIMIT)
    driver = None
    receipt = None
    failure = None
    try:
        driver = durable.Driver(ns / "ledger", contract, contract_sha, arm, "node", arm)
        leaf = leafmod.LeafAdapter(runtime["model"], driver, contract, contract_sha)
        original_emit = driver._emit

        def checked_emit(kind, payload):
            if kind == "intent":
                try:
                    runtime["check_limits"]()
                except BaseException:
                    original_emit("invalid", {})
                    raise
            return original_emit(kind, payload)

        driver._emit = checked_emit

        def emit(kind, payload):
            runtime["check_limits"]()
            store.check()
            trace.emit(kind, payload)

        result = control.plan(leaf, arm, task, config, policy, contract,
                              device=runtime["device"], emit=emit)
        emit("terminal", {"status": "complete_node"})
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        if driver is not None and not driver.failed and not driver.closed:
            try:
                driver.dispatch("setup", "__abort__", {}, lambda: None)
            except BaseException:
                pass
        try:
            trace.emit("terminal", {"status": "failed_or_partial", "error": failure})
        except BaseException:
            pass
    finally:
        trace.close()
        if driver is not None:
            receipt = driver.finish()
    account = None if receipt is None else consumer.consume(receipt["journal"], receipt["sha256"],
                                                             contract_sha, arm, "node", arm)
    info = {"status": "failed_or_partial" if failure else "complete", "error": failure,
            "contract_sha256": contract_sha, "ledger": receipt, "account": account,
            "trace": str(ns / "candidates.jsonl"), "trace_sha256": base.spectral.file_sha(ns / "candidates.jsonl")}
    store.put(name + "_receipt.json", info)
    if failure is not None:
        raise ValueError("node failed: " + str(failure))
    spec.require(account["status"] == "closed_unverified"
                 and account["charged_upper_bounds"] == control.cost(arm), "exact node accounting")
    packet = base.first_packet(result)
    return dict(info, commands=packet.cpu().tolist(), selected_index=result["selected_index"], scores=result["scores"])
