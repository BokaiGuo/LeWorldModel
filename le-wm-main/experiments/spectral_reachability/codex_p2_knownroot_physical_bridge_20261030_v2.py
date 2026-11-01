"""独立的 P2 known-root 物理桥接 seam。

该文件只消费已经外部绑定的物理计划包；不加载模型、不创建 CUDA、
不选择动作，也不把执行结果升级为 P2 admission。旧的 known-root
CUDA 报告没有物理 root/goal/warmup 绑定时，``compile_packet`` 会拒绝。
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time


NAME = "codex_p2_knownroot_physical_bridge_20261030_v2"
AUTH = "MAIN_COORDINATED_P2_KNOWNROOT_PHYSICAL_EXPLORATION_V2"
MAX_STEPS = 10


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_new(path, value):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _check_source_bindings(bindings):
    need(type(bindings) is dict and bindings, "source bindings required")
    for filename, expected in bindings.items():
        path = Path(filename)
        need(path.is_absolute() and path.is_file() and not path.is_symlink(), "source missing")
        need(file_sha(path) == expected, "source drift")


def compile_packet(*, plans, physical_cases, source_bindings):
    """Bind model raw actions to physical task/root/warmup identities.

    ``physical_cases`` is deliberately external: the existing CUDA plan does
    not contain enough information to infer it from post-hoc results.
    """
    need(type(plans) is list and plans, "plans required")
    need(type(physical_cases) is list and len(physical_cases) == len(plans),
         "one physical case per plan required")
    _check_source_bindings(source_bindings)
    rows = []
    for index, (plan, case) in enumerate(zip(plans, physical_cases, strict=True)):
        need(plan.get("selection_uses_physical_outcome") is False,
             "physical outcome cannot select plan")
        commands = deepcopy(plan.get("commands"))
        need(type(commands) is list and len(commands) == MAX_STEPS,
             "fixed raw10 action witness required")
        need(plan.get("root_sha256") == case.get("root_sha256"), "root identity")
        for key in ("task_id", "task_sha256", "goal_sha256", "warmup_sha256",
                    "renderer", "action_domain"):
            need(key in case, "missing physical binding: " + key)
        rows.append({
            "index": index,
            "root_sha256": case["root_sha256"],
            "task_id": case["task_id"],
            "task_sha256": case["task_sha256"],
            "goal_sha256": case["goal_sha256"],
            "warmup_sha256": case["warmup_sha256"],
            "renderer": case["renderer"],
            "action_domain": deepcopy(case["action_domain"]),
            "commands": commands,
            "selection_digest": digest(plan),
            "selection_uses_physical_outcome": False,
        })
    packet = {
        "schema": NAME + ".packet",
        "status": "bound_not_executed",
        "plans": rows,
        "source_bindings": deepcopy(source_bindings),
        "neural_calls": 0,
        "physical_release": False,
        "scientific_admission": False,
        "P1_admission": False,
        "P2_admission": False,
    }
    packet["packet_sha256"] = digest({k: v for k, v in packet.items() if k != "packet_sha256"})
    return packet


def save_packet(path, packet):
    path = Path(path)
    need(path.is_absolute() and path.name.startswith(NAME + "_"), "exclusive packet path")
    _write_new(path, packet)
    return file_sha(path)


class _Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open("x", encoding="utf-8")
        self.previous = None
        self.index = 0

    def emit(self, event):
        event = dict(event, index=self.index, previous_sha256=self.previous)
        event["event_sha256"] = digest({k: v for k, v in event.items() if k != "event_sha256"})
        self.stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.previous = event["event_sha256"]
        self.index += 1
        return event

    def close(self):
        self.stream.close()


def run_physical(*, packet, output, factory, authorization=None):
    """Run externally supplied physical environments, recording every boundary.

    The factory is the only physical callback. It must return an env with
    ``reset``, ``step``, ``render``, and ``close``. No neural callback exists
    in this function.
    """
    raise RuntimeError("BLOCKED: prototype lacks native root/warmup replay and physical resource enforcement")


def _fixture_only_run_physical(*, packet, output, factory, authorization=None):
    """Synthetic seam only; not an authorized native physical entry."""
    need(authorization == AUTH, "physical coordination authorization required")
    need(packet["schema"] == NAME + ".packet" and packet["status"] == "bound_not_executed",
         "unexecuted packet required")
    need(packet["packet_sha256"] == digest({k: v for k, v in packet.items()
                                            if k != "packet_sha256"}),
         "packet digest")
    need(packet["neural_calls"] == 0 and not packet["physical_release"] and
         not packet["scientific_admission"], "unadmitted packet")
    _check_source_bindings(packet["source_bindings"])
    output = Path(output)
    need(output.is_absolute() and output.name.startswith(NAME + "_") and
         not output.exists() and not output.is_symlink(), "exclusive output")
    output.mkdir(mode=0o700)
    started = time.monotonic()
    report = {
        "schema": NAME + ".report",
        "status": "partial_or_failed",
        "packet_sha256": packet["packet_sha256"],
        "rows": [],
        "errors": [],
        "neural_calls": 0,
        "physical_release": False,
        "scientific_admission": False,
        "P1_admission": False,
        "P2_admission": False,
    }
    journal = _Journal(output / "journal.jsonl")
    try:
        journal.emit({"kind": "run_started", "packet_sha256": packet["packet_sha256"]})
        for spec in packet["plans"]:
            row = dict(index=spec["index"], root_sha256=spec["root_sha256"],
                       task_id=spec["task_id"], selection_digest=spec["selection_digest"],
                       status="running", actions=[], observations=[], error=None)
            env = None
            try:
                env = factory(spec)
                need(all(hasattr(env, name) for name in ("reset", "step", "render", "close")),
                     "physical env interface")
                initial = env.reset()
                row["initial_state_digest"] = digest(initial)
                for step, action in enumerate(spec["commands"]):
                    journal.emit({"kind": "action_dispatch", "branch": spec["index"],
                                  "step": step, "action": deepcopy(action)})
                    returned = env.step(deepcopy(action))
                    rgb = env.render()
                    need(rgb is not None, "RGB observation required")
                    rgb_bytes = bytes(rgb)
                    obs = {"step": step, "rgb_sha256": hashlib.sha256(rgb_bytes).hexdigest(),
                           "state_digest": digest(returned), "nbytes": len(rgb_bytes)}
                    row["actions"].append(deepcopy(action))
                    row["observations"].append(obs)
                    journal.emit({"kind": "rgb_observation", "branch": spec["index"],
                                  "step": step, **obs})
                row["status"] = "completed"
            except BaseException as exc:
                row["status"] = "failed_partial"
                row["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
                report["errors"].append(row["error"])
            finally:
                if env is not None:
                    try:
                        env.close()
                    except BaseException as exc:
                        report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
                report["rows"].append(row)
                journal.emit({"kind": "branch_closed", "receipt": deepcopy(row)})
            if report["errors"]:
                break
        report["status"] = "physical_recorded_not_admitted" if not report["errors"] and \
            len(report["rows"]) == len(packet["plans"]) else "partial_or_failed"
        journal.emit({"kind": "run_finished", "status": report["status"],
                      "errors": deepcopy(report["errors"])})
    except BaseException as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
        report["status"] = "partial_or_failed"
    finally:
        report["journal_tail_sha256"] = journal.previous
        journal.close()
        report["wall_seconds"] = time.monotonic() - started
        _write_new(output / "report.json", report)
    return report


def consume(output, expected_report_sha256):
    """Strictly consume only this bridge's saved report and journal."""
    output = Path(output)
    report_path = output / "report.json"
    need(file_sha(report_path) == expected_report_sha256, "report hash")
    report = json.loads(report_path.read_text())
    need(report["schema"] == NAME + ".report" and report["neural_calls"] == 0 and
         report["physical_release"] is False and report["scientific_admission"] is False,
         "report contract")
    data = (output / "journal.jsonl").read_bytes()
    previous = None
    events = [json.loads(line) for line in data.splitlines()]
    need(events and events[0]["kind"] == "run_started" and events[-1]["kind"] == "run_finished",
         "journal bookends")
    for index, event in enumerate(events):
        need(event["index"] == index and event["previous_sha256"] == previous,
             "journal chain")
        expected = digest({k: v for k, v in event.items() if k != "event_sha256"})
        need(event["event_sha256"] == expected, "journal event hash")
        previous = event["event_sha256"]
    closed = [event["receipt"] for event in events if event["kind"] == "branch_closed"]
    need(closed == report["rows"], "saved row/journal binding")
    finished = events[-1]
    need(finished["status"] == report["status"] and
         finished["errors"] == report["errors"], "saved final binding")
    need(previous == report["journal_tail_sha256"], "journal tail")
    return {"status": report["status"], "rows": len(report["rows"]),
            "errors": deepcopy(report["errors"]), "P2_admission": False}
