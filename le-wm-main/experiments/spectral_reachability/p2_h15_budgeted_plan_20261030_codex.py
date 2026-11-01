"""Exclusive H15 plan using the observable replay/wall bridge.

The caller constructs the bridge before standalone preparation and supplies an
already constructed runner and explicit barrier. No model/environment launcher.
Only callbacks routed here are observed; this does not authorize production.
"""
import copy
from pathlib import Path

from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_h15_cuda_ledger_runner_20261030_codex as runner
from experiments.spectral_reachability import p2_h15_observable_budget_bridge_20261030_codex as budget
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec

NAME = "p2_h15_budgeted_plan_20261030_codex"
LIMIT = 1024 * 1024


def source_hash():
    return spec.sha(Path(__file__).read_bytes())


class Plan:
    def __init__(self, active_runner, bridge, *, expected_source_sha256):
        core.require(spec.hash_string(expected_source_sha256) and expected_source_sha256 == source_hash(),
                     "independently pinned plan source")
        core.require(type(active_runner) is runner.Runner and type(bridge) is budget.Bridge
                     and bridge.driver is None and not bridge.closed and not bridge.failed
                     and not active_runner.closed and not active_runner.driver.closed
                     and not active_runner.driver.failed and not active_runner.driver.ledger.receipt()["events"]
                     and not active_runner.cpu.compiled, "fresh runner and bridge")
        self.runner, self.bridge, self.source_sha = active_runner, bridge, expected_source_sha256
        self.used = False
        try:
            bridge.attach(active_runner.driver)
            self.journal = transport.Journal(active_runner.driver.directory / "budgeted_plan.jsonl", LIMIT)
            self.journal.emit("header", dict(schema=NAME, source_sha256=self.source_sha,
                                            bridge_sha256=budget.source_hash(),
                                            bridge_attach_sha256=bridge.journal.previous,
                                            contract_sha256=bridge.contract_sha, arm=bridge.arm,
                                            run_id=bridge.run_id, task_sha256=core.digest(active_runner.task)))
        except BaseException:
            bridge.failed = active_runner.driver.failed = True
            if hasattr(self, "journal"):
                self.journal.close()
            raise

    def execute(self, synchronize):
        r, b = self.runner, self.bridge
        try:
            core.require(not self.used and not r.closed and not b.closed and not b.failed
                         and source_hash() == self.source_sha, "single use and stable plan source")
            self.used = True
            response = r.signal()
            bank = r.cpu.compile(response["jacobian"], response["nominal"], r.task["goal"])
            scored = []
            for i, actions in enumerate(bank["raw15"]):
                path = b.replay(lambda: r.replay(actions, index=i))
                core.checked_path(r.task, r.task["root_context"], actions, path)
                score = r.cpu.score_candidate(i, path[-1]["latent_history"][-1], r.task["goal"])
                scored.append(dict(index=i, score=score, path=path))
            winner = min(range(40), key=lambda i: (scored[i]["score"], i))
            actions = bank["raw15"][winner]
            replays = [b.replay(lambda: r.replay(actions, index=40+i, segmented=segmented))
                       for i, segmented in enumerate((False, True))]
            core.require(replays[0] == replays[1] == scored[winner]["path"], "exact selected replay")
            result = dict(bank=bank, candidates=scored, winner=winner, replays=replays)
            r.event(dict(kind="plan_return", result=result))
            wall = b.finish_planning(synchronize)
            core.require(source_hash() == self.source_sha, "plan source drift")
            self.journal.emit("planned", dict(plan_sha256=core.digest(result), winner=winner, wall_ns=wall,
                                               replay_count=b.counts["replay_calls"]["completed"]))
            return result
        except BaseException:
            b.failed = r.driver.failed = True
            raise

    def finish(self, runner_result, bridge_result):
        r, b = self.runner, self.bridge
        try:
            core.require(self.used and r.closed and b.closed and not b.failed
                         and source_hash() == self.source_sha, "completed plan and stable source")
            core.require(runner_result["journal"] == str(r.driver.directory / "events.jsonl")
                         and runner_result["native_io"] == str(r.driver.directory / "native_io.jsonl")
                         and bridge_result["journal"] == str(b.path / "budget_events.jsonl"), "local journal paths")
            rc = runner.verify(runner_result, b.contract_sha)
            bc = budget.verify(bridge_result["journal"], bridge_result["sha256"], runner_result["journal"],
                               runner_result["sha256"], b.contract_sha, arm=b.arm, run_id=b.run_id)
            core.require(rc["status"] == bc["status"] == "complete"
                         and bc["counts"]["replay_calls"] == dict(attempted=42, completed=42),
                         "complete 42 bridged replays")
            self.journal.emit("terminal", dict(native_sha256=runner_result["native_io_sha256"],
                                                ledger_sha256=runner_result["sha256"],
                                                budget_sha256=bridge_result["sha256"]))
        finally:
            self.journal.close()
        path = self.journal.path
        return dict(journal=str(path), sha256=spec.sha(path.read_bytes()), coverage_verified=False,
                    budget_fairness=False, P1_admission=False, P2_admission=False, physical_release=False)
