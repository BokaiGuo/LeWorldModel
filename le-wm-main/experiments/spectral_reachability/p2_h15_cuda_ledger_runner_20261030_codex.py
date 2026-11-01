"""Exclusive native H15 wiring. No model loading, CUDA setup, or physical action.

The caller supplies the already-loaded native model and an independently fixed
resource contract. This is instrumentation preparation, not an admission gate.
"""
import copy
import hashlib
from pathlib import Path

from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import agent_p3_local_signal_cuda_20261028_v1 as signal
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as tree
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as durable
from experiments.spectral_reachability import p2_h15_neural_ledger_adapter_20261030_codex as adapter
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec

NAME = "p2_h15_cuda_ledger_runner_20261030_codex"
IO_LIMIT = 256 * 1024 * 1024


class Runner:
    """Single task, native model, durable ledger; all methods fail-stop."""
    def __init__(self, directory, contract, contract_sha, caps, arm, run_id, task, model, device):
        task = core.validate_task(task)
        core.require(getattr(device, "type", None) == "cuda", "native CUDA device required")
        before = signal.model_hash(model)
        core.require(before["original_state_dict"] == task["binding"]["model_sha256"], "model identity mismatch")
        self.cpu, self.ledger, self.driver = adapter.create(directory, contract, contract_sha, caps, arm, run_id, task)
        self.task, self.model, self.before = copy.deepcopy(task), model, before
        self.task_hash = core.digest(task)
        self.source_hashes = adapter.sources()
        self.runner_hash = spec.sha(Path(__file__).read_bytes())
        self.closed = False
        try:
            self.io = transport.Journal(Path(directory) / "native_io.jsonl", IO_LIMIT)
            self.io.emit("header", dict(schema=NAME, contract_sha256=contract_sha, arm=arm,
                                        run_id=run_id, task=self.task, task_sha256=self.task_hash,
                                        model_before=before, source_sha256=self.source_hashes,
                                        runner_sha256=self.runner_hash))
            self.backend = tree.NativeTree(model, device, task, self)
        except BaseException:
            self.driver.failed = True
            if hasattr(self, "io"):
                self.io.close()
            self.driver.finish()
            raise

    def event(self, event):
        try:
            self.io.emit("native", event)
        except BaseException:
            self.driver.failed = True
            raise

    def _check(self):
        core.require(not self.closed and not self.driver.closed and not self.driver.failed, "runner fail-stopped")
        core.require(core.digest(self.task) == self.task_hash and adapter.sources() == self.source_hashes
                     and spec.sha(Path(__file__).read_bytes()) == self.runner_hash, "task/source drift")
        core.require(signal.model_hash(self.model) == self.before, "model drift")

    def _invoke(self, method, request, scope, callback):
        try:
            self._check()
            self.ledger.scope = scope
            self.event(dict(kind="native_call_attempt", method=method, task=self.task, request=request))
            result = callback()
            self._check()
            self.event(dict(kind="native_call_return", result=result))
            return result
        except BaseException as exc:
            self.driver.failed = True
            try:
                self.event(dict(kind="native_call_failure", error=dict(type=type(exc).__name__, message=str(exc))))
            except BaseException:
                pass
            raise

    def signal(self):
        req = core.request(self.task, dict(context=self.task["root_context"]),
                           operation="compile_finite_horizon_J", horizon=3)
        return self._invoke("signal", req, dict(stage="signal", index=0),
                            lambda: self.backend.signal(req, self.ledger))

    def replay(self, actions, *, index, segmented=False):
        try:
            core.require(type(index) is int and index >= 0 and type(segmented) is bool, "replay scope")
            req = dict(parent_context=copy.deepcopy(self.task["root_context"]),
                       actions=copy.deepcopy(actions), segmented=segmented)
            return self._invoke("replay", req, dict(stage="replay", index=index, segmented=segmented),
                                lambda: self.backend.replay(req["parent_context"], req["actions"], segmented, self.ledger))
        except BaseException:
            self.driver.failed = True
            raise

    def plan(self):
        """One H15 bank, 40 paid candidates, two paid selected replays."""
        try:
            response = self.signal()
            bank = self.cpu.compile(response["jacobian"], response["nominal"], self.task["goal"])
            scored = []
            for index, actions in enumerate(bank["raw15"]):
                path = self.replay(actions, index=index)
                core.checked_path(self.task, self.task["root_context"], actions, path)
                score = self.cpu.score_candidate(index, path[-1]["latent_history"][-1], self.task["goal"])
                scored.append(dict(index=index, score=score, path=path))
            winner = min(range(len(scored)), key=lambda i: (scored[i]["score"], i))
            actions = bank["raw15"][winner]
            replays = [self.replay(actions, index=40+i, segmented=segmented)
                       for i, segmented in enumerate((False, True))]
            core.require(replays[0] == replays[1] == scored[winner]["path"], "exact selected replay")
            result = dict(bank=bank, candidates=scored, winner=winner, replays=replays)
            self.event(dict(kind="plan_return", result=result))
            return result
        except BaseException:
            self.driver.failed = True
            raise

    def finish(self):
        core.require(not self.closed, "runner already closed")
        try:
            if not self.driver.failed:
                try:
                    self._check()
                except BaseException:
                    self.driver.failed = True
                    raise
        finally:
            self.closed = True
            try:
                result = self.cpu.finish()
                self.io.emit("terminal", dict(ledger_sha256=result["sha256"], status=result["status"],
                                              model_after=signal.model_hash(self.model)))
            finally:
                self.io.close()
        result["native_io"] = str(self.io.path)
        result["native_io_sha256"] = hashlib.sha256(self.io.path.read_bytes()).hexdigest()
        result.update(coverage_verified=False, budget_fairness=False, P1_admission=False, P2_admission=False)
        return result


def verify(result, contract_sha):
    """Check externally anchored chains and cross-link, not completeness/fairness."""
    core.require(spec.hash_string(result["native_io_sha256"]), "external native digest")
    path = Path(result["native_io"])
    core.require(path.is_file() and not path.is_symlink() and path.stat().st_size <= IO_LIMIT, "bounded native journal")
    raw = path.read_bytes()
    core.require(hashlib.sha256(raw).hexdigest() == result["native_io_sha256"], "native journal SHA mismatch")
    rows = durable.read_chain(raw)
    core.require(len(rows) >= 2 and rows[0]["kind"] == "header" and rows[-1]["kind"] == "terminal"
                 and all(row["kind"] == "native" for row in rows[1:-1]), "native journal lifecycle")
    h, end = rows[0]["payload"], rows[-1]["payload"]
    core.require(h["schema"] == NAME and h["contract_sha256"] == contract_sha
                 and end["ledger_sha256"] == result["sha256"], "native/ledger binding")
    receipt = durable.verify(result["journal"], result["sha256"], contract_sha)
    core.require(receipt["status"] == end["status"], "terminal status binding")
    ledger_raw = Path(result["journal"]).read_bytes()
    core.require(hashlib.sha256(ledger_raw).hexdigest() == result["sha256"], "ledger changed during verification")
    ledger_rows = durable.read_chain(ledger_raw)
    ledger_header = ledger_rows[0]["payload"]
    task = core.validate_task(h["task"])
    core.require(core.digest(task) == h["task_sha256"] and h["arm"] == ledger_header["arm"]
                 and h["run_id"] == ledger_header["run_id"]
                 and all(ledger_header["contract"]["identity"][a] == task["binding"][b]
                         for a, b in adapter.IDENTITY_MAP.items())
                 and h["model_before"]["original_state_dict"] == task["binding"]["model_sha256"],
                 "task/model/ledger identity binding")
    core.require(h["source_sha256"] == adapter.sources()
                 and h["runner_sha256"] == spec.sha(Path(__file__).read_bytes()), "source drift")
    return dict(status=receipt["status"], native_events=len(rows)-2, resources=receipt,
                model_unchanged=h["model_before"] == end["model_after"],
                coverage_verified=False, budget_fairness=False, P1_admission=False, P2_admission=False)
