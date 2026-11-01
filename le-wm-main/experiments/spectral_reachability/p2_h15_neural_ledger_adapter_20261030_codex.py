"""Prospective native neural ledger interface; does not construct or run a model.

The facade exposes exactly the public fields used by NativeSignal and delegates
every call to the exclusive durable driver. It is not accepted by sealed host
guards and is not evidence that a model's hooks cover every operation.
"""
import copy
from pathlib import Path

from experiments.spectral_reachability import agent_p3_local_signal_cuda_20261028_v1 as native
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as tree
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as durable
from experiments.spectral_reachability import p2_h15_cpu_metered_backend_20261030_codex as cpu
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec

NAME = "p2_h15_neural_ledger_adapter_20261030_codex"
IDENTITY_MAP = {"root_sha256": "root_sha256", "task_sha256": "task_sha256",
                "model_sha256": "model_sha256", "action_domain_sha256": "action_domain_sha256"}


def file_hash(module):
    return spec.sha(Path(module.__file__).read_bytes())


def sources():
    return {"adapter_sha256": spec.sha(Path(__file__).read_bytes()),
            "native_signal_sha256": file_hash(native),
            "native_tree_sha256": file_hash(tree),
            "cpu_backend_sha256": file_hash(cpu)}


class NativeLedger:
    """Facade for actual native .scope/.events/.call ABI, with fixed task."""
    def __init__(self, driver, task):
        core.require(type(driver) is durable.Driver and not driver.closed and not driver.failed,
                     "active exclusive driver")
        self.task = core.validate_task(task)
        binding = self.task["binding"]
        identity = driver.ledger.contract["identity"]
        core.require(all(identity[a] == binding[b] for a, b in IDENTITY_MAP.items()),
                     "contract root/task/model/action domain mismatch")
        self.driver = driver
        self.source_hashes = sources()
        self.task_hash = core.digest(self.task)
        self._scope = None

    def _check(self):
        core.require(not self.driver.closed and not self.driver.failed, "driver closed or fail-stopped")
        core.require(sources() == self.source_hashes and core.digest(self.task) == self.task_hash,
                     "native source or task drift")

    @property
    def scope(self):
        return copy.deepcopy(self._scope)

    @scope.setter
    def scope(self, value):
        self._check()
        core.require(type(value) is dict and bool(value) and "neural_adapter" not in value,
                     "explicit nonempty native scope")
        self._scope = copy.deepcopy(value)
        self._scope["neural_adapter"] = dict(self.source_hashes, task_sha256=self.task_hash)

    @property
    def events(self):
        # NativeSignal uses len(events) and the last operation index for IO.
        return self.driver.ledger.receipt()["events"]

    @property
    def attempted(self):
        return self.driver.ledger.attempted

    @property
    def completed(self):
        return self.driver.ledger.completed

    @property
    def caps(self):
        return copy.deepcopy(self.driver.ledger.receipt()["caps"])

    def receipt(self):
        return self.driver.ledger.receipt()

    def call(self, resource, fn, *, shape=None, directions=None):
        try:
            self._check()
            core.require(self._scope is not None, "native operation scope required")
            return self.driver.call(self._scope, resource, fn, shape=shape, directions=directions)
        except BaseException:
            self.driver.failed = True
            raise


def create(directory, contract, contract_sha, original_caps, arm, run_id, task):
    """Create one shared future CPU+neural ledger; performs no model dispatch."""
    # Validate before creating a directory or any durable artifact.
    validated = core.validate_task(task)
    spec.validate_contract(contract)
    core.require(spec.sha(spec.canonical(contract)) == contract_sha, "contract hash")
    core.require(all(contract["identity"][a] == validated["binding"][b] for a, b in IDENTITY_MAP.items()),
                 "contract/task identity mismatch")
    d = cpu.make_driver(directory, contract, contract_sha, original_caps, arm, run_id)
    try:
        # CPU constructor requires a fresh ledger, so bind it before neural work.
        cpu_backend = cpu.Backend(d)
        neural_ledger = NativeLedger(d, validated)
    except BaseException:
        d.failed = True
        try:
            d.finish()
        except BaseException:
            pass
        raise
    return cpu_backend, neural_ledger, d
