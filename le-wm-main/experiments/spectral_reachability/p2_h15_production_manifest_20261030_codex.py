"""H15 opt-in source/precision/resource attestation; no device/model launcher.

Manifest commitments are caller-supplied and externally anchored. Class files
cover named module classes, not arbitrary Python/native dependencies or monkey
patching. The existing runner does not completely meter the resource vector;
production_gate intentionally remains closed.
"""
import copy
import inspect
from pathlib import Path

from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import agent_p3_local_signal_cuda_20261028_v1 as signal
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as native
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_h15_cuda_ledger_runner_20261030_codex as runner
from experiments.spectral_reachability import p2_h15_strict_saved_consumer_20261030_codex as strict
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec

NAME = "p2_h15_production_manifest_20261030_codex"
LIMIT = 4 * 1024 * 1024
AXIS_COVERAGE = {
    "predict_samples": "hook_guarded_partial_coverage",
    "vjp_directions": "hook_guarded_partial_coverage",
    "array_svd_calls": "hook_guarded_partial_coverage",
    "solve_calls": "hook_guarded_partial_coverage",
    "selection_evaluations": "metric_hook_guarded_partial_coverage",
    "replay_calls": "unmetered",
    "physical_steps": "outside_planning_unmetered",
    "flops": "unmetered",
    "planning_wall_ns": "incomplete_scope_unenforced",
    "peak_device_bytes": "unmetered",
}


def _hash_file(path):
    path = Path(path)
    core.require(path.is_absolute() and path.is_file() and not path.is_symlink()
                 and path.resolve() == path and path.stat().st_size <= 16 * 1024**2,
                 "bounded absolute class source")
    return spec.sha(path.read_bytes())


def capture_classes(model):
    """Read source for every named module class; reject per-instance forward overrides."""
    rows = []
    for name, module in model.named_modules(remove_duplicate=False):
        cls = type(module)
        path = Path(inspect.getfile(cls)).resolve()
        core.require("forward" not in vars(module), "instance forward override not allowed")
        rows.append(dict(name=name, class_name=cls.__module__ + "." + cls.__qualname__,
                         path=str(path), sha256=_hash_file(path)))
    core.require(rows and rows[0]["name"] == "" and len({x["name"] for x in rows}) == len(rows),
                 "nonempty unique named module graph")
    return rows


def budget_manifest(contract):
    spec.validate_contract(contract)
    core.require(set(AXIS_COVERAGE) == set(spec.AXES), "fixed full axis roster")
    return {axis: dict(contract["axes"][axis], instrumentation=AXIS_COVERAGE[axis])
            for axis in spec.AXES}


def manifest(contract, task, model_hashes, classes):
    """Construct data to pin externally before a run; never creates a model."""
    task = core.validate_task(task)
    spec.validate_contract(contract)
    core.require(all(contract["identity"][a] == task["binding"][b]
                     for a, b in runner.adapter.IDENTITY_MAP.items()), "contract task identity")
    core.require(type(model_hashes) is dict and set(model_hashes) == {"original_state_dict", "all_buffers"}
                 and all(spec.hash_string(x) for x in model_hashes.values())
                 and model_hashes["original_state_dict"] == task["binding"]["model_sha256"], "model hashes")
    core.require(type(classes) is list and bool(classes), "committed class graph")
    return dict(schema=NAME + ".manifest.v1", contract_sha256=core.digest(contract),
                task_sha256=core.digest(task), model_hashes=copy.deepcopy(model_hashes),
                classes=copy.deepcopy(classes), precision=native.precision_contract(),
                named_sources=strict.source_manifest(), attestor_sha256=_hash_file(Path(__file__).resolve()),
                axes=budget_manifest(contract))


def validate_manifest(value, expected_sha, contract, task):
    core.require(spec.hash_string(expected_sha) and core.digest(value) == expected_sha, "external manifest digest")
    core.keys(value, ("schema", "contract_sha256", "task_sha256", "model_hashes", "classes",
                      "precision", "named_sources", "attestor_sha256", "axes"))
    core.require(value["schema"] == NAME + ".manifest.v1"
                 and value == manifest(contract, task, value["model_hashes"], value["classes"]),
                 "manifest contract/task/precision/source/axes mismatch")
    for row in value["classes"]:
        core.keys(row, ("name", "class_name", "path", "sha256"))
        core.require(type(row["name"]) is str and type(row["class_name"]) is str
                     and bool(row["class_name"]) and spec.hash_string(row["sha256"])
                     and _hash_file(row["path"]) == row["sha256"], "class source commitment")
    core.require(value["classes"][0]["name"] == ""
                 and len({row["name"] for row in value["classes"]}) == len(value["classes"]),
                 "class graph names")
    return copy.deepcopy(value)


def production_gate(value):
    """Do not equate a manifest with complete instrumentation or authorization."""
    core.require(type(value) is dict and set(value.get("axes", {})) == set(AXIS_COVERAGE)
                 and all(value["axes"][k]["instrumentation"] == v for k, v in AXIS_COVERAGE.items()),
                 "fixed unaltered coverage declarations")
    return dict(ready_to_execute=False, missing_full_axis_coverage=list(AXIS_COVERAGE),
                physical_release=False, P1_admission=False, P2_admission=False)


def snapshot(model, torch, value):
    classes = capture_classes(model)
    core.require(classes == value["classes"], "actual model class graph/source mismatch")
    core.require(signal.model_hash(model) == value["model_hashes"], "model weights/buffers mismatch")
    parameters = list(model.parameters())
    core.require(bool(parameters) and all(not m.training for m in model.modules())
                 and all(not p.requires_grad and getattr(p.device, "type", None) == "cuda"
                         and p.dtype == torch.float32 for p in parameters)
                 and all(getattr(b.device, "type", None) == "cuda" for b in model.buffers()),
                 "frozen eval CUDA parameters float32 and CUDA buffers")
    precision = native.checked_precision(torch)
    core.require(precision == value["precision"], "precision mismatch")
    return dict(classes=classes, model_hashes=copy.deepcopy(value["model_hashes"]), precision=precision)


class Attestation:
    """Attach an exclusive sidecar before plan, and cross-link after finish.

    It records observations but does not call plan, load a model, set precision,
    launch CUDA, or authorize production. Exceptions fail-stop an active driver.
    """
    def __init__(self, active_runner, value, expected_sha, torch):
        self.runner, self.torch, self.manifest_sha = active_runner, torch, expected_sha
        try:
            core.require(type(active_runner) is runner.Runner and not active_runner.closed
                         and not active_runner.driver.closed and not active_runner.driver.failed,
                         "active exclusive runner")
            core.require(not active_runner.driver.ledger.receipt()["events"]
                         and not active_runner.cpu.compiled and active_runner.cpu.next_candidate == 0,
                         "attestation before any recorded planning")
            self.value = validate_manifest(value, expected_sha, active_runner.driver.ledger.contract, active_runner.task)
            active_runner._check()
            before = snapshot(active_runner.model, torch, self.value)
            path = active_runner.driver.directory / "production_attestation.jsonl"
            self.journal = transport.Journal(path, LIMIT)
            self.journal.emit("before", dict(schema=NAME, manifest_sha256=expected_sha, snapshot=before))
        except BaseException:
            if hasattr(active_runner, "driver"):
                active_runner.driver.failed = True
            if hasattr(self, "journal"):
                self.journal.close()
            raise

    def finish(self, result):
        try:
            core.require(self.runner.closed and self.runner.driver.closed, "runner must be finalized first")
            validate_manifest(self.value, self.manifest_sha, self.runner.driver.ledger.contract, self.runner.task)
            after = snapshot(self.runner.model, self.torch, self.value)
            directory = self.runner.driver.directory
            core.require(result["journal"] == str(directory / "events.jsonl")
                         and result["native_io"] == str(directory / "native_io.jsonl"),
                         "runner-local journal paths")
            check = runner.verify(result, self.value["contract_sha256"])
            core.require(check["status"] == "complete" and check["model_unchanged"], "complete runner required")
            self.journal.emit("after", dict(snapshot=after, native_sha256=result["native_io_sha256"],
                                             ledger_sha256=result["sha256"], status=check["status"]))
        finally:
            self.journal.close()
        data = self.journal.path.read_bytes()
        return dict(path=str(self.journal.path), sha256=spec.sha(data), manifest_sha256=self.manifest_sha,
                    **production_gate(self.value))


def consume(directory, *, attestation_sha256, manifest_value, manifest_sha256,
            expected_task, expected_task_sha256, contract, native_sha256, ledger_sha256, arm, run_id):
    """Verify anchored sidecar and strict saved plan; never infer full budget coverage."""
    value = validate_manifest(manifest_value, manifest_sha256, contract, expected_task)
    core.require(core.digest(expected_task) == expected_task_sha256, "external task snapshot digest")
    directory = Path(directory)
    core.require(directory.is_absolute() and directory.is_dir() and not directory.is_symlink(), "run directory")
    path = directory / "production_attestation.jsonl"
    rows = strict._read(path, attestation_sha256, LIMIT)
    core.require(len(rows) == 2 and [r["kind"] for r in rows] == ["before", "after"], "attestation lifecycle")
    before, after = (r["payload"] for r in rows)
    core.keys(before, ("schema", "manifest_sha256", "snapshot"))
    core.keys(after, ("snapshot", "native_sha256", "ledger_sha256", "status"))
    expected = dict(classes=value["classes"], model_hashes=value["model_hashes"], precision=value["precision"])
    core.require(before == dict(schema=NAME, manifest_sha256=manifest_sha256, snapshot=expected)
                 and after == dict(snapshot=expected, native_sha256=native_sha256,
                                   ledger_sha256=ledger_sha256, status="complete"), "sidecar cross-binding")
    check = strict.consume(directory, native_sha256=native_sha256, ledger_sha256=ledger_sha256,
                           contract_sha256=value["contract_sha256"], expected_task=expected_task,
                           expected_task_sha256=expected_task_sha256, expected_model_hashes=value["model_hashes"],
                           expected_manifest=value["named_sources"],
                           expected_manifest_sha256=core.digest(value["named_sources"]), arm=arm, run_id=run_id)
    return dict(schema=NAME + ".consume.v1", strict_saved=check,
                class_source_snapshots_recorded=True, precision_snapshots_recorded=True,
                continuous_precision_verified=False, runtime_dependency_closure_verified=False,
                attestation_sha256=attestation_sha256, manifest_sha256=manifest_sha256,
                **production_gate(value))
