"""Saved-only joint consumer for H15 production snapshots and budgeted plan.

The independently anchored envelope adds the bridge-aware sources to the
existing production manifest. It does not attest runtime chronology, complete
dependency coverage, GPU execution, or pre-registration timing.
"""
import copy
from pathlib import Path

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_h15_budgeted_plan_20261030_codex as plan
from experiments.spectral_reachability import p2_h15_budgeted_saved_consumer_20261030_codex as budgeted
from experiments.spectral_reachability import p2_h15_observable_budget_bridge_20261030_codex as bridge
from experiments.spectral_reachability import p2_h15_production_manifest_20261030_codex as production
from experiments.spectral_reachability import p2_h15_strict_saved_consumer_20261030_codex as saved
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec

NAME = "p2_h15_budgeted_manifest_consumer_20261030_codex"


def extension_sources():
    return dict(plan_sha256=plan.source_hash(), bridge_sha256=bridge.source_hash(),
                budget_consumer_sha256=spec.sha(Path(budgeted.__file__).read_bytes()),
                consumer_sha256=spec.sha(Path(__file__).read_bytes()))


def manifest(base, base_sha256, contract, task):
    """Create bytes to anchor externally; no model loading or device access."""
    value = production.validate_manifest(base, base_sha256, contract, task)
    return dict(schema=NAME + ".manifest.v1", base=value, base_sha256=base_sha256,
                extension_sources=extension_sources())


def validate_manifest(value, expected_sha256, contract, task):
    core.require(spec.hash_string(expected_sha256) and core.digest(value) == expected_sha256,
                 "external joint manifest digest")
    core.keys(value, ("schema", "base", "base_sha256", "extension_sources"))
    core.require(value["schema"] == NAME + ".manifest.v1"
                 and value == manifest(value["base"], value["base_sha256"], contract, task),
                 "joint manifest source/contract/task drift")
    return copy.deepcopy(value)


def consume(directory, budget_directory, *, manifest_value, manifest_sha256, contract,
            expected_task, expected_task_sha256, attestation_sha256, native_sha256,
            ledger_sha256, budget_sha256, plan_sha256, arm, run_id):
    """Jointly validate externally anchored snapshots and bridge-aware saved plan."""
    task = core.validate_task(expected_task)
    core.require(spec.hash_string(expected_task_sha256) and core.digest(task) == expected_task_sha256,
                 "external task snapshot digest")
    value = validate_manifest(manifest_value, manifest_sha256, contract, task)
    base, ext = value["base"], value["extension_sources"]
    directory = Path(directory)
    core.require(directory.is_absolute() and directory.is_dir() and not directory.is_symlink(), "run directory")
    attestation = saved._read(directory / "production_attestation.jsonl", attestation_sha256, production.LIMIT)
    core.require(len(attestation) == 2 and [x["kind"] for x in attestation] == ["before", "after"],
                 "attestation lifecycle")
    before, after = (x["payload"] for x in attestation)
    core.keys(before, ("schema", "manifest_sha256", "snapshot"))
    core.keys(after, ("snapshot", "native_sha256", "ledger_sha256", "status"))
    snapshot = dict(classes=base["classes"], model_hashes=base["model_hashes"], precision=base["precision"])
    core.require(before == dict(schema=production.NAME, manifest_sha256=value["base_sha256"], snapshot=snapshot)
                 and after == dict(snapshot=snapshot, native_sha256=native_sha256,
                                   ledger_sha256=ledger_sha256, status="complete"),
                 "source/precision attestation cross-binding")
    checked = budgeted.consume(
        directory, budget_directory, native_sha256=native_sha256, ledger_sha256=ledger_sha256,
        budget_sha256=budget_sha256, plan_sha256=plan_sha256, contract_sha256=base["contract_sha256"],
        expected_task=task, expected_task_sha256=expected_task_sha256,
        expected_model_hashes=base["model_hashes"], expected_manifest=base["named_sources"],
        expected_manifest_sha256=core.digest(base["named_sources"]),
        expected_plan_source_sha256=ext["plan_sha256"],
        expected_bridge_source_sha256=ext["bridge_sha256"], arm=arm, run_id=run_id)
    return dict(schema=NAME + ".consume.v1", joint_manifest_sha256=manifest_sha256,
                attestation_sha256=attestation_sha256, budgeted_saved=checked,
                model_class_source_snapshots_recorded=True, precision_snapshots_recorded=True,
                continuous_precision_verified=False, runtime_dependency_closure_verified=False,
                cross_journal_chronology_verified=False, preregistration_timing_verified=False,
                budget_fairness=False, **production.production_gate(base))
