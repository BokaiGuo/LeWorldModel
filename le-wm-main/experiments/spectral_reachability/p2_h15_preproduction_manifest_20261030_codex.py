"""Offline H15 source inventory and candidate manifest; never dispatches work.

Hashes bind bytes, not authorship or pre-registration time. The historical
report is an observation, not proof of the future runtime or of its complete
Python/native dependency closure.
"""
import copy
from pathlib import Path

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_h15_host_attestation_20261029_v1 as host
from experiments.spectral_reachability import p2_h15_budgeted_manifest_consumer_20261030_codex as joint
from experiments.spectral_reachability import p2_h15_production_manifest_20261030_codex as production
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec

NAME = "p2_h15_preproduction_manifest_20261030_codex"
REPORT = host.bridge.source.prep.ROOT / "results/p2_h15_controlled_host_20261029_v1_cuda_MAIN01/report.json"
# Hash observed from the existing saved report, not an independent timestamp.
REPORT_SHA = "188fe5552b269db22739a77042a318d1f60d1c1c69cddcf4c5aead20501fe6ce"


def checked_inventory(files):
    """Check each listed regular file, rejecting symlinked path components."""
    core.require(type(files) is dict and bool(files), "nonempty inventory")
    for name, digest in files.items():
        core.hash_value(digest)
        p = Path(name)
        core.require(p.is_absolute() and p.resolve() == p and p.is_file()
                     and not p.is_symlink() and host.bridge.sha(p) == digest,
                     "inventory file mismatch: " + name)
    return core.digest(files)


def preflight():
    """Rebuild two original tasks and rehash pinned historical source files."""
    mapped = host.bridge.build_bridge()
    files = host.historical_inventory()
    inventory_digest = checked_inventory(files)
    report = host.bridge.source.prep.read(REPORT, REPORT_SHA)
    a, b = report["attestation"], report["attestation_before"]
    model = b["model"]
    core.require(report["schema"] == "p2_h15_controlled_host_20261029_v1.report"
                 and report["status"] == "recorded_not_admitted"
                 and report["scientific_admission"] is False
                 and report["physical_release"] is False and report["errors"] == []
                 and report["root_status"] == ["sealed", "sealed"]
                 and b["fixture_only"] is False and a["fixture_only"] is False
                 and a["host_authorized"] is False and a["physical_release"] is False
                 and model == a["model_before"] == a["model_after"]
                 and set(model) == {"original_state_dict", "all_buffers"}
                 and all(spec.hash_string(x) for x in model.values())
                 and model["original_state_dict"] == mapped["roots"][0]["task"]["binding"]["model_sha256"]
                 and b["precision"] == a["precision_before"] == a["precision_after"] == host.native.precision_contract()
                 and b["sources"] == a["sources_before"] == a["sources_after"] == inventory_digest,
                 "historical report binding")
    return dict(schema=NAME + ".preflight.v1", historical_report_sha256=REPORT_SHA,
                bridge_sha256=core.digest(mapped), inventory_sha256=inventory_digest,
                inventory_file_count=len(files), builder_sha256=spec.sha(Path(__file__).read_bytes()),
                model_hashes=copy.deepcopy(model),
                precision=copy.deepcopy(b["precision"]),
                tasks=[dict(root_slot=r["root_slot"], task_sha256=core.digest(r["task"]),
                            physical_root_sha256=r["physical_root_sha256"],
                            checkpoint_file_sha256=r["checkpoint_file_sha256"])
                       for r in mapped["roots"]],
                class_graph_available=False, runtime_dependency_closure_verified=False,
                ready_to_execute=False, physical_release=False, P1_admission=False, P2_admission=False)


def candidate(root, contract, contract_sha256, task_sha256, class_evidence, class_evidence_sha256):
    """Construct the existing joint manifest from pinned inputs, still not authorized.

    class_evidence is a separately anchored capture declaration with keys
    model_hashes and classes. Its digest does not authenticate its author or
    establish completeness; a future real runtime snapshot must compare it.
    """
    info = preflight()
    core.require(type(root) is int and root in (0, 1), "root")
    mapped = host.bridge.build_bridge()
    task = mapped["roots"][root]["task"]
    core.require(spec.hash_string(task_sha256) and core.digest(task) == task_sha256,
                 "external task digest")
    core.require(spec.hash_string(contract_sha256) and core.digest(contract) == contract_sha256,
                 "external contract digest")
    core.require(spec.hash_string(class_evidence_sha256)
                 and core.digest(class_evidence) == class_evidence_sha256,
                 "external class evidence digest")
    core.keys(class_evidence, ("model_hashes", "classes"))
    core.require(class_evidence["model_hashes"] == info["model_hashes"], "historical model mismatch")
    rows = class_evidence["classes"]
    core.require(type(rows) is list and bool(rows), "missing class graph")
    files = host.historical_inventory()
    for row in rows:
        core.keys(row, ("name", "class_name", "path", "sha256"))
        core.require(type(row["path"]) is str and files.get(row["path"]) == row["sha256"],
                     "class source outside historical inventory")
    base = production.manifest(contract, task, info["model_hashes"], rows)
    base_sha = core.digest(base)
    envelope = joint.manifest(base, base_sha, contract, task)
    return dict(schema=NAME + ".candidate.v1", preflight=info, root_slot=root,
                class_evidence_sha256=class_evidence_sha256,
                class_evidence_authenticity_verified=False, class_graph_completeness_verified=False,
                base_sha256=base_sha, joint_manifest=envelope,
                joint_manifest_sha256=core.digest(envelope),
                **production.production_gate(base))


def verify_candidate(value, expected_sha256, contract, contract_sha256, task_sha256,
                     class_evidence, class_evidence_sha256):
    core.require(spec.hash_string(expected_sha256) and core.digest(value) == expected_sha256,
                 "external candidate digest")
    core.require(value == candidate(value["root_slot"], contract, contract_sha256, task_sha256,
                                    class_evidence, class_evidence_sha256),
                 "candidate reconstruction mismatch")
    return copy.deepcopy(value)
