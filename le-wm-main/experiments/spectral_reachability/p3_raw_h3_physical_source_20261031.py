"""Reconstruct fixed H3 batches from the independently pinned input manifest."""
import json
from pathlib import Path

from . import p3_raw_h3_physical_batch_20261031 as batch
from . import p3_raw_h3_physical_saved_20261031 as saved

prepared, host, require = batch.prepared, batch.host, batch.require
ROOT = prepared.ROOT
MANIFEST = ROOT / "results/p3_raw_h3_physical_prepare_20261031_RUN01.json"
MANIFEST_SHA = "3f2b8c34defb942242d2f66d1b46f83251125f07b871440b2ec6d8acde1189fd"


def reconstruct(index):
    """Bind a six-row slice to the original roster, model result, and physical roots."""
    require(type(index) is int and 0 <= index < 12, "batch index")
    require(host.sha(MANIFEST) == MANIFEST_SHA, "external manifest hash")
    manifest = json.loads(MANIFEST.read_bytes())
    require(manifest["schema"] == prepared.NAME
            and manifest["prefix_steps"] == 15 and manifest["suffix_steps"] == 5
            and manifest["physical_execution"] is False
            and len(manifest["plans"]) == 72, "manifest contract")
    _, bindings, _, data = prepared.h1.prepare()
    physical, _ = host.inputs()
    result = json.loads(prepared.h1.NONLINEAR.read_bytes())
    plans = prepared.build_plans(prepared.h1.source.load_roster(), result, physical)
    for path in (Path(prepared.__file__).resolve(), prepared.PROTOCOL.resolve()):
        bindings[str(path)] = host.sha(path)
    require(manifest["plans"] == plans and manifest["bindings"] == bindings
            and all(host.sha(path) == digest for path, digest in bindings.items()),
            "manifest source reconstruction")
    for path in (MANIFEST, Path(__file__), Path(batch.__file__), Path(batch.branch.__file__),
                 Path(saved.__file__), Path(__file__).with_name("p3_raw_h3_physical_run_20261031.py")):
        path = path.resolve()
        digest = host.sha(path)
        require(str(path) not in bindings or bindings[str(path)] == digest, "binding conflict")
        bindings[str(path)] = digest
    p = {"schema": batch.NAME + ".prepare", "contract": batch.contract(),
         "plans": plans[6 * index:6 * index + 6], "bindings": bindings,
         "manifest_sha256": MANIFEST_SHA, "batch_index": index,
         "physical_execution": False, "scientific_pass": False}
    batch.validate_plans(p["plans"])
    require(host.sha(MANIFEST) == MANIFEST_SHA
            and all(host.sha(path) == digest for path, digest in bindings.items()), "source drift")
    return p, data


def consume(directory, report_sha, prepare_sha, index):
    """Consume against independently reconstructed inputs, never report-provided plans."""
    expected, data = reconstruct(index)
    return saved.consume(directory, report_sha, prepare_sha, expected, data["references"])
