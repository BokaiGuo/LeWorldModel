"""Apply existing same-chain E0 interventions to sealed v8 selections."""
import json
import os
import sys
from pathlib import Path

import numpy as np

from experiments import pc_v5_e0_prospective_interventions_dev_20261104_codex as original
from experiments import pc_v8_search_induced_nominal_e0_dev_20261104_codex as plans
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = plans.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V8_E0_EXPOSED_INTERVENTION_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v8_e0_exposed_intervention_dev_20261104_codex"
MANIFEST, RESULT, AUDIT = [PREFIX.with_name(PREFIX.name + suffix) for suffix in
                           ("_manifest.json", "_result.json", "_audit.json")]


def paths(root):
    return [PREFIX.with_name(PREFIX.name + f"_root{root:02d}" + suffix) for suffix in (".json", ".npz")]


def calculate(row):
    inner = row["result"]["v7"]
    adapted = {"root": row["root"], "task": row["task"],
               "result": {"selected": row["result"]["selected"],
                          "first_bank": inner["first_bank"], "prefix_scores": inner["prefix_scores"]}}
    old = original.local
    class LocalAdapter:
        @staticmethod
        def build(value):
            response = old.build(value)
            response["controls"]["selected"]["candidate_index"] = -1
            return response
    try:
        original.local = LocalAdapter
        return original.calculate(adapted)
    finally:
        original.local = old


def main(mode):
    sha = seal.digest
    if json.loads(plans.AUDIT.read_text())["result_sha256"] != sha(plans.RESULT):
        raise ValueError("source plan audit mismatch")
    plan_manifest, source = json.loads(plans.MANIFEST.read_text()), json.loads(plans.RESULT.read_text())
    seal.verify(plan_manifest["dependencies"])
    if source["manifest_sha256"] != sha(plans.MANIFEST) or len(source["rows"]) != 18:
        raise ValueError("source plan binding mismatch")
    if mode == "intervene":
        outputs = [MANIFEST, RESULT, AUDIT] + [p for i in range(18) for p in paths(i)]
        if any(p.exists() for p in outputs):
            raise FileExistsError("existing or partial intervention")
        deps = seal.capture([Path(__file__), PROTOCOL, plans.MANIFEST, plans.RESULT, plans.AUDIT,
                             Path(original.__file__), Path(original.local.__file__), Path(original.core.__file__),
                             Path(original.exact.__file__), Path(original.exact.wall.__file__),
                             *map(Path, plan_manifest["dependencies"]["files"])])
        plans.panel.source.write(MANIFEST, {"seal": deps, "source_sha256": sha(plans.RESULT),
                                            "scope": "posthoc exposed-root intervention"})
        index = []
        for row in source["rows"]:
            data, arrays = calculate(row)
            meta, archive = paths(row["root"])
            with archive.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            data["arrays_sha256"] = sha(archive)
            plans.panel.source.write(meta, data)
            index.append({"root": row["root"], "metadata_sha256": sha(meta), "arrays_sha256": sha(archive),
                          "nonprogress": data["response"]["nonprogress"],
                          "base_intersection": data["base_intersection"], "all_intersection": data["all_intersection"]})
            print(f"root={row['root']} intervened", flush=True)
        seal.verify(deps)
        plans.panel.source.write(RESULT, {"manifest_sha256": sha(MANIFEST), "rows": index})
    elif mode == "audit":
        if AUDIT.exists():
            raise FileExistsError(AUDIT)
        manifest, result = json.loads(MANIFEST.read_text()), json.loads(RESULT.read_text())
        seal.verify(manifest["seal"])
        if result["manifest_sha256"] != sha(MANIFEST) or len(result["rows"]) != 18:
            raise ValueError("intervention binding mismatch")
        for row, entry in zip(source["rows"], result["rows"], strict=True):
            root = row["root"]
            meta, archive = paths(root)
            saved = json.loads(meta.read_text())
            if (entry["root"] != root or entry["metadata_sha256"] != sha(meta)
                    or entry["arrays_sha256"] != sha(archive) or saved["arrays_sha256"] != sha(archive)):
                raise ValueError("root binding mismatch")
            regenerated, arrays = calculate(row)
            regenerated["arrays_sha256"] = sha(archive)
            if regenerated != saved:
                raise ValueError(f"result mismatch root {root}")
            with np.load(archive, allow_pickle=False) as data:
                for key, value in arrays.items():
                    np.testing.assert_array_equal(data[key], value)
            print(f"root={root} audited", flush=True)
        seal.verify(manifest["seal"])
        plans.panel.source.write(AUDIT, {"result_sha256": sha(RESULT), "verified_roots": 18,
                                        "scope": "same-code exact backend deterministic reconstruction"})
    else:
        raise ValueError("choose intervene or audit")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("choose intervene or audit")
    main(sys.argv[1])
