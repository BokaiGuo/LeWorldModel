"""CPU-only source inventory for the replay-reserved P2 exploration.

This inventory is not a proof of complete transitive runtime closure or a
temporal preregistration. Verification does not authorize CUDA or physics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import p2_composition_predict_frontier_cuda_20261027_v2 as frontier
from experiments.spectral_reachability import p2_replay_reserved_runtime_20261101 as runtime
from experiments.spectral_reachability import codex_p2_host_physical_bridge_20261030_v1 as physical
from experiments.spectral_reachability import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as lock


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(prepared, expected):
    directory, protocol, state = frontier.load_prepared(prepared, expected)
    policy = Path(protocol["context"]["policy"]).resolve()
    files = set(map(Path, protocol["source_paths"]))
    files.update((Path(__file__), Path(runtime.__file__), Path(runtime.base.__file__),
                  Path(physical.__file__), Path(lock.__file__),
                  frontier.PROJECT.parent / "docs/P2_REPLAY_RESERVED_EXPLORATION_20261101.md",
                  directory / "protocol.json", directory / "prepared.json",
                  Path(protocol["context"]["compilation"]),
                  policy / "config.json", policy / "weights.pt"))
    records = {}
    for original in sorted(files):
        path = original.resolve(strict=True)
        if not path.is_file():
            raise ValueError("regular source file required: " + str(path))
        records[str(path)] = sha(path)
    return {"schema": "p2_replay_source_inventory_v1",
            "prepared": str(directory), "prepared_sha256": expected,
            "protocol_sha256": frontier.digest(protocol),
            "roots": [{"root_index": r["root_index"], "root_sha256": r["root_sha256"]}
                      for r in protocol["context"]["cases"]],
            "files": records, "transitive_closure_verified": False,
            "temporal_preregistration_verified": False, "P2_admission": False}


def verify(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError("external inventory SHA mismatch")
    value = json.loads(path.read_text())
    current = inventory(value["prepared"], value["prepared_sha256"])
    if value != current:
        raise ValueError("inventory or source drift")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--prepared", required=True)
    create.add_argument("--prepared-sha256", required=True)
    create.add_argument("--output", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--inventory", required=True)
    check.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "create":
        value = inventory(args.prepared, args.prepared_sha256)
        out = Path(args.output)
        if not out.is_absolute() or out.parent.resolve() != out.parent:
            raise ValueError("absolute canonical output required")
        with out.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"inventory": str(out), "sha256": sha(out), "P2_admission": False}))
    else:
        value = verify(args.inventory, args.sha256)
        print(json.dumps({"status": "inventory_verified", "files": len(value["files"]),
                          "P2_admission": False}))


if __name__ == "__main__":
    main()
