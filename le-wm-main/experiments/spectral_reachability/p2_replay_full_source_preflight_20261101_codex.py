"""CPU source-snapshot prerequisite for the exploratory P2 CUDA launcher.

Snapshots all project Python sources, plus the inherited inventory. This is
not a proof of external dependency, dynamic data, or temporal closure.
"""
import argparse
import json
import os
from pathlib import Path

from experiments.spectral_reachability import p2_replay_source_seal_20261101 as inherited

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "p2_replay_full_source_preflight_20261101_v1"
EXCLUDED = {".git", ".venv", "__pycache__", "results", "data", "wandb", "outputs"}


def sources(project=PROJECT):
    paths = set()
    for root, dirs, files in os.walk(project, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not d.startswith("."))
        for name in files:
            if name.endswith(".py"):
                path = Path(root) / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError("symlink or non-regular Python source: " + str(path))
                paths.add(str(path.resolve(strict=True)))
    return sorted(paths)


def snapshot(inventory, digest, project=PROJECT):
    value = inherited.verify(inventory, digest)
    files = sources(project)
    if not files:
        raise ValueError("empty project source inventory")
    return {
        "schema": SCHEMA,
        "inherited_inventory": str(Path(inventory).resolve(strict=True)),
        "inherited_inventory_sha256": digest,
        "prepared": value["prepared"],
        "prepared_sha256": value["prepared_sha256"],
        "project": str(project.resolve(strict=True)),
        "python_sources": {p: inherited.sha(p) for p in files},
        "external_dependencies_closed": False,
        "dynamic_data_closed": False,
        "temporal_preregistration_verified": False,
        "P2_admission": False,
    }


def verify(path, expected):
    if inherited.sha(path) != expected:
        raise ValueError("external full-source snapshot SHA mismatch")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("project") != str(PROJECT):
        raise ValueError("project identity mismatch")
    actual = snapshot(value["inherited_inventory"], value["inherited_inventory_sha256"])
    if value != actual:
        raise ValueError("full-source snapshot drift")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--inventory", required=True)
    create.add_argument("--inventory-sha256", required=True)
    create.add_argument("--output", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--snapshot", required=True)
    check.add_argument("--snapshot-sha256", required=True)
    run = sub.add_parser("execute")
    run.add_argument("--snapshot", required=True)
    run.add_argument("--snapshot-sha256", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device-uuid", required=True)
    run.add_argument("--operator-coordinated", action="store_true")
    args = parser.parse_args()
    if args.command == "create":
        value = snapshot(args.inventory, args.inventory_sha256)
        out = Path(args.output)
        if not out.is_absolute() or out.parent.resolve() != out.parent:
            raise ValueError("absolute canonical output required")
        with out.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"snapshot": str(out), "sha256": inherited.sha(out), "P2_admission": False}))
    elif args.command == "verify":
        value = verify(args.snapshot, args.snapshot_sha256)
        print(json.dumps({"status": "verified", "python_sources": len(value["python_sources"]), "P2_admission": False}))
    else:
        value = verify(args.snapshot, args.snapshot_sha256)
        from experiments.spectral_reachability import p2_replay_exclusive_cuda_20261101 as launcher
        original_verify = inherited.verify

        def guarded_verify(path, digest):
            # Avoid recursion: snapshot uses inherited.verify too.
            inherited.verify = original_verify
            try:
                current = verify(args.snapshot, args.snapshot_sha256)
                if (str(Path(path).resolve()) != current["inherited_inventory"]
                        or digest != current["inherited_inventory_sha256"]):
                    raise ValueError("launcher inventory binding mismatch")
                return original_verify(path, digest)
            finally:
                inherited.verify = guarded_verify

        inherited.verify = guarded_verify
        try:
            launcher.execute(inventory=value["inherited_inventory"],
                             inventory_sha256=value["inherited_inventory_sha256"],
                             output=args.output, device_uuid=args.device_uuid,
                             coordinated=args.operator_coordinated)
        finally:
            inherited.verify = original_verify


if __name__ == "__main__":
    main()
