"""Read-only explicit-file preflight; never loads a model or launches work.

This is a point-in-time metadata/readability check, not a reservation or lock.
No GPU probing, network, executor, dataset traversal or weights reads occur.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat

SCOPE = ("Read-only host input checks only. Does NOT prove GPU availability, "
         "authorization, model correctness, adapter compatibility or experiment admission. "
         "No output/disk reservation: recheck at launch; paths may change after inspection.")


def check_readiness(*, policy, adapter, output, sources, minbytes):
    """Return readiness and issues, without writing files or executing inputs.

    policy must contain regular nonempty weights.pt and config.json files.
    Weights are stat-only (including permission metadata), never opened/hashed.
    Config and adapter must be strict, nonempty JSON objects. Sources are
    explicit regular nonempty readable files; at most one byte is read each.
    The output's immediate parent must already exist; no mkdir/probe writes.
    minbytes is a caller-declared nonnegative integer, not an inferred estimate.
    """
    issues = []

    def path_of(value, label):
        if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
            issues.append(f"{label}: explicit nonempty path required")
            return None
        try:
            return Path(value)
        except (TypeError, ValueError):
            issues.append(f"{label}: invalid path")
            return None

    def regular(path, label):
        if path is None:
            return False
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                issues.append(f"{label}: nonempty regular file required (no symlinks)")
                return False
            if not os.access(path, os.R_OK):
                issues.append(f"{label}: unreadable file")
                return False
            return True
        except (OSError, ValueError) as exc:
            issues.append(f"{label}: cannot stat file ({type(exc).__name__})")
            return False

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def json_file(path, label):
        if not regular(path, label):
            return
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_object)
            if not isinstance(value, dict) or not value:
                raise ValueError("nonempty JSON object required")
        except (OSError, ValueError, RecursionError) as exc:
            issues.append(f"{label}: invalid/unreadable JSON object ({type(exc).__name__})")

    policy_path = path_of(policy, "policy")
    if policy_path is not None:
        try:
            if not stat.S_ISDIR(policy_path.lstat().st_mode):
                issues.append("policy: ordinary directory required (no symlink)")
            else:
                regular(policy_path/"weights.pt", "weights")
                json_file(policy_path/"config.json", "config")
        except (OSError, ValueError) as exc:
            issues.append(f"policy: inaccessible directory ({type(exc).__name__})")
    json_file(path_of(adapter, "adapter"), "adapter")
    if not isinstance(sources, (list, tuple)) or not sources:
        issues.append("sources: explicit nonempty list/tuple required")
    else:
        for index, source in enumerate(sources):
            label = f"source[{index}]"
            path = path_of(source, label)
            if regular(path, label):
                try:
                    with path.open("rb") as stream:
                        if not stream.read(1):
                            issues.append(f"{label}: file became empty")
                except OSError as exc:
                    issues.append(f"{label}: cannot read file ({type(exc).__name__})")
    valid_minbytes = type(minbytes) is int and minbytes >= 0
    if not valid_minbytes:
        issues.append("minbytes: explicit nonnegative integer required")
    output_path = path_of(output, "output")
    if output_path is not None:
        try:
            # lstat detects dangling symlinks as occupied output paths too.
            try:
                output_path.lstat()
            except FileNotFoundError:
                pass
            else:
                issues.append("output: path already exists")
            parent = output_path.parent
            if not parent.is_dir():
                issues.append("output: immediate parent directory must already exist")
            elif valid_minbytes:
                free = shutil.disk_usage(parent).free
                if type(free) is not int or free < 0:
                    issues.append("disk: invalid free-space measurement")
                elif free < minbytes:
                    issues.append(f"disk: free bytes {free} below caller minimum {minbytes}")
        except (OSError, ValueError) as exc:
            issues.append(f"output/disk: inaccessible ({type(exc).__name__})")
    return {"ready_for_host_cuda_request": not issues, "issues": issues, "scope": SCOPE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", dest="sources", action="append", required=True)
    parser.add_argument("--minbytes", type=int, required=True)
    result = check_readiness(**vars(parser.parse_args(argv)))
    print(json.dumps(result, allow_nan=False))
    return 0 if result["ready_for_host_cuda_request"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
