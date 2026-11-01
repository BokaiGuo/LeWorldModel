"""Strict continuation for a partially executed H10 shard directory.

This entry never rewrites existing anchors. It validates the fixed prepare and
existing shard reports, executes only missing anchors, and writes the aggregate
run report only after all 24 shards are present.
"""
import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import codex_p1_three_prefix_h10_runner_20261030 as runner
from experiments.spectral_reachability import codex_p1_three_prefix_h10_saved_20261030 as saved
from experiments.spectral_reachability import p1_matched_suffix_scale_20261030 as old

ROOT = runner.ROOT
PREPARED = ROOT / "results/codex_p1_three_prefix_h10_runner_20261030_prepare_CODEX01.json"
PREPARED_SHA = "ee136f0a7537c8328aa4ec7417e6d4d0e0c1324c50dc5f59bc4e0c871f020f30"
DEFAULT_OUTPUT = ROOT / "results/codex_p1_three_prefix_h10_runner_20261030_run_MAIN01"
require = runner.require


def _factory():
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
    return factory


def _write_new(path, value):
    path = Path(path)
    require(path.is_absolute() and not path.exists() and not path.is_symlink(), "exclusive output")
    data = runner.native.raw(value)
    with path.open("xb") as stream:
        require(stream.write(data) == len(data), "complete output")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    runner.base.paired.fsync_dir(path.parent)
    return runner.runner.physical.sha(path)


def _same_aggregate(existing_report, expected_report):
    """Compare immutable run facts, not per-invocation resume provenance."""
    keys = ("schema", "status", "shards", "run_binding_sha256", "fixed_branches",
            "independent_confirmation", "P1_admission")
    return all(existing_report.get(key) == expected_report[key] for key in keys)


def resume(output):
    output = Path(output)
    require(output == output.resolve() and output.parent == ROOT / "results"
            and output.is_dir() and not output.is_symlink(), "existing run directory")
    prepared = runner.verify_prepare(PREPARED, PREPARED_SHA)
    binding_path = output / "run_binding.json"
    require(binding_path.is_file() and not binding_path.is_symlink(), "run binding")
    binding = runner.native.read(binding_path, cap=16 * 1024**2)
    expected_bindings = dict(prepared["bindings"], **{str(PREPARED): PREPARED_SHA})
    require(binding == dict(prepared=str(PREPARED), prepare_sha256=PREPARED_SHA,
                            bindings=expected_bindings, fixed_branches=1008,
                            independent_confirmation=False, P1_admission=False), "fixed binding")
    for path, digest in binding["bindings"].items():
        require(runner.runner.physical.sha(path) == digest, f"source drift: {path}")

    source_report = runner.native.read(old.prep.DIRECTORY / "report.json", cap=64 * 1024**2)
    _, data = runner.runner.physical.verify(
        source_report["execution_binding"]["prepared"], old.prep.PREP_SHA)
    runner.native.constructor_contract()

    existing = {}
    for anchor in prepared["original"]["anchors"]:
        idx = anchor["index"]
        path = output / f"anchor_{idx:02d}"
        report_path = path / "report.json"
        if report_path.exists():
            require(path.is_dir() and not path.is_symlink(), "existing shard directory")
            digest = runner.runner.physical.sha(report_path)
            saved.check_shard(path, anchor, prepared["original"]["probe_banks"][anchor["logical_pair"]], digest)
            existing[idx] = dict(anchor=idx, report_sha256=digest, status="recorded_not_consumed")

    require(set(existing).issubset(set(range(24))), "anchor roster")
    factory = _factory()
    created = []
    for anchor in prepared["original"]["anchors"]:
        idx = anchor["index"]
        if idx in existing:
            continue
        path = output / f"anchor_{idx:02d}"
        require(not path.exists() and not path.is_symlink(), "missing anchor namespace")
        bank = prepared["original"]["probe_banks"][anchor["logical_pair"]]
        report = runner.execute_shard(anchor, bank, data, path, factory, binding["bindings"])
        require(report["status"] == "recorded_not_consumed", "new shard status")
        created.append(idx)
        existing[idx] = dict(anchor=idx, report_sha256=runner.runner.physical.sha(path / "report.json"),
                             status=report["status"])

    require(sorted(existing) == list(range(24)), "all anchors")
    require(all(x["status"] == "recorded_not_consumed" for x in existing.values()), "all shard status")
    for path, digest in binding["bindings"].items():
        require(runner.runner.physical.sha(path) == digest, f"final source drift: {path}")
    result = dict(schema=runner.NAME + ".run.v1", status="recorded_not_consumed",
                  shards=[existing[i] for i in range(24)], run_binding_sha256=runner.runner.physical.sha(binding_path),
                  fixed_branches=1008, independent_confirmation=False, P1_admission=False,
                  resumed_existing=sorted(set(existing) - set(created)), created_anchors=created)
    report_path = output / "run_report.json"
    if report_path.exists():
        old_report = runner.native.read(report_path, cap=16 * 1024**2)
        require(_same_aggregate(old_report, result), "existing aggregate report differs")
        digest = runner.runner.physical.sha(report_path)
    else:
        digest = _write_new(report_path, result)
    return dict(status=result["status"], sha256=digest, created_anchors=created)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    print(json.dumps(resume(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
