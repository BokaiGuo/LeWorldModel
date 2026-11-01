"""Fixed known-root, matched multi-scale physical suffix exploration."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p1_backtrack_capability_prepare_20261030 as prep
from experiments.spectral_reachability import p1_backtrack_capability_runner_20261030 as runner
from experiments.spectral_reachability import p1_backtrack_capability_consume_20261030 as consume

NAME = "p1_matched_suffix_scale_20261030"
ROOT = runner.physical.ROOT
PROTOCOL = ROOT.parent / "docs/D2P_P1_MATCHED_SUFFIX_SCALE_PROTOCOL_20261030.md"
BASE = ROOT / "results/p1_backtrack_capability_prepare_20261030_MAIN01.json"
BASE_SHA = "21d117651b44ebcf2e89aa07d280bca47ab661d3b5bcc300323eb0d00e12294e"
require = runner.require


def make_probes(incumbent, selected):
    actions = [("zero", np.zeros((5, 2), dtype="<f4"))]
    for label, value in (("incumbent", incumbent), ("selected", selected)):
        a = np.asarray(value, dtype="<f4")
        require(a.shape == (5, 2) and np.isfinite(a).all() and (np.abs(a) <= 1).all(), "source actions")
        for scale in (.25, -.25, .5, -.5, .75, -.75, 1., -1.):
            actions.append((f"{label}_{scale:+.2f}", (a * np.float32(scale)).astype("<f4")))
    for name, direction in (("+x", (.5, 0)), ("-x", (-.5, 0)),
                            ("+y", (0, .5)), ("-y", (0, -.5))):
        actions.append((name, np.tile(np.asarray(direction, dtype="<f4"), (5, 1))))
    require(len(actions) == 21, "fixed suffix denominator")
    return [dict(index=i, name=name, raw5_float32_hex=a.tobytes().hex(),
                 raw5_sha256=hashlib.sha256(a.tobytes()).hexdigest())
            for i, (name, a) in enumerate(actions)]


def build():
    require(runner.physical.sha(BASE) == BASE_SHA, "base prepare SHA")
    base = runner.physical.native.read(BASE, cap=32 * 1024**2)
    require(base == prep.build(), "source and original preparation closure")
    report = runner.physical.native.read(prep.DIRECTORY / "report.json", cap=64 * 1024**2)
    p, _ = runner.physical.verify(report["execution_binding"]["prepared"], prep.PREP_SHA)
    plans = p["plans"]
    require(len(plans) == len(base["anchors"]) == 16, "all fixed anchors")
    probe_sets = []
    for i in range(0, 16, 2):
        a, b = plans[i:i + 2]
        require((a["role"], b["role"]) == ("incumbent", "model_selected") and
                all(a[k] == b[k] for k in ("root_slot", "condition", "arm")), "paired plan identity")
        def suffix(plan):
            raw = np.frombuffer(bytes.fromhex(plan["raw15_float32_hex"]), dtype="<f4").reshape(15, 2)
            return raw[5:10]
        bank = make_probes(suffix(a), suffix(b))
        probe_sets.extend((deepcopy(bank), deepcopy(bank)))
    bindings = deepcopy(base["bindings"])
    for path in (BASE, PROTOCOL, Path(__file__), Path(runner.__file__), Path(consume.__file__),
                 Path(runner.prior.__file__)):
        bindings[str(path.resolve())] = runner.physical.sha(path)
    for path, digest in bindings.items():
        require(runner.physical.sha(path) == digest, "source drift")
    return dict(schema=NAME + ".prepare", anchors=base["anchors"], probe_sets=probe_sets,
                base_prepare_sha256=BASE_SHA, bindings=bindings, fixed_anchors=16,
                fixed_branches=336, known_roots=2, training=False, P1_admission=False,
                independent_confirmation=False)


def prepare(output):
    path = Path(output)
    require(path.is_absolute() and path.parent == ROOT / "results" and path.name.startswith(NAME + "_prepare_")
            and not path.exists() and not path.is_symlink(), "exclusive prepare")
    value = runner.physical.native.raw(build())
    with path.open("xb") as stream:
        require(stream.write(value) == len(value), "prepare write")
        stream.flush()
        os.fsync(stream.fileno())
    runner.prior.base.paired.fsync_dir(path.parent)
    return runner.physical.sha(path)


def run(source, expected, output):
    source = Path(source)
    require(runner.physical.sha(source) == expected, "prepared SHA")
    p = runner.physical.native.read(source, cap=32 * 1024**2)
    require(p == build(), "prepare/source closure")
    report = runner.physical.native.read(prep.DIRECTORY / "report.json", cap=64 * 1024**2)
    _, data = runner.physical.verify(report["execution_binding"]["prepared"], prep.PREP_SHA)
    runner.physical.native.constructor_contract()
    directory = Path(output)
    require(directory.is_absolute() and directory.parent == ROOT / "results" and
            directory.name.startswith(NAME + "_run_") and not directory.exists() and not directory.is_symlink()
            and shutil.disk_usage(directory.parent).free >= runner.FLOOR + 16 * runner.CAP,
            "exclusive full-run reservation")
    directory.mkdir()
    runner.prior.base.paired.fsync_dir(directory.parent)
    bindings = deepcopy(p["bindings"])
    bindings[str(source.resolve())] = expected
    binding = runner.physical.native.raw(dict(prepared=str(source.resolve()), prepare_sha256=expected,
                                               source_code_bindings=bindings, fixed_branches=336))
    with (directory / "run_binding.json").open("xb") as stream:
        require(stream.write(binding) == len(binding), "binding write")
        stream.flush()
        os.fsync(stream.fileno())
    runner.prior.base.paired.fsync_dir(directory)
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
    print("Execute all 336 fixed known-root matched scale probes; no neural calls or P1 admission.", flush=True)
    for anchor, bank in zip(p["anchors"], p["probe_sets"], strict=True):
        shard = directory / f"anchor_{anchor['index']:02d}"
        report = runner.execute_shard(anchor, bank, data, shard, factory, bindings)
        print(json.dumps(dict(anchor=anchor["index"], status=report["status"],
                              report_sha256=runner.physical.sha(shard / "report.json"))), flush=True)
        if report["status"] != "recorded_not_independently_consumed":
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--prepared")
    ap.add_argument("--sha256")
    ap.add_argument("--authorize-physics", action="store_true")
    args = ap.parse_args()
    if args.command == "prepare":
        print(json.dumps(dict(path=args.output, sha256=prepare(args.output))))
        return 0
    require(args.authorize_physics, "physical execution authorization")
    return run(args.prepared, args.sha256, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
