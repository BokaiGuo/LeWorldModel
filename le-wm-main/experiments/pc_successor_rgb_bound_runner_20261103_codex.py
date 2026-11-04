"""Prospective, explicit-scope source receipts for successor RGB development."""
import hashlib
import json
from pathlib import Path

from experiments import pc_successor_rgb_execution_dev_20261103_codex as pilot
from experiments import pc_fixed_terminal_fair_feedback_dev_20261103_codex as runner
from experiments import pc_pointmaze_shadow_rgb_bridge_20261103_codex as bridge
from experiments import pointmaze_action_history_unroll_20261103 as unroll
from experiments import a0_pointmaze_native_render_probe_20261102 as scene
from experiments.pc_budget_allocation_20261103_codex import optimize

ROOT, h, old = runner.ROOT, runner.h, runner.old
PROTOCOL = ROOT.parent / "docs/PC_SUCCESSOR_RGB_SOURCE_BINDING_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_successor_rgb_bound_runner_20261103_codex"
PRE = PREFIX.with_name(PREFIX.name + "_source_before.json")
POST = PREFIX.with_name(PREFIX.name + "_source_after.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths():
    """Declared scope only; does not claim an exhaustive environment closure."""
    declared = [Path(__file__), PROTOCOL, pilot.PROTOCOL, Path(pilot.__file__), Path(pilot.fallback.__file__),
            Path(runner.__file__), Path(pilot.policy.__file__), Path(pilot.policy.generator.__file__),
            Path(optimize.__code__.co_filename), Path(unroll.__file__), Path(bridge.__file__),
            ROOT / "experiments/pc_pointmaze_shadow_rgb_bridge_worker_20261103_codex.py",
            ROOT / "experiments/pc_pointmaze_shadow_renderer_worker_v4_20261103_codex.py",
            Path(h.__file__), Path(old.__file__), Path(scene.__file__),
            Path(runner.previous.__file__), Path(runner.previous.prior.__file__),
            ROOT / "external/local_dinov2/hubconf.py",
            h.DEFAULT_CHECKPOINT, h.DEFAULT_CONFIG,
            h.CAMERA, h.OLD, h.OBSTACLE.with_suffix(".npz"), old.HISTORY, old.DECISIONS,
            *(h.DEFAULT_DATASET / name for name in ("states.pth", "actions.pth", "seq_lengths.pth"))]
    # These helpers are imported into h from separate modules and therefore
    # are not covered by hashing h.__file__ alone.
    declared += [Path(getattr(h, name).__code__.co_filename) for name in
                 ("encode_with_state", "estimate_state", "causal_state", "projected_cost",
                  "get_encoder", "load_model_and_dataset", "parameter_hash", "require_cuda")]
    # Several imported functions share their defining file.
    return list(dict.fromkeys(declared))


def snapshot(paths):
    paths = [Path(path).resolve(strict=True) for path in paths]
    if len(paths) != len(set(paths)) or any(not path.is_file() for path in paths):
        raise ValueError("duplicate or non-file dependency")
    return {str(path): digest(path) for path in paths}


def exclusive_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def finish(before, sources, outputs):
    if snapshot(sources) != before:
        raise ValueError("dependency changed during execution")
    return {"source_before_sha256": digest(PRE), "outputs": snapshot(outputs),
            "scope": "declared file set only; not complete environment or training provenance"}


def main():
    if PRE.exists() or POST.exists():
        raise FileExistsError("receipt exists")
    runner.PROTOCOL, runner.PREFIX, runner.ARMS, runner.decide = PROTOCOL, PREFIX, pilot.ARMS, pilot.decide
    outputs = [p for root in range(4) for arm in pilot.ARMS
               for group in runner.paths(root, arm) for p in (group if isinstance(group, list) else [group])]
    if any(path.exists() for path in outputs):
        raise FileExistsError("partial or complete outputs exist")
    sources = source_paths()
    before = snapshot(sources)
    exclusive_json(PRE, {"sources": before, "scope": "declared file set; not exhaustive"})
    runner.main()
    exclusive_json(POST, finish(before, sources, outputs))
    print(f"Bound {len(before)} declared sources and {len(outputs)} output files.", flush=True)


if __name__ == "__main__":
    main()
