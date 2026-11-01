"""Read-only CPU preflight for the specifically pinned prospective E1 run.

This does not reserve a GPU or authorize execution. Run immediately before
execution as well as at sealing time; changes after preflight remain possible.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "agent_hilewm_rgb_high_low_20261028_v1_prepare_MAIN01/manifest.json"
MANIFEST_SHA = "a3255d38e9cf718f9560c1b9e381abe38750edd9e91d3f04e5ad332cf414d7e9"
OUTPUT = RESULTS / "agent_hilewm_rgb_high_low_20261028_v1_cuda_CODEX_E1_DURABLE_EXCLUSIVE_01"
JOURNAL = RESULTS / "codex_e1_learned_durable_EXCLUSIVE_01.events.jsonl"
RUN_ID = "codex_e1_learned_durable_EXCLUSIVE_01"
PINS = {
    "codex_e1_learned_durable_runner_20261101_exclusive.py":
        "f9bb88cb410d73b37da9ee1653efb602d74e1122837dd9be9930c0ce50f38b25",
    "codex_e1_proposal_durable_events_20261101.py":
        "aab370f280c0a714eb8cfc65711cc4e04cbbc4a98fde55f87181363bd712fdf6",
}


def preflight(runner):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preflight requires CUDA_VISIBLE_DEVICES=''")
    if RESULTS.resolve() != RESULTS or not RESULTS.is_dir():
        raise ValueError("results directory must exist without symlink aliases")
    if any(os.path.lexists(p) for p in (OUTPUT, JOURNAL)):
        raise FileExistsError("sealed output or journal already exists")
    for name, expected in PINS.items():
        path = Path(__file__).resolve().parent / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("instrumentation changed: " + name)
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != MANIFEST_SHA:
        raise ValueError("manifest digest mismatch")
    manifest = runner.verify(str(MANIFEST), MANIFEST_SHA)
    slots = [(slot["root_slot"], slot["condition"]) for slot in manifest["slots"]]
    if slots != [(0, "off"), (0, "on"), (1, "off"), (1, "on")]:
        raise ValueError("unexpected four-slot execution order")
    if any(os.path.lexists(p) for p in (OUTPUT, JOURNAL)):
        raise FileExistsError("sealed output or journal appeared during verification")
    return {"status": "cpu_preflight_verified_not_authorized",
            "manifest_sha256": MANIFEST_SHA, "slots": slots,
            "output": str(OUTPUT), "journal": str(JOURNAL), "run_id": RUN_ID}


def main():
    from experiments.spectral_reachability import agent_hilewm_rgb_high_low_20261028_v1 as runner
    print(json.dumps(preflight(runner), sort_keys=True))


if __name__ == "__main__":
    main()
