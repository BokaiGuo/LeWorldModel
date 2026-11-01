"""Bounded saved-metadata join; absence of a collision is not root independence."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "codex_p1_root_source_join_20261030"
FILES = {
    "history618": ("results/root_exclusion_authority_resolution_20261027_v1_inventory618_MAIN01.json",
                   "e2afed12a1d8f788ad16d0223482377b5f54d281e67739766475228b53ac2591"),
    "history8": ("results/moving_root_exclusion_manifest_20261027_v1_inventory8_main01.json",
                 "a17bc732c9a021a4ca81a7e4da573c752ad1f66eafe08b1a3dad2b64464d7060"),
    "config": ("results/agent_p1_configuration_provenance_20261028_v1_MAIN01.json",
               "43841acd6b2f0f293d041b2c2b871076b425cc3e090e1adfa21e96cfece7dd57"),
    "official_review": ("results/p1_checkpoint_historical_contract_review_20261027_v1_readonly_v1/report.json",
                        "07f935a7fe1abdf936545673393fca0f1c290f4b74c8107311ebad7413919948"),
    "gap": ("results/codex_p1_source_authority_gap_20261030_LOCAL02.json",
            "da197b3a6586e31b1c11f0353831be64bb3ecdb6e74adf972a42acd155ba1b24"),
}
CHECKPOINT = Path("/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm")
WEIGHTS_SHA = "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"
CONFIG_SHA = "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"
PROPOSED = [(203027101+i, 203027201+i) for i in range(4)]


def require(ok, why):
    if not ok:
        raise ValueError(why)


def file_hash(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path, "canonical file")
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "file changed")
    return h.hexdigest()


def compare(history8, history618):
    require(history8["freshness_certified"] is False and history8["source_run_preconditions_met"] is False
            and history618["freshness_certified"] is False and history618["P1_admission"] is False,
            "non-admission inputs")
    seeds = history618["stable_identity_projections"]["seed_projections"]
    require(all(type(x["seed"]) is int and x["kind"] in ("root_seed", "reset_seed", "calibration_root_seed")
                for x in seeds), "seed projection")
    known = set(history8["used_or_reserved_root_seeds"]) | set(history8["used_or_reserved_reset_seeds"])
    known |= {x["seed"] for x in seeds}
    spans = history8["reserved_reset_spans"]
    require(all(type(x) is list and len(x) == 2 and all(type(v) is int for v in x)
                and x[0] < x[1] for x in spans), "half-open spans")
    rows = []
    for slot, (root, reset) in enumerate(PROPOSED):
        collisions = []
        for kind, value in (("root_seed", root), ("reset_seed", reset)):
            if value in known:
                collisions.append({"candidate_field": kind, "value": value, "kind": "known_seed"})
            for lo, hi in spans:
                if lo <= value < hi:
                    collisions.append({"candidate_field": kind, "value": value,
                                       "kind": "known_reset_span", "span": [lo, hi]})
        rows.append({"slot": slot, "root_seed": root, "reset_seed": reset, "conflicts": collisions,
                     "no_known_conflict_is_not_independence": True})
    return {"candidates": rows, "history618_projected_seed_records": len(seeds),
            "history8_known_distinct_seed_count": len(set(history8["used_or_reserved_root_seeds"]) |
                                                      set(history8["used_or_reserved_reset_seeds"])),
            "history8_reset_span_count": len(spans),
            "scope": "fixed historical summaries only; no future or outcome fields, no new reset",
            "history_complete": False, "training_disjoint_certified": False,
            "freshness_certified": False, "source_run_authorized": False, "P1_admission": False}


def build():
    values = {}
    for name, (rel, expected) in FILES.items():
        path = ROOT / rel
        require(file_hash(path) == expected, "pinned input drift: " + name)
        values[name] = json.loads(path.read_bytes())
        require(file_hash(path) == expected, "input drift on read: " + name)
    require(file_hash(CHECKPOINT / "weights.pt") == WEIGHTS_SHA
            and file_hash(CHECKPOINT / "config.json") == CONFIG_SHA, "checkpoint drift")
    config, hist = values["config"], values["history618"]
    require(config["training_identity"]["verified"] is False
            and config["value_binding"]["all_float32_bits_equal"] is True, "normalizer evidence boundary")
    result = compare(values["history8"], hist)
    result.update({
        "schema": NAME + ".v1",
        "input_bindings": {name: {"path": str(ROOT / rel), "sha256": h}
                           for name, (rel, h) in FILES.items()},
        "checkpoint_weights_sha256": WEIGHTS_SHA, "checkpoint_config_sha256": CONFIG_SHA,
        "history618_summary": hist["summary"],
        "history_authority_requests": [{k: row[k] for k in ("family", "status", "remaining")}
                                       for row in hist["authority_requests"]],
        "training_identity": config["training_identity"],
        "normalizer_float32_value_agreement": True,
        "official_review_status": values["official_review"]["status"],
        "local_gap_inventory": values["gap"]["inventory_assessment"],
        "limitations": "Metadata-level collision check only; history completeness, pose/history equality, "
                       "training membership and historical normalizer provenance not established. "
                       "Input summaries do not prove the historical bytes of every referenced source.",
        "no_outcome_based_selection": True, "neural_calls": 0, "physical_calls": 0,
    })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    path = args.output
    require(path.is_absolute() and path.parent == ROOT / "results"
            and path.name.startswith(NAME + "_") and not path.is_symlink(), "exclusive namespace")
    result = build()
    result["source_sha256"] = file_hash(Path(__file__).resolve())
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as f:
        require(f.write(data) == len(data), "complete write")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"output": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                      "conflicts": [x["conflicts"] for x in result["candidates"]]}))


if __name__ == "__main__":
    main()
