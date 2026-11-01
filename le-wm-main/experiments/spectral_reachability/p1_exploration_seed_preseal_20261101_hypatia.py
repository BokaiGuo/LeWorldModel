"""One-shot random calibration seed proposal; no environment or outcomes.

The new seed distribution does not prove historical or training independence.
All draws are retained before collision checks; no rejection sampling.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "innerdocs/p1_conditional_procedural_source_20261027_v1_calibration2_config.json"
TEMPLATE_SHA = "86a6a95a0ef0e3bec36daf1537e6eaf91982a83f4473d5e46d85384a7617e8bc"
NAME = "p1_exploration_seed_preseal_20261101_hypatia"
LOW = 2**30
WIDTH = 2**30 - 8


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def construct(template, draws):
    if len(draws) != 3 or any(type(x) is not int or not LOW <= x < LOW + WIDTH for x in draws):
        raise ValueError("three predeclared uniform draws required")
    if template["proposal_limit_per_seed"] != 2 or template["reset_attempts_per_proposal"] != 2:
        raise ValueError("template budget drift")
    config = copy.deepcopy(template)
    config["root_seeds"] = draws[:2]
    config["reset_seed_start"] = draws[2]
    candidates = []
    for i, seed in enumerate(draws[:2]):
        candidates.append({"candidate_id": f"root_seed_{i}", "root_seed": seed,
                           "reset_seed": None, "state7": None})
    for i in range(8):
        candidates.append({"candidate_id": f"random_reset_{i}", "root_seed": None,
                           "reset_seed": draws[2] + i, "state7": None})
    return config, {"schema": "p1_explicit_collision_check_20261101_hypatia.candidates.v1",
                    "candidates": candidates}


def write_exclusive(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        if stream.write(data) != len(data):
            raise OSError("short write")
        stream.flush()
        os.fsync(stream.fileno())
    return digest(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-output", required=True, type=Path)
    parser.add_argument("--candidates-output", required=True, type=Path)
    args = parser.parse_args()
    for path, suffix in ((args.config_output, "_config.json"), (args.candidates_output, "_candidates.json")):
        if not path.is_absolute() or path.parent.resolve() != path.parent or path.name != NAME + suffix:
            raise ValueError("exclusive fixed output namespace required")
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    raw = TEMPLATE.read_bytes()
    if digest(raw) != TEMPLATE_SHA:
        raise ValueError("template hash drift")
    template = json.loads(raw)
    draws = [LOW + secrets.randbelow(WIDTH) for _ in range(3)]
    config, candidates = construct(template, draws)
    config_sha = write_exclusive(args.config_output, config)
    candidate_sha = write_exclusive(args.candidates_output, candidates)
    print(json.dumps({"config": str(args.config_output), "config_sha256": config_sha,
                      "candidates": str(args.candidates_output), "candidates_sha256": candidate_sha,
                      "scope": "seed proposals only; no root generated; freshness unknown",
                      "draws": "three independent secrets.randbelow calls; no retry"}))


if __name__ == "__main__":
    main()
