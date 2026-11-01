"""Development-only dev303 receipt to blind NPZ adapter."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from . import p1_queue_goal_blind_adapter_20261101_codex as core


ROOT = core.ROOT
NAME = "p1_dev303_blind_adapter_20261101_codex"
QUEUE = ROOT / "results/p1_queue_goal_history_20261101_codex_QUEUE_DEV03.json"
QUEUE_SHA = "ec5656ea3c907844d3f8374af7f55aee9c65c9fe5196bbcd195717248618eb0e"
REPORT_SHA = "871048f32696639fea89abe07e40403c132f02281f88e920e537f12aa688056f"
SLOT = "dev_holdout03"
ATTEMPT = 0


@contextmanager
def _patched():
    values = {
        "QUEUE": QUEUE,
        "QUEUE_SHA": QUEUE_SHA,
        "REPORT_SHA": REPORT_SHA,
        "SLOT": SLOT,
        "ATTEMPT": ATTEMPT,
    }
    previous = {key: getattr(core, key) for key in values}
    for key, value in values.items():
        setattr(core, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(core, key, value)


def source():
    with _patched():
        return core.source()


def build(output):
    with _patched():
        value = core.build(output)
    value["schema"] = NAME + ".v1"
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest).absolute()
    core.blind.require(
        not manifest.exists() and manifest != Path(args.output).absolute(),
        "exclusive manifest",
    )
    value = build(args.output)
    with manifest.open("xb") as stream:
        stream.write(core.blind.canonical(value))
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
