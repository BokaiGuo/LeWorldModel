"""Compare actual RGB/action input receipts of the bound and historical runs."""
import json
from pathlib import Path

from experiments import pc_bound_rgb_closed_loop_replay_20261107_codex as replay
from experiments.pc_input_permissions import RGB, file_hash
from experiments.research_runtime.artifacts import atomic_write_json

OUT = replay.source.ROOT / "results/pc_bound_rgb_closed_loop_receipt_audit_20261107_codex.json"


def compare_receipts(old, new):
    if len(old) != len(new):
        raise ValueError("forward receipt count mismatch")
    for a, b in zip(old, new, strict=True):
        if (a["operation"] != b["operation"] or a["inputs"] != b["inputs"]
                or a["track_id"] != RGB or b["track_id"] != RGB):
            raise ValueError("RGB, goal, action or operation receipt mismatch")
    return len(old)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    old, new = (json.loads(p.read_text()) for p in (replay.source.OUTPUT, replay.OUT))
    if (new["source_sha256"] != file_hash(replay.source.OUTPUT)
            or new["runner_sha256"] != file_hash(Path(replay.__file__))
            or new["protocol_sha256"] != file_hash(replay.PROTOCOL)
            or new["model_sha256"] != old["model_sha256"]
            or new["normalizer_sha256"] != old["normalizer_sha256"]):
        raise ValueError("provenance mismatch")
    count = decisions = 0
    for a, b in zip(old["rows"][:4], new["rows"], strict=True):
        if a["episode"] != b["episode"]:
            raise ValueError("episode mismatch")
        for arm in replay.ARMS:
            x, y = a["arms"][arm]["decisions"], b["arms"][arm]["decisions"]
            if len(x) != len(y):
                raise ValueError("decision count mismatch")
            for u, v in zip(x, y, strict=True):
                count += compare_receipts(u["receipts"], v["receipts"])
                decisions += 1
    if count != new["binding"]["runtime_receipts"] or count != 126 or decisions != 40:
        raise ValueError("receipt count mismatch")
    atomic_write_json(OUT, {"scope": "same-backend replay; input receipt equality only",
                            "source_sha256": file_hash(replay.source.OUTPUT), "replay_sha256": file_hash(replay.OUT),
                            "runner_sha256": file_hash(Path(__file__)), "decisions": decisions,
                            "matching_receipts": count})
    print(f"decisions={decisions} identical RGB/action receipts={count}")


if __name__ == "__main__":
    main()
