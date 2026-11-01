"""Join separately audited physical, encoder and planner replays by exact boundary identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from . import e3_cell03_replay_join_codex_20261030_v1 as replay
from . import e3_cell03_planner_parity_consume_codex_20261030_v1 as planner

ROOT = replay.ROOT
NAME = "e3_cell03_end_to_end_replay_join_20261031_main"


def join(replay_result, planner_result, manifest, reconstructed):
    need = replay.require
    need(manifest["original_report_sha256"] == replay.source.REPORT_SHA
         and manifest["original_completion_sha256"] == replay.source.COMPLETION_SHA
         and manifest["authority_sha256"] == replay.source.AUTHORITY_SHA, "source identity")
    need(manifest["rows"] == reconstructed, "planner reconstructed source drift")
    rr, pr, mr = replay_result["rows"], planner_result["rows"], manifest["rows"]
    need([x["boundary"] for x in rr] == [x["boundary"] for x in pr]
         == [x["boundary"] for x in mr] == [10, 15], "boundaries")
    result = []
    for physical, node, entry in zip(rr, pr, mr, strict=True):
        action = bytes.fromhex(entry["original_action_bits"])
        need(len(action) == 40 and hashlib.sha256(action).hexdigest() == physical["next_action_sha256"]
             and physical["physical_frames_match"] is True
             and physical["encoder_recompute_bitwise"] is True
             and node["action_exact"] is True and node["result_exact"] is True,
             "replay action or parity")
        result.append({"boundary": physical["boundary"], "pixel_sha256": physical["pixel_sha256"],
                       "next_action_sha256": physical["next_action_sha256"],
                       "source_body_sha256": entry["body_sha256"],
                       "original_node_row_sha256": entry["original_row_sha256"],
                       "recomputed_node_row_sha256": node["row_sha256"],
                       "physical_frames_match": True, "encoder_recompute_bitwise": True,
                       "planner_result_exact": True, "planner_action_exact": True,
                       "planner_axes": node["axes"]})
    return {"schema": NAME, "rows": result, "original_report_sha256": replay.source.REPORT_SHA,
            "original_completion_sha256": replay.source.COMPLETION_SHA,
            "authority_sha256": replay.source.AUTHORITY_SHA,
            "planner_manifest_sha256": planner.MANIFEST_SHA, "planner_report_sha256": planner.REPORT_SHA,
            "replay_sources": replay_result["input_sha256"],
            "full_saved_strict": False, "scientific_admission": False,
            "scope": "Separate physical replay and CUDA encoder/planner recomputations on one known task; "
                     "not a new live closed-loop replay, independent task, or cost-fair scientific result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    replay.require(out.is_absolute() and out.parent == ROOT and out.name.startswith(NAME + "_")
                   and not out.exists() and not out.is_symlink(), "output")
    physical = replay.audit()
    numerical = planner.consume()
    manifest = planner.read(planner.p.MANIFEST, planner.MANIFEST_SHA)
    original = planner.p.reconstruct()
    result = join(physical, numerical, manifest, original)
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"sha256": hashlib.sha256(data).hexdigest(), "boundaries": len(result["rows"])}))


if __name__ == "__main__":
    main()
