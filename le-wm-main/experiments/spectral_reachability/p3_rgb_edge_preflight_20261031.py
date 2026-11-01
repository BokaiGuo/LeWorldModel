"""RGB-only, conservative preflight for edge-clipped PushT observations."""
import argparse
import hashlib
import json
from pathlib import Path

from . import codex_p1_rgb_task_score_20261030_v2 as features
from . import p3_h3_real_context_replay_20261031 as source

NAME = "p3_rgb_edge_preflight_20261031"
ROOT, need = source.ROOT, source.need
PROTOCOL = ROOT.parent / "docs/D2P_P3_RGB_EDGE_PREFLIGHT_PROTOCOL_20261031.md"


def preflight(rgb):
    """Reject unreliable or near-edge pusher markers, without reading state."""
    try:
        image = features._image(rgb)
        _, marker = features._mask(image, features.BLUE, 8, single=True)
    except ValueError as exc:
        return {"admitted": False, "reason": str(exc), "bounds_xyxy": None}
    x0, y0, x1, y1 = marker["bounds_xyxy"]
    distances = [x0, y0, 223 - x1, 223 - y1]
    if min(distances) < 8:
        return {"admitted": False, "reason": "pusher bbox within 8 pixels of image edge",
                "bounds_xyxy": marker["bounds_xyxy"]}
    return {"admitted": True, "reason": "pusher marker passes exploratory edge preflight",
            "bounds_xyxy": marker["bounds_xyxy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    entries, excluded, bindings = source.collect()
    need(len(entries) == 52 and len(excluded) == 10, "fixed roster")
    for path in (Path(__file__).resolve(), Path(features.__file__).resolve(), PROTOCOL):
        bindings[str(path)] = source.base.sha(path)
    rows = []
    for entry in entries:
        rgb = entry["frames"][2]
        rows.append({"logical_pair": entry["logical_pair"], "variant": entry["variant"],
                     "root_slot": entry["key"][3], "cp15_pixel_sha256": hashlib.sha256(
                         rgb.tobytes()).hexdigest(), **preflight(rgb)})
    need(all(source.base.sha(path) == digest for path, digest in bindings.items()), "source drift")
    result = {"schema": NAME, "rows": rows, "fixed_rows": 52, "excluded": excluded,
              "source_hashes": bindings, "P1_admission": False, "P3_admission": False,
              "limitations": "Post-incident exploratory RGB-only input preflight; passing does not "
                             "guarantee safe later actions. No physical or neural calls."}
    digest = source.base.write(out, result)
    print(json.dumps({"rows": len(rows), "admitted": sum(r["admitted"] for r in rows),
                      "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
