"""Export observed RGB and model inputs only; no CUDA, physics, or dispatch."""
import argparse
import base64
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from experiments.spectral_reachability import codex_p1_backtrack_observed_adapter_20261030_v1 as adapter
from experiments.spectral_reachability import codex_p1_blind_bundle_consumer_20261030_v1 as consumer

ROOT = adapter.ROOT
SOURCE = ROOT / "results/codex_p1_backtrack_observed_adapter_20261030_v1_prepare_MAIN02.json"
SOURCE_SHA = "ef39471f50da11eb67e6ad196ee5d74e11a5bb8868a105b78675dd60b57b3bbe"
NAME = "codex_p1_blind_bundle_export_20261030_v1"
need = consumer.need


def b64(data):
    return base64.b64encode(data).decode("ascii")


def blind_inputs(export, task):
    contract = export["task_contract"]
    path = Path(contract["blind_path"])
    need(path.is_absolute() and path.is_file() and not path.is_symlink() and
         path.resolve() == path and path.stat().st_size <= 8*1024**2, "blind source path")
    data = path.read_bytes()
    need(consumer.sha(data) == contract["blind_sha256"], "blind source SHA")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        need(len(members) == 3 and {x.filename for x in members} ==
             {"rgb_history.npy", "goal_rgb.npy", "raw_action_history.npy"} and
             sum(x.file_size for x in members) <= 4*1024**2, "blind archive ABI")
    with np.load(io.BytesIO(data), allow_pickle=False) as pack:
        history, goal, raw = (pack[x] for x in ("rgb_history", "goal_rgb", "raw_action_history"))
    need(history.shape == (3, 224, 224, 3) and history.dtype == np.uint8 and
         goal.shape == (224, 224, 3) and goal.dtype == np.uint8 and
         raw.shape == (10, 2) and raw.dtype == np.float32, "blind array ABI")
    need(np.array_equal(raw, np.asarray(task["root_raw_action_history"], np.float32)), "history actions")
    frames = [np.ascontiguousarray(x).tobytes() for x in (*history, goal)]
    need([consumer.sha(x) for x in frames] == contract["pixel_sha256"], "original four RGB identities")
    return [dict(rgb_base64=b64(x), pixel_sha256=consumer.sha(x)) for x in frames]


def build(prepared, original):
    need(prepared["fixed_existing_rows"] == 16 and len(prepared["rows"]) == 16 and
         len(original["requests"]) == 4, "source roster")
    frames = {}
    rows = []
    for item in prepared["rows"]:
        slot = item["slot"]
        task = item["task"]
        if slot not in frames:
            compact = original["requests"][slot]["task"]
            paths = [p for p, h in original["bindings"].items() if h == compact["binding"]["task_sha256"]]
            need(len(paths) == 1 and compact["binding"]["task_sha256"] == item["original_export_sha256"],
                 "bound source export")
            export = adapter.export_load(paths[0], item["original_export_sha256"])
            need(export["task"] == task and export["task_id"] == item["original_task_id"], "exact export task")
            frames[slot] = blind_inputs(export, task)
        cp = item["cp5"]
        pixels = adapter.saved.rgb(cp)
        png = Path(cp["path"]).read_bytes()
        need(consumer.sha(png) == cp["file_sha256"], "cp5 source bytes")
        rows.append(dict(**{k: deepcopy(item[k]) for k in
                           ("index", "slot", "root_slot", "condition", "arm", "role", "task", "prefix")},
                         task_id=item["original_task_id"], history_rgb=deepcopy(frames[slot][:3]),
                         goal_rgb=deepcopy(frames[slot][3]),
                         cp5_rgb=dict(rgb_base64=b64(pixels.tobytes()),
                                      source_png_sha256=cp["file_sha256"], pixel_sha256=cp["pixel_sha256"])))
    result = dict(schema=consumer.NAME, source_prepare_sha256=SOURCE_SHA,
                  model_before=deepcopy(prepared["model_before"]), model_files=deepcopy(prepared["model_files"]),
                  rows=rows, formal_roster=deepcopy(prepared["formal_roster"]),
                  producer_sha256=adapter.saved.sha(Path(__file__).resolve()),
                  consumer_sha256=adapter.saved.sha(Path(consumer.__file__).resolve()),
                  physical_authorized=False, P1_admission=False)
    consumer.validate(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    path = args.output
    need(path.is_absolute() and path.parent == ROOT/"results" and path.parent.resolve() == path.parent and
         path.name.startswith(NAME+"_") and not os.path.lexists(path), "exclusive output path")
    print("Verifying saved sources and exporting only historical RGB and model inputs; no neural/physical calls.",
          flush=True)
    adapter.verify(SOURCE, SOURCE_SHA)
    prepared = adapter.saved.native.read(SOURCE, cap=32*1024**2)
    replay = adapter.saved.replay
    report = replay.one.saved.read_verified(replay.one.source.REPORT, replay.one.source.REPORT_SHA)
    original = replay.one.saved.read_verified(report["prepared"], report["prepare_sha256"])
    result = build(prepared, original)
    need(adapter.saved.sha(SOURCE) == SOURCE_SHA, "source drift")
    payload = adapter.saved.encode_json(result)
    need(len(payload) <= 32*1024**2, "bundle cap")
    adapter.saved.write_exclusive(path, payload)
    print(json.dumps(dict(path=str(path), sha256=consumer.sha(payload), rows=16,
                          greedy_unknown=8, physical_authorized=False)))


if __name__ == "__main__":
    main()
