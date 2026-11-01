"""Bounded LOCAL identity evidence; never import model code or unpickle tensors.

Pickle opcode strings are claims, not an executed state_dict or tensor-shape proof.
Only ZIP metadata and a <=1 MiB data.pkl are read; no archive extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import pickletools
import re
import stat
import struct
import subprocess
import zipfile

NAME = "p5_architecture_readiness_20261027_v2"
MAX_ARCHIVE = 512 * 1024**2
MAX_METADATA = 1024**2
MAX_MEMBERS = 4096


def safe_name(name):
    parts = name.split("/")
    if (not name or len(name) > 1024 or "\\" in name or "\x00" in name
            or PurePosixPath(name).is_absolute() or ".." in parts
            or re.match(r"^[A-Za-z]:", name)):
        raise ValueError("unsafe archive member path")


def opcode_strings(data):
    if len(data) > MAX_METADATA:
        raise ValueError("pickle metadata limit")
    strings, globals_seen = [], []
    last = None
    for count, (op, arg, pos) in enumerate(pickletools.genops(data), 1):
        if count > 200000:
            raise ValueError("opcode count limit")
        last = (op.name, pos)
        if op.name in {"UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"}:
            strings.append(arg)
        elif op.name == "GLOBAL":
            globals_seen.append(arg)
        elif op.name in {"EXT1", "EXT2", "EXT4", "STACK_GLOBAL", "INST", "OBJ"}:
            raise ValueError("unsupported identity opcode: " + op.name)
    if last != ("STOP", len(data) - 1):
        raise ValueError("missing STOP or trailing pickle data")
    return strings, sorted(set(globals_seen))


def inspect_archive(path):
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_ARCHIVE:
        raise ValueError("archive byte limit")
    # Bound central-directory allocation BEFORE ZipFile builds its member list.
    with path.open("rb") as handle:
        handle.seek(max(0, size - 65557))
        tail = handle.read(65557)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or len(tail) - offset < 22:
        raise ValueError("not a supported ZIP (no tar/legacy pickle execution)")
    _, disk, cd_disk, disk_n, total_n, cd_size, cd_offset, comment_n = struct.unpack(
        "<4s4H2LH", tail[offset:offset + 22])
    if (disk or cd_disk or disk_n != total_n or total_n > MAX_MEMBERS
            or cd_size > 4 * 1024**2 or cd_offset + cd_size > size
            or len(tail) - offset != 22 + comment_n):
        raise ValueError("ZIP central directory limits/multidisk/ZIP64")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != total_n:
            raise ValueError("member count mismatch")
        seen, total = set(), 0
        for info in members:
            safe_name(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError("nonregular archive member")
            if info.filename in seen or info.flag_bits & 1:
                raise ValueError("duplicate/encrypted archive member")
            seen.add(info.filename)
            total += info.file_size
            if (info.file_size > MAX_ARCHIVE or total > 1024**3
                    or info.file_size > max(1, info.compress_size) * 64):
                raise ValueError("archive expansion limit")
        metadata = [i for i in members if i.filename.endswith("/data.pkl")]
        if len(metadata) != 1 or metadata[0].file_size > MAX_METADATA:
            raise ValueError("expected one bounded data.pkl")
        with archive.open(metadata[0]) as handle:
            data = handle.read(MAX_METADATA + 1)
        strings, globals_seen = opcode_strings(data)
        keys = sorted(set(s for s in strings if s.startswith("module.")))
        blocks = sorted({int(m.group(1)) for s in keys
                         if (m := re.match(r"module.predictor_blocks\.(\d+)\.", s))})
        markers = {s: s in strings for s in (
            "predictor", "module.predictor_embed.weight", "module.action_encoder.weight",
            "module.predictor_blocks.0.adaLN_modulation.1.weight", "proprio_encoder")}
        match = all(markers.values()) and blocks == list(range(6))
        return {"format": "pytorch_zip_not_tar", "member_count": len(members),
                "declared_uncompressed_bytes": total, "metadata_member": metadata[0].filename,
                "metadata_bytes_read": len(data), "metadata_sha256": hashlib.sha256(data).hexdigest(),
                "opcode_globals_not_executed": globals_seen, "identity_markers": markers,
                "predictor_block_indices": blocks, "parameter_name_literals": keys,
                "identity": "JEPA_WM_AdaLN_action_dynamics_static_match" if match else "unresolved",
                "tensor_shapes_verified": False, "state_dict_loaded": False,
                "tensor_payloads_read": False}


def record(path, hash_limit=MAX_ARCHIVE):
    path = Path(path)
    out = {"path": str(path), "exists": path.is_file()}
    if out["exists"]:
        out["bytes"] = path.stat().st_size
        if out["bytes"] <= hash_limit:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024**2), b""):
                    digest.update(chunk)
            out["sha256"] = digest.hexdigest()
            out["hash_scope"] = "full_file_bytes_no_deserialization"
    return out


def build_report(root):
    root = Path(root).resolve()
    baseline_path = root / "results/p5_local_asset_binding_20261027_v1_assets/manifest.json"
    baseline = json.loads(baseline_path.read_text())
    prior = {r["path"]: r.get("sha256") for r in baseline["files"]}
    archives = []
    for task in ("pusht", "pointmaze", "wall"):
        path = root / f"data/jepa_wm_checkpoints/jepa_wm_{task}.pth.tar"
        item = record(path)
        item["matches_v1_sha256"] = item.get("sha256") == prior.get(str(path)) and bool(item.get("sha256"))
        try:
            item["inspection"] = inspect_archive(path)
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, NotImplementedError) as exc:
            item["inspection_error"] = str(exc)
        archives.append(item)
    relative_sources = [
        "external/jepa_acquisition_manifest.json",
        "external/jepa-wms/app/vjepa_wm/utils.py",
        "external/jepa-wms/app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py",
        "external/jepa-wms/app/plan_common/models/AdaLN_vit.py",
        "external/jepa-wms/app/plan_common/models/dino.py",
        "external/jepa-wms/app/plan_common/models/prop_embedding.py",
        "external/jepa-wms/app/plan_common/models/vit.py",
        "external/jepa-wms/app/plan_common/datasets/pusht_dset.py",
        "external/local_dinov2/hubconf.py", 
        ".cache/dinov2/dinov2-small/config.json",
        ".cache/dinov2/dinov2-small/model.safetensors",
        "experiments/theory_spectral_transfer/run_jepa_wm_temporal_eval.py",
    ]
    configs = [f"external/jepa-wms/configs/vjepa_wm/{folder}/{filename}" for folder, filename in (
        ("pt_sweep", "pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml"),
        ("mz_sweep", "mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml"),
        ("wall_sweep", "wall_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml"))]
    sources = [record(root / p) for p in relative_sources + configs]
    sources += [record(root.parent / "docs/D2P_PARALLEL_CUDA_PROGRESS_20261027.md"), record(baseline_path)]
    source_git = {}
    for name, args in (("commit", ["rev-parse", "HEAD"]), ("dirty_status", ["status", "--short"])):
        proc = subprocess.run(["git", "-C", str(root / "external/jepa-wms"), *args],
                              capture_output=True, text=True, timeout=10, check=False)
        source_git[name] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    dependencies = {}
    for name in ("torch", "torchvision", "tensordict", "transformers", "einops", "PyYAML", "decord", "opencv-python"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    dataset = root / "data/jepa_wm_osf/extracted/pusht_noise/val"
    data_records = [record(dataset / name) for name in (
        "states.pth", "rel_actions.pth", "velocities.pth", "seq_lengths.pkl")]
    # Bounded directory inspection; do not traverse videos or load dataset pickle.
    import itertools
    video_paths = list(itertools.islice((dataset / "obses").glob("*.mp4"), 4097))
    lewm = []
    for task in ("tworoom", "reacher", "pusht", "cube"):
        path = Path("/home/gbk/.stable_worldmodel/checkpoints") / task / "lewm/config.json"
        cfg = json.loads(path.read_text()) if path.is_file() else {}
        lewm.append({**record(path), "target": cfg.get("_target_"), "predictor": cfg.get("predictor")})
    missing_dino = [record(root / f"data/dino_wm_osf/extracted/outputs/{task}/checkpoints/model_latest.pth")
                    for task in ("point_maze", "pusht", "wall_single")]
    return {"schema": NAME, "scope": "fixed local references only; not exhaustive installed-model discovery",
            "status": "independent_dynamics_candidate_identified_runtime_not_certified",
            "P5_admission": False, "cuda_executed": False, "neural_execution": False,
            "native_full_d2p_runtime_ready": False, "cuda_validation_command": None,
            "archives": archives, "sources": sources, "source_git": source_git,
            "dependency_distribution_metadata_only": dependencies,
            "pusht_dataset": {"files": data_records, "mp4_count_bounded": len(video_paths),
                              "listing_truncated": len(video_paths) > 4096, "content_loaded": False},
            "lewm_configs": lewm, "dino_dynamics_paths": missing_dino,
            "finding": "JEPA-WM has separate AdaLN action-conditioned patch dynamics, not encoder-only or same LeWM; metadata alone is not a load/shape/runtime proof. DINOv2 cache alone is encoder-only.",
            "prerequisites": [
                "Safe restricted tensor/dataset loading and exact checkpoint-to-module shape/key validation; existing fetch_checkpoint uses torch.load with no explicit weights_only and strict=False, dataset uses pickle.load.",
                "Audit and freeze dirty native JEPA source and HF DINOv2 shim equivalence; current commit alone does not identify executed source.",
                "Explicit native action/RGB/proprio context binding: PushT relative action /100 then /ACTION_STD, packet5x2; 384-d patch latent plus 4-d privileged proprio, versus LeWM192 RGB-only. No silent removal of proprio or replacement by LeWM.",
                "A CUDA-only safe validation entrypoint recording exact missing/unexpected keys, shapes, offline dependencies, finite action-sensitive prediction and receipts; existing temporal evaluator is not Full-D2P P5 replay.",
                "Frozen independent-architecture Full-D2P adapter/protocol, task metric and resource accounting; upstream P1/scientific admission remains closed."],
            "safety": {"archive_extracted": False, "pickle_deserialized": False,
                       "downloads": False, "training": False, "cpu_neural_fallback": False,
                       "limits": {"archive_bytes": MAX_ARCHIVE, "metadata_bytes": MAX_METADATA,
                                  "members": MAX_MEMBERS, "central_directory_bytes": 4*1024**2,
                                  "declared_expanded_bytes": 1024**3, "expansion_ratio": 64,
                                  "pickle_opcodes": 200000}},
            "interpretation": "Static independent architecture candidate, not independent P5 efficacy or a usable certified safe native runtime. No CUDA command asserted until prerequisites are satisfied."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"report": str(args.output), "status": report["status"]}))


if __name__ == "__main__":
    main()
