"""Data-only raw-DEFLATE decoding. No object interpretation or inference authority."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import time
import zlib

NAME = "agent_hilewm_bounded_deflate_decode_20261028_v1"
ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "results/agent_hilewm_asset_resume_20261028_v1"
INPUT = ASSETS / "continuation02/pusht_hi_lewm_epoch15_object.ckpt.deflate"
GIB = 1024**3
DISK_CAP, FREE_FLOOR, REPORT_CAP = 3 * GIB, 10 * GIB, 2 * 1024**2
IN_BLOCK, OUT_BLOCK = 128 * 1024, 64 * 1024
PINS = {
    "request.json": "17f73073da75479cae5907c029b284a61c8f20c4a0bd22e4d2d3d7a9e72d4867",
    "continuation02/report.json": "c215ea4210d42329b8d833b8cbc61f270053d03936f8aff0836087f3de5dfb1b",
    "saved_verification.json": "e87ea1737144a33bea4f28094a541553511cca6d688619839516f83e1ba6dff6",
    "handoff.json": "a256cfbf658dddf48754380602938cc117e44d3bd7d3a296ed71ae541216cd28",
}
TEXT_BASE = ROOT / "results/p1_hilewm_text_contract_20261027_v1"
CONTRACT_SHA = "c25edf542f38faa9cf0150df4822198d28753e2cf77a0d243e990ffabe849ded"
API = ROOT / "results/p1_learned_subgoal_remote_inventory_20261027_v1/zenodo_api.json"
API_SHA = "2fc221128c12b609b86dfb7810eda46f2399bc796348f36c5d7e016ab25dd773"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: need(False, "nonfinite JSON " + x))


@dataclass(frozen=True)
class Spec:
    packed: int
    digest: str
    unpacked: int
    crc32: int


SPEC = Spec(113104429, "d169cf3c4a8d668ac5b0dcb17c6fd1682360764ce578d38daac7fec6dc76d086",
            122424517, 0xd59f865b)


def identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)


def dir_fd(path):
    """Traverse every absolute directory component without following symlinks."""
    path = Path(path)
    need(path.is_absolute() and ".." not in path.parts, "absolute nontraversing path")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def reader(path):
    path = Path(path)
    parent = dir_fd(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        need(stat.S_ISREG(os.fstat(fd).st_mode), "regular input only")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def unchanged(stream, path, before):
    parent = dir_fd(Path(path).parent)
    try:
        current = os.stat(Path(path).name, dir_fd=parent, follow_symlinks=False)
        need(identity(before) == identity(os.fstat(stream.fileno())) == identity(current),
             "file/fd identity changed")
    finally:
        os.close(parent)


def small(path, expected):
    with reader(path) as f:
        before = os.fstat(f.fileno())
        raw = f.read(REPORT_CAP + 1)
        need(len(raw) <= REPORT_CAP and sha(raw) == expected, "pinned metadata SHA/cap " + str(path))
        unchanged(f, path, before)
    return decode_json(raw)


def hash_stream(stream, tick=lambda: None):
    h = hashlib.sha256()
    count = 0
    while True:
        tick()
        block = stream.read(IN_BLOCK)
        if not block:
            break
        count += len(block)
        h.update(block)
    return count, h.hexdigest()


def decode_stream(source, sink, spec, tick=lambda: None):
    """RAM-testable bounded kernel; production CLI cannot override SPEC.

    Every decompress call has max_length <=64KiB; no unbounded flush. Limit
    overflow is rejected before writing even one over-limit output byte.
    """
    need(type(spec.packed) is int and spec.packed > 0 and
         type(spec.unpacked) is int and spec.unpacked > 0, "positive exact byte limits")
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    packed_hash, out_hash = hashlib.sha256(), hashlib.sha256()
    packed = count = crc = 0

    def emit(block):
        nonlocal count, crc
        need(count + len(block) <= spec.unpacked, "decoded output cap / bomb")
        if block:
            tick()
            need(sink.write(block) == len(block), "short output write")
            count += len(block)
            crc = zlib.crc32(block, crc)
            out_hash.update(block)

    while True:
        tick()
        data = source.read(IN_BLOCK)
        if not data:
            break
        need(not decompressor.eof, "trailing bytes after DEFLATE EOF")
        packed += len(data)
        need(packed <= spec.packed, "compressed length cap")
        packed_hash.update(data)
        pending = data
        while pending:
            tick()
            previous = len(pending)
            block = decompressor.decompress(pending, min(OUT_BLOCK, spec.unpacked - count + 1))
            emit(block)
            pending = decompressor.unconsumed_tail
            need(not decompressor.unused_data, "trailing / concatenated stream")
            need(not pending or block or len(pending) < previous, "no decompressor progress")
    # Drain bounded internal output, never use unbounded decompress.flush().
    while not decompressor.eof:
        tick()
        block = decompressor.decompress(b"", min(OUT_BLOCK, spec.unpacked - count + 1))
        emit(block)
        if not block:
            break
    need(decompressor.eof, "missing DEFLATE EOF / truncated input")
    need(not decompressor.unused_data and not decompressor.unconsumed_tail, "trailing input")
    need(packed == spec.packed and packed_hash.hexdigest() == spec.digest, "compressed length/SHA mismatch")
    need(count == spec.unpacked, "decoded exact length mismatch")
    need(crc & 0xffffffff == spec.crc32, "decoded CRC32 mismatch")
    return dict(compressed_bytes=packed, compressed_sha256=packed_hash.hexdigest(),
                decoded_bytes=count, decoded_crc32=f"{crc & 0xffffffff:08x}",
                decoded_sha256=out_hash.hexdigest(), deflate_eof=True, trailing_bytes=0)


def scope_bytes(path):
    """Only explicitly scoped trees; count duplicate chunks/hardlinks separately."""
    total = 0
    with os.scandir(path) as items:
        for item in items:
            s = item.stat(follow_symlinks=False)
            need(not stat.S_ISLNK(s.st_mode), "budget tree symlink refused")
            if stat.S_ISDIR(s.st_mode):
                total += scope_bytes(item.path)
            else:
                need(stat.S_ISREG(s.st_mode), "budget tree special file refused")
                total += s.st_size
    return total


def budget_bytes():
    total = scope_bytes(ASSETS)
    for p in (ROOT / "results").iterdir():
        if p.name.startswith(NAME + "_"):
            need(not p.is_symlink() and p.is_dir(), "decoder scope must be real directory")
            total += scope_bytes(p)
    return total


def capacity(used, free, reserve):
    need(used + reserve <= DISK_CAP, "combined asset+decoder disk cap 3GiB")
    need(free - reserve >= FREE_FLOOR, "preserve >=10GiB free after reserved writes")


def deadline(seconds):
    need(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 600,
         "wall seconds in (0,600]")
    stop = time.monotonic() + seconds
    def tick():
        need(time.monotonic() < stop, "wall deadline exceeded")
    return tick


def provenance():
    """Read only already saved, pinned JSON/text; never parse YAML or import it."""
    closure = {str(ASSETS / key): expected for key, expected in PINS.items()}
    saved = {key: small(ASSETS / key, expected) for key, expected in PINS.items()}
    need(saved["continuation02/report.json"]["compressed_member_sha256"] == SPEC.digest and
         saved["continuation02/report.json"]["status"] == "compressed_member_transport_complete",
         "terminal saved transport required")
    contract = small(TEXT_BASE / "contract.json", CONTRACT_SHA)
    api = small(API, API_SHA)
    closure.update({str(TEXT_BASE / "contract.json"): CONTRACT_SHA, str(API): API_SHA})
    locators = (
        ("package/checkpoints/pusht/main/config.yaml", 1, 117, "released config; not inspected object values"),
        ("package/code/THIRD_PARTY_LEWM.md", 19, 39, "unbound upstream commit/license"),
        ("package/code/h_le_wm/eval/hierarchical.py", 34, 38, "dynamic pickle module compatibility"),
        ("package/code/h_le_wm/eval/hierarchical.py", 318, 330, "original StandardScaler.fit; not authorized"),
        ("package/code/h_le_wm/eval/hierarchical.py", 514, 533, "high/low device and AutoCostModel loader"),
        ("package/code/h_le_wm/baseline/adapter.py", 11, 16, "required upstream module.py/utils.py"),
        ("package/code/h_le_wm/baseline/adapter.py", 39, 63, "dynamic exec_module; not run here"),
        ("package/code/environment-gpu.yml", 1, 11, "unpinned dependency versions"),
        ("package/code/h_le_wm/checkpoint_registry.yaml", 14, 18, "canonical registry path differs from downloaded main member"),
    )
    refs = []
    for member, first, last, role in locators:
        record = contract["sources"][member]
        path = TEXT_BASE / record["local_text"]
        with reader(path) as f:
            before = os.fstat(f.fileno())
            raw = f.read(REPORT_CAP + 1)
            need(len(raw) <= REPORT_CAP and sha(raw) == record["sha256"], "source text SHA")
            unchanged(f, path, before)
        lines = raw.decode().splitlines()
        need(0 < first <= last <= len(lines), "source line locator")
        closure[str(path)] = record["sha256"]
        refs.append(dict(path=str(path), sha256=record["sha256"], archive_member=member,
                         first_line=first, last_line=last, role=role,
                         quote="\n".join(lines[first - 1:last])))
    return dict(bindings_sha256=closure, source_locators=refs,
                record_license=api["metadata"]["license"], license_scope="record-level only",
                trusted_pickle_class_manifest=None, immutable_upstream_revision=None,
                embedded_low_tensor_pairing=None, frozen_dataset_normalizer_identity=None,
                canonical_registry_to_main_object_equivalence=None,
                baseline_runnable=None, official_equivalence=False, scientific_admission=False)


def own_closure():
    files = (Path(__file__), ROOT / "tests" / ("test_" + NAME + ".py"),
             ROOT / "innerdocs" / (NAME + ".md"))
    return {str(p): sha(p.read_bytes()) for p in files}


def output_directory(path):
    path = Path(path)
    need(path.is_absolute() and path.parent == ROOT / "results" and
         re.fullmatch(re.escape(NAME) + r"_[A-Za-z0-9_-]+", path.name), "exclusive scoped output directory")
    fd = dir_fd(path.parent)
    try:
        os.mkdir(path.name, mode=0o700, dir_fd=fd)
        outfd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        os.fsync(fd)
        return outfd
    finally:
        os.close(fd)


def new_file(directory_fd, name):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=directory_fd)
    return os.fdopen(fd, "wb")


def run(output, wall_seconds=120):
    started = time.monotonic()
    tick = deadline(wall_seconds)
    refs = provenance()
    source_code = own_closure()
    capacity(budget_bytes(), shutil.disk_usage(ROOT / "results").free, SPEC.unpacked + REPORT_CAP)
    tick()
    # Full hash prepass, before any decompression or output directory creation.
    with reader(INPUT) as src:
        before = os.fstat(src.fileno())
        need(before.st_size == SPEC.packed, "compressed file exact size")
        need(hash_stream(src, tick) == (SPEC.packed, SPEC.digest), "compressed prepass SHA")
        unchanged(src, INPUT, before)
        src.seek(0)
        fd = output_directory(output)
        report = dict(schema=NAME, status="rejected_partial_preserved", input=str(INPUT),
                      input_sha256=SPEC.digest, output=str(Path(output) / "decoded.unverified.ckpt"),
                      provenance=refs, own_sha256=source_code, errors=[],
                      deserialized=False, foreign_code_executed=False, baseline_runnable=None,
                      official_equivalence=False, scientific_admission=False, gpu_calls=0, physics_calls=0)
        try:
            with new_file(fd, "decoded.unverified.ckpt") as dst:
                def guarded():
                    tick()
                    capacity(budget_bytes(), shutil.disk_usage(ROOT / "results").free, OUT_BLOCK + REPORT_CAP)
                    need(os.fstat(dst.fileno()).st_nlink == 1, "unexpected output hardlink")
                decoded = decode_stream(src, dst, SPEC, guarded)
                dst.flush()
                os.fsync(dst.fileno())
            unchanged(src, INPUT, before)
            tick()
            with reader(report["output"]) as f:
                out_before = os.fstat(f.fileno())
                need(hash_stream(f, tick) == (SPEC.unpacked, decoded["decoded_sha256"]), "output readback SHA")
                unchanged(f, report["output"], out_before)
            need(provenance() == refs and own_closure() == source_code, "metadata/code closure changed")
            report.update(decoded, status="decoded_bytes_verified_not_trusted_loadable")
        except Exception as exc:
            report["errors"].append(type(exc).__name__ + ": " + str(exc))
        finally:
            try:
                report.update(wall_seconds=time.monotonic() - started, accounted_bytes=budget_bytes())
                raw = encoded(report)
                need(len(raw) <= REPORT_CAP, "report cap")
                capacity(budget_bytes(), shutil.disk_usage(ROOT / "results").free, len(raw))
                with new_file(fd, "report.json") as f:
                    need(f.write(raw) == len(raw), "short report write")
                    f.flush()
                    os.fsync(f.fileno())
                os.fsync(fd)
            finally:
                os.close(fd)
    return report


def consume(report_path, report_sha):
    r = small(report_path, report_sha)
    need(r["schema"] == NAME and r["status"] == "decoded_bytes_verified_not_trusted_loadable"
         and r["errors"] == [], "successful data-only report required")
    need(r["provenance"] == provenance() and r["own_sha256"] == own_closure(), "saved closure")
    need(r["input"] == str(INPUT) and r["input_sha256"] == SPEC.digest and
         r["compressed_bytes"] == SPEC.packed and r["compressed_sha256"] == SPEC.digest and
         r["decoded_bytes"] == SPEC.unpacked and r["decoded_crc32"] == f"{SPEC.crc32:08x}" and
         r["deflate_eof"] is True and type(r["trailing_bytes"]) is int and r["trailing_bytes"] == 0,
         "exact production ABI")
    need(r["deserialized"] is False and r["foreign_code_executed"] is False and
         r["baseline_runnable"] is None and r["official_equivalence"] is False and
         r["scientific_admission"] is False, "data verification is not loading/science approval")
    directory = Path(report_path).parent
    need(Path(report_path).is_absolute() and directory.parent == ROOT / "results" and
         re.fullmatch(re.escape(NAME) + r"_[A-Za-z0-9_-]+", directory.name) and
         Path(report_path).name == "report.json" and
         r["output"] == str(directory / "decoded.unverified.ckpt"), "fixed exclusive output binding")
    tick = deadline(120)
    for path, expected_size, expected_sha, expected_crc in (
            (INPUT, SPEC.packed, SPEC.digest, None),
            (Path(r["output"]), SPEC.unpacked, r["decoded_sha256"], SPEC.crc32)):
        with reader(path) as f:
            before = os.fstat(f.fileno())
            h = hashlib.sha256()
            size = crc = 0
            for b in iter(lambda: f.read(IN_BLOCK), b""):
                tick()
                size += len(b)
                need(size <= expected_size, "saved length cap")
                h.update(b)
                crc = zlib.crc32(b, crc)
            need(size == expected_size and h.hexdigest() == expected_sha, "saved size/SHA")
            need(expected_crc is None or crc & 0xffffffff == expected_crc, "saved CRC")
            unchanged(f, path, before)
    return dict(schema=NAME + ".consume", status="saved_bytes_verified_not_trusted_loadable",
                decoded_sha256=r["decoded_sha256"], report_sha256=report_sha,
                deserialized=False, baseline_runnable=None, scientific_admission=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("plan", help="metadata/text only; never opens compressed member")
    dec = subs.add_parser("decode", help="MAIN-reviewed data decode, no object interpretation")
    dec.add_argument("--output-dir", type=Path, required=True)
    dec.add_argument("--wall-seconds", type=float, default=120)
    con = subs.add_parser("consume", help="hash/CRC saved bytes without decoding or loading")
    con.add_argument("--report", type=Path, required=True)
    con.add_argument("--sha256", required=True)
    args = p.parse_args(argv)
    if args.command == "plan":
        out = dict(schema=NAME + ".plan", input=str(INPUT), expected=SPEC.__dict__,
                   provenance=provenance(), own_sha256=own_closure(),
                   disk_cap_bytes=DISK_CAP, free_floor_bytes=FREE_FLOOR, decode_executed=False)
    elif args.command == "decode":
        out = run(args.output_dir, args.wall_seconds)
    else:
        out = consume(args.report, args.sha256)
    print(encoded(out).decode(), end="")
    return int(out.get("status") == "rejected_partial_preserved")


if __name__ == "__main__":
    raise SystemExit(main())
