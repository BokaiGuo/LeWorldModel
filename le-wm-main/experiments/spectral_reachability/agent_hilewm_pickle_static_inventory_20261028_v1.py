"""Bounded ZIP/pickle opcode inventory, never deserialization or loading approval."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import pickletools
import re
import shutil
import stat
import struct
import time
import zipfile

from experiments.spectral_reachability import agent_hilewm_bounded_deflate_decode_20261028_v1 as base

NAME = "agent_hilewm_pickle_static_inventory_20261028_v1"
ROOT = base.ROOT
HELPER_SHA = "81aa18081378ee073e267702742566b49744365d321409dff0dccfbe6d45d30c"
DECODE_REPORT = ROOT / "results/agent_hilewm_bounded_deflate_decode_20261028_v1_MAIN01/report.json"
DECODE_REPORT_SHA = "f399b23e07f37960127433d10e14c8c7b7eae6c75ff329accb75a52f701edcc9"
INPUT = DECODE_REPORT.parent / "decoded.unverified.ckpt"
INPUT_BYTES = 122424517
INPUT_SHA = "b87805747d40037841877ce7b99b7dda3ebe7a52202c0ba46bf0006ab5d6f008"
REPORT_CAP = 32 * 1024**2
UNKNOWN, MARK = ("unknown",), ("mark",)


@dataclass(frozen=True)
class Caps:
    members: int = 8192
    central_bytes: int = 4 * 1024**2
    pickle_bytes: int = 16 * 1024**2
    total_declared: int = 512 * 1024**2
    member_bytes: int = 256 * 1024**2
    opcodes: int = 200000
    stack: int = 200000
    memo: int = 200000
    name_chars: int = 4096


CAPS = Caps()


def need(ok, message):
    base.need(ok, message)


def path_issues(name, limit=CAPS.name_chars):
    out = []
    if not name or len(name) > limit:
        out.append("empty_or_oversize_name")
    if name.startswith("/") or "\\" in name or ":" in name:
        out.append("absolute_backslash_or_drive_path")
    if any(ord(c) < 32 or ord(c) == 127 for c in name):
        out.append("control_or_NUL_path")
    parts = name[:-1].split("/") if name.endswith("/") else name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        out.append("empty_dot_or_parent_segment")
    return out


def pickle_inventory(raw, caps=CAPS, tick=lambda: None):
    """Conservative symbolic *strings* only; no value/callable construction.

    Objects, reducers, extension registries, storage IDs and buffers are opaque.
    Only string literals and their memo/DUP copies can resolve STACK_GLOBAL.
    """
    need(len(raw) <= caps.pickle_bytes, "data.pkl byte cap")
    stack, memo, ops, symbols, unresolved, extensions = [], {}, [], [], [], []
    counts, sensitive = Counter(), []
    errors = []
    stopped = False

    def popn(n):
        need(len(stack) >= n and all(x != MARK for x in stack[-n:]) if n else True,
             "symbolic stack underflow/mark")
        if n:
            out = stack[-n:]
            del stack[-n:]
            return out
        return []

    def symbol(module, name, pos, opcode):
        need(bool(module and name) and len(module) <= caps.name_chars and
             len(name) <= caps.name_chars, "invalid/oversize GLOBAL symbol")
        symbols.append(dict(position=pos, opcode=opcode, module=module, name=name,
                            evidence="literal declaration; not imported or class-certified"))

    def declared(pos, opcode):
        # genops joins module/name with a space; decode the original two lines
        # so a literal module containing spaces is never silently re-partitioned.
        middle = raw.find(b"\n", pos + 1)
        end = raw.find(b"\n", middle + 1)
        need(middle >= 0 and end >= 0, "GLOBAL literal lines")
        symbol(raw[pos + 1:middle].decode("utf-8"), raw[middle + 1:end].decode("utf-8"), pos, opcode)

    try:
        for op, arg, pos in pickletools.genops(raw):
            tick()
            need(len(ops) < caps.opcodes, "opcode cap")
            n = op.name
            ops.append(dict(position=pos, opcode=n))
            counts[n] += 1
            if n in {"REDUCE", "BUILD", "NEWOBJ", "NEWOBJ_EX", "OBJ", "INST",
                     "PERSID", "BINPERSID", "NEXT_BUFFER", "READONLY_BUFFER"}:
                sensitive.append(dict(position=pos, opcode=n))
            if n in {"UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"}:
                stack.append(("literal", arg, pos) if len(arg) <= caps.name_chars else UNKNOWN)
            elif n == "GLOBAL":
                declared(pos, n)
                stack.append(UNKNOWN)
            elif n == "STACK_GLOBAL":
                module, name = popn(2)
                if module[0] == name[0] == "literal":
                    symbol(module[1], name[1], pos, n)
                    symbols[-1]["string_origins"] = [module[2], name[2]]
                else:
                    unresolved.append(dict(position=pos, opcode=n,
                                           reason="operands not proven unicode literals/memo copies"))
                stack.append(UNKNOWN)
            elif n in {"PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"}:
                need(stack and stack[-1] != MARK, "memo store without value")
                index = len(memo) if n == "MEMOIZE" else int(arg)
                need(0 <= index < caps.memo and len(memo) < caps.memo, "memo cap/index")
                memo[index] = stack[-1]
            elif n in {"GET", "BINGET", "LONG_BINGET"}:
                need(int(arg) in memo, "unknown memo reference")
                stack.append(memo[int(arg)])
            elif n == "DUP":
                need(stack and stack[-1] != MARK, "DUP without concrete stack entry")
                stack.append(stack[-1])
            elif n == "MARK":
                stack.append(MARK)
            elif n == "POP":
                need(stack, "POP underflow")
                stack.pop()
            elif n in {"EXT1", "EXT2", "EXT4"}:
                extensions.append(dict(position=pos, opcode=n, code=arg,
                                       resolution=None, reason="extension registry not consulted"))
                stack.append(UNKNOWN)
            elif n == "PROTO":
                need(pos == 0 and 0 <= arg <= 5, "unsupported/misplaced pickle protocol")
            elif n == "FRAME":
                need(0 <= arg <= len(raw) - pos - 9, "frame exceeds data.pkl")
            else:
                before = [x.name for x in op.stack_before]
                if "mark" in before:
                    need(MARK in stack, "missing MARK")
                    index = len(stack) - 1 - stack[::-1].index(MARK)
                    required = sum(x not in {"mark", "stackslice"} for x in before[before.index("mark") + 1:])
                    need(len(stack) - index - 1 >= required, "MARK slice underflow")
                    del stack[index:]
                    popn(before.index("mark"))
                else:
                    popn(len(before))
                if n == "INST":
                    declared(pos, n)
                stack.extend(UNKNOWN for _ in op.stack_after)
                if n == "STOP":
                    need(not stack and pos + 1 == len(raw), "STOP stack/trailing pickle bytes")
                    stopped = True
            need(len(stack) <= caps.stack, "symbolic stack cap")
    except Exception as exc:
        errors.append(type(exc).__name__ + ": " + str(exc)[:1000])
    if not stopped:
        errors.append("complete STOP not observed")
    return dict(opcodes=ops, opcode_counts=dict(sorted(counts.items())),
                opcode_count=len(ops), opcode_inventory_complete=stopped and not errors,
                declared_globals=symbols,
                unique_declared_symbols=[dict(module=a, name=b) for a, b in
                                         sorted({(x["module"], x["name"]) for x in symbols})],
                unresolved_stack_globals=unresolved, extension_codes=extensions,
                global_resolution_complete=stopped and not errors and not unresolved and not extensions,
                sensitive_opcodes=sensitive, errors=errors,
                executes_reducers=False, persistent_ids_resolved=False,
                declared_symbols_are_not_a_trusted_class_manifest=True)


def zip_preflight(stream, caps):
    stream.seek(0, 2)
    size = stream.tell()
    need(size >= 22, "not a ZIP container")
    stream.seek(0)
    need(stream.read(4) == b"PK\x03\x04", "no raw-pickle/SFX fallback")
    stream.seek(max(0, size - 65557))
    tail = stream.read(65557)
    pos = tail.rfind(b"PK\x05\x06")
    need(pos >= 0 and len(tail) - pos >= 22, "missing ZIP EOCD")
    _, disk, cd_disk, disk_n, entries, cd_size, cd_offset, comment = struct.unpack(
        "<4s4H2IH", tail[pos:pos + 22])
    need(pos + 22 + comment == len(tail), "ZIP trailing bytes/invalid EOCD")
    need(disk == cd_disk == 0 and disk_n == entries, "multi-disk ZIP refused")
    eocd_offset = size - len(tail) + pos
    directory_end_limit = eocd_offset
    zip64 = False
    # ZipFile can override ordinary EOCD fields using a ZIP64 locator. Bound
    # those fields BEFORE ZipFile allocates/reads its central directory.
    if pos >= 20 and tail[pos - 20:pos - 16] == b"PK\x06\x07":
        _, locator_disk, offset64, disks = struct.unpack("<4sIQI", tail[pos - 20:pos])
        need(locator_disk == 0 and disks == 1 and 0 <= offset64 <= eocd_offset - 76,
             "ZIP64 locator bounds/disk")
        stream.seek(offset64)
        body64 = stream.read(56)
        need(len(body64) == 56, "ZIP64 header truncated")
        z = struct.unpack("<4sQ2H2I4Q", body64)
        need(z[0] == b"PK\x06\x06" and 44 <= z[1] <= 4096 and
             offset64 + 12 + z[1] == eocd_offset - 20, "ZIP64 record bounds")
        need(z[4] == z[5] == 0 and z[6] == z[7], "ZIP64 multi-disk refused")
        need((entries in (65535, z[7])) and (cd_size in (0xffffffff, z[8])) and
             (cd_offset in (0xffffffff, z[9])), "ZIP64/EOCD disagreement")
        entries, cd_size, cd_offset = z[7:10]
        directory_end_limit, zip64 = offset64, True
    else:
        need(entries != 65535 and cd_size != 0xffffffff and cd_offset != 0xffffffff,
             "ZIP64 sentinel without bounded locator")
    need(0 < entries <= caps.members and cd_size <= caps.central_bytes,
         "ZIP member/central-directory engineering cap")
    need(cd_offset + cd_size == directory_end_limit, "central-directory bounds/gap")
    stream.seek(0)
    return dict(file_bytes=size, member_count=entries, central_directory_bytes=cd_size,
                central_directory_offset=cd_offset, archive_comment_bytes=comment, zip64=zip64)


def inspect_stream(stream, caps=CAPS, tick=lambda: None):
    report = dict(members=[], member_inventory_complete=False, data_pickle=None,
                  errors=[], storage_payloads_read=False, all_member_CRC_verified=False)
    try:
        tick()
        metadata = zip_preflight(stream, caps)
        report["zip_metadata"] = metadata
        with zipfile.ZipFile(stream, "r") as archive:
            infos = archive.infolist()
            need(archive.start_dir == metadata["central_directory_offset"], "ZIP central-directory offset mismatch")
            need(len(infos) == metadata["member_count"], "directory count mismatch")
            names = Counter(i.filename for i in infos)
            declared = 0
            for i in infos:
                tick()
                issues = path_issues(i.orig_filename, caps.name_chars)
                mode = (i.external_attr >> 16) & 0xffff
                kind = stat.S_IFMT(mode)
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    issues.append("symlink_or_special_type")
                if names[i.filename] != 1:
                    issues.append("duplicate_member_name")
                if i.flag_bits & 1:
                    issues.append("encrypted_member")
                if i.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    issues.append("unsupported_compression_method")
                if not (0 <= i.header_offset < metadata["central_directory_offset"]):
                    issues.append("invalid_local_header_offset")
                if not (0 <= i.file_size <= caps.member_bytes and
                        0 <= i.compress_size <= metadata["file_bytes"]):
                    issues.append("member_size_cap")
                declared += i.file_size
                report["members"].append(dict(name=i.orig_filename, file_bytes=i.file_size,
                    compressed_bytes=i.compress_size, crc32=f"{i.CRC:08x}",
                    compression_method=i.compress_type, flags=i.flag_bits, external_attr=i.external_attr,
                    local_header_offset=i.header_offset, create_system=i.create_system,
                    issues=issues, CRC_verified=False, payload_read=False))
            report.update(member_inventory_complete=True, total_declared_bytes=declared)
            need(declared <= caps.total_declared, "total declared expansion cap")
            need(not any(x["issues"] for x in report["members"]), "ZIP metadata hazards; see member issues")
            candidates = [i for i in infos if not i.is_dir() and
                          (i.filename == "data.pkl" or i.filename.endswith("/data.pkl"))]
            need(len(candidates) == 1, "exactly one data.pkl required; no filename guessing")
            info = candidates[0]
            need(info.file_size <= caps.pickle_bytes and info.compress_size <= caps.pickle_bytes,
                 "data.pkl engineering byte cap")
            raw = bytearray()
            with archive.open(info, "r") as f:
                while True:
                    tick()
                    block = f.read(min(65536, caps.pickle_bytes - len(raw) + 1))
                    if not block:
                        break
                    need(len(raw) + len(block) <= caps.pickle_bytes and
                         len(raw) + len(block) <= info.file_size, "data.pkl expansion cap")
                    raw.extend(block)
            need(len(raw) == info.file_size, "data.pkl exact length")
            entry = next(x for x in report["members"] if x["name"] == info.orig_filename)
            entry.update(CRC_verified=True, payload_read=True)
            report["data_pickle"] = dict(member=info.orig_filename, bytes=len(raw),
                                         sha256=base.sha(raw), CRC_verified=True,
                                         scan=pickle_inventory(bytes(raw), caps, tick))
    except Exception as exc:
        report["errors"].append(type(exc).__name__ + ": " + str(exc)[:1000])
    scan = report["data_pickle"]["scan"] if report["data_pickle"] else None
    complete = (not report["errors"] and scan is not None and
                scan["opcode_inventory_complete"] and scan["global_resolution_complete"])
    report.update(status="static_inventory_recorded_not_trusted_loadable" if complete else
                  "static_inventory_incomplete_not_trusted_loadable", trusted_loading_allowed=False,
                  static_scan_proves_safety=False, deserialized=False, foreign_code_executed=False,
                  baseline_runnable=None, official_equivalence=False, scientific_admission=False)
    return report


def binding():
    with base.reader(Path(base.__file__)) as f:
        need(base.hash_stream(f)[1] == HELPER_SHA, "frozen helper source SHA")
    r = base.small(DECODE_REPORT, DECODE_REPORT_SHA)
    need(r["schema"] == base.NAME and r["status"] == "decoded_bytes_verified_not_trusted_loadable"
         and r["errors"] == [] and r["output"] == str(INPUT) and
         r["decoded_bytes"] == INPUT_BYTES and r["decoded_sha256"] == INPUT_SHA and
         r["decoded_crc32"] == "d59f865b" and r["deflate_eof"] is True and
         r["trailing_bytes"] == 0 and r["deserialized"] is False, "MAIN decode binding")
    need(r["provenance"] == base.provenance() and r["own_sha256"] == base.own_closure(),
         "decode metadata/source closure")
    return dict(input=str(INPUT), input_bytes=INPUT_BYTES, input_sha256=INPUT_SHA,
                decode_report=str(DECODE_REPORT), decode_report_sha256=DECODE_REPORT_SHA,
                helper_source_sha256=HELPER_SHA, author_object_sha256=None,
                prior_consume_receipt_not_required_for_static_scan=True)


def own_closure():
    return {str(p): base.sha(p.read_bytes()) for p in (
        Path(__file__), ROOT / "tests" / ("test_" + NAME + ".py"),
        ROOT / "innerdocs" / (NAME + ".md"))}


def run(wall_seconds=120):
    tick = base.deadline(wall_seconds)
    start = time.monotonic()
    bound, own = binding(), own_closure()
    with base.reader(INPUT) as f:
        before = os.fstat(f.fileno())
        need(before.st_size == INPUT_BYTES and base.hash_stream(f, tick) == (INPUT_BYTES, INPUT_SHA),
             "fixed decoded file size/SHA prepass")
        base.unchanged(f, INPUT, before)
        f.seek(0)
        report = inspect_stream(f, CAPS, tick)
        base.unchanged(f, INPUT, before)
    need(binding() == bound and own_closure() == own, "input/code closure changed")
    report.update(schema=NAME, binding=bound, own_sha256=own,
                  wall_seconds=time.monotonic() - start, engineering_caps=CAPS.__dict__,
                  gpu_calls=0, cpu_neural_calls=0, physics_calls=0, network_calls=0)
    need(len(base.encoded(report)) <= REPORT_CAP, "report cap; refusing silent truncation")
    return report


def save(path, report):
    """Exclusive new metadata only. Never writes model/data.pkl/storage files."""
    path = Path(path)
    need(path.is_absolute() and path.parent == ROOT / "results" and
         re.fullmatch(re.escape(NAME) + r"_[A-Za-z0-9_-]+\.json", path.name), "exclusive scoped report path")
    raw = base.encoded(report)
    need(len(raw) <= REPORT_CAP, "report cap")
    used = base.budget_bytes()
    for p in (ROOT / "results").glob(NAME + "_*.json"):
        need(not p.is_symlink() and p.is_file(), "report scope symlink/special file")
        used += p.stat().st_size
    base.capacity(used, shutil.disk_usage(path.parent).free, len(raw))
    fd = base.dir_fd(path.parent)
    try:
        with base.new_file(fd, path.name) as f:
            need(f.write(raw) == len(raw), "short report write")
            f.flush()
            os.fsync(f.fileno())
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(path), bytes=len(raw), sha256=base.sha(raw),
                status=report["status"], trusted_loading_allowed=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("plan", help="only saved JSON/text, never opens checkpoint")
    scan = subs.add_parser("scan", help="MAIN-only static byte inspection, no object loading")
    scan.add_argument("--output", type=Path, required=True)
    scan.add_argument("--wall-seconds", type=float, default=120)
    args = p.parse_args(argv)
    if args.command == "plan":
        result = dict(schema=NAME + ".plan", binding=binding(), own_sha256=own_closure(),
                      engineering_caps=CAPS.__dict__, real_checkpoint_scanned=False,
                      trusted_loading_allowed=False, scientific_admission=False)
        print(base.encoded(result).decode(), end="")
        return 0
    report = run(args.wall_seconds)
    print(base.encoded(save(args.output, report)).decode(), end="")
    return int(report["status"] != "static_inventory_recorded_not_trusted_loadable")


if __name__ == "__main__":
    raise SystemExit(main())
