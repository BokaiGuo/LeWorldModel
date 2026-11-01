"""Fixed-checkpoint data-only storage/NPY export; no pickle or model loader."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
from pathlib import Path
import re
import shutil
import struct
import time
import zipfile
import zlib

from experiments.spectral_reachability import agent_hilewm_inert_pickle_ir_20261028_v1 as ir

base, scan, ROOT = ir.base, ir.scan, ir.ROOT
NAME = "agent_hilewm_storage_npy_export_20261028_v1"
IR_PATH = ROOT / "results/agent_hilewm_inert_pickle_ir_20261028_v1_MAIN01.json"
IR_SHA = "709b31e28bee8039d577668cbd9613347be1d070a7619e3db3ca61e3674cabbe"
IR_BYTES = 6103904
IR_SOURCE_SHA = "6752c9eb0b4351e780a62198ab15948b4cb2bd181ebfd8325d07db253c8aec10"
BLOCK, ARRAY_CAP, JSON_CAP = 65536, 128 * 1024**2, 32 * 1024**2
DESCRIPTORS = {("torch", "FloatStorage"): ("f4", 4, "float32"),
               ("torch", "LongStorage"): ("i8", 8, "int64")}


def need(ok, message):
    base.need(ok, message)


def read_json(path, expected, cap):
    with base.reader(path) as f:
        before = os.fstat(f.fileno())
        need(before.st_size <= cap, "JSON byte cap")
        raw = f.read(cap + 1)
        need(len(raw) <= cap and base.sha(raw) == expected, "JSON SHA/cap")
        base.unchanged(f, path, before)
    return base.decode_json(raw)


def own_closure():
    return {str(p): base.sha(p.read_bytes()) for p in (
        Path(__file__), ROOT/"tests"/("test_"+NAME+".py"), ROOT/"innerdocs"/(NAME+".md"))}


def binding():
    with base.reader(Path(ir.__file__)) as f:
        need(base.hash_stream(f)[1] == IR_SOURCE_SHA, "frozen IR source SHA")
    saved = read_json(IR_PATH, IR_SHA, IR_BYTES)
    static = ir.binding()
    need(saved["schema"] == ir.NAME and saved["own_sha256"] == ir.own_closure(), "IR source closure")
    need(saved["status"] == "inert_metadata_recorded_not_loadable" and
         saved["input"] == str(scan.INPUT) and saved["input_sha256"] == scan.INPUT_SHA and
         saved["static_report_sha256"] == ir.STATIC_SHA and
         saved["data_pickle_sha256"] == ir.PICKLE_SHA and
         saved["trusted_loading_allowed"] is False and saved["foreign_code_executed"] is False,
         "IR binding/admission")
    # Traverse saved symbolic nodes again, NEVER parse or execute pickle.
    need(saved["nn"] == ir.nn_inventory(saved["ir"], static["members"], ir.PICKLE_NAME),
         "saved nn metadata must equal symbolic graph traversal")
    return saved, static


def integer(value):
    need(type(value) is int and 0 <= value <= 2**63-1, "bounded nonnegative integer")
    return value


def view_layout(meta, definition):
    need(meta["storage_type"] == definition["storage_type"] and
         meta["storage_numel"] == definition["storage_numel"] and
         meta["storage_bytes"] == definition["storage_bytes"] and
         meta["storage_member"] == definition["storage_member"] and
         meta["location"] == definition["location"], "view storage identity")
    dtype, itemsize, declaration = DESCRIPTORS[tuple(definition["storage_type"])]
    need(meta["dtype_declaration"] == declaration, "dtype declaration")
    numel, offset = integer(definition["storage_numel"]), integer(meta["storage_offset"])
    need(definition["storage_bytes"] == numel*itemsize <= ARRAY_CAP, "storage size cap")
    shape, stride = meta["shape"], meta["stride"]
    need(type(shape) is list and type(stride) is list and len(shape) == len(stride) <= 64,
         "shape/stride dimension cap")
    for value in shape + stride:
        integer(value)
    count = math.prod(shape)
    need(count*itemsize <= ARRAY_CAP, "logical view byte cap")
    high = offset + sum((s-1)*t for s, t in zip(shape, stride)) if count else offset
    need(offset <= numel if count == 0 else high < numel, "view range")
    return dtype, itemsize, count


def npy_header(descr, shape):
    """NPY v1, C-order plain numeric bytes; no object dtype/np.save/pickle."""
    need(descr in {"<f4", ">f4", "<i8", ">i8"}, "plain numeric descriptor")
    shape = tuple(integer(x) for x in shape)
    need(len(shape) <= 64, "NPY shape dimension cap")
    text = repr(dict(descr=descr, fortran_order=False, shape=shape)).encode("ascii")
    padding = (-(10 + len(text) + 1)) % 64
    raw = text + b" "*padding + b"\n"
    need(len(raw) < 65536, "NPY v1 header cap")
    return b"\x93NUMPY\x01\x00"+struct.pack("<H", len(raw))+raw


def view_blocks(data, meta, definition, tick=lambda: None):
    """C-index iteration copies exact original bits, including NaN and signed zero."""
    _, size, count = view_layout(meta, definition)
    need(len(data) == definition["storage_bytes"], "view storage byte count")
    if not count:
        return
    shape, stride, offset = meta["shape"], meta["stride"], meta["storage_offset"]
    expected, contiguous = 1, True
    for s, t in reversed(list(zip(shape, stride))):
        if s > 1 and t != expected:
            contiguous = False
        expected *= s
    if contiguous:
        for pos in range(offset*size, (offset+count)*size, BLOCK):
            tick()
            yield memoryview(data)[pos:min(pos+BLOCK, (offset+count)*size)]
    else:
        # O(dimensions) index memory: itertools.product would materialize huge ranges.
        index, block = [0]*len(shape), bytearray()
        for _ in range(count):
            if len(block) == 0:
                tick()
            pos = (offset+sum(i*t for i, t in zip(index, stride)))*size
            block.extend(memoryview(data)[pos:pos+size])
            if len(block) >= BLOCK:
                yield bytes(block)
                block.clear()
            for d in range(len(shape)-1, -1, -1):
                index[d] += 1
                if index[d] < shape[d]:
                    break
                index[d] = 0
        if block:
            yield bytes(block)


def validate_metadata(saved, static):
    nn = saved["nn"]
    need(nn["complete"] is True and nn["errors"] == [] and len(nn["entries"]) <= 20000,
         "complete bounded nn entries required")
    members = {x["name"]: x for x in static["members"]}
    need(len(members) == len(static["members"]), "duplicate static member")
    definitions, groups, paths = nn["storage_definitions"], {}, set()
    reserve = JSON_CAP + 1024**2  # manifest and bounded journal
    for key, definition in definitions.items():
        need(re.fullmatch(r"[A-Za-z0-9_-]+", key), "safe storage key")
        need(tuple(definition["storage_type"]) in DESCRIPTORS, "unsupported storage type")
        _, size, _ = DESCRIPTORS[tuple(definition["storage_type"])]
        n = integer(definition["storage_numel"])
        need(n*size == definition["storage_bytes"] <= ARRAY_CAP, "storage declared size")
        need(definition["storage_member"] == ir.PICKLE_NAME.rsplit("/", 1)[0]+"/data/"+key,
             "exact storage member identity")
        member = members[definition["storage_member"]]
        need(member["file_bytes"] == n*size and member["issues"] == [], "static storage size/issues")
        reserve += n*size + 512
        groups[key] = []
    for index, row in enumerate(nn["entries"]):
        need(row["collection"] in {"_parameters", "_buffers"} and type(row["module_path"]) is list
             and all(type(p) is str for p in row["module_path"]+[row["name"]]), "literal nn path")
        path = tuple(row["module_path"]+[row["collection"], row["name"]])
        need(path not in paths, "duplicate nn path")
        paths.add(path)
        if row["status"] == "explicit_none":
            need(row["metadata"] is None, "None metadata")
            continue
        need(row["status"] == "literal_tensor_metadata", "unknown entry refused")
        meta = row["metadata"]
        key = meta["storage_key"]
        _, size, count = view_layout(meta, definitions[key])
        need(row["node"] == meta["node"], "entry node identity")
        groups[key].append(index)
        reserve += count*size + 512
    need(all(groups.values()), "unreferenced storage definition")
    aliases, node_aliases = {}, {}
    for key, indices in groups.items():
        for i in indices:
            row = nn["entries"][i]
            pointer = {k: row[k] for k in ("module_path", "collection", "name")}
            aliases.setdefault(key, []).append(pointer)
            node_aliases.setdefault(str(row["node"]), []).append(pointer)
    need({k:v for k,v in aliases.items() if len(v)>1} == nn["storage_alias_paths"] and
         {k:v for k,v in node_aliases.items() if len(v)>1} == nn["node_alias_paths"], "all alias paths")
    data_names = {m for m in members if m.startswith(ir.PICKLE_NAME.rsplit("/",1)[0]+"/data/")}
    need(data_names == {d["storage_member"] for d in definitions.values()}, "all ZIP storage members accounted")
    return groups, reserve


def state_dict_paths(saved):
    """Read literal persistence flags, not nn.state_dict()/class hooks or methods."""
    nodes, modules = saved["ir"]["nodes"], {}

    def mapping(index):
        n = nodes[index]
        need(n["kind"] == "dict" or n.get("container") == "ordered_dict", "literal state map")
        result = {}
        for key, value in n["entries"]:
            k = nodes[key]
            need(k["kind"] == "str" and k["value"] not in result, "literal unique state key")
            result[k["value"]] = value
        return result

    def strings(index):
        n = nodes[index]
        need(n["kind"] in {"list", "tuple", "set", "frozenset"}, "literal persistence set")
        result = []
        for i in n["items"]:
            need(nodes[i]["kind"] == "str", "literal persistence name")
            result.append(nodes[i]["value"])
        need(len(set(result)) == len(result), "duplicate persistence name")
        return result

    for module in saved["nn"]["modules"]:
        if module.get("explicit_none"):
            continue
        states = nodes[module["node"]]["states"]
        need(len(states) == 1, "module BUILD state count")
        state = nodes[states[0]]
        parts = state["items"] if state["kind"] == "tuple" else states
        fields = {}
        for part in parts:
            if nodes[part]["kind"] == "none":
                continue
            values = mapping(part)
            need(not fields.keys() & values.keys(), "duplicate state field")
            fields.update(values)
        need("_non_persistent_buffers_set" in fields, "missing persistence flags; no default")
        node = nodes[fields["_non_persistent_buffers_set"]]
        if node["kind"] == "reduce":
            symbol, args = nodes[node["callable"]], nodes[node["args"]]
            need(symbol["kind"] == "global" and symbol["module"] in {"builtins", "__builtin__"}
                 and symbol["name"] == "set" and args["kind"] == "tuple" and len(args["items"]) == 1
                 and not node["states"], "literal set reducer layout; never executed")
            names = strings(args["items"][0])
        else:
            names = strings(node["id"])
        buffer_names = mapping(fields["_buffers"])
        need(set(names) <= buffer_names.keys(), "nonpersistent names absent from buffers")
        modules[tuple(module["path"])] = names
    rows, flat = [], set()
    for i, row in enumerate(saved["nn"]["entries"]):
        nonpersistent = row["collection"] == "_buffers" and row["name"] in modules[tuple(row["module_path"])]
        components = row["module_path"]+[row["name"]]
        need(all(x and "." not in x for x in components), "ambiguous dotted state_dict key")
        key = ".".join(components)
        need(key not in flat, "duplicate flattened state_dict key")
        flat.add(key)
        rows.append(dict(index=i, key=key, include_in_literal_state_dict=row["metadata"] is not None
                         and not nonpersistent, nonpersistent_buffer=nonpersistent))
    return rows


def budget_bytes():
    used = base.budget_bytes()
    for p in (ROOT/"results").iterdir():
        if p.name.startswith((scan.NAME+"_", ir.NAME+"_", NAME+"_")):
            need(not p.is_symlink(), "budget symlink")
            if p.is_dir():
                used += base.scope_bytes(p)
            else:
                with base.reader(p) as f:
                    used += os.fstat(f.fileno()).st_size
    return used


def read_member(z, member, cap, tick=lambda: None):
    info = z.getinfo(member["name"])
    need(info.file_size == member["file_bytes"] <= cap and f"{info.CRC:08x}" == member["crc32"]
         and info.compress_type == member["compression_method"] and info.flag_bits == member["flags"]
         and info.header_offset == member["local_header_offset"], "ZIP member metadata")
    data, crc = bytearray(), 0
    with z.open(info) as f:
        while True:
            tick()
            block = f.read(min(BLOCK, cap-len(data)+1))
            if not block:
                break
            data.extend(block)
            need(len(data) <= cap, "member byte cap")
            crc = zlib.crc32(block, crc)
    need(len(data) == member["file_bytes"] and f"{crc & 0xffffffff:08x}" == member["crc32"],
         "ZIP member CRC/length")
    return data


def emit_array(directory, filename, descr, shape, blocks, verify=False):
    need(re.fullmatch(r"(?:storage|entry)_[0-9]{4,5}\.npy", filename),
         "exclusive numeric array filename")
    header, digest, values, size, value_size = npy_header(descr, shape), hashlib.sha256(), hashlib.sha256(), 0, 0
    path = directory/filename
    if verify:
        stream = base.reader(path)
    else:
        fd = base.dir_fd(directory)
        try:
            stream = base.new_file(fd, filename)
        finally:
            os.close(fd)
    with stream as f:
        before = os.fstat(f.fileno())
        for n, block in enumerate(itertools.chain((header,), blocks)):
            digest.update(block)
            size += len(block)
            if n:
                values.update(block)
                value_size += len(block)
            if verify:
                need(f.read(len(block)) == block, "saved NPY differs from original storage/view")
            else:
                need(f.write(block) == len(block), "short NPY write")
        need(value_size == math.prod(shape)*int(descr[-1]), "NPY logical byte length")
        if verify:
            need(f.read(1) == b"", "NPY trailing bytes")
            base.unchanged(f, path, before)
        else:
            f.flush()
            os.fsync(f.fileno())
            base.unchanged(f, path, os.fstat(f.fileno()))
    return dict(file=filename, file_bytes=size, file_sha256=digest.hexdigest(),
                value_bytes=value_size, value_sha256=values.hexdigest(), descr=descr, shape=list(shape),
                fortran_order=False, value_hash_order="C-index exact source-endian bytes")


def write_json(directory, name, value, cap=JSON_CAP):
    raw = base.encoded(value)
    need(len(raw) <= cap, "output JSON cap")
    base.capacity(budget_bytes(), shutil.disk_usage(ROOT/"results").free, len(raw))
    fd = base.dir_fd(directory)
    try:
        with base.new_file(fd, name) as f:
            need(f.write(raw) == len(raw), "short JSON write")
            f.flush()
            os.fsync(f.fileno())
        os.fsync(fd)
    finally:
        os.close(fd)
    return base.sha(raw)


def output_scope(path):
    path = Path(path)
    need(path.is_absolute() and path.parent == ROOT/"results" and
         re.fullmatch(re.escape(NAME)+r"_[A-Za-z0-9_-]+", path.name), "exclusive output directory")
    return path


def run(output, wall_seconds=300, verify_sha=None):
    tick, start = base.deadline(wall_seconds), time.monotonic()
    output = output_scope(output)
    saved, static = binding()
    own = own_closure()
    groups, reserve = validate_metadata(saved, static)
    key_paths = state_dict_paths(saved)
    verify = verify_sha is not None
    old = read_json(output/"report.json", verify_sha, JSON_CAP) if verify else None
    if not verify:
        base.capacity(budget_bytes(), shutil.disk_usage(ROOT/"results").free, reserve)
        parent = base.dir_fd(output.parent)
        try:
            os.mkdir(output.name, mode=0o700, dir_fd=parent)
            os.fsync(parent)
        finally:
            os.close(parent)
    rows, storages, journal, errors = [], [], [], []
    try:
        with base.reader(scan.INPUT) as source:
            before = os.fstat(source.fileno())
            need(base.hash_stream(source, tick) == (scan.INPUT_BYTES, scan.INPUT_SHA), "original ZIP SHA/size")
            base.unchanged(source, scan.INPUT, before)
            source.seek(0)
            scan.zip_preflight(source, scan.CAPS)
            with zipfile.ZipFile(source) as z:
                members = {x["name"]:x for x in static["members"]}
                need([x.filename for x in z.infolist()] == list(members), "exact ZIP inventory")
                order_name = ir.PICKLE_NAME.rsplit("/",1)[0]+"/byteorder"
                order_bytes = read_member(z, members[order_name], 6, tick)
                need(order_bytes in (b"little", b"big"), "explicit byteorder required; no native default")
                order = order_bytes.decode("ascii")
                prefix = "<" if order == "little" else ">"
                for ordinal, (key, indices) in enumerate(groups.items()):
                    tick()
                    definition = saved["nn"]["storage_definitions"][key]
                    journal.append(dict(storage_key=key, operation="storage_crc_and_arrays", phase="attempt"))
                    if not verify:
                        base.capacity(budget_bytes(), shutil.disk_usage(ROOT/"results").free,
                                      definition["storage_bytes"]+512+sum(
                                          math.prod(saved["nn"]["entries"][i]["metadata"]["shape"])*
                                          DESCRIPTORS[tuple(definition["storage_type"])][1]+512 for i in indices))
                    data = read_member(z, members[definition["storage_member"]], ARRAY_CAP, tick)
                    dtype = prefix+DESCRIPTORS[tuple(definition["storage_type"])][0]
                    storage = emit_array(output, f"storage_{ordinal:04d}.npy", dtype,
                                         [definition["storage_numel"]],
                                         (memoryview(data)[i:i+BLOCK] for i in range(0,len(data),BLOCK)), verify)
                    storages.append(dict(storage_key=key, definition=definition, array=storage,
                                         crc32=members[definition["storage_member"]]["crc32"],
                                         crc_verified=True, aliases=indices))
                    for index in indices:
                        tick()
                        row = saved["nn"]["entries"][index]
                        meta = row["metadata"]
                        array = emit_array(output, f"entry_{index:04d}.npy", dtype, meta["shape"],
                                           view_blocks(data, meta, definition, tick), verify)
                        rows.append(dict(index=index, source_entry=row, array=array,
                                         original_shape_stride_offset_preserved_in_source_entry=True))
                    del data
                    journal.append(dict(storage_key=key, operation="storage_crc_and_arrays", phase="return"))
                for index, row in enumerate(saved["nn"]["entries"]):
                    if row["metadata"] is None:
                        rows.append(dict(index=index, source_entry=row, array=None))
            base.unchanged(source, scan.INPUT, before)
            source.seek(0)
            need(base.hash_stream(source, tick) == (scan.INPUT_BYTES, scan.INPUT_SHA), "ZIP post-export SHA")
            base.unchanged(source, scan.INPUT, before)
        need(binding() == (saved, static) and own_closure() == own, "source metadata closure changed")
        rows.sort(key=lambda r:r["index"])
        need(len(rows) == len(saved["nn"]["entries"]), "complete entry denominator")
        package = dict(schema=NAME, status="literal_nn_data_exported_not_model_equivalence",
                       ir_path=str(IR_PATH), ir_sha256=IR_SHA, zip_sha256=scan.INPUT_SHA,
                       zip_path=str(scan.INPUT), static_report_sha256=ir.STATIC_SHA,
                       own_sha256=own, byteorder=order, byteorder_member=members[order_name],
                       byteorder_crc_verified=True, rows=rows, storages=storages,
                       modules=saved["nn"]["modules"], module_alias_paths=saved["nn"]["module_alias_paths"],
                       node_alias_paths=saved["nn"]["node_alias_paths"],
                       storage_alias_paths=saved["nn"]["storage_alias_paths"],
                       state_dict_paths=key_paths, class_state_dict_hooks_executed=False,
                       custom_extra_state_and_class_version_not_inferred=True,
                       runtime_state_dict_equivalence=False, low_high_pairing_verified=False,
                       trusted_pickle_loading_allowed=False, scientific_admission=False,
                       pickle_executed=False, foreign_code_executed=False,
                       gpu_calls=0, physics_calls=0, all_storage_crc_verified=True)
        if verify:
            for key, value in package.items():
                need(old[key] == value, "saved manifest mismatch: "+key)
            need(old["errors"] == [], "successful saved export required")
            need(old["journal"] == journal and old["planned_upper_new_bytes"] == reserve,
                 "exact attempt/return ledger and reservation")
            expected_files = {"report.json"} | {s["array"]["file"] for s in storages} | {
                r["array"]["file"] for r in rows if r["array"] is not None}
            need({p.name for p in output.iterdir()} == expected_files, "unexpected/missing bundle files")
            return dict(status="saved_data_verified_not_model_equivalence", report_sha256=verify_sha,
                        arrays_verified=len(storages)+sum(r["array"] is not None for r in rows),
                        ir_sha256=IR_SHA, trusted_pickle_loading_allowed=False, scientific_admission=False)
        package.update(errors=[], journal=journal, wall_seconds=time.monotonic()-start,
                       planned_upper_new_bytes=reserve)
        digest = write_json(output, "report.json", package)
        return dict(path=str(output/"report.json"), sha256=digest, status=package["status"])
    except Exception as exc:
        errors.append(dict(type=type(exc).__name__, message=str(exc)[:2000]))
        if not verify:
            write_json(output, "failure.json", dict(schema=NAME, status="failed_partial_files_preserved",
                       ir_sha256=IR_SHA, zip_path=str(scan.INPUT), expected_zip_sha256=scan.INPUT_SHA,
                       own_sha256=own, errors=errors, journal=journal, rows=rows, storages=storages,
                       wall_seconds=time.monotonic()-start, scientific_admission=False))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="saved JSON only; does not open production ZIP/storage")
    for command in ("export", "verify"):
        c = sub.add_parser(command)
        c.add_argument("--output", required=True, type=Path)
        c.add_argument("--wall-seconds", type=float, default=300)
        if command == "verify":
            c.add_argument("--report-sha256", required=True)
    a = p.parse_args(argv)
    if a.command == "plan":
        saved, static = binding()
        groups, reserve = validate_metadata(saved, static)
        paths = state_dict_paths(saved)
        used, free = budget_bytes(), shutil.disk_usage(ROOT/"results").free
        base.capacity(used, free, reserve)
        result = dict(schema=NAME+".plan", ir_sha256=IR_SHA, storage_count=len(groups),
                      entries=len(saved["nn"]["entries"]), planned_upper_new_bytes=reserve,
                      literal_state_dict_entries=sum(x["include_in_literal_state_dict"] for x in paths),
                      used_bytes=used, free_bytes=free, storage_payloads_read=False,
                      actual_byteorder=None, model_equivalence=False)
    else:
        result = run(a.output, a.wall_seconds, a.report_sha256 if a.command == "verify" else None)
    print(base.encoded(result).decode(), end="")


if __name__ == "__main__":
    main()
