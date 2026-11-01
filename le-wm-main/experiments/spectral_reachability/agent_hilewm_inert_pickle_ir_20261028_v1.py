"""Inert pickle IR and literal nn/Tensor metadata, not a Python object loader."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import io
import os
from pathlib import Path
import pickletools
import re
import shutil
import time
import zipfile

from experiments.spectral_reachability import agent_hilewm_pickle_static_inventory_20261028_v1 as scan

base = scan.base
NAME = "agent_hilewm_inert_pickle_ir_20261028_v1"
ROOT = scan.ROOT
SCANNER_SHA = "b4fa5b3304a7939ee573f57c16108f839a92fe73bac70d58fc91c33fa2daea67"
STATIC_REPORT = ROOT / "results/agent_hilewm_pickle_static_inventory_20261028_v1_MAIN01.json"
STATIC_SHA = "7c2658ea69caae3117a376d9804357b851d44b4e7babbed9fa3f91fe0fdde5c2"
STATIC_BYTES = 6392458
PICKLE_NAME = "hi_lewm_p2_train_hope2_22253175_epoch_15_object/data.pkl"
PICKLE_BYTES = 170471
PICKLE_SHA = "df11e246e2b0e8cd7ab654a4bde0f3e1f090fcf567168f5dd44e73afabd8d480"
REPORT_CAP = 48 * 1024**2
MARK = -1
# Fixed data-layout recognizers, NEVER an import/unpickle allowlist.
RECOGNIZED_REDUCERS = {("collections", "OrderedDict"), ("__builtin__", "set"),
                       ("builtins", "set"), ("torch._utils", "_rebuild_tensor_v2"),
                       ("torch._utils", "_rebuild_parameter")}
STORAGE_TYPES = {("torch", "FloatStorage"): ("float32", 4),
                 ("torch", "LongStorage"): ("int64", 8)}


@dataclass(frozen=True)
class Caps:
    raw_bytes: int = 16 * 1024**2
    nodes: int = 100000
    opcodes: int = 200000
    memo: int = 200000
    edges: int = 500000
    literal_bytes: int = 65536
    paths: int = 20000
    depth: int = 128


CAPS = Caps()


def need(ok, message):
    base.need(ok, message)


def parse_ir(raw, caps=CAPS, tick=lambda: None):
    """Record node IDs, literal containers and call syntax; never invoke calls."""
    need(len(raw) <= caps.raw_bytes, "pickle byte cap")
    nodes, stack, memo, declarations, errors, semantic_unknowns = [], [], {}, [], [], []
    count, edges, stopped = 0, 0, False
    opcounts = Counter()

    def node(kind, pos, **fields):
        need(len(nodes) < caps.nodes, "IR node cap")
        value = dict(id=len(nodes), kind=kind, position=pos, **fields)
        nodes.append(value)
        return value["id"]

    def link(n):
        nonlocal edges
        edges += n
        need(edges <= caps.edges, "IR edge cap")

    def take(n):
        need(len(stack) >= n and all(x != MARK for x in stack[-n:]) if n else True,
             "IR stack underflow/mark")
        out = stack[-n:] if n else []
        if n:
            del stack[-n:]
        return out

    def marked():
        need(MARK in stack, "missing IR MARK")
        i = len(stack) - 1 - stack[::-1].index(MARK)
        values = stack[i + 1:]
        del stack[i:]
        return values

    def string(i):
        n = nodes[i]
        need(n["kind"] == "str", "literal string required")
        return n["value"]

    def symbol(pos, opcode, module, name):
        need(module and name and len(module) <= caps.literal_bytes and len(name) <= caps.literal_bytes,
             "GLOBAL string bound")
        declarations.append(dict(position=pos, opcode=opcode, module=module, name=name))
        return node("global", pos, module=module, name=name)

    def global_lines(pos, opcode):
        mid, end = raw.find(b"\n", pos+1), -1
        if mid >= 0:
            end = raw.find(b"\n", mid+1)
        need(mid >= 0 and end >= 0, "GLOBAL literal lines")
        return symbol(pos, opcode, raw[pos+1:mid].decode("utf-8"), raw[mid+1:end].decode("utf-8"))

    def setitems(target, values):
        need(len(values) % 2 == 0, "odd dictionary pairs")
        n = nodes[target]
        need(n["kind"] == "dict" or n.get("container") == "ordered_dict",
             "SETITEM target not a supported inert mapping")
        link(len(values))
        n["entries"].extend([values[i:i+2] for i in range(0, len(values), 2)])

    try:
        for op, arg, pos in pickletools.genops(raw):
            tick()
            count += 1
            need(count <= caps.opcodes, "IR opcode cap")
            name = op.name
            opcounts[name] += 1
            if name == "PROTO":
                need(pos == 0 and 0 <= arg <= 5, "unsupported/misplaced PROTO")
            elif name == "FRAME":
                need(0 <= arg <= len(raw)-pos-9, "FRAME bound")
            elif name == "MARK":
                stack.append(MARK)
            elif name == "POP":
                need(stack, "POP underflow")
                stack.pop()
            elif name == "POP_MARK":
                marked()
            elif name == "DUP":
                need(stack and stack[-1] != MARK, "DUP underflow")
                stack.append(stack[-1])
            elif name in {"PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"}:
                need(stack and stack[-1] != MARK, "memo value missing")
                index = len(memo) if name == "MEMOIZE" else int(arg)
                need(0 <= index < caps.memo and len(memo) < caps.memo, "memo cap/index")
                memo[index] = stack[-1]
            elif name in {"GET", "BINGET", "LONG_BINGET"}:
                need(int(arg) in memo, "unknown memo index")
                stack.append(memo[int(arg)])
            elif name in {"UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"}:
                need(len(arg.encode("utf-8")) <= caps.literal_bytes, "literal string cap")
                stack.append(node("str", pos, value=arg))
            elif name in {"STRING", "BINSTRING", "SHORT_BINSTRING", "BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8"}:
                # Protocol-0 STRING / old BINSTRING decoding depends on loader
                # encoding. Preserve raw syntax, not an invented Python str.
                need(len(arg) <= caps.literal_bytes, "literal byte/string cap")
                fields = dict(opcode=name)
                if isinstance(arg, str):
                    fields["parsed_text"] = arg
                else:
                    fields["hex"] = bytes(arg).hex()
                stack.append(node("encoded_literal", pos, **fields))
            elif name in {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "NEWTRUE", "NEWFALSE"}:
                need(type(arg) in (int, bool) or name in {"NEWTRUE", "NEWFALSE"}, "integer literal")
                value = name == "NEWTRUE" if name in {"NEWTRUE", "NEWFALSE"} else arg
                # Decimal strings avoid JSON integer conversion caps without
                # losing the exact arbitrary-width literal value.
                stack.append(node("bool" if type(value) is bool else "int", pos,
                                  value=value if type(value) is bool else str(value)))
            elif name in {"FLOAT", "BINFLOAT"}:
                stack.append(node("float", pos, hex=arg.hex()))
            elif name == "NONE":
                stack.append(node("none", pos))
            elif name in {"EMPTY_TUPLE", "EMPTY_LIST", "EMPTY_DICT", "EMPTY_SET"}:
                kind = {"EMPTY_TUPLE": "tuple", "EMPTY_LIST": "list", "EMPTY_DICT": "dict", "EMPTY_SET": "set"}[name]
                stack.append(node(kind, pos, **({"entries": []} if kind == "dict" else {"items": []})))
            elif name in {"TUPLE", "TUPLE1", "TUPLE2", "TUPLE3", "LIST", "DICT", "FROZENSET"}:
                values = take(int(name[-1])) if name in {"TUPLE1", "TUPLE2", "TUPLE3"} else marked()
                kind = "tuple" if name.startswith("TUPLE") else name.lower()
                link(len(values))
                if kind == "dict":
                    need(len(values) % 2 == 0, "odd DICT")
                    stack.append(node(kind, pos, entries=[values[i:i+2] for i in range(0, len(values), 2)]))
                else:
                    stack.append(node(kind, pos, items=values))
            elif name in {"APPEND", "APPENDS", "ADDITEMS"}:
                values = take(1) if name == "APPEND" else marked()
                need(stack and stack[-1] != MARK and nodes[stack[-1]]["kind"] in ({"set"} if name == "ADDITEMS" else {"list"}),
                     "literal container mutation target")
                link(len(values))
                nodes[stack[-1]]["items"].extend(values)
            elif name in {"SETITEM", "SETITEMS"}:
                values = take(2) if name == "SETITEM" else marked()
                need(stack and stack[-1] != MARK, "SETITEM mapping missing")
                setitems(stack[-1], values)
            elif name == "GLOBAL":
                stack.append(global_lines(pos, name))
            elif name == "STACK_GLOBAL":
                mod, cls = take(2)
                stack.append(symbol(pos, name, string(mod), string(cls)))
            elif name in {"REDUCE", "NEWOBJ", "NEWOBJ_EX"}:
                refs = take(3 if name == "NEWOBJ_EX" else 2)
                cls, args = refs[:2]
                need(nodes[args]["kind"] == "tuple", "constructor args must be literal tuple")
                link(len(refs))
                fields = dict(callable=cls, args=args, states=[], executed=False)
                if name == "NEWOBJ_EX":
                    fields["kwargs"] = refs[2]
                callable_node = nodes[cls]
                sym = (callable_node.get("module"), callable_node.get("name"))
                if name == "REDUCE":
                    if callable_node["kind"] != "global" or sym not in RECOGNIZED_REDUCERS:
                        semantic_unknowns.append(dict(position=pos, callable_node=cls,
                                                      reason="unknown REDUCE; retained opaque, never invoked"))
                    elif sym == ("collections", "OrderedDict"):
                        need(not nodes[args]["items"], "only zero-argument OrderedDict syntax supported")
                        fields.update(container="ordered_dict", entries=[])
                else:
                    need(callable_node["kind"] == "global", "NEWOBJ class must be literal global symbol")
                stack.append(node(name.lower(), pos, **fields))
            elif name == "BUILD":
                state, = take(1)
                need(stack and stack[-1] != MARK and nodes[stack[-1]]["kind"] in {"newobj", "newobj_ex", "reduce"}, "BUILD target unsupported")
                link(1)
                nodes[stack[-1]]["states"].append(state)
            elif name in {"BINPERSID", "PERSID"}:
                if name == "BINPERSID":
                    pid, = take(1)
                else:
                    need(len(arg) <= caps.literal_bytes, "persistent literal cap")
                    pid = node("str", pos, value=arg)
                link(1)
                stack.append(node("persistent_ref", pos, pid=pid, resolved=False))
            elif name == "STOP":
                need(len(stack) == 1 and stack[0] != MARK and pos+1 == len(raw), "STOP stack/trailing bytes")
                stopped = True
            else:
                raise ValueError("unsupported inert opcode " + name)
            need(len(stack) <= caps.nodes, "IR stack cap")
    except Exception as exc:
        errors.append(type(exc).__name__ + ": " + str(exc)[:1000])
    if not stopped:
        errors.append("complete STOP not observed")
    return dict(nodes=nodes, root_node=stack[0] if stopped and not errors else None,
                declarations=declarations, opcode_counts=dict(sorted(opcounts.items())), opcode_count=count,
                memo_entries=len(memo), memo_table={str(k): v for k, v in sorted(memo.items())},
                edges=edges, complete=stopped and not errors,
                errors=errors, unknown_reducers=semantic_unknowns, constructors_executed=False)


def nn_inventory(ir, members, pickle_name, caps=CAPS, tick=lambda: None):
    """Literal nn-state paths and fixed Tensor layout records, not tensor values."""
    nodes = ir["nodes"]
    module_rows, rows, errors, storage_definitions = [], [], [], {}
    member_map = {x["name"]: x for x in members}

    def literal(i, kind):
        n = nodes[i]
        need(n["kind"] == kind, "expected literal " + kind)
        return int(n["value"]) if kind == "int" else n.get("value")

    def mapping(i):
        n = nodes[i]
        need(n["kind"] == "dict" or n.get("container") == "ordered_dict", "literal mapping required")
        out = {}
        for k, v in n["entries"]:
            key = literal(k, "str")
            need(key not in out, "duplicate state key; no guessed overwrite semantics")
            out[key] = v
        return out

    def state(i):
        n = nodes[i]
        need(n["kind"] in {"newobj", "newobj_ex"} and len(n["states"]) == 1,
             "exact one inert NEWOBJ BUILD state required")
        s = nodes[n["states"][0]]
        if s["kind"] == "tuple":
            need(len(s["items"]) == 2, "BUILD state tuple layout")
            out = {}
            for part in s["items"]:
                if nodes[part]["kind"] == "none":
                    continue
                fields = mapping(part)
                need(not set(out).intersection(fields), "duplicate BUILD state field")
                out.update(fields)
            return out
        return mapping(s["id"])

    def symbol(i):
        n = nodes[i]
        need(n["kind"] == "global", "literal callable symbol required")
        return n["module"], n["name"]

    def sequence(i):
        n = nodes[i]
        need(n["kind"] == "tuple", "literal tuple required")
        return n["items"]

    def tensor(i):
        original, parameter_grad, parameter_hooks = i, None, None
        n = nodes[i]
        if n["kind"] == "reduce" and symbol(n["callable"]) == ("torch._utils", "_rebuild_parameter"):
            args = sequence(n["args"])
            need(len(args) == 3, "parameter syntax arity")
            i, parameter_grad, parameter_hooks = args[0], literal(args[1], "bool"), args[2]
            n = nodes[i]
        need(n["kind"] == "reduce" and symbol(n["callable"]) == ("torch._utils", "_rebuild_tensor_v2"),
             "unknown Tensor callable; not executed")
        args = sequence(n["args"])
        need(len(args) == 6, "only six-argument _rebuild_tensor_v2 layout supported")
        storage = nodes[args[0]]
        need(storage["kind"] == "persistent_ref", "literal storage reference required")
        pid = sequence(storage["pid"])
        need(len(pid) == 5 and literal(pid[0], "str") == "storage", "storage PID layout")
        kind = symbol(pid[1])
        need(kind in STORAGE_TYPES, "unknown storage type; no dynamic resolver")
        key, location, numel = literal(pid[2], "str"), literal(pid[3], "str"), literal(pid[4], "int")
        need(re.fullmatch(r"[A-Za-z0-9_-]+", key) and numel >= 0, "storage key/length")
        offset = literal(args[1], "int")
        shape = [literal(x, "int") for x in sequence(args[2])]
        stride = [literal(x, "int") for x in sequence(args[3])]
        need(len(shape) == len(stride) <= 64 and offset >= 0 and all(x >= 0 for x in shape+stride),
             "shape/stride/offset layout")
        empty = any(x == 0 for x in shape)
        highest = offset if empty else offset + sum((s-1)*t for s, t in zip(shape, stride))
        need((offset <= numel if empty else highest < numel), "view exceeds declared storage")
        dtype, itemsize = STORAGE_TYPES[kind]
        member_name = pickle_name.rsplit("/", 1)[0] + "/data/" + key
        need(member_name in member_map and member_map[member_name]["file_bytes"] == numel*itemsize,
             "storage member identity/declared byte size")
        definition = dict(storage_type=list(kind), storage_numel=numel, location=location,
                          storage_member=member_name, storage_bytes=numel*itemsize)
        need(key not in storage_definitions or storage_definitions[key] == definition,
             "conflicting storage-key declarations")
        storage_definitions[key] = definition
        return dict(node=original, tensor_node=i, storage_key=key, storage_ref_node=storage["id"],
                    dtype_declaration=dtype, storage_offset=offset, shape=shape, stride=stride,
                    tensor_requires_grad=literal(args[4], "bool"), parameter_requires_grad=parameter_grad,
                    hooks_node=args[5], parameter_hooks_node=parameter_hooks,
                    storage_crc_verified=False, tensor_values_read=False, **definition)

    if not ir["complete"]:
        return dict(complete=False, errors=["IR incomplete; no parameter completeness claim"], modules=[], entries=[])
    todo = [(ir["root_node"], [], frozenset())]
    while todo:
        tick()
        module, path, ancestors = todo.pop()
        try:
            need(len(module_rows) < caps.paths and len(rows) < caps.paths and len(todo) < caps.paths,
                 "nn path count cap")
            need(len(path) <= caps.depth and module not in ancestors, "module depth/cycle")
            fields = state(module)
            cls = nodes[module]["callable"]
            module_rows.append(dict(path=path, node=module, class_declaration=list(symbol(cls))))
            need(all(k in fields for k in ("_modules", "_parameters", "_buffers")),
                 "nn state missing explicit _modules/_parameters/_buffers")
            for collection in ("_parameters", "_buffers"):
                for name, value in mapping(fields[collection]).items():
                    tick()
                    need(len(rows) < caps.paths, "nn parameter path count cap")
                    entry = dict(module_path=path, collection=collection, name=name, node=value)
                    if nodes[value]["kind"] == "none":
                        entry.update(status="explicit_none", metadata=None)
                    else:
                        try:
                            entry.update(status="literal_tensor_metadata", metadata=tensor(value))
                        except Exception as exc:
                            entry.update(status="unknown", metadata=None, error=str(exc)[:1000])
                            errors.append(dict(path=path+[collection, name], reason=str(exc)[:1000]))
                    rows.append(entry)
            for name, child in reversed(list(mapping(fields["_modules"]).items())):
                if nodes[child]["kind"] == "none":
                    module_rows.append(dict(path=path+[name], node=child, explicit_none=True))
                else:
                    todo.append((child, path+[name], ancestors | {module}))
        except Exception as exc:
            errors.append(dict(path=path, reason=type(exc).__name__ + ": " + str(exc)[:1000]))
            if "cap" in str(exc) or "deadline" in str(exc):
                break
    aliases, storage_aliases, module_aliases = {}, {}, {}
    for row in module_rows:
        if not row.get("explicit_none"):
            module_aliases.setdefault(str(row["node"]), []).append(row["path"])
    for row in rows:
        if row["metadata"] is not None:
            pointer = dict(module_path=row["module_path"], collection=row["collection"], name=row["name"])
            aliases.setdefault(str(row["node"]), []).append(pointer)
            storage_aliases.setdefault(row["metadata"]["storage_key"], []).append(pointer)
    return dict(complete=not errors and not ir["unknown_reducers"], errors=errors, modules=module_rows,
                entries=rows, storage_definitions=storage_definitions,
                node_alias_paths={k: v for k, v in aliases.items() if len(v) > 1},
                storage_alias_paths={k: v for k, v in storage_aliases.items() if len(v) > 1},
                module_alias_paths={k: v for k, v in module_aliases.items() if len(v) > 1},
                tensor_pairing_verified=False, state_dict_extracted=False, storage_payloads_read=False,
                module_labels_are_literal_not_low_high_classification=True)


def build(raw, members, pickle_name=PICKLE_NAME, caps=CAPS, tick=lambda: None):
    ir = parse_ir(raw, caps, tick)
    parameters = nn_inventory(ir, members, pickle_name, caps, tick)
    return dict(ir=ir, nn=parameters, status="inert_metadata_recorded_not_loadable" if
                ir["complete"] and parameters["complete"] else "inert_metadata_incomplete_not_loadable",
                deserialized=False, foreign_code_executed=False, constructors_executed=False,
                trusted_loading_allowed=False, tensor_pairing_verified=False,
                state_dict_extracted=False, baseline_runnable=None, scientific_admission=False)


def static_metadata(path, expected, cap=STATIC_BYTES):
    """Bounded saved JSON only; the opcode inventory exceeds decoder's 2MiB cap."""
    need(0 < cap <= STATIC_BYTES, "static metadata cap")
    with base.reader(path) as f:
        before = os.fstat(f.fileno())
        need(before.st_size <= cap, "static metadata byte cap")
        raw = f.read(cap + 1)
        need(len(raw) <= cap and base.sha(raw) == expected, "static metadata SHA/cap")
        base.unchanged(f, path, before)
    return base.decode_json(raw)


def binding():
    with base.reader(Path(scan.__file__)) as f:
        need(base.hash_stream(f)[1] == SCANNER_SHA, "frozen scanner source SHA")
    saved = static_metadata(STATIC_REPORT, STATIC_SHA)
    need(saved["binding"] == scan.binding() and saved["own_sha256"] == scan.own_closure(), "static closure")
    need(saved["status"] == "static_inventory_recorded_not_trusted_loadable" and saved["errors"] == []
         and saved["member_inventory_complete"] is True and saved["trusted_loading_allowed"] is False,
         "complete prior static inventory required, not loading authorization")
    p = saved["data_pickle"]
    need((p["member"], p["bytes"], p["sha256"]) == (PICKLE_NAME, PICKLE_BYTES, PICKLE_SHA),
         "data.pkl source binding")
    need(len(p["scan"]["declared_globals"]) == 49, "all49 declarations retained")
    return saved


def own_closure():
    return {str(p): base.sha(p.read_bytes()) for p in (
        Path(__file__), ROOT / "tests" / ("test_" + NAME + ".py"), ROOT / "innerdocs" / (NAME + ".md"))}


def run(wall_seconds=120):
    tick, start = base.deadline(wall_seconds), time.monotonic()
    saved, own = binding(), own_closure()
    with base.reader(scan.INPUT) as f:
        before = os.fstat(f.fileno())
        need(base.hash_stream(f, tick) == (scan.INPUT_BYTES, scan.INPUT_SHA), "fixed decoded SHA")
        base.unchanged(f, scan.INPUT, before)
        f.seek(0)
        scan.zip_preflight(f, scan.CAPS)
        with zipfile.ZipFile(f, "r") as z:
            i = z.getinfo(PICKLE_NAME)
            need(i.file_size == PICKLE_BYTES, "exact data.pkl byte cap")
            with z.open(i, "r") as data:
                raw = data.read(PICKLE_BYTES+1)
            need(len(raw) == PICKLE_BYTES and base.sha(raw) == PICKLE_SHA, "data.pkl length/SHA")
        result = build(raw, saved["members"], tick=tick)
        actual = result["ir"]["declarations"]
        expected = [{k: x[k] for k in ("position", "opcode", "module", "name")} for x in
                    saved["data_pickle"]["scan"]["declared_globals"]]
        need(actual == expected, "all49 exact declaration identity; no dynamic allowlist")
        base.unchanged(f, scan.INPUT, before)
    need(binding() == saved and own_closure() == own, "source/report closure changed")
    result.update(schema=NAME, own_sha256=own, static_report=str(STATIC_REPORT), static_report_sha256=STATIC_SHA,
                  input=str(scan.INPUT), input_sha256=scan.INPUT_SHA, data_pickle_sha256=PICKLE_SHA,
                  original_declarations=saved["data_pickle"]["scan"]["declared_globals"],
                  wall_seconds=time.monotonic()-start, caps=CAPS.__dict__, gpu_calls=0, physics_calls=0)
    need(len(base.encoded(result)) <= REPORT_CAP, "IR report cap; no silent pruning")
    return result


def save(path, report):
    path = Path(path)
    need(path.is_absolute() and path.parent == ROOT/"results" and
         re.fullmatch(re.escape(NAME)+r"_[A-Za-z0-9_-]+\.json", path.name), "exclusive IR report scope")
    raw = base.encoded(report)
    need(len(raw) <= REPORT_CAP, "IR report cap")
    used = base.budget_bytes()
    for prefix in (scan.NAME, NAME):
        for old in (ROOT/"results").glob(prefix+"_*.json"):
            need(not old.is_symlink() and old.is_file(), "metadata scope symlink/special file")
            used += old.stat().st_size
    base.capacity(used, shutil.disk_usage(path.parent).free, len(raw))
    fd = base.dir_fd(path.parent)
    try:
        with base.new_file(fd, path.name) as f:
            need(f.write(raw) == len(raw), "short IR report write")
            f.flush()
            os.fsync(f.fileno())
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(path), sha256=base.sha(raw), bytes=len(raw), status=report["status"],
                trusted_loading_allowed=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="saved metadata only, no checkpoint/storage read")
    cmd = commands.add_parser("build", help="MAIN-reviewed inert IR, never object loading")
    cmd.add_argument("--output", required=True, type=Path)
    cmd.add_argument("--wall-seconds", type=float, default=120)
    a = p.parse_args(argv)
    if a.command == "plan":
        binding()
        result = dict(schema=NAME+".plan", static_report_sha256=STATIC_SHA, input_sha256=scan.INPUT_SHA,
                      data_pickle_sha256=PICKLE_SHA, own_sha256=own_closure(), caps=CAPS.__dict__,
                      real_checkpoint_scanned=False, trusted_loading_allowed=False)
    else:
        result = save(a.output, run(a.wall_seconds))
    print(base.encoded(result).decode(), end="")
    return int(result.get("status") == "inert_metadata_incomplete_not_loadable")


if __name__ == "__main__":
    raise SystemExit(main())
