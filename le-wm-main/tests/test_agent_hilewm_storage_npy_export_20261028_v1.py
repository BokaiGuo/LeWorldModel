"""Pure stdlib RAM/synthetic ZIP checks; no production storage access."""
import ast
import copy
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

import pytest

from experiments.spectral_reachability import agent_hilewm_storage_npy_export_20261028_v1 as m


def fixture(order="little"):
    prefix = m.ir.PICKLE_NAME.rsplit("/", 1)[0]
    endian = "<" if order == "little" else ">"
    # Includes payload NaN and negative zero; never convert to Python floats.
    data = struct.pack(endian+"6I", 0x7fc01234, 0x80000000, 0x3f800000,
                       0x40000000, 0x40400000, 0x40800000)
    f = io.BytesIO()
    with zipfile.ZipFile(f, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(m.ir.PICKLE_NAME, b"not read as pickle")
        z.writestr(prefix+"/byteorder", order.encode())
        z.writestr(prefix+"/data/0", data)
    raw = f.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members = [dict(name=i.filename, file_bytes=i.file_size, crc32=f"{i.CRC:08x}",
                        flags=i.flag_bits, local_header_offset=i.header_offset,
                        compression_method=i.compress_type, issues=[]) for i in z.infolist()]
    definition = dict(storage_type=["torch", "FloatStorage"], storage_numel=6, location="cuda:0",
                      storage_member=prefix+"/data/0", storage_bytes=24)
    meta = dict(**definition, dtype_declaration="float32", storage_offset=0, shape=[2, 2],
                stride=[1, 3], storage_key="0", node=10)
    rows = [dict(module_path=[], collection="_parameters", name="weight", node=10,
                 status="literal_tensor_metadata", metadata=meta),
            dict(module_path=[], collection="_buffers", name="temporary", node=11,
                 status="explicit_none", metadata=None)]
    nodes = []
    def node(kind, **kw):
        index = len(nodes)
        nodes.append(dict(id=index, kind=kind, **kw))
        return index
    empty = node("dict", entries=[])
    buf_key = node("str", value="temporary")
    buffers = node("dict", entries=[[buf_key, node("none")]])
    fields = []
    for name, value in [("_buffers", buffers), ("_non_persistent_buffers_set", node("set", items=[]))]:
        fields.append([node("str", value=name), value])
    state = node("dict", entries=fields)
    module = node("newobj", states=[state])
    saved = dict(ir=dict(nodes=nodes), nn=dict(complete=True, errors=[], entries=rows,
                   storage_definitions={"0": definition}, storage_alias_paths={}, node_alias_paths={},
                   module_alias_paths={}, modules=[dict(path=[], node=module)]))
    return saved, dict(members=members), raw, data


@pytest.fixture
def ram(tmp_path, monkeypatch, request):
    saved, static, raw, data = fixture(getattr(request, "param", "little"))
    (tmp_path/"results").mkdir()
    source = tmp_path/"original.zip"
    source.write_bytes(raw)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m.scan, "INPUT", source)
    monkeypatch.setattr(m.scan, "INPUT_BYTES", len(raw))
    monkeypatch.setattr(m.scan, "INPUT_SHA", m.base.sha(raw))
    monkeypatch.setattr(m, "binding", lambda: (saved, static))
    monkeypatch.setattr(m, "own_closure", lambda: {"RAM": "fixed"})
    monkeypatch.setattr(m, "budget_bytes", lambda: sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()))
    monkeypatch.setattr(m.shutil, "disk_usage", lambda p: type("U", (), {"free": 20*1024**3})())
    return tmp_path/"results"/(m.NAME+"_RAM01"), saved, static, raw, data


def unpack_npy(raw):
    assert raw[:8] == b"\x93NUMPY\x01\x00"
    n = struct.unpack("<H", raw[8:10])[0]
    return ast.literal_eval(raw[10:10+n].decode()), raw[10+n:]


@pytest.mark.parametrize("order", ["little", "big"])
def test_view_bits_and_endian(order, tmp_path):
    saved, static, raw, data = fixture(order)
    meta = saved["nn"]["entries"][0]["metadata"]
    blocks = list(m.view_blocks(data, meta, saved["nn"]["storage_definitions"]["0"]))
    expected = b"".join(data[i*4:(i+1)*4] for i in [0, 3, 1, 4])
    assert b"".join(blocks) == expected
    desc = ("<" if order == "little" else ">")+"f4"
    receipt = m.emit_array(tmp_path, "entry_0000.npy", desc, [2, 2], iter(blocks))
    header, payload = unpack_npy((tmp_path/"entry_0000.npy").read_bytes())
    assert header == dict(descr=desc, fortran_order=False, shape=(2, 2))
    assert payload == expected
    assert receipt["value_sha256"] == m.base.sha(expected)
    assert m.emit_array(tmp_path, "entry_0000.npy", desc, [2, 2], iter(blocks), True) == receipt


@pytest.mark.parametrize("shape,stride,offset,indices", [([], [], 2, [2]), ([0, 2], [2, 1], 6, []),
    ([2, 2], [2, 1], 1, [1,2,3,4]), ([2,2], [0,1], 1, [1,2,1,2]), ([1,3], [999,1], 1, [1,2,3])])
def test_empty_scalar_contiguous_broadcast(shape, stride, offset, indices):
    saved, _, _, data = fixture()
    meta = saved["nn"]["entries"][0]["metadata"]
    meta.update(shape=shape, stride=stride, storage_offset=offset)
    assert b"".join(m.view_blocks(data, meta, saved["nn"]["storage_definitions"]["0"])) == b"".join(
        data[i*4:(i+1)*4] for i in indices)


@pytest.mark.parametrize("field,value", [("shape", [-1]), ("stride", [-1,1]),
    ("storage_offset", 6), ("storage_offset", True), ("shape", [2**62,2**62]),
    ("dtype_declaration", "float64"), ("location", "cpu"), ("shape", [2]),
    ("stride", [True,1]), ("storage_offset", -1)])
def test_range_dtype_and_identity_refusals(field, value):
    saved, _, _, _ = fixture()
    meta = saved["nn"]["entries"][0]["metadata"]
    meta[field] = value
    with pytest.raises((ValueError, KeyError)):
        m.view_layout(meta, saved["nn"]["storage_definitions"]["0"])


@pytest.mark.parametrize("descr", ["|O", "<f8", "=f4", "|S4"])
def test_only_numeric_npy(descr):
    with pytest.raises(ValueError):
        m.npy_header(descr, [1])


def test_long_storage_bits(tmp_path):
    raw = struct.pack(">2q", -9223372036854775808, 9223372036854775807)
    receipt = m.emit_array(tmp_path, "entry_0000.npy", ">i8", [2], iter([raw]))
    assert receipt["value_bytes"] == 16
    assert unpack_npy((tmp_path/"entry_0000.npy").read_bytes())[1] == raw


def test_metadata_denominator_and_persistence():
    saved, static, _, _ = fixture()
    groups, reserve = m.validate_metadata(saved, static)
    assert groups == {"0": [0]} and reserve > 24
    paths = m.state_dict_paths(saved)
    assert paths[0] == dict(index=0, key="weight", include_in_literal_state_dict=True, nonpersistent_buffer=False)
    assert paths[1]["include_in_literal_state_dict"] is False


def test_aliases_must_be_complete():
    saved, static, _, _ = fixture()
    row = copy.deepcopy(saved["nn"]["entries"][0])
    row["name"] = "tied"
    saved["nn"]["entries"].append(row)
    with pytest.raises(ValueError, match="alias"):
        m.validate_metadata(saved, static)
    pointers = [{k:r[k] for k in ("module_path", "collection", "name")} for r in
                [saved["nn"]["entries"][0], row]]
    saved["nn"]["storage_alias_paths"] = {"0":pointers}
    saved["nn"]["node_alias_paths"] = {"10":pointers}
    assert m.validate_metadata(saved, static)[0] == {"0": [0,2]}


@pytest.mark.parametrize("case", ["missing", "dotted", "duplicate", "foreign", "nonpersistent_unknown"])
def test_persistence_failclosed(case):
    saved, _, _, _ = fixture()
    nodes = saved["ir"]["nodes"]
    module = saved["nn"]["modules"][0]
    fields = nodes[nodes[module["node"]]["states"][0]]["entries"]
    if case == "missing":
        fields.pop()
    elif case == "dotted":
        saved["nn"]["entries"][0]["name"] = "bad.name"
    elif case == "duplicate":
        saved["nn"]["entries"].append(copy.deepcopy(saved["nn"]["entries"][0]))
    elif case == "foreign":
        nodes[fields[-1][1]]["kind"] = "global"
    else:
        index = len(nodes)
        nodes.append(dict(id=index, kind="str", value="missing"))
        nodes[fields[-1][1]]["items"].append(index)
    with pytest.raises((ValueError, KeyError)):
        m.state_dict_paths(saved)


def test_nonpersistent_actual_buffer_excluded():
    saved, _, _, _ = fixture()
    saved["nn"]["entries"][1] = copy.deepcopy(saved["nn"]["entries"][0])
    saved["nn"]["entries"][1].update(name="temporary", collection="_buffers")
    nodes = saved["ir"]["nodes"]
    names = next(n for n in nodes if n["kind"] == "set")
    names["items"] = [next(n["id"] for n in nodes if n.get("value") == "temporary")]
    assert m.state_dict_paths(saved)[1]["nonpersistent_buffer"] is True
    assert m.state_dict_paths(saved)[1]["include_in_literal_state_dict"] is False


@pytest.mark.parametrize("ram", ["little", "big"], indirect=True)
def test_RAM_export_verify_all_and_no_overwrite(ram):
    out, saved, _, _, data = ram
    receipt = m.run(out)
    report = json.loads((out/"report.json").read_bytes())
    assert len(report["rows"]) == 2 and len(report["storages"]) == 1
    assert report["rows"][1]["array"] is None
    assert report["storages"][0]["crc_verified"] is True
    assert report["byteorder"] in {"little", "big"} and report["all_storage_crc_verified"] is True
    assert len(report["journal"]) == 2
    assert report["runtime_state_dict_equivalence"] is False
    assert report["scientific_admission"] is False and report["pickle_executed"] is False
    assert unpack_npy((out/"storage_0000.npy").read_bytes())[1] == data
    assert m.run(out, verify_sha=receipt["sha256"])["arrays_verified"] == 2
    with pytest.raises(FileExistsError):
        m.run(out)


@pytest.mark.parametrize("case", ["payload", "trailing", "manifest", "sha", "symlink", "extra", "ledger"])
def test_saved_mutation_refused(ram, case):
    out, _, _, _, _ = ram
    receipt = m.run(out)
    array = out/"entry_0000.npy"
    if case == "payload":
        raw = bytearray(array.read_bytes()); raw[-1] ^= 1; array.write_bytes(raw)
    elif case == "trailing":
        array.write_bytes(array.read_bytes()+b"x")
    elif case == "manifest":
        report = json.loads((out/"report.json").read_bytes())
        report["rows"][0]["array"]["value_sha256"] = "0"*64
        raw = m.base.encoded(report); (out/"report.json").write_bytes(raw)
        receipt["sha256"] = m.base.sha(raw)
    elif case == "sha":
        receipt["sha256"] = "0"*64
    elif case == "symlink":
        target = out/"existing.npy"
        target.write_bytes(array.read_bytes())
        array.unlink()
        array.symlink_to(target)
    elif case == "extra":
        (out/"unexpected.npy").write_bytes(b"not a declared array")
    else:
        report = json.loads((out/"report.json").read_bytes())
        report["journal"].pop()
        raw = m.base.encoded(report); (out/"report.json").write_bytes(raw)
        receipt["sha256"] = m.base.sha(raw)
    with pytest.raises((ValueError, OSError)):
        m.run(out, verify_sha=receipt["sha256"])
    assert not (out/"failure.json").exists()


def test_CRC_member_metadata_and_bomb_cap():
    _, static, raw, _ = fixture()
    member = copy.deepcopy(static["members"][-1])
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with pytest.raises(ValueError):
            m.read_member(z, member, 1)
        member["crc32"] = "00000000"
        with pytest.raises(ValueError):
            m.read_member(z, member, 128)


def test_bad_payload_CRC():
    _, static, raw, _ = fixture()
    member = static["members"][-1]
    pos = member["local_header_offset"]
    name_len, extra_len = struct.unpack_from("<HH", raw, pos+26)
    damaged = bytearray(raw)
    damaged[pos+30+name_len+extra_len] ^= 1
    with zipfile.ZipFile(io.BytesIO(damaged)) as z:
        with pytest.raises(zipfile.BadZipFile):
            m.read_member(z, member, 128)


def test_export_failure_preserves_directory(ram, monkeypatch):
    out, _, _, _, _ = ram
    original = m.emit_array
    def failing(directory, filename, *args, **kwargs):
        if filename.startswith("entry_"):
            raise OSError("RAM interrupted")
        return original(directory, filename, *args, **kwargs)
    monkeypatch.setattr(m, "emit_array", failing)
    with pytest.raises(OSError, match="interrupted"):
        m.run(out)
    assert (out/"storage_0000.npy").is_file()
    report = json.loads((out/"failure.json").read_bytes())
    assert report["status"] == "failed_partial_files_preserved"
    assert len(report["journal"]) == 1 and report["errors"]
    assert not (out/"report.json").exists()


@pytest.mark.parametrize("kind", ["floor", "budget", "deadline", "relative", "escape"])
def test_preflight_refuses_before_output_creation(ram, monkeypatch, kind):
    out, _, _, _, _ = ram
    wall = 300
    if kind == "floor":
        monkeypatch.setattr(m.shutil, "disk_usage", lambda p: type("U", (), {"free": 10*1024**3})())
    elif kind == "budget":
        monkeypatch.setattr(m, "budget_bytes", lambda: 3*1024**3)
    elif kind == "deadline":
        wall = 601
    elif kind == "relative":
        out = Path("relative")
    else:
        out = out.parent/"foreign"
    with pytest.raises(ValueError):
        m.run(out, wall)
    assert not out.exists()


def test_cold_metadata_plan_forbids_loading_and_storage():
    code = '''
import builtins, pickle, sys
def forbidden(*a, **k): raise AssertionError("pickle execution forbidden")
pickle.load = pickle.loads = pickle.Unpickler = forbidden
original = builtins.__import__
def checked(name,*a,**k):
    if name.split(".")[0] in {"torch","numpy","hi_jepa","hi_module","transformers"}:
        raise AssertionError("foreign import " + name)
    return original(name,*a,**k)
builtins.__import__ = checked
def audit(event,args):
    if event == "open" and str(args[0]).endswith((".ckpt",".zip",".deflate",".npy")):
        raise AssertionError("production storage access forbidden")
    if event in {"socket.connect","subprocess.Popen"}: raise AssertionError(event)
sys.addaudithook(audit)
from experiments.spectral_reachability import agent_hilewm_storage_npy_export_20261028_v1 as m
m.main(["plan"])
'''
    proc = subprocess.run([sys.executable,"-B","-c",code], cwd=m.ROOT, capture_output=True,
                          text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["storage_count"] == 425 and plan["entries"] == 491
    assert plan["storage_payloads_read"] is False and plan["actual_byteorder"] is None


def test_no_input_dtype_or_pickle_override():
    with pytest.raises(SystemExit):
        m.main(["export", "--output", "/tmp/foreign", "--input", "/tmp/object", "--dtype", "object"])


@pytest.mark.parametrize("case", ["extra_storage", "missing_definition", "missing_entry", "bad_none", "duplicate_path"])
def test_metadata_denominator_mutations(case):
    saved, static, _, _ = fixture()
    if case == "extra_storage":
        extra = copy.deepcopy(static["members"][-1]); extra["name"] += "extra"
        static["members"].append(extra)
    elif case == "missing_definition":
        saved["nn"]["storage_definitions"] = {}
    elif case == "missing_entry":
        saved["nn"]["entries"] = []
    elif case == "bad_none":
        saved["nn"]["entries"][1]["metadata"] = {"unknown": True}
    else:
        saved["nn"]["entries"].append(copy.deepcopy(saved["nn"]["entries"][0]))
    with pytest.raises((ValueError, KeyError)):
        m.validate_metadata(saved, static)


def test_explicit_byteorder_required_no_fallback(ram):
    out, _, static, _, _ = ram
    static["members"] = [x for x in static["members"] if not x["name"].endswith("/byteorder")]
    with pytest.raises(ValueError):
        m.run(out)
    assert (out/"failure.json").is_file() and not (out/"report.json").exists()


def test_no_arbitrary_array_destination(tmp_path):
    with pytest.raises(ValueError):
        m.emit_array(tmp_path, "../escape.npy", "<f4", [0], iter([]))
