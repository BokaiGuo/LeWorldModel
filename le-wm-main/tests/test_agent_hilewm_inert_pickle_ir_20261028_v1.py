"""Handwritten inert pickle byte fixtures only; no production checkpoint scan."""
from dataclasses import replace
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_hilewm_inert_pickle_ir_20261028_v1 as m


def text(s):
    raw = s.encode()
    return b"X" + struct.pack("<I", len(raw)) + raw


def integer(i):
    return b"J"+struct.pack("<i", i)


def tup(*args):
    return b"("+b"".join(args)+b"t"


def glob(module, name):
    return b"c"+module.encode()+b"\n"+name.encode()+b"\n"


def mapping(items):
    return b"}("+b"".join(text(k)+v for k, v in items)+b"u"


def ordered():
    return glob("collections", "OrderedDict")+b")R"


def tensor(offset=0, shape=(2, 2), stride=(2, 1), key="0", numel=4,
           storage_type="FloatStorage", fn="_rebuild_tensor_v2"):
    pid = tup(text("storage"), glob("torch", storage_type), text(key), text("cpu"), integer(numel))+b"Q"
    args = tup(pid, integer(offset), tup(*(integer(i) for i in shape)),
               tup(*(integer(i) for i in stride)), b"\x89", ordered())
    return glob("torch._utils", fn)+args+b"R"


def parameter(value=None):
    return glob("torch._utils", "_rebuild_parameter")+tup(value or tensor(), b"\x88", ordered())+b"R"


def obj(fields, module="inert_fixture", name="NeverImported"):
    return glob(module, name)+b")\x81"+mapping(fields)+b"b"


def nn(parameters=None, buffers=None, children=None, extra=None):
    return obj([("_modules", mapping(children or [])),
                ("_parameters", mapping(parameters or [])),
                ("_buffers", mapping(buffers or []))]+(extra or []))


MEMBERS = [{"name": "ram/data/0", "file_bytes": 16, "crc32": "00000000"}]


def build(raw, members=MEMBERS, caps=m.CAPS):
    return m.build(b"\x80\x02"+raw+b".", members, "ram/data.pkl", caps)


def closed(result):
    assert result["deserialized"] is False and result["foreign_code_executed"] is False
    assert result["constructors_executed"] is False and result["trusted_loading_allowed"] is False
    assert result["tensor_pairing_verified"] is False and result["state_dict_extracted"] is False
    assert result["baseline_runnable"] is None and result["scientific_admission"] is False


def test_literal_module_paths_tensors_storage():
    raw = nn(children=[("encoder", nn(parameters=[("weight", parameter())])),
                       ("predictor_high", nn(buffers=[("running", tensor())]))])
    r = build(raw)
    assert r["status"] == "inert_metadata_recorded_not_loadable"
    assert r["ir"]["complete"] is True and r["nn"]["complete"] is True
    assert len(r["nn"]["entries"]) == 2
    a, b = r["nn"]["entries"]
    assert a["module_path"] == ["encoder"] and a["collection"] == "_parameters"
    assert b["module_path"] == ["predictor_high"] and b["collection"] == "_buffers"
    assert a["metadata"]["shape"] == [2, 2] and a["metadata"]["stride"] == [2, 1]
    assert a["metadata"]["storage_key"] == "0" and a["metadata"]["storage_offset"] == 0
    assert a["metadata"]["parameter_requires_grad"] is True
    assert b["metadata"]["parameter_requires_grad"] is None
    assert r["nn"]["storage_alias_paths"]["0"]
    assert r["nn"]["storage_payloads_read"] is False
    closed(r)


def test_alias_memo_parameter_identity():
    first = parameter()+b"q\x00"
    r = build(nn(parameters=[("weight", first), ("tied", b"h\x00")]))
    assert r["nn"]["complete"]
    entries = r["nn"]["entries"]
    assert entries[0]["node"] == entries[1]["node"]
    aliases = r["nn"]["node_alias_paths"][str(entries[0]["node"])]
    assert [x["name"] for x in aliases] == ["weight", "tied"]
    assert r["ir"]["memo_table"]["0"] == entries[0]["node"]


def test_shared_module_paths_not_deduplicated():
    child = nn(parameters=[("w", parameter())])+b"q\x00"
    r = build(nn(children=[("left", child), ("right", b"h\x00")]))
    assert r["nn"]["complete"]
    assert {tuple(x["module_path"]) for x in r["nn"]["entries"]} == {("left",), ("right",)}
    assert len(r["nn"]["module_alias_paths"]) == 1


def test_none_entries_not_zero_or_tensor_alias():
    r = build(nn(parameters=[("a", b"N"), ("b", b"N")], buffers=[("c", b"N")],
                 children=[("disabled", b"N")]))
    assert r["nn"]["complete"]
    assert all(x["status"] == "explicit_none" and x["metadata"] is None for x in r["nn"]["entries"])
    assert not r["nn"]["node_alias_paths"] and not r["nn"]["storage_definitions"]


def test_generic_self_reference_finite_json_no_object_allocation():
    ir = m.parse_ir(b"]q\x00h\x00a.")
    assert ir["complete"] and ir["nodes"][0]["items"] == [0]
    assert ir["memo_table"] == {"0": 0}
    json.dumps(ir)


def test_unknown_reducer_retained_never_invoked():
    raw = nn(extra=[("uninterpreted", glob("os", "system")+tup(text("DO_NOT_EXECUTE"))+b"R")])
    r = build(raw)
    assert r["ir"]["complete"] and len(r["ir"]["unknown_reducers"]) == 1
    assert r["status"] == "inert_metadata_incomplete_not_loadable"
    assert any(n.get("module") == "os" for n in r["ir"]["nodes"])
    assert r["ir"]["constructors_executed"] is False
    closed(r)


def test_all_declarations_preserved_not_allowlist():
    raw = tup(*(glob("foreign_not_imported", "Class"+str(i)) for i in range(49)))+b"."
    ir = m.parse_ir(raw)
    assert ir["complete"] and len(ir["declarations"]) == 49
    assert ir["declarations"][48]["name"] == "Class48"
    assert len(m.RECOGNIZED_REDUCERS) == 5


@pytest.mark.parametrize("kwargs", [dict(fn="unknown_rebuild"), dict(storage_type="HalfStorage"),
                                    dict(key="../bad"), dict(numel=3), dict(offset=-1),
                                    dict(shape=(2,), stride=(1, 1)), dict(stride=(-1, 1)),
                                    dict(offset=4), dict(shape=(-1, 2))])
def test_tensor_unknown_or_invalid_layout_failclosed(kwargs):
    r = build(nn(parameters=[("w", parameter(tensor(**kwargs)))]))
    assert r["status"] == "inert_metadata_incomplete_not_loadable"
    assert r["nn"]["entries"][0]["metadata"] is None
    assert r["nn"]["entries"][0]["status"] == "unknown"
    assert r["nn"]["errors"]


@pytest.mark.parametrize("shape,stride,offset", [((0, 2), (2, 1), 0), ((), (), 0), ((2, 2), (1, 2), 0)])
def test_empty_scalar_noncontiguous_metadata(shape, stride, offset):
    r = build(nn(buffers=[("b", tensor(shape=shape, stride=stride, offset=offset))]))
    assert r["nn"]["complete"]
    assert r["nn"]["entries"][0]["metadata"]["shape"] == list(shape)


def test_missing_storage_bytes_remain_unknown():
    r = build(nn(parameters=[("w", parameter())]), members=[])
    assert not r["nn"]["complete"] and r["nn"]["entries"][0]["metadata"] is None


def test_conflicting_storage_declarations_refused():
    raw = nn(buffers=[("a", tensor()), ("b", tensor(numel=2, shape=(2,), stride=(1,), storage_type="LongStorage"))])
    r = build(raw)
    assert r["nn"]["entries"][0]["metadata"] is not None
    assert r["nn"]["entries"][1]["metadata"] is None
    assert "conflicting storage" in r["nn"]["errors"][0]["reason"]


def test_missing_nn_fields_not_invented():
    r = build(obj([("_modules", mapping([]))]))
    assert not r["nn"]["complete"] and "missing explicit" in r["nn"]["errors"][0]["reason"]


def test_duplicate_state_key_not_guessed_overwrite():
    r = build(nn(extra=[("_parameters", mapping([]))]))
    assert not r["nn"]["complete"] and "duplicate state key" in r["nn"]["errors"][0]["reason"]


def test_module_cycle_refused_without_recursion():
    # Memoize root before BUILD, then refer to root in _modules.
    raw = glob("fixture", "Root")+b")\x81q\x00"+mapping([
        ("_modules", mapping([("cycle", b"h\x00")])),
        ("_parameters", mapping([])), ("_buffers", mapping([]))])+b"b"
    r = build(raw)
    assert not r["nn"]["complete"] and any("cycle" in x["reason"] for x in r["nn"]["errors"])


@pytest.mark.parametrize("raw", [b"\x82\x01.", b"\x97.", b"h\x00.", b"N.N.", b"NN.",
                                  b"N", b"!", b"\x93.", b"\x80\x06N."])
def test_unsupported_malformed_partial_ir(raw):
    ir = m.parse_ir(raw)
    assert not ir["complete"] and ir["root_node"] is None and ir["errors"]


def test_unknown_stack_global_not_resolved_or_executed():
    ir = m.parse_ir(glob("builtins", "str")+b")R"+text("C")+b"\x93.")
    assert not ir["complete"] and ir["unknown_reducers"]


def test_unicode_stack_global_memo():
    ir = m.parse_ir(text("literal_module")+b"q\x00"+text("C")+b"\x93.")
    assert ir["complete"] and ir["declarations"][0]["module"] == "literal_module"


@pytest.mark.parametrize("key,value", [("nodes", 1), ("opcodes", 1), ("edges", 1),
                                       ("memo", 1), ("literal_bytes", 1)])
def test_parse_caps(key, value):
    caps = replace(m.CAPS, **{key: value})
    raw = b"\x80\x02"+text("long")+b"q\x00"+text("more")+b"q\x01\x86."
    ir = m.parse_ir(raw, caps)
    assert not ir["complete"] and ir["errors"]


def test_raw_byte_cap():
    with pytest.raises(ValueError, match="byte cap"):
        m.parse_ir(b"N.", replace(m.CAPS, raw_bytes=1))


def test_nn_path_cap():
    r = build(nn(children=[("child", nn())]), caps=replace(m.CAPS, depth=0))
    assert not r["nn"]["complete"]


def test_byte_and_float_literal_preservation():
    ir = m.parse_ir(b"(C\x03abcG"+struct.pack(">d", float("nan"))+b"t.")
    assert ir["complete"]
    assert ir["nodes"][0]["hex"] == "616263" and ir["nodes"][1]["hex"] == "nan"
    json.dumps(ir, allow_nan=False)


def test_build_tuple_state_record_no_setstate_execution():
    raw = glob("foreign", "NoImport")+b")\x81"+tup(mapping([
        ("_modules", mapping([])), ("_parameters", mapping([])), ("_buffers", mapping([]))]), b"N")+b"b"
    r = build(raw)
    assert r["nn"]["complete"] and r["ir"]["constructors_executed"] is False


def test_saved_binding_metadata_only():
    r = m.binding()
    assert len(r["data_pickle"]["scan"]["declared_globals"]) == 49
    assert r["data_pickle"]["sha256"] == m.PICKLE_SHA


def test_static_metadata_large_RAM(tmp_path):
    path = tmp_path / "inventory.json"
    raw = json.dumps({"literal": "x" * (2 * 1024**2)}).encode()
    path.write_bytes(raw)
    assert m.static_metadata(path, m.base.sha(raw))["literal"] == "x" * (2 * 1024**2)


@pytest.mark.parametrize("case", ["cap", "hash", "duplicate", "symlink"])
def test_static_metadata_refuses_RAM(tmp_path, case):
    path = tmp_path / "inventory.json"
    raw = b'{"a":1,"a":2}' if case == "duplicate" else b'{"a":1}'
    path.write_bytes(raw)
    if case == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(path)
        path = link
    with pytest.raises((ValueError, OSError)):
        m.static_metadata(path, "0" * 64 if case == "hash" else m.base.sha(raw),
                          cap=1 if case == "cap" else m.STATIC_BYTES)


def test_cold_plan_and_IR_no_load_no_foreign_no_real_scan():
    code = '''
import builtins, pickle, sys
def forbidden(*a, **k): raise AssertionError("pickle execution forbidden")
pickle.load = pickle.loads = pickle.Unpickler = forbidden
original = builtins.__import__
def checked(name, *a, **k):
    if name.split(".")[0] in {"torch", "numpy", "h_le_wm", "hi_jepa", "hi_module", "transformers", "foreign"}:
        raise AssertionError("foreign import " + name)
    return original(name, *a, **k)
builtins.__import__ = checked
def audit(event, args):
    if event == "open" and str(args[0]).endswith((".ckpt", ".deflate")):
        raise AssertionError("production scan forbidden")
    if event in {"socket.connect", "subprocess.Popen"}: raise AssertionError(event)
sys.addaudithook(audit)
from experiments.spectral_reachability import agent_hilewm_inert_pickle_ir_20261028_v1 as m
assert m.main(["plan"]) == 0
ir = m.parse_ir(b"cos\\nsystem\\n)R.")
assert ir["complete"] and ir["unknown_reducers"] and ir["constructors_executed"] is False
'''
    p = subprocess.run([sys.executable, "-B", "-c", code], cwd=m.ROOT,
                       text=True, capture_output=True, timeout=20)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["real_checkpoint_scanned"] is False


def test_save_new_report_RAM(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m.base, "budget_bytes", lambda: 100)
    monkeypatch.setattr(m.shutil, "disk_usage", lambda p: type("U", (), {"free": 20*1024**3})())
    r = build(nn())
    path = results / (m.NAME+"_RAM.json")
    receipt = m.save(path, r)
    assert receipt["sha256"] == m.base.sha(path.read_bytes())
    with pytest.raises(FileExistsError):
        m.save(path, r)
    with pytest.raises(ValueError):
        m.save(Path("relative"), r)


def test_cli_forbids_input_and_reducer_override():
    with pytest.raises(SystemExit):
        m.main(["build", "--output", "/tmp/no", "--input", "/tmp/foreign", "--allow-reducer", "os.system"])
