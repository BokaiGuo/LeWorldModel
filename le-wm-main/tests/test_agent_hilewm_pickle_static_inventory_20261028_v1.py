"""Synthetic RAM ZIP/pickle opcode bytes only; no real checkpoint scanning."""
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import warnings
import zipfile

import pytest

from experiments.spectral_reachability import agent_hilewm_pickle_static_inventory_20261028_v1 as m


def container(raw=b"\x80\x02N.", entries=None, method=zipfile.ZIP_STORED):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", compression=method) as z:
        for name, data in (entries if entries is not None else
                           [("ram/data.pkl", raw), ("ram/data/0", b"RAM storage")]):
            z.writestr(name, data)
    return b.getvalue()


def inspect(raw, caps=m.CAPS):
    return m.inspect_stream(io.BytesIO(raw), caps)


def pscan(raw, caps=m.CAPS):
    return m.pickle_inventory(raw, caps)


def assert_closed(report):
    assert report["trusted_loading_allowed"] is False
    assert report["static_scan_proves_safety"] is False
    assert report["deserialized"] is False and report["foreign_code_executed"] is False
    assert report["baseline_runnable"] is None and report["scientific_admission"] is False


def test_exact_bindings():
    assert m.INPUT_BYTES == 122424517
    assert m.INPUT_SHA == "b87805747d40037841877ce7b99b7dda3ebe7a52202c0ba46bf0006ab5d6f008"
    assert m.DECODE_REPORT_SHA == "f399b23e07f37960127433d10e14c8c7b7eae6c75ff329accb75a52f701edcc9"


@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_complete_members_but_not_storage_crc(method):
    r = inspect(container(method=method))
    assert r["status"] == "static_inventory_recorded_not_trusted_loadable"
    assert r["member_inventory_complete"] is True and len(r["members"]) == 2
    assert r["members"][0]["CRC_verified"] is True
    assert r["members"][1]["CRC_verified"] is False
    assert r["storage_payloads_read"] is False and r["all_member_CRC_verified"] is False
    assert_closed(r)


@pytest.mark.parametrize("name", ["../bad", "/absolute", "C:/drive", "back\\slash",
                                  "a//bad", "a/./bad", "a/../bad", "a/\x00bad", "a/\nbad", ""])
def test_hazard_path_names(name):
    assert m.path_issues(name)


@pytest.mark.parametrize("name", ["ram/data.pkl", "ram/.data/serialization_id", "ram/", "version"])
def test_safe_names(name):
    assert m.path_issues(name) == []


def test_dangerous_path_full_list_and_no_pickle_read():
    r = inspect(container(entries=[("../bad", b"evil"), ("ram/data.pkl", b"N.")]))
    assert r["member_inventory_complete"] is True and len(r["members"]) == 2
    assert r["data_pickle"] is None and r["errors"]
    assert r["members"][0]["issues"]
    assert_closed(r)


def test_duplicate_names_retained_and_refused():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = container(entries=[("a/data.pkl", b"N."), ("a/data.pkl", b"N.")])
    r = inspect(raw)
    assert len(r["members"]) == 2 and all("duplicate_member_name" in x["issues"] for x in r["members"])
    assert r["data_pickle"] is None


def test_symlink_metadata():
    i = zipfile.ZipInfo("ram/link")
    i.create_system = 3
    i.external_attr = (stat.S_IFLNK | 0o777) << 16
    r = inspect(container(entries=[("ram/data.pkl", b"N."), (i, b"/bad")]))
    assert "symlink_or_special_type" in r["members"][1]["issues"]
    assert r["data_pickle"] is None


def test_encrypted_metadata():
    raw = bytearray(container())
    pos = raw.find(b"PK\x01\x02")
    flags = struct.unpack_from("<H", raw, pos+8)[0]
    struct.pack_into("<H", raw, pos+8, flags | 1)
    r = inspect(bytes(raw))
    assert "encrypted_member" in r["members"][0]["issues"] and r["data_pickle"] is None


@pytest.mark.parametrize("entries", [[("ram/data/0", b"x")],
                                     [("a/data.pkl", b"N."), ("b/data.pkl", b"N.")]])
def test_missing_ambiguous_pickle(entries):
    r = inspect(container(entries=entries))
    assert r["data_pickle"] is None and "exactly one" in r["errors"][0]


@pytest.mark.parametrize("change", ["member_count", "central_size", "pickle_size", "member_size", "total_size"])
def test_engineering_caps(change):
    caps = m.CAPS
    if change == "member_count":
        caps = replace(caps, members=1)
    elif change == "central_size":
        caps = replace(caps, central_bytes=1)
    elif change == "pickle_size":
        caps = replace(caps, pickle_bytes=1)
    elif change == "member_size":
        caps = replace(caps, member_bytes=1)
    else:
        caps = replace(caps, total_declared=1)
    r = inspect(container(), caps)
    assert r["errors"] and r["data_pickle"] is None
    assert_closed(r)


@pytest.mark.parametrize("raw", [b"N.", b"", container()+b"trailing", b"SFX"+container()])
def test_non_zip_or_trailing_refused(raw):
    r = inspect(raw)
    assert r["errors"] and r["member_inventory_complete"] is False


def zip64_fixture(huge=False):
    raw = container()
    e = struct.unpack("<4s4H2IH", raw[-22:])
    offset = len(raw)-22
    record = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0,
                         e[4], e[4], 2**40 if huge else e[5], e[6])
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, offset, 1)
    return raw[:-22]+record+locator+raw[-22:]


def test_bounded_zip64_metadata():
    r = inspect(zip64_fixture())
    assert r["zip_metadata"]["zip64"] is True and not r["errors"]


def test_zip64_override_rejected_before_zipfile_allocation(monkeypatch):
    raw = zip64_fixture(huge=True)
    def forbidden(*a, **kw):
        raise AssertionError("ZipFile must not be called")
    monkeypatch.setattr(m.zipfile, "ZipFile", forbidden)
    r = inspect(raw)
    assert r["errors"] and "ZIP64/EOCD disagreement" in r["errors"][0]


def test_crc_corruption_data_pickle_only():
    raw = bytearray(container())
    pos = raw.find(b"\x80\x02N.")
    assert pos >= 0
    raw[pos+2] ^= 1
    r = inspect(raw)
    assert r["errors"] and "CRC" in r["errors"][0]
    assert r["member_inventory_complete"] is True and r["data_pickle"] is None


def test_full_opcode_positions_and_global_no_execution():
    raw = b"\x80\x02cos\nsystem\nX\x04\x00\x00\x00echo\x85R."
    s = pscan(raw)
    assert s["opcode_inventory_complete"] is True
    assert s["unique_declared_symbols"] == [{"module": "os", "name": "system"}]
    assert s["opcode_counts"]["REDUCE"] == 1 and s["executes_reducers"] is False
    assert s["opcodes"][0] == {"position": 0, "opcode": "PROTO"}
    assert all(set(x) == {"position", "opcode"} for x in s["opcodes"])


def test_global_module_space_not_silently_repartitioned():
    s = pscan(b"cmodule with space\nName\n.")
    assert s["declared_globals"][0]["module"] == "module with space"
    assert s["declared_globals"][0]["name"] == "Name"


def test_stack_global_unicode_memo_and_origins():
    raw = b"\x80\x04\x8c\x03pkg\x94\x8c\x05Class\x94\x930h\x00h\x01\x93."
    s = pscan(raw)
    assert s["global_resolution_complete"] is True
    assert len(s["declared_globals"]) == 2
    assert s["declared_globals"][0]["string_origins"] == s["declared_globals"][1]["string_origins"]


def test_stack_global_with_space_preserves_exact_literals():
    s = pscan(b"\x80\x04\x8c\x03a b\x8c\x01C\x93.")
    assert s["declared_globals"][0]["module"] == "a b"
    assert s["declared_globals"][0]["name"] == "C"


def test_dynamic_stack_global_remains_unknown():
    # Reducer output could be a string at load time; scanner must not evaluate.
    raw = b"cbuiltins\nstr\n)R\x8c\x01C\x93."
    s = pscan(raw)
    assert s["opcode_inventory_complete"] is True and s["global_resolution_complete"] is False
    assert len(s["unresolved_stack_globals"]) == 1
    assert len(s["declared_globals"]) == 1


def test_extension_registry_not_resolved():
    s = pscan(b"\x80\x02\x82\x01.")
    assert s["extension_codes"][0]["code"] == 1 and s["extension_codes"][0]["resolution"] is None
    assert s["global_resolution_complete"] is False


def test_persistent_id_not_resolved_or_printed():
    s = pscan(b"Pstorage-secret\n.")
    assert s["opcode_counts"]["PERSID"] == 1 and s["persistent_ids_resolved"] is False
    assert "storage-secret" not in json.dumps(s)


@pytest.mark.parametrize("raw", [b"h\x01.", b"\x93.", b"N.N.", b"NN.", b"N", b"!",
                                  b"\x80\x06N.", b"\x95"+struct.pack("<Q", 1000)+b"N."])
def test_malformed_pickle_retains_partial_inventory(raw):
    s = pscan(raw)
    assert s["errors"] and s["opcode_inventory_complete"] is False


@pytest.mark.parametrize("field,value", [("opcodes", 1), ("stack", 1), ("memo", 1), ("pickle_bytes", 1)])
def test_pickle_caps(field, value):
    caps = replace(m.CAPS, **{field: value})
    raw = b"\x80\x04\x8c\x01a\x94\x8c\x01b\x94\x93."
    if field == "pickle_bytes":
        with pytest.raises(ValueError, match="cap"):
            pscan(raw, caps)
    else:
        s = pscan(raw, caps)
        assert s["errors"] and not s["opcode_inventory_complete"]


def test_deadline_partial_report():
    calls = 0
    def tick():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("synthetic deadline")
    s = m.pickle_inventory(b"\x80\x02N.", tick=tick)
    assert s["errors"] and "deadline" in s["errors"][0]


def test_saved_binding_metadata_only():
    b = m.binding()
    assert b["input_sha256"] == m.INPUT_SHA and b["decode_report_sha256"] == m.DECODE_REPORT_SHA
    assert b["author_object_sha256"] is None


def test_fresh_process_plan_forbids_real_object_or_loading():
    code = '''
import sys, builtins, pickle
def no_load(*a, **k): raise AssertionError("deserialization forbidden")
pickle.load = pickle.loads = pickle.Unpickler = no_load
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "numpy", "hydra", "stable_worldmodel", "h_le_wm", "requests"}:
        raise AssertionError("foreign/model import " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
def audit(event, args):
    if event == "open" and str(args[0]).endswith((".ckpt", ".deflate")):
        raise AssertionError("real byte scan forbidden in worker test")
    if event in {"socket.connect", "subprocess.Popen"}:
        raise AssertionError("external execution forbidden")
sys.addaudithook(audit)
from experiments.spectral_reachability import agent_hilewm_pickle_static_inventory_20261028_v1 as m
assert m.main(["plan"]) == 0
assert m.pickle_inventory(b"cos\\nsystem\\n)R.")["executes_reducers"] is False
'''
    p = subprocess.run([sys.executable, "-B", "-c", code], cwd=m.ROOT,
                       capture_output=True, text=True, timeout=20)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["real_checkpoint_scanned"] is False


def test_scoped_exclusive_report_save_RAM(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m.base, "budget_bytes", lambda: 100)
    monkeypatch.setattr(m.shutil, "disk_usage", lambda p: type("U", (), {"free": 20*1024**3})())
    path = results / (m.NAME + "_RAM.json")
    r = inspect(container())
    receipt = m.save(path, r)
    assert receipt["sha256"] == m.base.sha(path.read_bytes())
    with pytest.raises(FileExistsError):
        m.save(path, r)
    with pytest.raises(ValueError, match="scoped"):
        m.save(tmp_path / "wrong.json", r)


def test_cli_no_input_or_cap_override():
    with pytest.raises(SystemExit):
        m.main(["scan", "--output", "/tmp/wrong", "--input", "/tmp/other"])
