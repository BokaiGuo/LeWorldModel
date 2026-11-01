"""RAM/synthetic bytes and pinned saved text only; never decode real checkpoint."""
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zlib

import pytest

from experiments.spectral_reachability import agent_hilewm_bounded_deflate_decode_20261028_v1 as m


def fixture(data=b"RAM fixture\x00" * 100):
    c = zlib.compressobj(wbits=-15)
    raw = c.compress(data) + c.flush()
    return data, raw, m.Spec(len(raw), m.sha(raw), len(data), zlib.crc32(data))


def test_fixed_production_binding():
    assert m.SPEC == m.Spec(113104429, "d169cf3c4a8d668ac5b0dcb17c6fd1682360764ce578d38daac7fec6dc76d086",
                            122424517, 0xd59f865b)
    assert m.DISK_CAP == 3 * 1024**3 and m.FREE_FLOOR == 10 * 1024**3


@pytest.mark.parametrize("data", [b"x", b"abc" * 100000, bytes(range(256)) * 2000,
                                  hashlib.shake_256(b"RAM only").digest(300000)])
def test_exact_stream(data):
    data, raw, spec = fixture(data)
    output = io.BytesIO()
    r = m.decode_stream(io.BytesIO(raw), output, spec)
    assert output.getvalue() == data
    assert r["decoded_sha256"] == m.sha(data) and r["decoded_bytes"] == len(data)
    assert r["decoded_crc32"] == f"{zlib.crc32(data):08x}"
    assert r["deflate_eof"] is True and r["trailing_bytes"] == 0


@pytest.mark.parametrize("change", ["truncated", "trailing", "concat", "bad_sha", "bad_crc",
                                    "small_packed", "large_packed", "small_decoded", "large_decoded",
                                    "corrupt", "zlib_wrapper", "empty"])
def test_reject_mutations(change):
    data, raw, spec = fixture()
    if change == "truncated":
        raw = raw[:-1]
        spec = dataclasses.replace(spec, packed=len(raw), digest=m.sha(raw))
    elif change == "trailing":
        raw += b"trailing"
        spec = dataclasses.replace(spec, packed=len(raw), digest=m.sha(raw))
    elif change == "concat":
        raw *= 2
        spec = dataclasses.replace(spec, packed=len(raw), digest=m.sha(raw))
    elif change == "bad_sha":
        spec = dataclasses.replace(spec, digest="0" * 64)
    elif change == "bad_crc":
        spec = dataclasses.replace(spec, crc32=spec.crc32 ^ 1)
    elif change == "small_packed":
        spec = dataclasses.replace(spec, packed=len(raw) - 1)
    elif change == "large_packed":
        spec = dataclasses.replace(spec, packed=len(raw) + 1)
    elif change == "small_decoded":
        spec = dataclasses.replace(spec, unpacked=len(data) - 1)
    elif change == "large_decoded":
        spec = dataclasses.replace(spec, unpacked=len(data) + 1)
    elif change == "corrupt":
        raw = bytes([raw[0] ^ 0xff]) + raw[1:]
        spec = dataclasses.replace(spec, digest=m.sha(raw))
    elif change == "zlib_wrapper":
        raw = zlib.compress(data)
        spec = dataclasses.replace(spec, packed=len(raw), digest=m.sha(raw))
    else:
        raw = b""
    with pytest.raises((ValueError, zlib.error)):
        m.decode_stream(io.BytesIO(raw), io.BytesIO(), spec)


def test_trailing_in_later_read():
    data, raw, spec = fixture(b"AB" * 20)
    class Split(io.BytesIO):
        def read(self, n=-1):
            return super().read(len(raw))
    all_raw = raw + b"late garbage"
    with pytest.raises(ValueError, match="trailing"):
        m.decode_stream(Split(all_raw), io.BytesIO(), dataclasses.replace(
            spec, packed=len(all_raw), digest=m.sha(all_raw)))


def test_bomb_never_writes_beyond_cap():
    _, raw, spec = fixture(b"A" * 5_000_000)
    target = io.BytesIO()
    with pytest.raises(ValueError, match="bomb"):
        m.decode_stream(io.BytesIO(raw), target, dataclasses.replace(spec, unpacked=1024))
    assert len(target.getvalue()) <= 1024


def test_large_valid_output_never_allocated_in_one_emit():
    data, raw, spec = fixture(b"A" * 1_000_000)
    sizes = []
    class Sink(io.BytesIO):
        def write(self, b):
            sizes.append(len(b))
            return super().write(b)
    out = Sink()
    m.decode_stream(io.BytesIO(raw), out, spec)
    assert out.getvalue() == data and max(sizes) <= 65536 and len(sizes) > 10


def test_short_write_rejected():
    _, raw, spec = fixture()
    class Sink:
        def write(self, b):
            return len(b) - 1
    with pytest.raises(ValueError, match="short output write"):
        m.decode_stream(io.BytesIO(raw), Sink(), spec)


def test_tick_failure_preserves_written_bytes():
    _, raw, spec = fixture(b"A" * 1_000_000)
    calls = 0
    def tick():
        nonlocal calls
        calls += 1
        if calls == 7:
            raise ValueError("mock deadline")
    target = io.BytesIO()
    with pytest.raises(ValueError, match="mock deadline"):
        m.decode_stream(io.BytesIO(raw), target, spec, tick)
    assert 0 < len(target.getvalue()) < spec.unpacked


@pytest.mark.parametrize("seconds", [0, -1, 601, float("inf"), float("nan"), True])
def test_wall_bounds(seconds):
    with pytest.raises(ValueError):
        m.deadline(seconds)


def test_expired_clock(monkeypatch):
    clock = iter([100., 102.])
    monkeypatch.setattr(m.time, "monotonic", lambda: next(clock))
    with pytest.raises(ValueError, match="deadline"):
        m.deadline(1)()


@pytest.mark.parametrize("used,free,reserve", [(m.DISK_CAP, 20*m.GIB, 1),
                                               (0, m.FREE_FLOOR, 1),
                                               (m.DISK_CAP+1, 20*m.GIB, 0)])
def test_budget_refuse(used, free, reserve):
    with pytest.raises(ValueError):
        m.capacity(used, free, reserve)


def test_budget_exact_boundary():
    m.capacity(m.DISK_CAP-1, m.FREE_FLOOR+1, 1)


def test_disk_counts_duplicates(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    (tmp_path / "b").write_bytes(b"abc")
    os.link(tmp_path / "a", tmp_path / "hard")
    assert m.scope_bytes(tmp_path) == 9
    (tmp_path / "link").symlink_to(tmp_path / "a")
    with pytest.raises(ValueError, match="symlink"):
        m.scope_bytes(tmp_path)


def test_scoped_special_file_refused(tmp_path):
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(ValueError, match="special"):
        m.scope_bytes(tmp_path)
    with pytest.raises(ValueError, match="regular"):
        m.reader(tmp_path / "fifo")


def test_reader_symlink_and_ancestor_rejected(tmp_path):
    (tmp_path / "data").write_bytes(b"RAM")
    (tmp_path / "link").symlink_to(tmp_path / "data")
    with pytest.raises(OSError):
        m.reader(tmp_path / "link")
    (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        m.reader(tmp_path / "alias/data")


def test_fd_identity_changes(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"RAM")
    with m.reader(path) as f:
        initial = os.fstat(f.fileno())
        path.write_bytes(b"changed")
        with pytest.raises(ValueError, match="identity"):
            m.unchanged(f, path, initial)


def test_exclusive_output_file(tmp_path):
    fd = m.dir_fd(tmp_path)
    try:
        with m.new_file(fd, "data") as f:
            f.write(b"keep")
        with pytest.raises(FileExistsError):
            m.new_file(fd, "data")
    finally:
        os.close(fd)
    assert (tmp_path / "data").read_bytes() == b"keep"


def test_exclusive_directory_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    path = tmp_path / "results" / (m.NAME + "_fixture")
    os.close(m.output_directory(path))
    with pytest.raises(FileExistsError):
        m.output_directory(path)
    for bad in (tmp_path / (m.NAME + "_x"), tmp_path / "results/unrelated", Path("relative")):
        with pytest.raises(ValueError):
            m.output_directory(bad)


def test_json_duplicate_nonfinite_reject():
    for data in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        with pytest.raises(ValueError):
            m.decode_json(data)


def test_pinned_small_hash_mutation(tmp_path):
    p = tmp_path / "meta"
    p.write_bytes(b'{"a":1}')
    assert m.small(p, m.sha(p.read_bytes())) == {"a": 1}
    with pytest.raises(ValueError, match="SHA"):
        m.small(p, "0"*64)


def test_saved_plan_text_only():
    # Reads only already saved metadata/text, never compressed/object files.
    p = m.provenance()
    assert p["record_license"] == {"id": "cc-by-4.0"}
    assert p["trusted_pickle_class_manifest"] is None
    assert p["immutable_upstream_revision"] is None
    assert p["baseline_runnable"] is None and p["scientific_admission"] is False
    assert len(p["source_locators"]) == 9
    assert any("processor.fit" in x["quote"] for x in p["source_locators"])
    assert all(Path(x["path"]).is_absolute() for x in p["source_locators"])


def test_cold_plan_no_object_model_or_network_imports():
    # Independent interpreter: union test runs may already have imported torch.
    code = '''
import builtins, sys
real_import = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "numpy", "pickle", "hydra", "stable_worldmodel", "requests"} or name in {"urllib.request", "http.client"}:
        raise AssertionError("forbidden import " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = checked
def audit(event, args):
    if event == "open" and (str(args[0]).endswith(".deflate") or str(args[0]).endswith(".ckpt")):
        raise AssertionError("checkpoint bytes must not open during plan")
    if event in {"socket.connect", "subprocess.Popen"}:
        raise AssertionError("no external execution/network")
sys.addaudithook(audit)
from experiments.spectral_reachability import agent_hilewm_bounded_deflate_decode_20261028_v1 as m
assert m.main(["plan"]) == 0
'''
    p = subprocess.run([sys.executable, "-B", "-c", code], cwd=m.ROOT,
                       capture_output=True, text=True, timeout=20)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["decode_executed"] is False


def test_consume_rejected_report_before_binary_open(tmp_path):
    p = tmp_path / "report.json"
    raw = m.encoded(dict(schema=m.NAME, status="rejected_partial_preserved"))
    p.write_bytes(raw)
    with pytest.raises(ValueError, match="successful"):
        m.consume(p, m.sha(raw))


def test_cli_cannot_override_scientific_bindings():
    with pytest.raises(SystemExit) as exc:
        m.main(["decode", "--output-dir", "/tmp/no", "--crc32", "00000000"])
    assert exc.value.code != 0


def file_fixture(tmp_path, monkeypatch, bad_crc=False):
    data, raw, spec = fixture()
    results = tmp_path / "results"
    assets = results / "synthetic_assets"
    assets.mkdir(parents=True)
    src = assets / "RAM.deflate"
    src.write_bytes(raw)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "ASSETS", assets)
    monkeypatch.setattr(m, "INPUT", src)
    monkeypatch.setattr(m, "SPEC", dataclasses.replace(spec, crc32=spec.crc32 ^ 1) if bad_crc else spec)
    monkeypatch.setattr(m, "provenance", lambda: {"RAM_mock": True})
    monkeypatch.setattr(m, "own_closure", lambda: {"RAM_code": "fixed"})
    usage = type("Usage", (), {"free": 20 * m.GIB})()
    monkeypatch.setattr(m.shutil, "disk_usage", lambda path: usage)
    return data, results / (m.NAME + "_RAM")


def test_RAM_file_run_and_consume(tmp_path, monkeypatch):
    data, out = file_fixture(tmp_path, monkeypatch)
    r = m.run(out)
    assert r["status"] == "decoded_bytes_verified_not_trusted_loadable"
    assert (out / "decoded.unverified.ckpt").read_bytes() == data
    report = out / "report.json"
    c = m.consume(report, m.sha(report.read_bytes()))
    assert c["decoded_sha256"] == m.sha(data) and c["baseline_runnable"] is None
    assert len(list(out.iterdir())) == 2


def test_RAM_failure_retains_output_and_report(tmp_path, monkeypatch):
    data, out = file_fixture(tmp_path, monkeypatch, bad_crc=True)
    r = m.run(out)
    assert r["status"] == "rejected_partial_preserved" and "CRC32" in r["errors"][0]
    assert (out / "decoded.unverified.ckpt").read_bytes() == data
    assert (out / "report.json").exists()


@pytest.mark.parametrize("mutation", ["output", "input", "decoded_bytes", "decoded_crc32",
                                     "decoded_sha256", "deflate_eof", "trailing_bytes",
                                     "deserialized", "official_equivalence", "scientific_admission"])
def test_RAM_consume_mutations(tmp_path, monkeypatch, mutation):
    _, out = file_fixture(tmp_path, monkeypatch)
    r = m.run(out)
    if mutation in ("output", "input", "decoded_sha256", "decoded_crc32"):
        r[mutation] = "changed"
    elif mutation in ("decoded_bytes", "trailing_bytes"):
        r[mutation] += 1
    else:
        r[mutation] = not r[mutation]
    # Synthetic own report only, not any sealed/main record.
    p = out / "report.json"
    p.write_bytes(m.encoded(r))
    with pytest.raises(ValueError):
        m.consume(p, m.sha(p.read_bytes()))


def test_RAM_prepass_rejects_without_output(tmp_path, monkeypatch):
    _, out = file_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(m, "SPEC", dataclasses.replace(m.SPEC, digest="0"*64))
    with pytest.raises(ValueError, match="prepass"):
        m.run(out)
    assert not out.exists()


def test_RAM_second_run_does_not_overwrite(tmp_path, monkeypatch):
    _, out = file_fixture(tmp_path, monkeypatch)
    m.run(out)
    p = out / "decoded.unverified.ckpt"
    old = p.read_bytes()
    with pytest.raises(FileExistsError):
        m.run(out)
    assert p.read_bytes() == old
