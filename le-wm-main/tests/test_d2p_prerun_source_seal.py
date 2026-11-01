import json

import pytest

from experiments.spectral_reachability import d2p_prerun_source_seal as seal


def setup_seal(tmp_path):
    source = tmp_path / "runner.py"
    source.write_bytes(b"# explicit fixture\n")
    receipt = seal.create_seal(files=[source], args_json='{"seed": 4}',
                               output_dir=tmp_path / "seal")
    return source, receipt


def verify(receipt):
    return seal.verify_seal(output_dir=receipt["output_dir"],
                            expected_manifest_sha256=receipt["manifest_sha256"])


def test_roundtrip_and_nooverwrite(tmp_path):
    source, receipt = setup_seal(tmp_path)
    assert verify(receipt)["status"] == "verified"
    output = tmp_path / "seal"
    assert (output / "sources/000.source").read_bytes() == source.read_bytes()
    assert json.loads((output / "args.json").read_text()) == {"seed": 4}
    with pytest.raises(FileExistsError):
        seal.create_seal(files=[source], args_json="{}", output_dir=output)
    assert verify(receipt)["source_count"] == 1


@pytest.mark.parametrize("target", ["source", "sources/000.source", "args.json", "manifest.json"])
def test_tampering_detected(tmp_path, target):
    source, receipt = setup_seal(tmp_path)
    path = source if target == "source" else tmp_path / "seal" / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="mismatch"):
        verify(receipt)


@pytest.mark.parametrize("kind", ["dir", "symlink", "ancestor", "duplicate", "empty", "missing"])
def test_invalid_inputs_leave_no_output(tmp_path, kind):
    source = tmp_path / "input.py"
    source.write_text("fixture")
    files = [source]
    if kind == "dir":
        files = [tmp_path]
    elif kind == "symlink":
        link = tmp_path / "link"
        link.symlink_to(source)
        files = [link]
    elif kind == "ancestor":
        link = tmp_path / "link"
        link.symlink_to(tmp_path, target_is_directory=True)
        files = [link / source.name]
    elif kind == "duplicate":
        files *= 2
    elif kind == "empty":
        files = []
    else:
        files = [tmp_path / "missing"]
    with pytest.raises((ValueError, OSError)):
        seal.create_seal(files=files, args_json="{}", output_dir=tmp_path / "seal")
    assert not (tmp_path / "seal").exists()


@pytest.mark.parametrize("args", ["[]", '{"x": NaN}', '{"x": Infinity}', "not JSON"])
def test_bad_args(tmp_path, args):
    source = tmp_path / "input"
    source.write_bytes(b"x")
    with pytest.raises(ValueError):
        seal.create_seal(files=[source], args_json=args, output_dir=tmp_path / "seal")
    assert not (tmp_path / "seal").exists()


def test_fixed_size_and_count_limits(tmp_path):
    large = tmp_path / "weights"
    with large.open("wb") as stream:
        stream.truncate(seal.MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match="bounded"):
        seal.create_seal(files=[large], args_json="{}", output_dir=tmp_path / "seal")
    with pytest.raises(ValueError, match="too many"):
        seal.create_seal(files=[large] * 65, args_json="{}", output_dir=tmp_path / "seal")
    paths = []
    for i in range(9):
        path = tmp_path / str(i)
        with path.open("wb") as stream:
            stream.truncate(seal.MAX_FILE_BYTES)
        paths.append(path)
    with pytest.raises(ValueError, match="total"):
        seal.create_seal(files=paths, args_json="{}", output_dir=tmp_path / "seal")
    assert not (tmp_path / "seal").exists()


def test_replacement_symlink_and_wrong_external_pin(tmp_path):
    source, receipt = setup_seal(tmp_path)
    with pytest.raises(ValueError, match="manifest"):
        seal.verify_seal(output_dir=receipt["output_dir"], expected_manifest_sha256="0" * 64)
    source.unlink()
    source.symlink_to(tmp_path / "seal/sources/000.source")
    with pytest.raises(ValueError, match="symlink"):
        verify(receipt)


def test_output_symlink_rejected(tmp_path):
    source = tmp_path / "input"
    source.write_text("fixture")
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        seal.create_seal(files=[source], args_json="{}", output_dir=link / "seal")


def test_same_basename_distinct_sources_and_missing_archive(tmp_path):
    paths = []
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        path = folder / "runner.py"
        path.write_text(name)
        paths.append(path)
    receipt = seal.create_seal(files=paths, args_json="{}", output_dir=tmp_path / "seal")
    assert verify(receipt)["source_count"] == 2
    assert (tmp_path / "seal/sources/000.source").read_text() == "a"
    assert (tmp_path / "seal/sources/001.source").read_text() == "b"
    (tmp_path / "seal/sources/001.source").unlink()
    with pytest.raises(FileNotFoundError):
        verify(receipt)


def test_fifo_rejected_without_blocking(tmp_path):
    import os

    path = tmp_path / "fifo"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular"):
        seal.create_seal(files=[path], args_json="{}", output_dir=tmp_path / "seal")


@pytest.mark.parametrize("args", ['{"seed":1,"seed":2}', '{"nested":{"x":1,"x":2}}'])
def test_duplicate_json_keys_rejected(tmp_path, args):
    with pytest.raises(ValueError, match="duplicate JSON"):
        seal.create_seal(files=[], args_json=args, output_dir=tmp_path / "seal")
    assert not (tmp_path / "seal").exists()


def test_parent_traversal_rejected_before_normalization(tmp_path):
    source = tmp_path / "input"
    source.write_text("fixture")
    with pytest.raises(ValueError, match="traversal"):
        seal.create_seal(files=[tmp_path / "missing/../input"], args_json="{}",
                         output_dir=tmp_path / "seal")


def test_creation_detects_source_change_and_failed_directory_not_reusable(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.write_bytes(b"original")
    original_verify = seal.verify_seal

    def changed_before_verify(**kwargs):
        source.write_bytes(b"modified")
        return original_verify(**kwargs)

    monkeypatch.setattr(seal, "verify_seal", changed_before_verify)
    with pytest.raises(ValueError, match="source SHA256"):
        seal.create_seal(files=[source], args_json="{}", output_dir=tmp_path / "seal")
    assert (tmp_path / "seal").is_dir()
    monkeypatch.setattr(seal, "verify_seal", original_verify)
    with pytest.raises(FileExistsError):
        seal.create_seal(files=[source], args_json="{}", output_dir=tmp_path / "seal")


def test_args_limits_and_invalid_files_container(tmp_path):
    with pytest.raises(ValueError, match="bounded args"):
        seal.create_seal(files=[], args_json='{"x":"' + 'x' * seal.MAX_ARGS_BYTES + '"}',
                         output_dir=tmp_path / "seal")
    # Compact input can expand beyond the canonical serialized limit.
    with pytest.raises(ValueError, match="canonical args"):
        seal.create_seal(files=[], args_json=json.dumps({"x": [0] * 50000}, separators=(",", ":")),
                         output_dir=tmp_path / "seal")
    with pytest.raises(ValueError, match="file list"):
        seal.create_seal(files="input.py", args_json="{}", output_dir=tmp_path / "seal")


def test_relative_source_and_external_receipts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "input.py"
    source.write_text("fixture")
    args = {"weights_receipt": {"sha256": "a" * 64, "verified_by_caller": True}}
    receipt = seal.create_seal(files=["input.py"], args_json=json.dumps(args), output_dir="seal")
    assert verify(receipt)["status"] == "verified"
    manifest = json.loads((tmp_path / "seal/manifest.json").read_text())
    assert manifest["files"][0]["source"] == str(source)
    assert json.loads((tmp_path / "seal/args.json").read_text()) == args
