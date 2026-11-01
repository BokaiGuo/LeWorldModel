"""Non-neural adversarial archive and static identity tests only."""
import importlib.util
from pathlib import Path
import pickle
import stat
import struct
import zipfile

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/p5_architecture_readiness_20261027_v2.py"
SPEC = importlib.util.spec_from_file_location("p5_readiness_v2", SOURCE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def archive(tmp_path, data=None, extras=()):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("checkpoint/data.pkl", data if data is not None else pickle.dumps({"encoder": {}}, protocol=2))
        for name, payload in extras:
            handle.writestr(name, payload)
    return path


@pytest.mark.parametrize("name", ["../data.pkl", "/tmp/x", "C:/x", "C:x", "a/../../x", "a\\..\\x", "a\x00x", ""])
def test_unsafe_paths(name):
    with pytest.raises(ValueError):
        m.safe_name(name)


def test_encoder_is_not_dynamics(tmp_path):
    result = m.inspect_archive(archive(tmp_path))
    assert result["identity"] == "unresolved"
    assert not result["state_dict_loaded"]


def test_separate_dynamics_metadata(tmp_path):
    names = ["predictor", "module.predictor_embed.weight", "module.action_encoder.weight",
             "module.predictor_blocks.0.adaLN_modulation.1.weight", "proprio_encoder"]
    names += [f"module.predictor_blocks.{i}.norm1.weight" for i in range(6)]
    result = m.inspect_archive(archive(tmp_path, pickle.dumps(names, protocol=2)))
    assert result["identity"] == "JEPA_WM_AdaLN_action_dynamics_static_match"
    assert not result["tensor_shapes_verified"]  # marker spoofing is not certified identity


def test_never_executes_pickle_reduce(tmp_path):
    marker = tmp_path / "executed"
    # Arbitrary GLOBAL/REDUCE instructions are scanned, never invoked.
    payload = b"cos\nsystem\n(S'touch " + str(marker).encode() + b"'\ntR."
    result = m.inspect_archive(archive(tmp_path, payload))
    assert "os system" in result["opcode_globals_not_executed"]
    assert not marker.exists()


@pytest.mark.parametrize("payload", [b"broken", b"N", b"N.extra", b"\x80\x04\x93."])
def test_malformed_or_unsupported_pickle(tmp_path, payload):
    with pytest.raises((ValueError, EOFError)):
        m.inspect_archive(archive(tmp_path, payload))


def test_path_traversal_archive(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        m.inspect_archive(archive(tmp_path, extras=[("../bad", b"x")]))


def test_symlink(tmp_path):
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError, match="nonregular"):
        m.inspect_archive(archive(tmp_path, extras=[(link, b"../../x")]))


def test_duplicate(tmp_path):
    with pytest.warns(UserWarning):
        path = archive(tmp_path, extras=[("checkpoint/data.pkl", b"N.")])
    with pytest.raises(ValueError, match="duplicate"):
        m.inspect_archive(path)


def test_metadata_size(tmp_path):
    with pytest.raises(ValueError, match="bounded"):
        m.inspect_archive(archive(tmp_path, b"x" * (m.MAX_METADATA + 1)))


def test_zip_bomb_ratio(tmp_path):
    path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("checkpoint/data.pkl", b"x" * 100000)
    with pytest.raises(ValueError, match="expansion"):
        m.inspect_archive(path)


def test_central_directory_preallocation_bound(tmp_path):
    path = tmp_path / "oversized-central.zip"
    path.write_bytes(struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 1, 1, 5000000, 0, 0))
    with pytest.raises(ValueError, match="central"):
        m.inspect_archive(path)


def test_no_archive_extraction_or_model_imports():
    import ast
    tree = ast.parse(SOURCE.read_text())
    forbidden = {"torch", "pickle", "tarfile", "transformers"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.split(".")[0] in forbidden for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] not in forbidden
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"extract", "extractall", "Unpickler"}
            if node.attr == "loads":
                assert isinstance(node.value, ast.Name) and node.value.id == "json"


def test_non_zip_refused(tmp_path):
    path = tmp_path / "legacy.pth.tar"
    path.write_bytes(b"not a ZIP archive")
    with pytest.raises(ValueError, match="supported ZIP"):
        m.inspect_archive(path)
