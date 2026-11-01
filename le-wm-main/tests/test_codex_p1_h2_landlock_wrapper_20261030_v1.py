import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/codex_p1_h2_landlock_wrapper_20261030_v1.py"
spec = importlib.util.spec_from_file_location("h2_landlock_wrapper", SCRIPT)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def config(tmp_path):
    source, model, output, denied = [tmp_path / name for name in ("src", "model", "out", "old")]
    for directory in (source, model, output, denied):
        directory.mkdir()
    bundle = denied / "blind.json"
    bundle.write_bytes(b"blind")
    for name in ("weights.pt", "config.json"):
        (model / name).write_bytes(name.encode())
    return dict(python=str(Path(sys.executable).resolve()), runtime_dirs=["/usr"],
                source_dir=str(source), bundle=str(bundle), bundle_sha256=digest(bundle),
                model_dir=str(model), model_sha256={n: digest(model / n)
                for n in ("weights.pt", "config.json")}, output_dir=str(output),
                deny_paths=[str(denied)], cuda_device="0")


def test_policy_and_actual_exec_enforcement(config, tmp_path):
    rules = wrapper.compile_policy(config)
    hidden = Path(config["deny_paths"][0]) / "secret"
    hidden.write_bytes(b"hidden")
    leaked = os.open(hidden, os.O_RDONLY)
    os.set_inheritable(leaked, True)
    program = """
import os,sys
assert open(sys.argv[1],'rb').read() == b'blind'
try: open(sys.argv[2],'rb')
except PermissionError: pass
else: raise AssertionError('hidden readable')
try: os.fstat(int(sys.argv[3]))
except OSError: pass
else: raise AssertionError('inherited fd')
try: open(sys.argv[1], 'wb')
except PermissionError: pass
else: raise AssertionError('blind writable')
open(sys.argv[4], 'w').write('ok')
"""
    try:
        result = wrapper.invoke(
            [config["python"], "-I", "-c", program, config["bundle"], str(hidden),
             str(leaked), str(Path(config["output_dir"]) / "receipt")],
            config["source_dir"], {"PATH": "/usr/bin:/bin", "LANG": "C"}, rules)
    finally:
        os.close(leaked)
    assert result.returncode == 0, result.stderr
    assert (Path(config["output_dir"]) / "receipt").read_text() == "ok"


def test_reject_broad_allowlist_and_tampering(config):
    config["runtime_dirs"] = [str(Path(config["deny_paths"][0]).parent)]
    with pytest.raises(ValueError, match="overlaps"):
        wrapper.compile_policy(config)
    config["runtime_dirs"] = ["/usr"]
    config["bundle_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        wrapper.compile_policy(config)


def test_no_overwrite_or_unreviewed_field(config):
    (Path(config["output_dir"]) / "existing").touch()
    with pytest.raises(ValueError, match="empty"):
        wrapper.compile_policy(config)
    (Path(config["output_dir"]) / "existing").unlink()
    config["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        wrapper.compile_policy(config)


def test_cli_rejects_invalid_manifest_nonzero(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({}))
    result = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(manifest),
                             "--acknowledge", "EXPLORATORY_HOST_CUDA_NOT_PRODUCTION_APPROVED"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert json.loads(result.stderr)["production_authorized"] is False


def test_virtualenv_executable_symlink_and_child_failure(config, tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(Path(config["python"]))
    config["python"] = str(venv / "bin" / "python")
    config["runtime_dirs"].append(str(venv))
    rules = wrapper.compile_policy(config)
    result = wrapper.invoke([config["python"], "-I", "-c", "raise SystemExit(9)"],
                            config["source_dir"], {"PATH": "/usr/bin:/bin"}, rules)
    assert result.returncode == 9
