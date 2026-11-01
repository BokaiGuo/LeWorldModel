import json
import os

import pytest

from experiments.spectral_reachability import codex_p1_planner_isolation_probe_20261030_v1 as probe


def test_kernel_mount_isolation_and_whitelisted_stdin(tmp_path):
    secret = tmp_path / "outcome.txt"
    secret.write_text("SECRET", encoding="utf8")
    program = tmp_path / "probe.py"
    program.write_text(
        "import json,os,sys\n"
        "x=json.loads(sys.stdin.buffer.read())\n"
        "print(json.dumps({'input':x,'secret_visible':os.path.exists(x['secret']),"
        "'gpu_visible':os.path.exists('/dev/nvidia0'),'home_visible':os.path.exists('/home/gbk')}))\n",
        encoding="utf8")
    result = probe.probe(program, json.dumps({"secret": str(secret), "cp5": 5}).encode())
    assert result.returncode == 0, result.stderr.decode()
    obj = json.loads(result.stdout)
    assert obj["input"]["cp5"] == 5
    assert not obj["secret_visible"] and not obj["home_visible"] and not obj["gpu_visible"]


def test_rejects_symlink_and_oversize(tmp_path):
    program = tmp_path / "p.py"
    program.write_text("print(1)", encoding="utf8")
    link = tmp_path / "link.py"
    os.symlink(program, link)
    with pytest.raises(ValueError):
        probe.probe(link, b"")
    with pytest.raises(ValueError):
        probe.probe(program, b"x" * (2 * 1024 * 1024 + 1))
