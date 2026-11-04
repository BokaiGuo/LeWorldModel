import pytest

from experiments.pc_runtime_preflight_v1 import run_before_artifacts


def test_failure_does_not_start_or_create_output(tmp_path):
    output = tmp_path / "manifest.json"

    def fail():
        raise RuntimeError("device unavailable")

    with pytest.raises(RuntimeError, match="device unavailable"):
        run_before_artifacts([lambda: None, fail], lambda: output.write_text("{}"))
    assert not output.exists()


def test_checks_precede_start():
    calls = []
    result = run_before_artifacts(
        [lambda: calls.append("cuda"), lambda: calls.append("render")],
        lambda: calls.append("start") or "ok",
    )
    assert result == "ok"
    assert calls == ["cuda", "render", "start"]
