import json

from experiments.gpu_preflight import main
from experiments.gpu_preflight import require_cuda_allocation


def test_gpu_preflight_rejects_cpu(monkeypatch):
    monkeypatch.setattr("sys.argv", ["gpu_preflight", "--device", "cpu"])
    try:
        main()
    except SystemExit as exc:
        assert "requires a CUDA device" in str(exc)
    else:
        raise AssertionError("CPU must not be accepted by the CUDA preflight")


def test_gpu_preflight_reports_sandbox_device_isolation(monkeypatch):
    monkeypatch.setattr("experiments.gpu_preflight.visible_nvidia_nodes", lambda: [])
    try:
        require_cuda_allocation("cuda")
    except RuntimeError as exc:
        message = str(exc)
        assert "no NVIDIA device nodes are visible" in message
        assert "/dev/nvidia" in message
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("missing device nodes must fail closed")


def test_gpu_preflight_persists_structured_failure_without_cpu_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("experiments.gpu_preflight.visible_nvidia_nodes", lambda: [])
    output = tmp_path / "cuda_preflight.json"
    monkeypatch.setattr(
        "sys.argv",
        ["gpu_preflight", "--device", "cuda", "--output", str(output)],
    )
    try:
        main()
    except SystemExit as exc:
        assert exc.code == 1
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("missing device nodes must fail closed")
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "no NVIDIA device nodes are visible" in report["error"]
    assert report["visible_nvidia_nodes"] == []
