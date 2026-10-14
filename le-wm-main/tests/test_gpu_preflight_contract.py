from experiments.gpu_preflight import main


def test_gpu_preflight_rejects_cpu(monkeypatch):
    monkeypatch.setattr("sys.argv", ["gpu_preflight", "--device", "cpu"])
    try:
        main()
    except SystemExit as exc:
        assert "requires a CUDA device" in str(exc)
    else:
        raise AssertionError("CPU must not be accepted by the CUDA preflight")
