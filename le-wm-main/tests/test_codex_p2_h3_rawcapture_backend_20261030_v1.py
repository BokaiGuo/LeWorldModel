from experiments.spectral_reachability import codex_p2_h3_rawcapture_backend_20261030_v1 as m
from experiments.spectral_reachability import codex_p2_symmetric_native_bridge_20261030_v1 as old


def test_future_backend_overrides_terminal_without_patching_sealed_source():
    assert issubclass(m.RawCaptureBackend, old.NativeCudaBackend)
    assert m.RawCaptureBackend._terminal is not old.NativeCudaBackend._terminal

