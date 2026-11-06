from experiments.spectral_reachability import pc_recursive_response_v14_timed as timed


def test_timing_covers_outer_compiler_without_mutating_child(monkeypatch):
    original = {"method": "old", "resource": {"wall_seconds": 1.5, "prediction_tokens": 42},
                "selected": [[0.]]}
    monkeypatch.setattr(timed.v14, "compile_program", lambda *args, **kwargs: original)
    ticks = iter((10., 14.))
    monkeypatch.setattr(timed.time, "monotonic", lambda: next(ticks))
    result = timed.compile_program(None, None, None)
    assert result["resource"]["wall_seconds"] == 4.
    assert result["resource"]["child_wall_seconds"] == 1.5
    assert result["resource"]["prediction_tokens"] == 42
    assert original["resource"]["wall_seconds"] == 1.5
    assert result["selected"] == original["selected"]
