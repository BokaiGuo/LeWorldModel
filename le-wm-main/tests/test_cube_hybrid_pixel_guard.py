from experiments.run_cube_hybrid_contact_latent import _pixel_guard_should_fallback


def test_pixel_guard_triggers_only_after_fixed_error_margin():
    assert not _pixel_guard_should_fallback(None, 4.0, False, 0.5)
    assert not _pixel_guard_should_fallback(4.0, 4.5, False, 0.5)
    assert _pixel_guard_should_fallback(4.0, 4.51, False, 0.5)


def test_pixel_guard_is_irreversible_after_trigger():
    assert _pixel_guard_should_fallback(9.0, 0.0, True, 0.5)


def test_pixel_guard_rejects_negative_threshold():
    try:
        _pixel_guard_should_fallback(1.0, 2.0, False, -0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative pixel guard threshold must be rejected")
