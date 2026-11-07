from experiments.pc_rgb_gradient_public_filetrace_audit_20261107_codex import FORBIDDEN, REQUIRED, inspect


def test_trace_requires_positive_controls_and_rejects_even_denied_forbidden_access():
    lines = [f'1 openat(AT_FDCWD, "{path}", O_RDONLY) = 3\n' for path in REQUIRED]
    assert inspect(lines)["pass"]
    assert not inspect(lines[:-1])["pass"]
    assert not inspect(lines + [f'1 openat(AT_FDCWD, "{FORBIDDEN[0]}", O_RDONLY) = -1 EACCES\n'])["pass"]
