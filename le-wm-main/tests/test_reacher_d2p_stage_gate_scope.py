"""Guard that the scalar stage admission fix did not edit neural execution."""
import ast
from pathlib import Path

from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


def test_gate_patch_preserves_every_neural_and_execution_function():
    archived = Path("results/reacher_d2p_certified_task_full_drive_fresh8_20261025_cuda.implementation/p0.py")
    if not archived.exists():
        import pytest
        pytest.skip("local immutable runtime source snapshot is not shipped with unit tests")
    current_ast = ast.parse(Path(p0.__file__).read_text())
    archived_ast = ast.parse(archived.read_text())
    def functions(tree):
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name not in ("_p1_admission", "main")}
    assert len(functions(current_ast)) >= 20
    assert functions(current_ast) == functions(archived_ast)


def test_p1_gate_consumer_patch_preserves_compilation_and_execution_functions():
    from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as selector
    from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
    folder = Path("results/reacher_d2p_certified_task_full_drive_fresh8_20261025_cuda.implementation")
    for current, archived, excluded in ((Path(selector.__file__), folder / "prefix_suffix_backend.py", {"main", "_validate_upstream_p0"}),
                                       (Path(runner.__file__), folder / "runner.py", {"main"})):
        if not archived.exists():
            import pytest
            pytest.skip("local immutable implementation snapshot not shipped with unit tests")
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name not in excluded}
        assert len(functions(current)) >= 10
        assert functions(current) == functions(archived)
