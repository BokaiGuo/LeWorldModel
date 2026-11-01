"""Bank conversion/entrypoint contracts only; no model or physical execution."""
import inspect

import numpy as np
import pytest

from experiments.spectral_reachability.run_pusht_d2p_metric_program_preflight import prefix_parameters, run


def bank():
    return [{"proposal_index": i, "prefix_action": [i/32., -i/32.]} for i in range(16)]


def test_every_prefix_preserved_in_original_order_and_parameter_chart():
    rows = bank()[::-1]
    params = prefix_parameters(rows)
    assert params.shape == (16, 1, 2)
    np.testing.assert_allclose(np.tanh(params[:, 0]), [r["prefix_action"] for r in bank()])


@pytest.mark.parametrize("kind", ["missing", "duplicate", "endpoint", "nonfinite", "complex", "nozero"])
def test_bad_bank_rejected_without_clipping_or_dropping(kind):
    rows = bank()
    if kind == "missing": rows.pop()
    elif kind == "duplicate": rows[-1]["proposal_index"] = 0
    elif kind == "endpoint": rows[-1]["prefix_action"][0] = 1.
    elif kind == "nonfinite": rows[-1]["prefix_action"][0] = float("nan")
    elif kind == "complex": rows[-1]["prefix_action"][0] = 1j
    else: rows[0]["prefix_action"] = [.1, .2]
    with pytest.raises(ValueError):
        prefix_parameters(rows)


def test_source_seal_precedes_cuda_and_no_new_physical_future_labels():
    source = inspect.getsource(run)
    assert source.index("create_seal(files=") < source.index('require_cuda_allocation("cuda")')
    assert '"physical_future_labels_used": False' in source
    assert '"P1_admission": False' in source
    assert "execute_packet(" not in source and "env.step(" not in source
    assert "compiler.compile_programs(" in source and "verification = audit(compiled)" in source
