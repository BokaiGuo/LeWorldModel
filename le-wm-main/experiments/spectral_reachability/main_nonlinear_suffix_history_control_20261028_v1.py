"""Saved history3 control arithmetic, separate from the frozen primary protocol."""
import numpy as np
from experiments.spectral_reachability import main_nonlinear_suffix_midpoint_effect_20261028_v1 as primary


def history_diagnostics(root_history, imagined_mid_history, observed_mid_history):
    """Matched old2 controls; no threshold or batch-layout cause attribution."""
    histories = []
    for h in (root_history, imagined_mid_history, observed_mid_history):
        primary.require(np.asarray(h).shape == (3, 192), 'complete native history3')
        histories.append(np.stack([primary.vector(row) for row in h]))
    root, imagined, observed = histories
    primary.require(np.array_equal(root[1:], imagined[:2]),
                    'imagined midpoint retains original two causal RGB coordinates')
    rows = []
    for i in range(3):
        delta = observed[i]-imagined[i]
        rows.append(dict(position=i, role='matched_old_observation' if i < 2 else 'new_actual_midpoint',
                         exact_equal=bool(np.array_equal(observed[i], imagined[i])),
                         MSE=float(np.mean(delta**2)), max_abs=float(np.max(np.abs(delta)))))
    return dict(rows=rows, matched_old2_exact=all(r['exact_equal'] for r in rows[:2]),
                treatment='replace complete observed history3',
                drift_cause_attributed=False, last_only_intervention=False)
