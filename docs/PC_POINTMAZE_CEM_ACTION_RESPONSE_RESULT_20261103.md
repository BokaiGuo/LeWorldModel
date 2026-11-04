# Same-action response geometry on fixed CEM candidates

Re-encoded the 384 saved cp10 RGB observations and recomputed the matching
two-packet model predictions on host CUDA. The source model and actual RGB
goal scores were reproduced within the fixed tolerance; checkpoint parameters
were unchanged. For each of eight root/arm groups, both feature sets were
centered over their 48 fixed candidates before comparison.

| Root | Arm | Centered response cosine | Predicted/actual response norm | Relative squared error |
| --- | --- | ---: | ---: | ---: |
| 1884 | random | 0.0090 | 0.2386 | 1.0526 |
| 1884 | spectral | 0.0178 | 0.2471 | 1.0523 |
| 1885 | random | 0.0115 | 0.2975 | 1.0817 |
| 1885 | spectral | -0.0027 | 0.2911 | 1.0863 |
| 1886 | random | 0.0776 | 0.2308 | 1.0174 |
| 1886 | spectral | 0.0761 | 0.2305 | 1.0180 |
| 1887 | random | 0.0333 | 0.2641 | 1.0522 |
| 1887 | spectral | -0.0079 | 0.2410 | 1.0619 |

The cosine is one joint feature-space inner product after candidate-mean
subtraction, not an average of independent action correlations. The norm
ratio compares total centered variation, not a calibrated physical gain.
In this panel, near-zero cosines and ratios around 0.23–0.30 show that a
single action-independent additive offset cannot explain all the observed
feature discrepancy. They do **not** prove a particular model defect, a
unique causal mechanism, or failure of every task-relevant direction.
The raw task-mask feature vectors and projections on the initial-to-goal
direction are saved for further analysis.

Real candidate futures were used in centering, so these statistics cannot be
used as a deployable scorer. The four roots and CEM banks are already exposed,
and candidate samples share starts and search ancestry. The model's constant
raw-zero proprio and archive-wide normalizer are not authenticated as the
checkpoint's original training interface. No new physical outcome, control
advantage, independent generalization or P1 passage is claimed.

- Protocol: `docs/PC_POINTMAZE_CEM_ACTION_RESPONSE_PROTOCOL_20261103.md`
- Runner: `le-wm-main/experiments/a0_pointmaze_cem_action_response_20261103.py`
- Report SHA-256:
  `0f7492d9be9c39d07ee4805c359c252f50a354d20214ce1f4634fce39023b471`
- Array SHA-256:
  `a0c3d2cccbf9d3790a9a46cf57b40e7ffb6a288e6b9f0869710c35ac3b80335e`
- Validation: `tests/test_a0_pointmaze_cem_action_response_20261103.py` and
  the same-action feedback tests: five passed. The tests recompute saved
  feature arithmetic, not an independent model or simulator implementation.
