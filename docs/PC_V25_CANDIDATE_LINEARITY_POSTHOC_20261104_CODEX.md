# v25 candidate-scale linearity: post-hoc E0 audit

The audit reads the four exposed roots of the stopped v25 fixed-bank panel;
it does not run new trajectories or change their selections. For each of
the eight first-layer nominal points it verifies the recorded 0.05
centered-difference probe actions and reconstructs the recorded augmented
response matrix. It then compares each actual spectral candidate's
recorded endpoint and score with the local linear prediction using that
candidate's actual bounded action difference.

| Requested absolute scale | Candidates | Median position absolute error | Median relative position-change error | Median score absolute error | Maximum position absolute error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 32 | 0.0911 | 0.4075 | 0.0437 | 0.1858 |
| 0.50 | 32 | 0.3322 | 0.8431 | 0.2102 | 0.7164 |

Relative error divides by the norm of the *observed change from the
corresponding nominal*, and would be omitted for norms at or below
`1e-8`. Among 224 within-nominal candidate pairs, the linearized score
ordering agreed for 190, disagreed for 34, with no ties at the declared
`1e-10` tolerance. Pairs share candidates and roots, and are not
independent observations.

The observed scale dependence is an engineering warning: a numerically
correct local derivative is not by itself an accurate prediction at the
actual 0.25/0.5 candidate amplitudes. The score component is itself part
of v25's augmented response. This retrospective, same-exact-dynamics,
first-layer analysis does not test visual model versus physical response,
later states, or PC1 as a whole; nor does it show that this error caused
the planning failure. Re-linearization, smaller steps, or nonlocal
fallback are hypotheses requiring a new cost-matched protocol.

Raw per-candidate data and source hashes:
`le-wm-main/results/pc_v25_fixed_bank_suffix_e0_dev_20261104_codex/candidate_linearity_posthoc.json`.
