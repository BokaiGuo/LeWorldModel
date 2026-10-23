# HTSC target-domain recalibration: resolving the transport-coverage failure

## Original negative result

The sealed fixed-bank source-domain calibration used 14 calibration roots at
`alpha=0.1`, producing a valid finite-sample root-maximum contrast bound of
`0.33369`. On 100 disjoint target roots its simultaneous candidate-bank
coverage was only `80/100`. The train-only latent Mahalanobis support rule
abstained on 25 roots, but failures occurred at the same `20%` rate in both
the supported and abstained subsets. Consequently, source-latent OOD scoring
does not explain or repair this transport failure.

## Root cause and intervention

The failure is a calibration-population mismatch, not the old conformal rank
bug: the implementation already uses the valid
`ceil((n+1)(1-alpha))` order statistic and fails closed when it is unavailable.
The source-domain contrast-error distribution simply does not transport to
the deployment root distribution.

We tested a different, explicit deployment contract. Under a lexicographic
root-arrival order, target roots `1900`--`1923` form a 24-root *target*
calibration panel; they are used only to calibrate the root-maximum contrast
threshold. Later roots `1924`--`1999` remain untouched until after that
threshold is frozen. With `alpha=0.1`, the resulting threshold is `0.49899`.

On the 76 untouched target roots:

| Quantity | Result |
| --- | ---: |
| Simultaneous fixed-bank coverage | `73/76 = 96.05%` |
| Wilson 95% interval | `[88.91%, 98.67%]` |
| Target calibration size | `24` roots |
| Target outcome size | `76` roots |

## Interpretation

This is evidence that the observed negative result is repairable by collecting
and using a small, distribution-matched calibration panel before deployment.
It is not evidence that the original source-to-target certificate was valid,
that coverage is conditional on release, that the policy may certify adaptive
CEM candidates, or that HTSC improves closed-loop control. The original
source-only transport result remains a necessary failure boundary.

The intervention does **not** yet solve HTSC utility: even the smaller
source-domain bound had zero released candidate banks in this 16-action
evaluation (and only 10% non-vacuous joint-regret bounds). The target bound is
larger, as expected for a shifted domain, so it cannot increase release under
the same predicted margins. The next HTSC method problem is therefore
tightening the action-effect contrast model while preserving this independent
target-domain calibration, rather than weakening the release condition.

## Permanent implementation change

`audit_htsc_target_domain_recalibration.py` records an immutable sequential
split, calibrates only target-prefix root errors, refuses an insufficient
prefix, and evaluates only the untouched suffix. It keeps root-level maximum
contrast errors as the resampling/coverage unit.

## Artifacts

- Source-domain failure: `le-wm-main/results/htsc_finite_sample_calibration_20261021_cuda_external100_v4/finite_sample_audit.json`
- Target-domain sequential audit: `le-wm-main/results/htsc_target_domain_recalibration_20261021_external100_prefix24/target_recalibration.json`
