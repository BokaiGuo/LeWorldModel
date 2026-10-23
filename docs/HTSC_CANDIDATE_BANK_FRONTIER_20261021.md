# HTSC candidate-bank frontier: why calibration repair alone remains vacuous

## Experiment

Using the same frozen latent-plus-action effect adapter family, train roots,
calibration roots, 100 external roots, and fixed action ordering, we evaluated
four predeclared nested banks: candidate-ID prefixes of size `2`, `4`, `8`,
and `16`. Each arm was run on host CUDA with five deterministic ensemble
members. The runtime certificate still fail-closes because an independently
valid deployment support gate has not been established; a separate
support-passed counterfactual reports whether score margins would otherwise
permit release.

## Result

| Bank size | Source-calibration contrast bound | Source-to-target coverage | Non-vacuous root bounds | Hypothetical source permits | Target-recalibrated bound | Target-outcome coverage | Target permits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.15068 | 85% | 100/100 | 13/100 | 0.22130 | 97.37% | 0/76 |
| 4 | 0.28134 | 87% | 65/100 | 0/100 | 0.43618 | 96.05% | 0/76 |
| 8 | 0.27633 | 80% | 52/100 | 0/100 | 0.43172 | 94.74% | 0/76 |
| 16 | 0.32733 | 78% | 11/100 | 0/100 | 0.54906 | 97.37% | 0/76 |

The 2-action restriction reduces the simultaneous contrast bound sharply and
would permit 13 roots under the source-only threshold if support were valid.
However, the sequential target calibration required by the observed transport
shift raises the bound to `0.22130`, while the largest later-root predicted
margin is only `0.19508`. Thus not one of the 76 untouched target roots can
be released without weakening the certificate.

## Mechanistic conclusion

There are two distinct failure mechanisms:

1. source-domain calibration does not transport, which is repaired by target
   calibration; and
2. even after that repair, predicted action-effect margins are smaller than
   the irreducible calibrated contrast envelope, which leaves the certificate
   correctly nonempty in coverage but empty in control utility.

Candidate-bank pruning addresses only part of the second mechanism. It is not
a sufficient HTSC repair. Future work must reduce state-conditioned
action-effect contrast error (or use an information-gathering probe) while
maintaining a separate target-domain calibration panel. Relaxing the release
rule would invalidate the stated safety interpretation.

## Artifacts

- GPU frontier: `le-wm-main/results/htsc_candidate_bank_frontier_20261021_cuda/`
- Sequential target recalibration: `le-wm-main/results/htsc_candidate_bank_frontier_20261021_target_recalibration/`
