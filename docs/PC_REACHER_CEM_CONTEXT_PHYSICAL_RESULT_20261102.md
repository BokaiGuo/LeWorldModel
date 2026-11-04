# Saved-context CEM tail physical replay: development results

All 48 scheduled branches returned 15 physical steps. The saved-only
consumer checked the journal hash chain, all 720 intent/return action pairs,
the complete branch order, action hashes, root and goal identities, each
stored RGB payload and joint-distance calculation, stop flags, and identical
root/prefix traces. This is consistency of saved data, not an independent
simulator attestation.

The entries below are `observed-context final distance - predicted-context
final distance`; negative values favor the observed-context tail. Distances
are measured in the existing wrapped joint-distance metric. In the small
`16 × 2` CEM setting the largest absolute difference across 12 pairs was
`3.81e-8`. For `32 × 3`, the six pairs with absolute differences above
the descriptive `1e-6` tolerance are:

| Root | Prefix | CEM seed | Predicted tail | Observed tail | Difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20263160 | 0 | 20268160 | 0.606452 | 0.511719 | -0.094734 |
| 20263160 | 12 | 20268160 | 0.536844 | 0.559084 | +0.022240 |
| 20263160 | 12 | 20268161 | 0.497896 | 0.516210 | +0.018314 |
| 20263161 | 0 | 20268163 | 0.742866 | 0.745016 | +0.002150 |
| 20263161 | 12 | 20268161 | 0.719230 | 0.668234 | -0.050996 |
| 20263161 | 12 | 20268162 | 0.734284 | 0.666119 | -0.068165 |

Across all twelve larger-budget pairs, three differences are negative,
three positive and six within `1e-6`; the descriptive mean difference is
`-0.014266`. The tolerance and this sign summary are descriptive, not a
preregistered statistical test. These 12 pairs share only two exposed roots
and must not be treated as 12 independent tasks. The complete 24-pair table
is in the saved receipt.

The selected suffixes were computed on archived contexts before this replay;
this does not establish a new online RGB-feedback controller, feedback
advantage, P1 mechanism, independent generalization, or fair strong-baseline
victory. CPU decoding of stored tail parameters is not certified bitwise
identical to the CUDA transform. No outcomes or arms were excluded, no neural
calls were made, and no model was trained in the replay.

- Plan SHA-256: `d640b146bb78125eac19b9a940c50cb70fe05cdf30c3a307ff90ba3f2bd81a8f`.
- Journal SHA-256: `f5f0663c49889d83fad103c1102bbfe9661d55ef10e29bb5af0d1967a8657721`.
- Saved receipt SHA-256: `84756c414cc15b28132c571a5c048a25f010558b05ebd88f054e11c2c1faf1b1`.
- Fourteen targeted tests passed, including branch ordering, action, stop and
  RGB tampering, plus the earlier context-alignment tests.
