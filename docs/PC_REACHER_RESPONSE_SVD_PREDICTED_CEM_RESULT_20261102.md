# Response-SVD prefixes with predicted-successor CEM: development

The sealed, model-only development run completed on the two exposed Reacher
roots using host CUDA. It retained the full existing 13-prefix bank and ran
the same 32-candidate, two-packet CEM at each predicted successor. The frozen
model hash and plan postcheck matched. The saved consumer independently
recomputed the eligibility and tie-breaking from saved scores and checked
the full roster, selected parameters and nominal model-call counts. It did
not rerun the neural computations; the original run did not save full arrays
for independent verification of its reconstructed action history.

| Root | Eligible of 12 | Model-selected prefix | Immediate model cost: selected / neutral | Model best suffix cost: selected / neutral |
| --- | ---: | ---: | ---: | ---: |
| 20263160 | 5 | 12 | 1.207776 / 1.191466 | 0.789241 / 0.797402 |
| 20263161 | 6 | 12 | 2.205125 / 2.197972 | 1.945145 / 1.949508 |

The predicted suffixes in this new run were not physically executed. After
model selection, the fixed prefix IDs were joined to **previously opened**
physical records with independently replanned real-cp5-RGB L2 suffixes:

| Root | Initial joint distance | cp5 selected / neutral | Final selected / neutral |
| --- | ---: | ---: | ---: |
| 20263160 | 1.003500 | 1.025701 / 1.010031 | 0.524471 / 0.546408 |
| 20263161 | 1.182600 | 1.256223 / 1.245444 | 0.612893 / 0.598310 |

For these two joint-distance measurements the prefix was not immediately
better; real-feedback suffix replanning was better than neutral on root 0 and
worse on root 1. This comparison does not establish a complete P1 task
criterion, a native success result or an independent planning advantage.
The five prior shared-suffix interventions also show mixed response changes;
for selected prefix 12, the four active-tail neutral-adjusted differences
have signs `[-,+,-,+]` on both roots. Shared-suffix response is not a
substitute for independently optimized suffixes or a proof of useful
controllability. The old all-tail exploratory selector had returned null; this
new rule is a separate development selector, not a modification of that
contract or the P1 admission gate.

These roots, candidates and prior outcomes were already exposed before this
new experiment, so this is not independent confirmation. Both arms use the
same suffix proposal count per prefix, but inherited SVD, encoding, internal
backward work, complete wall time and full baseline costs remain unverified.
The checkpoint training provenance remains unresolved.

- Plan SHA-256:
  `7d7e3160b7154f2c01301bdc36c10a3c764da8ebc074482d3b8a4320bd17aa95`.
- Model report SHA-256:
  `489f72132db81469ff02dd40f95474b188e3845b87f3070923920dd0aabf0c06`.
- Saved selection and prior-result join SHA-256:
  `d47604072fd5e113336af9e4ae156545f4e7426eafbf887089d76e7ac970dd21`.
- Selector, SVD, saved-consumer and bound-analysis tests: 18 passed.

The first plan-preparation attempt wrote its immutable plan before failing
to print the SHA because the CLI supplied a string where a Path was expected.
No model or physics was run in that attempt. The source was corrected and
the above v2 plan separately sealed; v1 remains untouched.
