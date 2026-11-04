# P3 point-30 fixed winners: known-root physical development

The four saved model winners were bound to a new physical plan after the model
outcomes had been opened. Both roots are previously exposed. The physical
executor completed all four ten-action branches; the independent saved-data
consumer checked source binding, journal chain, action prefixes, saved RGB,
termination and resource counts. This is not an independent simulator
attestation or a confirmatory P3 experiment.

| Root | Allocation | Model cost | Final native distance | Object position error | Wrapped object angle error | Native success |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | signal | 1.008468 | 191.686161 | 70.786706 | 0.475903 | no |
| 0 | reverse | 0.877305 | 335.019586 | 70.786706 | 0.475903 | no |
| 1 | signal | 0.229728 | 278.345078 | 76.402680 | 0.082232 | no |
| 1 | reverse | 0.229728 | 278.345078 | 76.402680 | 0.082232 | no |

On root 0 the reverse winner's model cost is lower, but its native distance is
higher; the measured object pose is identical across the two arms. Thus this
instance does not show a task-relevant object-control advantage for either
allocation. Root 1 uses the same winner in both arms and is not a distinct
action comparison. All four ran the complete horizon, with 0/4 native
successes. This does not identify a single cause of the mismatch.

The plan follows the original saved candidate selections, not a new selector.
Its costs retain separate counters, and full model internals, loading, energy,
and end-to-end FLOPs are not certified equal. The consumer first checks the
actual embedded plan exactly; it then augments in-memory copies with a
presentation-only model accounting field for the existing validator. It
does not modify the saved plan or journal. P3/C3 admission remains false.

- Sealed physical plan SHA-256:
  `152e0e6ab143d26a48cde8d99b84ab7815d9aaffafa823eeac8cd66c34e94dda`.
- Physical report SHA-256:
  `0343a14644f74de19d328ea9d11398e9cfc32c0166482d4bd0e775173f217f01`.
- Saved-consumer SHA-256:
  `81bf42b48a870cebc6c1f49d88cd9c123c321e54bba0c3aaca1ae3ec4409e5ed`.
- Plan reconstruction, saved verification, wrong-hash rejection and Q/L2
  context pairing fixtures: 7 passed.
