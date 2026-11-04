# Q-consistent suffix search: known-root model result

The fixed-bank protocol in `PC_REACHER_Q_CONSISTENT_SUFFIX_PROTOCOL_20261102.md`
completed on the frozen model with host CUDA. All 8 and 10 original/refined
prefix candidates were retained. Each prefix's immediate Q cost passed the
saved-score consistency check; model parameter hashes matched before/after.

| Known root | Previous Q-selected prefix | New Q-consistent selected prefix | Eligible prefixes |
| --- | ---: | ---: | ---: |
| 20263160 | 3 | 3 | 2 |
| 20263161 | none, zero fallback | 3 | 2 |

For root 20263161, prefix 3 has model immediate Q cost 0.536031 versus
zero's 0.526716, and optimized terminal Q cost 0.198142 versus zero's
0.336721. This is a model-space comparison against zero, not evidence of
physical non-immediate progress or future control capability. Its same-bank
Q-greedy prefix 4 has a lower model terminal Q cost, 0.193148, so the fixed
zero-reference selector does not establish superiority to greedy.

The run used 520 and 650 newly counted predictor sample-transitions
respectively (prefix plus two-token suffix samples), and 40 and 50 predictor
API calls. The reported model counts exclude goal and context encoding,
the upstream prefix construction, and the inherited goal-Q 192 VJPs per
root. The same finite-budget suffix search uses the same random seed for
each candidate within a root; it does not prove globally optimal suffixes.
No physical action, independent root or same-total-budget comparison was
performed. Prior physical results cannot be assigned to newly selected
actions or newly optimized suffixes.

Report SHA-256:
`890accde2e8268677175ad8c0333c4111f88b09128b10c5faa2fec4fd257267a`.
Runner SHA-256:
`4059c60699f3327157cccdf49f66be4399f9f0cf9402afe39ee48a8630d890c2`.
