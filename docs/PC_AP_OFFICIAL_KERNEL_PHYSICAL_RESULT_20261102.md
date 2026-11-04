# AP-compatible torch CEM: real-RGB development outcome

The four fixed arms in `PC_AP_OFFICIAL_KERNEL_PHYSICAL_PROTOCOL_20261102.md`
completed ten physical actions each. Frozen model parameter hashes matched
before/after; paired root and goal image hashes matched in both roots.
All eight saved raw action arrays reproduce their executed-action SHA-256,
and each second planning input's RGB hash equals its arm's recorded first
successor hash. Each decision recorded three model prediction batches and
34 candidate blocks. No arm reported environment termination or truncation.

| Exposed root | Initial qpos distance | Observed-target AP kernel | Final-goal kernel |
| --- | ---: | ---: | ---: |
| 20263160 | 1.003500 | 0.804990 | 0.779041 |
| 20263161 | 1.182600 | 1.346018 | 1.019378 |

These are evaluator-only final joint distances, not a certified native
success rate. They describe a two-root, short-horizon development run, with
a three-episode memory and a synthetic goal whose specified retrieval
offset is not an authenticated recorded offset. The terminal reward field
is zero in all four arms; do not use that alone as a replacement success
predicate. Results are not an official AP reproduction, a D2P comparison,
or independent evidence.

Both arms have equal prediction counts in this run, but memory preparation,
retrieval, model loading, encoding and full cost have not been established
as equal resources. The earlier NumPy-kernel development run remains
separate: changing RNG and sample-variance semantics together means any
physical difference cannot be assigned to either alone. This run tests the
new kernel in a real RGB feedback path, not the general theory or P1 gate.

Result SHA-256:
`f2c95637b9dda338bf7ed4165934ac384c4c78308cc0aa932ade776fd04acd91`.
Runner SHA-256:
`a4929eb31f66fcc7d4e5177c1dabe0e7937ac254e6d1fce50f6224f5574f3e44`.
Kernel SHA-256:
`6ddfa22a333a5f899794085dc30604157541dcb9a8d995c33e5e3110fbb1f129`.
