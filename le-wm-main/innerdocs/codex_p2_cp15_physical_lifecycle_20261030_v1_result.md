# CP15 joint versus sequential suffix: physical exploration

Four fixed jobs were executed once, joint then suffix, with the original
native reset, goal and warmup. All eight trajectories reached plan step 30,
all eight crossed the exact 106-event native/RGB cp15 guard, and none
terminated, truncated, or achieved native success. Each job's separate
saved-only process validated the original sources, action bits, journal,
feedback guard and 252 RGB PNGs. The four receipts were joined in fixed order.
The consumer shares the lifecycle module's validation code; it is a separate
process, not an independently authored implementation.

| Base | Arm | Object position error | Wrapped angle error (rad) | Native success |
|---:|---|---:|---:|---|
| 0 | joint | 69.139 | 0.5593 | false |
| 0 | suffix | 69.139 | 0.5593 | false |
| 1 | joint | 41.265 | 0.1673 | false |
| 1 | suffix | 36.522 | 0.2490 | false |
| 2 | joint | 131.221 | 0.2684 | false |
| 2 | suffix | 131.759 | 0.2808 | false |
| 3 | joint | 131.221 | 0.2684 | false |
| 3 | suffix | 131.759 | 0.2808 | false |

For base 1, suffix has lower endpoint object-position error but larger
angle error. Base 0 endpoints have the same object pose despite different
native-distance values. Bases 2 and 3 duplicate the same saved context and
outcomes, so they are not independent observations. These are descriptive
physical endpoints, not model-Q comparisons or evidence of success.

Each job counted attempted=completed: constructor/close 2 each, reset 4,
public steps 80, physics steps 808, state sets 8, goal setters 6, goal renders
4, goal binds 2 and explicit renders 252. Across all jobs these are respectively
8, 8, 16, 320, 3232, 32, 24, 16, 8 and 1008. These operation axes nest and
are not additive measures of compute. Four inclusive job wall times sum to
28.747 seconds, excluding gaps and separate consumer time. Eight slots used
126 actual RGB captures each. The physical phase made no neural calls.

The prior CUDA panel's matched requests do not establish equal total compute:
model loading, historical acquisition, internal backward operations, energy
and other inherited costs are separate or unknown. The CUDA cooperative lock
was released before process exit and explicit CUDA context cleanup; no claim
of complete lock-lifecycle exclusivity is made. Roots and the panel are
known-result exploration, not a fresh independent confirmation.
Sequential suffix freezing is not complete-program composition. Neither
P1 nor P2 is scientifically admitted by this record.

Joined receipt:
`results/codex_p2_cp15_physical_lifecycle_20261030_v1_JOIN_01.json`
SHA-256 `357beffd7740af13c6ede2d373d02ea1bf4a57a25c442df66310ac4ad0f920d5`.
Per-job run, prepare and consumer receipt filenames contain `B0` through `B3`;
the original CUDA evidence and prior physical outputs were not overwritten.
