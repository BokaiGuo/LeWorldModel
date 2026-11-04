# P2 conditional initialization: development physical execution

## Scope and intervention

The two fixed model-selected suffixes from the conditional P2 handoff were
executed on the same exposed B0 PushT root after the same 15-step prefix.
The composition and no-composition arms used distinct fresh environments.
The result is exploratory: the model-side costs are unequal and there is only
one development root. A later saved-output consistency audit is described
below. This is not P1 or P2 admission.

The first execution failed in the environment constructor because the
read-only bubblewrap view left no usable temporary directory. Its report
SHA-256 is `f34bdb44a9a067723ecb85bb1ab79940daf85d725dd04134a4192aea0f98c767`;
it recorded zero public and physics steps. It is an engineering failure, not
a physical negative result. The original session was preserved.

The separately named `p2_conditional_physical_executor_tmpfix_20261102_exclusive.py`
sets `TMPDIR` to the new exclusive writable session inside the same read-only
view. Its independently prepared plan SHA-256 is
`11efba074c71716e66cd64ecfa85916e903d3ae8ce572ab14d68572f35b90ac0`.
The model-selected actions and physical protocol were not changed.

## Recorded result

Report:
`le-wm-main/results/p2_conditional_physical_executor_tmpfix_20261102_exclusive_SESSION_01/codex_p2_multistart_joint_physical_20261030_v1_CONDITIONAL_01/report.json`

Report SHA-256:
`dfcbfbbb0b9f1561f29c6069a7a36ac30b9314b03766f7e58e4e659712f80dda`.
Status is `recorded_development_unconsumed`; the execution-level postchecks
reported no errors. Both arms completed 25 plan steps and neither reported
native success. Both returned the same bound actual/compile context hash
at the common prefix. Total recorded operations include 70 public steps
and 708 physics steps across both branches.

| Metric at plan step 25 | Composition | No composition |
| --- | ---: | ---: |
| Native success | false | false |
| Native distance | 133.8481 | 150.1829 |
| Object position error | 83.7078 | 82.6193 |
| Wrapped object angle error (rad) | 1.1573 | 1.0390 |

Metric definitions and goal references are those in the saved native report;
these columns are not interchangeable. The lower native distance in the
composition arm does not establish improved object control, task success,
equal-budget superiority, or generalization. The earlier model-space
advantage also cannot be promoted to such a claim.

## Saved-output finalization correction

The initial saved consumer incorrectly required the separate `completion.json`
of the historical producer. This tmpfix producer instead finalizes with
`conditional_finished` and then writes `report.json`; it never specifies or
writes that separate completion file. The missing file alone is therefore
not evidence of failed tmpfix execution. The prior consumer and its receipt
are preserved; neither is retroactively altered.

The separately implemented `p2_conditional_tmpfix_finalization_consumer_20261102_exclusive`
checked the pinned report, 2,410-event chain, 1,028 operation receipts, 222 PNG
file/pixel hashes, two literal action sequences, prefix/witness binding, and
final event/report consistency. Five CPU tests passed, including tampering
cases. This certifies saved-record consistency on this known root, not a
separate producer completion certificate, post-report wall-time bound, or
physical execution from bytes alone. It reuses historical validation helpers.
The independent-root, equal-budget and strong-baseline comparisons remain.
