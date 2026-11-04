# Predicted CEM tails: known-root physical execution and saved verification

This is a development experiment on two previously exposed Reacher roots, not
an independent confirmation or a Planning Compiler P1 admission. The
model-selected prefix and its predicted CEM tail were sealed before this
physical execution. Each selected or neutral prefix was followed by its own
predicted ten-step tail without a cp5 observation or replanning.

The host execution returned four complete 15-step branches; no branch stopped
early. The saved-only consumer checks the plan and journal hashes, journal
chain, source-bound reconstructed plan, ordered branch roster, action bytes
and per-step action binding, RGB payload hashes, state/target binding,
recomputed joint distances, and termination fields. It does not independently
replay the simulator. Its receipt SHA-256 is
`5d581f122ac10ded662964fc5349dd7e4effee34149fafb020243f475dffcf00`.

| Root | Prefix | Initial joint distance | cp5 | Final |
| --- | --- | ---: | ---: | ---: |
| 20263160 | selected | 1.0035004 | 1.0257013 | 0.5244712 |
| 20263160 | neutral | 1.0035004 | 1.0100306 | 0.5464076 |
| 20263161 | selected | 1.1826005 | 1.2562233 | 0.6128934 |
| 20263161 | neutral | 1.1826005 | 1.2454437 | 0.5983105 |

Both selected prefixes increase the measured joint distance relative to
their own starting state. Selected beats neutral at the first root by
approximately 0.02194 in final distance, and loses at the second by 0.01458.
These are two correlated development cases, not a significance test or
evidence of a general capability change. The arms use different tails,
so this comparison is not the fixed-common-tail mechanism intervention.

## Important correction about the earlier feedback comparison

The previously saved cp5 real-RGB replanning run used the same nominal CEM
seed and settings. Its saved selected and neutral suffix actions differ from
the respective predicted-tail actions here by at most
`5.9604645e-08`, `5.9604645e-08`, `1.1920929e-07`, and `1.1920929e-07`,
in table order. They are not bitwise identical, but are effectively the
same action sequences at this resolution. The near-identical final distances
therefore do **not** constitute evidence that real feedback improves or
fails to improve control: the two procedures did not generate meaningfully
different executed tails in these four cases. This does not establish why
the optimizer chose the same actions. The saved actions are not generally
saturated at the box bounds. The seed, small CEM budget and selection
sensitivity warrant further controlled investigation.

## Engineering and provenance limitations

The first sandbox execution failed root identity checks before dispatching
actions. The subsequent host run verified both roots and completed all arms;
the exact reason for the sandbox mismatch has not been independently
identified. On one selected prefix, CPU decoding differs from the
previously saved CUDA prefix by `1.4901161e-08`; the execution used the
original saved raw prefix, with decoding checked at the pre-existing `1e-6`
tolerance. The predicted tail was decoded on CPU from saved parameters;
bitwise equivalence to CUDA decoding has not been established.

The journal durably records branch start and branch return, not each step
before it is sent to the environment. The physical source and historical
roots remain development evidence. A jointly measured response intervention,
independent roots, full source certification and equal-resource strong
baselines are still required.

The consumer tests include action, ordering, RGB, stop and goal tampering,
and the saved-tail comparison; nine combined tests passed, including the
existing frozen-plan tests.
