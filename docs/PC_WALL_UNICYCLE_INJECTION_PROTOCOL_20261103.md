# Exact wall-unicycle successful-chain injection

Post-pilot development diagnosis: for each of the three exposed seeds, take
the corresponding successful full 16-command action chain selected by the
previous random-waypoint control. Split into four four-command packets and
inject the appropriate packet into Atlas's bank only when the full incoming
context equals that chain's actual prefix context. At the initial root add
the first packet to the existing bank. Never modify the original banks,
candidate ordering, scores, beam width, or stable-tie behavior.

Replay the chain to build context bindings; record whether Atlas actually
visits the matching incoming context for packets 2–4 and whether a
context-identical complete chain wins a branch or overall. Multiple paths can
reach one context; a visit is not proof that the uniquely named path survived.
If a context was not visited, the continuation was not offered. Compare
model terminal costs, selected action witness, full replay, operation counts,
and collision/FD transitions to the ordinary saved Atlas run. All injection
construction and audit replays are additional to search operations.

Injection uses solutions obtained by an external planner, adds branches and
unequal work, and is prohibited as a performance or independent evidence
claim. The setup has exact analytical state, not RGB. A dropped path does
not itself prove a code bug: finite-width, current-cost beam pruning may
deliberately eliminate temporarily costly branches. It also cannot identify
the precise competing branch without separately tracing the actual beam.
