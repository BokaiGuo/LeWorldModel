# Joint prefix / conditional Vreach multi-start: callback core

Exploratory, model-only callback implementation. It has no real source
authority, native GPU runner, saved consumer, model closure, or physical
release. The four fixed tasks are the two known roots and off/on conditions.
Nothing trains the frozen model or uses physical outcome to choose a root,
arm, candidate, scale or action. This is not a P1 result or a retrospective
preregistration; an actual run requires MAIN to approve and freeze a new host.

Each slot has **two fully paid arms**, with four fixed prefix starts: the
original winner prefix, zero, +0.25 e0 and -0.25 e0. For each start, evaluate
the starting prefix and four eta=1 ascent updates (20 candidates per arm).
At *every* candidate prefix the backend must obtain a fresh conditional
zero-action H1 nominal and its 192×10 action-logit Jacobian. The frozen original
`bounded_ridge(J, goal-nominal)` with damping .002 and original packet ball
produces own Vreach and suffix logits. Execute the resulting suffix and an
independent full two-packet replay. Prefix and suffix are thus jointly generated
by alternating prefix updates and fresh conditional ridge, not by claiming to
differentiate Vreach through its SVD or reuse a stale J.

The first arm uses the original projected stage-Q ascent. The second removes
the outward radial component of the ascent at **every nonzero radius**, then
uses the same original projection. It is a deliberately specified
outward-free tangent variant, not an exact boundary-only tangent cone:
inward components are kept and no post-hoc `1e-6` threshold is used. The
recent boundary statistics motivate this prospective comparison but do not
prove a bug or select a root. Both arms perform the same number of callback
operations; neither equal FLOPs nor equal wall is asserted.

Within each arm, the same original winner raw10 is independently replayed for
the terminal baseline. Winner-prefix fresh conditional Vreach is the reach
baseline. A candidate is eligible only if **all** `stage_Q > initial_Q`,
`own_Vreach < winner_prefix_Vreach`, and `terminal_Q < original_winner_Q` hold
strictly. This reach criterion is an additional exploratory constraint, not a
replacement for the original P1 gate or an admission decision. Stable
`(terminal_Q,start_index,iteration)` selects among eligible model candidates;
otherwise null, no fallback. All 160 candidate identities and 8 arms remain
in the panel. Model witnesses carry explicit raw10 float32 bits, but are not
physical evidence and cannot be released by this core.

Per arm the fixed envelope is 138 predict, 16 scalar reverse, 20 Jacobian
calls/3840 mathematical VJP directions, 20 SVD and 20 ridge solves: baseline
2 + 20*(prefix 1 + conditional signal 2 + suffix 1 + independent replay 2)
+ 16 stage-gradient forwards. Eight arms total 1104 predict, 128 scalar
reverse, 160 Jacobian/30720 mathematical VJP, 160 SVD/solve. VJP and callback
counts are nested, not additive FLOPs. This is the **declared callback cost**
of a signal ABI that must later prove its two actual forwards and 192 VJP per
call; the kernel ledger charges atomic envelopes before callbacks and cannot
independently certify nested native operations. Failures retain paid attempted
and completed envelopes via events; sink or persistence failures require a
future host to preserve them. It is not the fixed-prefix 1504/640 budget and
is not an equal-total-budget comparison to the old method.

RAM tests use a synthetic linear backend and are not scientific or production
evidence. A future independently reviewed host must authenticate original
winner/task/goal/normalizer, map the real NativeSignal output into the explicit
`nominal_context,J` callback ABI, reconstruct zero-action complete history,
audit every native IO/resource/event and scalar, seal source/model/precision,
persist all failure denominators, and enforce explicit engineering caps before
MAIN may execute CUDA. Physics, fresh-root confirmation and P1 remain separate.

Test: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_joint_vreach_multistart_20261029_v1.py`.
