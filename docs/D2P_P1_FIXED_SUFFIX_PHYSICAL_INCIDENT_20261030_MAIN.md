# P1 fixed-prefix suffix physical pilot incident

This is an exploratory known-root pilot, not P1 admission, independent
confirmation, a common-suffix mechanism intervention, or an equal-budget
comparison. All 32 candidates were fixed before physical execution.

The prepared manifest is
`results/codex_p1_fixed_prefix_suffix_physical_20261030_main_prepare_MAIN01.json`
with SHA256 `fe5606c27303ada8739e5c58ed612273c26c4d1257c345779cd38aafc90cfc3e`.
It contains 32 plans and no new neural calls.

The physical attempt
`results/codex_p1_fixed_prefix_suffix_physical_20261030_main_run_MAIN01`
stopped at branch 1, plan step 4. Its retained partial report SHA256 is
`30072c619be79f80b8763d1fddaae1ae30beac8464e1725f7fb0a144f3437b78`.
Branch 0 reached the horizon; the other 30 branches were not attempted.
The saved fresh-observation event shows the agent at y=520.4985249270607,
outside the guard's 0–512 canvas coordinate range. The guard checks both
the agent/block coordinates and the presence of blue pixels, so this
observation establishes an out-of-domain coordinate but does not by itself
establish the blue-pixel condition. The guard correctly prevented continuation
under this execution contract; this is not evidence of a Python bug or a
general theoretical failure.

Before retrying, a separately specified fail-closed censoring protocol must
verify the triggering transition, action bytes, state and original PNG,
retain the out-of-domain outcome without extrapolating future steps, and
continue the remaining fixed branches. The existing H10 censoring machinery
is a relevant reference but must not be assumed compatible without tests.
The current 32-row saved consumer only accepts complete uncensored reports and
is not an independent validation of this partial attempt.

The runner was also hardened before this attempt: cleanup is attempted if
journaling fails, and a 32-branch operation census is checked. Twelve tests
passed, including tampered task/action rejection and existing early-stop
tests. This does not certify the full physical pipeline.
