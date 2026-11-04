# Fixed finite-difference suffix development launch

Run the fixed search in `PC_REACHER_P1_FD_SUFFIX_DEV_20261102_CODEXEXCLUSIVE.md`
on the two preflight-bound development roots, in the saved order. Seal the
launcher, search, strict-selector code, preflight, action statistics, frozen
checkpoint, goal Q and protocol before any inference. Save every probe and
candidate and separately report development and unchanged strict selectors.
The search's callback uses only RGB, goal, executed action history, frozen
model, action bounds and goal Q; simulation state is not a selector input.

Each completed root uses 29 prefix and 290 suffix-token logical samples.
The post-search nine-row endpoint census adds nine and 90 respectively.
These exclude encoding, inherited goal-Q construction and complete internal
compute. No equal-budget or physical claim follows from this development run.
Retain per-root exceptions and the full denominator; never substitute a
fallback candidate for a null selection.
