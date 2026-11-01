# Fixed-suffix known-root exploration: bounded canvas censoring

This follow-up was designed after observing the first failed physical attempt;
it is exploratory, not preregistered confirmation. The same complete 32-plan
roster is rerun in its original order with the same initial states, actions,
goals, rendering and native stop rules. The prior partial result remains
unchanged. No outcome-dependent candidate or root selection is allowed.

Only a plan-step rejection originating in the existing RGB/canvas guard can be
handled as censored. The runner verifies the exact requested/returned action,
native transition, goal and renderer invariants, finite state bits, a coordinate
outside [0,512], and the original saved RGB file and pixel hash. It then stops
that branch immediately, records the action, state and frame, closes its
environment and continues to the next fixed branch. It does not infer the
unobserved endpoint or grant an accepted RGB observation. Any other error
remains a fail-stop engineering error. Native early stops remain distinct.

The 32-plan denominator includes censored and early-stopped branches; no
endpoint is padded. The post-run consumer must verify action/transition/frame
identities, censor proofs and counts independently before any result is used.
These open-loop plans use separately optimized suffixes, with no matched
feedback or equal total budget, and are not the common-suffix capability
intervention, a strong-baseline comparison, or P1 admission.
