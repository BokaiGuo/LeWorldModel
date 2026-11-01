# P3 early-stop physical replay result

The fixed known-root two-point panel ran 12 physical branches, with a fresh
environment per branch and no new neural calls. Both point reports are
`recorded_not_admitted`; both independent saved consumers returned
`saved_semantics_checked_not_scientifically_admitted`. The report SHA-256
values are:

- P20: `39344766d76d6163b7dc3f17e5c4beccde1ba86b457ca3a9f244fdfaace2789b`
- P30: `4833d631081791b7ad48d53dfd41746dbecca11492138e1d90e98e44482f650b`

Each point returned all 60 planned actions across six branches, with no
native termination or truncation and no native success. Per-point actual
attempted/completed counts were construct 6, close 6, reset 12, set-state 24,
goal-render 12, goal-bind 6, public-step 120, hooked physics-step 1224 and
explicit-render 132. Public steps include 60 warmup steps. No physical
resource axes are added to neural axes.

Across both points every arm selected the same candidate within each root:
root 0 index 0 and root 1 index 4. Each root's final recorded state is the
same across its six runs. Root 0 final object position error is
70.78670592324957 and wrapped angle error is 0.47590319286748706 radians;
root 1 values are 76.40267978400153 and 0.08223166638733746 radians.
These are exploratory instance observations, not evidence of scheduling
advantage, independence, equal compute, or P3 admission. The source manifests
record actual early-stop allocations and each model axis independently.
