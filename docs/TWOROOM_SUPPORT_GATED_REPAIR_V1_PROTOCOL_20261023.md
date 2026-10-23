# TwoRoom support-gated repair V1

This component is the prospective C2 route prepared after the V4 replication
panel.  It does not replace V4 and cannot read its outcome roots.

1. Fit the full effect adapter and action-only fallback on train roots.
2. Construct support solely from full-adapter input features on train roots.
3. On calibration roots, freeze a support threshold using paired true-ranking
   regret of full adapter versus fallback.
4. On untouched outcome roots, release the adapter only if every candidate is
   within the frozen support envelope; otherwise use action-only scores.

The gate is authorized only when a nonzero calibration release region has
positive mean gain.  It is not a conformal guarantee: the decisive evidence is
the subsequent root-disjoint outcome comparison of gated, always-adapter, and
always-fallback policies.
