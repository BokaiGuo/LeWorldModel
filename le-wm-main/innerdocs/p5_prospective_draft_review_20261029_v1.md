# P5 prospective draft review: advisory interface

This is an independent review aid for a future P5 protocol draft, not a frozen
protocol schema, scientific gate, freshness authority, or collection command.
It neither scans historical attempts nor reads outcomes. The section labels
are organizational prompts from the existing precondition resolver; nonblank
text only means supplied, not complete, valid, pre-outcome, or approved.

The input is an optional JSON object with `sections` and `separation_claims`.
`sections` maps any of the seven names printed in `sections_missing` to draft
text or null. Each optional separation claim maps one of `event_new`,
`state_set_disjoint`, `sampling_independence`, `training_lineage` to
`{"declaration":"unknown|asserted","citations":[{"path":"/absolute/path",
"sha256":"<64 lowercase hex>","pointer":"/json/pointer"}]}`. Citations are
locators only, never opened or authenticated; even a populated draft cannot
grant execution, fresh certification, or P1/P5 admission. A declaration on
one axis never fills another. Empty or missing sections remain visible.

Run from the project root with no input to show the current unfilled template:

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_prospective_draft_review_20261029_v1
```

For a future independently supplied draft add `--draft /absolute/draft.json`.
Output is stdout only, includes the actual draft SHA when supplied, and is not
an approval receipt. The format is deliberately tentative: MAIN must separately
freeze the real protocol, decide its N/ordering and budget/statistics, and
review original authorities and evidence before any collection. No protocol
section or four-slot pilot in this aid modifies existing scientific thresholds.

Current snapshot: `results/agent_p5_fresh_confirmation_preconditions_20261029_v1_MAIN01.json`
has SHA `d453e4cf70518839844916566fa274959c515d05310ae090fb5699e589cf3133`,
zero certified fresh roots and null P5 denominator. Its upstream requests name
original reset closure, post-cutoff scope/identity, external scope approval,
P5 prospective protocol and actual candidate identity. This is a pinned
snapshot, not a claim that no later evidence exists. The old four-slot LeWM
design cannot be transferred as P5's denominator. The alternative source
separation proposal distinguishes new event, state-set disjointness, independent
sampling and training lineage; it explicitly does not replace the old gate.
