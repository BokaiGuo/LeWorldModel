# E1 forced-candidate same-target native paired summary

Saved-only fixed six slots: two known roots × bank winner/zero extra/witness
extra, cp0/5/10. For each root, require exact task/root/target identity from
the parent physical prepare and same actual native goal when both checkpoints
exist. Output all 18 cells and all 18 within-root same-time method-pair slots,
including nulls after early stop or failure; never replace a missing cp5/10
with an endpoint. Each pair reports signed b-minus-a native7, joint/agent/object
XY, velocity, raw/native/wrapped angle and native success separately, plus
whether raw5/raw10 are identical. cp0 is the verified native initial baseline,
not a later control checkpoint. No winner, threshold fitting, IID treatment or
post-outcome selection; two known roots are not independent confirmation.

The scalar code recomputes from saved state7 and actual native goal using fsum;
the original native success rule uses joint xy <20 and native angle <pi/9.
Raw native7 angle and wrapped-angle diagnostic stay separate, including the
original min-angle behavior outside one turn. All original physical resource
counts and parent errors are retained. No target arrival success is invented,
and latent-predicted or actual-RGB tracking requires a separately authenticated
encoder bridge. FLOPs are null and total costs are not claimed equal.

External actual physical report/receipt SHA and physical-source SHA are
required: source must also match the physical prepare's binding, receipt must
bind report and agree exactly on rows/counts/checkpoints; the prepare's six
plans must match every branch. This inherits the MAIN full physical strict
receipt, not a rerun of fullstrict or an independent re-execution of PNG/native
physics. A self-declared fake receipt cannot replace MAIN authority. Parent
`partial_or_failed` remains explicit; missing values do not count as success.
No physical, NN or fullstrict calls occur in this analyzer.

After actual report and full receipt exist, MAIN can use new absolute paths and
actual SHA values, without changing the old adapter:

```
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_forced_candidate_paired_summary_20261029_v1 analyze --receipt /ABS/RECEIPT.json --receipt-sha RECEIPT_SHA --report /ABS/OUTPUT/report.json --report-sha REPORT_SHA --physical-source-sha SOURCE_SHA --output /ABS/NEW_SUMMARY.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_forced_candidate_paired_summary_20261029_v1 verify --analysis /ABS/NEW_SUMMARY.json --analysis-sha SUMMARY_SHA
```

Test: `.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e1_forced_candidate_paired_summary_20261029_v1.py`. Fixtures and temporary file chains only; no actual future authority is fabricated.
