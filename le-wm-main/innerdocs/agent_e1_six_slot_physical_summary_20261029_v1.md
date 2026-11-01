# E1 six-slot physical saved summary

This is a prospective, saved-only consumer; no future E1 physical report has
been read while specifying or testing it. MAIN supplies both the actual external
SHA of the persisted successful physical strict receipt and the SHA of its
bound report. The analyzer verifies both bytes, exact rows/counts, the receipt's
checkpoint projection, fixed root/arm order, and the frozen physical source
SHA. It does **not** rerun the physical fullstrict consumer or claim that SHA
alone proves receipt authenticity: MAIN's actual successful strict receipt is
the trust boundary. It binds its own source, tests and this document in output.

The predefined denominator is 2 roots × 3 arms × cp0/cp5/cp10 = 18 checkpoint
cells, plus 2 roots × 3 fixed arm pairs × 3 times = 18 pair cells. Checkpoint
time is the number of own physical commands after the shared warmup, not wall
time. Early termination keeps unavailable cells `null`. Pair values are `null`
if either checkpoint is unavailable or the two saved native seven-dimensional
goals differ. No endpoint substitution, padding, root selection or subset-based
aggregate occurs. Arm-pair deltas are descriptive native final-goal differences
within known root and same time; they do not compare the quality of different
latent/intermediate targets, establish causality or establish an independent
confirmation.

For each observed cell, seven state and native goal coordinates, their signed
differences and per-axis squares are saved, alongside agent XY, object XY,
combined XY, full native7 Euclidean distance, velocity squared, original native
angle term, wrapped angle error, and the original native success predicate.
Sums use Python `math.fsum`, so the derived floating-point distances need not
be byte-identical to the native NumPy norm. The native success predicate is
combined XY < 20 and `min(abs(angle), 2*pi-abs(angle)) < pi/9`; the latter can
be negative for out-of-range angles and is reported without silently repairing
the source semantics. It is not an object-only success indicator. There is no
significance test on two known roots, no model-target metric, no experiment,
and no change to any scientific gate.

MAIN commands (from `/home/gbk/code/LeWorldModel/le-wm-main`; replace the four
absolute paths and the externally verified literal hashes):

```sh
.venv/bin/python -m experiments.spectral_reachability.agent_e1_six_slot_physical_summary_20261029_v1 analyze --receipt /ABS/STRICT_RECEIPT.json --receipt-sha RECEIPT_SHA --report /ABS/PHYSICAL_OUTPUT/report.json --report-sha REPORT_SHA --output /ABS/NEW_SUMMARY.json
.venv/bin/python -m experiments.spectral_reachability.agent_e1_six_slot_physical_summary_20261029_v1 verify --analysis /ABS/NEW_SUMMARY.json --analysis-sha SUMMARY_SHA
```

`analyze` writes exclusively and prints the output SHA. `verify` re-reads the
same SHA-bound inputs and independently repeats this analysis, not the upstream
fullstrict. Tests are synthetic RAM and temporary-disk fixtures only:

```sh
.venv/bin/python -m pytest -q tests/test_agent_e1_six_slot_physical_summary_20261029_v1.py
```
