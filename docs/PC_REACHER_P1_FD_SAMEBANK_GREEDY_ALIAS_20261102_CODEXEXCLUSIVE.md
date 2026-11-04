# FD same-bank immediate-Q greedy: saved physical-feedback alias

The model-only input is the sealed FD report and its nine indexed candidate
rows. The rule minimizes saved model immediate Q over indices 0 through 8;
exact ties use the lowest index. It does not inspect physical outcomes,
longer-horizon scores or the development/strict eligibility flags. These
probes and FD directions were themselves produced by the original FD search:
the comparison is same bank, not a separately optimized greedy search.

Both exposed roots select index 0, which has exactly the same float64
parameters and float32 raw prefix as the sealed parameter-zero arm. The
already completed float64 run therefore supplies the actual cp5 RGB/history,
physical prefix and CEM successor for this greedy choice. Re-running it is
not required to create another label; no second execution or independent
sample is claimed. If greedy is not index 0 or the bytes do not bind, the
alias fails closed and this code does not schedule a replacement.

This rule and its selection are being sealed *after* physical outcomes for
these roots were exposed, despite requiring no physical values as inputs.
Do not describe it as prospective pre-registration. Immediate Q is a saved
model score; the existing FD consumer checks internal consistency but cannot
independently recompute its latent Q from saved inputs. The strict selector
remains null and the original gate is unchanged.

The read-only consumer checks the model-only plan, archived host plan and
source hashes, journal hash and chain, and recomputes the existing saved
consumer before copying the exact zero arm witness and metrics. Its output
retains stop receipts and references the existing journal; `greedy` and
`parameter_zero` are one physical arm, not two denominator entries. The
source closure is not complete and saved consistency does not independently
prove live execution.

The inherited FD search cost is 29 prefix samples and 290 suffix-token
samples per root, plus the separately saved census of 9 and 90, respectively.
The already executed zero arm used 32 nominal CEM candidate rollouts and
physically recorded 5 prefix + 5 zero-control + 10 suffix steps per root.
Encoding, Q construction, loading, synchronization, control execution and
other costs require separate accounting. The alias adds no model or physical
execution and establishes no equal-total-compute advantage. The two roots are
development-exposed; no independent or P1 claim follows.
