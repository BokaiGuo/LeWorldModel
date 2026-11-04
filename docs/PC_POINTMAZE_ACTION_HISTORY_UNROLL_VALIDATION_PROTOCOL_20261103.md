# Action-history unroll: frozen-model interface validation

Check the explicit normalized-action sliding-window implementation using the
existing fixed-tail history diagnostic and source artifacts, without selecting
or executing new actions. On host CUDA, for all eight exposed prefix groups,
compare its two-frame one-step visual output against the sealed direct
`forward_pred` history output. Separately, on the cp5 encoded context, compare
its one-frame two-step visual and proprio outputs against the current
single-frame wrapper's `unroll`, using the first two already executed tails
as a fixed two-packet sequence. Record max absolute differences, tensor
shapes, source hashes and frozen model hashes. Fail the run if any comparison
exceeds `rtol=1e-5, atol=1e-6`; never overwrite old outcomes.

This tests implementation equivalence for the selected examples, not
checkpoint training provenance, independent data, physical prediction
accuracy, equal-budget planning, or P1. Two-step sequences are diagnostic
inputs, not executed action chains.
