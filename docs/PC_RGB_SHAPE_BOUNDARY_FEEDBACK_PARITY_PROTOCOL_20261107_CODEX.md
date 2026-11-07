# RGB image boundary closed-loop parity protocol

Engineering regression on previously exposed TwoRoom roots 10–13 and all four
existing retention strategies. Reuse the original frozen model, inputs, seeds,
search, resets and feedback loop; change only the model boundary to
`RGBImageLeWM`. The reset state is used by the evaluator, never passed to the
planner. Save a fresh result before reading the historical result.

Compare model and input fingerprints, every decision's input receipts and
candidate hashes, choice, dispatched and planned actions, old-tail hashes,
query counts, termination and physical traces. Model scores use absolute
tolerance 1e-4 and relative tolerance 1e-5; actions and receipts must match
exactly, and physical distances use absolute tolerance 1e-6. Timing and audit
fingerprints are not parity criteria. A mismatch is reported, not suppressed.
This exposed four-root test is not evidence of algorithmic improvement, input
provenance, security isolation, independent confirmation, or gate passage.
