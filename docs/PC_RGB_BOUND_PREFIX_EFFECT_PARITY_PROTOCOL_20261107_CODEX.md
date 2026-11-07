# RGB fixed-prefix hold-relative descriptor parity protocol

Before running, reuse the sealed four-root, five-prefix, five-tail TwoRoom RGB
matrix, including the known reset-border exception. Rebuild all 100 action
chains with the new adapter, using zero as the declared comparison action, not
asserting that zero is a physically valid stationary hold at every reached
state. Score the exact full chains with the same frozen LeWM and normalizer.
Require model and normalizer fingerprints to match the sealed prediction and
compare the full 5-by-5 matrices with absolute tolerance `1e-3`, plus compare
the hold-minus-probe effects with absolute tolerance `2e-3`. Capture the
individual costs, action and observation receipts, input hashes and query
counts. This checks interface/numerical parity only; no fresh physical
validation, adaptive retention test, independent task, or gate pass is claimed.
