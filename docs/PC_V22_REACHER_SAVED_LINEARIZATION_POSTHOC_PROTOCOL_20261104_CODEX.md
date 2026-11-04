# v22 Reacher saved-response linearization: post hoc protocol

Analyze the three already-executed v22 Reacher feedback decisions, with no
new model calls, physical actions or task search. These are exposed
development data. For every generated candidate, reconstruct the nominal
probe and response matrix recorded for its retained parent and nominal,
then compare the local predicted feature change with the recorded actual
candidate feature change. The feature vector contains the two-dimensional
chart and scaled full-latent score: report their errors separately rather
than combining coordinates with different semantics.

Report absolute vector error and predicted/actual norms, with relative
error and cosine only when the actual and predicted norms permit them.
Group by decision, depth and actual maximum action perturbation. Verify
all input hashes and candidate-to-parent mappings. Identical nominal
actions may have small batch-dependent floating point differences: take
the first recorded matching evaluation and report the maximum spread
instead of assuming exact score equality. This measures model-internal
first-order fit on candidate scales; it does not establish AD correctness,
physical response, task effectiveness, causality or independent PC1.
