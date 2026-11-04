# Affine-score saved-trace replay

The earlier random-CEM feedback receipt did not save its action normalizer
or bind the sequence-length file. To assess whether this affects the
comparison, replay **all** saved candidate batches from both the random and
spectral arms, using the currently hash-bound dataset, scene, checkpoint,
RGB goal, current RGB history, five-step packet transform and frozen model.
Reconstruct initial and cp5 contexts exactly as specified by the original
protocol, then calculate the affine score for every saved proposal.

Record each batch's maximum absolute and relative error, whether the stable
score ordering is identical, and whether its selected elite indices match
the original trace. Use an absolute tolerance of `1e-6` and relative
tolerance of `1e-4` for score numerical agreement, but report ordering
separately; numerical agreement cannot excuse an altered optimizer update.
Check the historical cp5 input against a fresh reconstruction from its saved
cp4/cp5 images. Count calls and bind both reports, saved arrays, current
normalizer input files, model parameters, and scoring code. No physical
labels influence scoring or selection; this audit does not execute an
environment or create new planning outcomes.

Even exact functional reproduction cannot prove the historical contents
or training provenance of `seq_lengths.pth`. Conversely, a mismatch is an
interface or reproducibility diagnostic, not automatic proof that either
planning method failed theoretically.
