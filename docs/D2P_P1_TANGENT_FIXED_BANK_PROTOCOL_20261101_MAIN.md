# Fixed-bank score decomposition

After fixing the geometric metric independently of physical outcomes, replay
all eight previously sealed equal-candidate-Q own plans, exactly15 steps.
Use their original root/history/action normalization and verify original-Q
score reproduction before interpreting new scores. Do not optimize actions.
Twenty-four host-CUDA action-encoder/predict calls, no new RGB encoding;
reuse already-sealed true RGB embeddings only for offline evaluation.

For every complete own-plan endpoint, compare model-vs-actual RGB and
actual-RGB-vs-jointXY margins under both fixed metrics. Preserve all censored
branches; no endpoint padding. Pairs share one known root, not independent
samples. No outcome labels enter a deployed candidate selection, metric fitting
or the concurrent fixed eight-arm control panel.
