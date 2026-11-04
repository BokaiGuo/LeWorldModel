# Actual-RGB task-metric diagnostic on six fixed development chains

Use the two previously exposed roots and the same six completed 15-step
physical chains as the CEM32 same-chain experiment. On each chain compare the
recorded final RGB with the root-bound goal image using full-frame RGB mean
squared error and mean absolute error, both on uint8 values scaled to [0, 1].
No masks, weighting, tuning, cropping, thresholds, or fitted parameters.
Compare lower-is-better pairwise rankings with the original model score,
actual encoded latent score, and actual wrapped joint distance. Report all
three arms, pairwise concordance with the physical metric and the physical
regret of each metric's selected arm; ties are reported explicitly.

This protocol is exploratory and fixed after the physical/latent outcomes
were already opened. The final real RGB is an outcome available only after
execution, not a future image predicted by LeWM. The pixel scores therefore
cannot be used as a deployable pre-execution selector, nor can they establish
a P1 mechanism. Two roots are not a basis for a population estimate.
There are no new neural calls, physical actions, training, or gate changes.
