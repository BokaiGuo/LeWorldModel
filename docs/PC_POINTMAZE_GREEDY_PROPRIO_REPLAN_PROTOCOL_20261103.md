# Greedy one-packet proprio CEM with RGB replanning

On all eight exposed roots 1888–1895, use the same checkpoint, RGB
projection, goal token, predicted-proprio score and source noises as the
saved spectral experiment. At the initial RGB, run 3×16 CEM over a single
five-step packet using the first packet component of the saved initial
innovations. Select the minimum predicted terminal score: this is
model-greedy, not physical-oracle greedy. Execute that packet, update from
its own real cp4/cp5 RGB with causal velocity, and run the identical
3×16 tail optimizer using saved tail innovations. Execute the resulting
tail; no early stopping.

Save traces, raw actions, real RGB histories, normalized input provenance,
counts and time. Verify frozen model hashes and compare terminal RGB and
position with separate legacy/position replays. Record cp5 position and
the strict offline `<0.5` terminal metric only after all decisions.

Compare with sealed zero, random and spectral prefix results. Initial
candidate counts match random/spectral (48), but greedy predicts one packet
per initial candidate while those arms predict two, and spectral incurs two
JVPs plus SVD. Tail counts match; this does not certify equal total cost or
a full-horizon strong baseline. All roots and prior methods were exposed;
do not claim independent P1, causal mechanism or native online success.
