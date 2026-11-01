# Supplied-witness model diagnostic

After the four-arm H3 panel, evaluate the single already-sealed V3 scripted
witness without searching, selecting or altering any action. This is a
privileged offline diagnostic, never a planner initialization or performance arm.

At5/10/15/20 compare open predicted latents, same-action predictions refreshed
from true three-frame RGB histories, and actual encoded frames. Also predict a
zero-action20-step trajectory, explicitly without physical counterfactual data.
Retain full predicted and observed vectors, Q and latent MSE. Re-encode the
three root-history frames and verify them against the prepared context; final
frame must encode identically to the goal artifact.

Seven RGB encodes and twelve action-encode/predict calls, no gradients,
frozen host CUDA only, full source/checkpoint binding. This can distinguish
whether the model gives a feasible supplied route favorable scores, but cannot
prove the search can discover it, non-greedy capability change, or generalization.
