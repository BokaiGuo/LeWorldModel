# Actual final-RGB encoding of CEM32 model-scored action chains

For every one of the six saved, complete development physical chains, encode
the actual frames at steps 5, 10 and 15 using the existing frozen Reacher
checkpoint and the actual executed action history, and separately encode the
bound goal image. Compute the same latent goal MSE used by the model selector
and compare its ranking with both the original model predicted score and the
physical final joint-distance ranking. No new physical execution, selection,
training or simulator-state inputs to the encoder. Host CUDA only.

Check the physical saved receipt and action journal, original goal identity,
model and action-statistics hashes, exact 3-frame/three-packet time alignment
and model before/after hash. Preserve all six cases. Saved model scores
evaluated CUDA-decoded parameters, while this physical replay used CPU-decoded
parameters; bitwise equivalence of the transform remains uncertified.
These are exposed development roots; no causal attribution or P1 admission.
