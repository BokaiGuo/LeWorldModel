# Imagined-cp12 first attempt: TensorDict batch metadata

The first run loaded the model, encoded the goal and first cp7 frame, and
made one cp7-to-cp12 model prediction. It then failed before opening the
physical environment because the returned TensorDict had no declared batch
dimensions and `[-1:]` attempted to index that container. No physical
actions or result files were produced.

The runner now takes the last tensor slice separately for `visual` and
`proprio`, validates shapes and finiteness, and constructs an explicitly
batched `[1,1]` context. The candidate bank, score, optimizer and protocol
are unchanged. A targeted test covers the original failure. The failed
model work is additional to the rerun, not experimental evidence.
