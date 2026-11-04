# PointMaze registry-normalization development contrast

Before running, fix roots 1880–1883, the sealed 8-by-8 action bank, frame-0
and frame-10 RGB and saved position-only physical outcomes from the previous
constant-proprio development experiment. Repeat its 64 pair scores per root
with the frozen checkpoint on host CUDA, but replace the global dataset-derived
action/proprio normalizers with the checked-in JEPA-WM `DATA_STATS` pointmaze
registry. Use its raw proprio mean as the same fixed input for every root and
both frames. Normalize physical actions exactly once with registry statistics.
Record all scores, argmins, stat values, file hashes, parameter hashes and
serial-versus-batch check before accessing physical outcome arrays (the
sealed source file is parsed earlier for input identities).

Compare selected physical costs and regrets against the prior development
run, without choosing roots or models by observed outcome. The registry source
claims its statistics were derived from training data, but its exact mapping to
the local checkpoint and overlap with these roots are not established. Thus
this is a known-root sensitivity test, not a training-disjoint certification,
new physical outcome, RGB-only online experiment or P1 confirmation. Count
model scores separately from encoding, preprocessing and wall time.
