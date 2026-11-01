# P1 planner input bundle, development only

The CPU exporter first invokes the previous MAIN02 source verifier, then joins
the original compressed task export and its blind NPZ by SHA. It checks the
three historical RGB arrays, goal RGB, raw warmup history and all pixel
digests against the original task contract. It checks the actual cp5 PNG bytes
and decoded pixels, then exports only raw 224×224×3 RGB pixels, not PNG
metadata or old filesystem paths. Original source PNG SHA is recorded as a
provenance assertion; the standalone consumer checks the exported pixel bytes,
not the historical PNG file. The trusted expected bundle SHA must be supplied
out of band.

Each of 16 records contains its row identity, original observed compiler task,
five already-executed prefix actions, three historical RGB observations, goal
RGB and actual cp5 RGB. The task includes the saved RGB-derived latent
history, goal latent, action normalization and fixed model/task identities.
This does not contain a physical state, snapshot, outcome, reward, future
frame or path. The complete 24-condition roster keeps eight greedy entries
unknown. This is not a physical or P1 authorization.

The independent consumer imports no physical source or planner modules. It
accepts only an exact schema, verifies external bundle SHA, every RGB byte and
pixel hash, original task digest, dimensions, normalization and full fixed
roster. Extra keys, duplicate JSON keys and malformed bytes are rejected.
It does not establish historical provenance on its own: that derives from the
exporter source audit and the trusted out-of-band bundle digest.

This is data minimization, not an OS security sandbox. A planner running under
the same account with access to the repository can still open old files.
Before actual planning, mount/copy only the bundle and approved model/runtime
artifacts into an independently restricted process and test denied access to
the physical sources, outcomes, subsequent frames, and non-allowlisted paths.
No cp5 latent has been encoded in this work; host CUDA exclusivity remains
externally coordinated.

The existing observed H2 kernel requires an original imagined suffix candidate,
which is a planned future action, not executed action history. It is excluded
from this bundle under the requested allowlist. A reviewed lawful input and
method contract is required before wiring that kernel: do not silently inject
the old candidate, change the candidate set, or claim matched budget. The
greedy provenance, cp5 observations, physical replay guard, new suffix
executor, strict saved consumer, and unchanged scientific gates remain open.
