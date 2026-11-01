# E3 original encoder IO draft

HOST-CUDA-only `encode` reads the current captured PNG, checks bytes and decoded
pixels, installs temporary encoder/projector hooks, saves full original seven
arrays and records operation attempt/return in the unified event stream. It uses
the frozen original preprocessing and model.encode, checks actual preprocessed
bytes, forbids action/predict modules during encoding and removes hooks finally.
The caller must separately authenticate model/weight/source/precision, physical
state, input frame authority, native guard/limits, failure/cleanup and publication.

`verify_arrays` independently checks saved pixels, preprocessing, encoder input,
CLS and projector-to-latent connections, not neural parity. Fixture tests use
synthetic saved arrays; no encoder or CUDA executed. This is not a complete
source/model identity verifier, same-environment dispatcher, or production CLI.
