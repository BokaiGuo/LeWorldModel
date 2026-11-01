# Future cp10 native transcript validator component

`validate_native(p,r,events,load)` implements the reviewed five-image encoder/projector native IO and exact 52-event chronology checks without the old bridge task namespace. It validates the literal input roster, off PNG/preprocessing, full encoder hidden and projector native IO, guards, counts and ordering, then reconstructs five own-namespace tasks and seven signals. Its `load` callback must be an independently bounded SHA-verifying saved-array loader; this function alone does not authenticate descriptors, event files, sources or receipt publication and explicitly returns `full_saved_consumer=false`.

This is a component, **not** a future producer, runnable CUDA bridge, or full saved consumer. A real implementation still needs: externally authenticated future physical prepare/report/completion/full receipt and preselected cohort; complete physical prefix51/source/PNG/history joins without hardcoded old references; source/model closure; CUDA-only runtime and actual native trace writer with partial/error preservation and storage/wall limits; durable own-schema report/completion/receipt; and a full consumer authenticating the entire artifact set before calling this validator. Existing future receipt join has a candidate ABI and cannot replace these checks. Original known-root physical inputs are historical and cannot establish fresh roots or pre-outcome timing. No historical namespace spoofing, new science gate, fallback for missing witness, or actual authorization is supplied here.

Tests run the previous reviewed temporary-disk synthetic native IO fixture with real Store and fake tensors; no GPU/NN inference or physics occurs. They do not authenticate a future producer or production outcome.

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_future_cp10_native_transcript_20261029_v1.py
```
