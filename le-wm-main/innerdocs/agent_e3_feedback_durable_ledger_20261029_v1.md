# E3 durable operation journal

Bounded flat append-only JSONL, single LF, fsync each event, hash chain and
independent bounded saved audit. Attempt/return/failure and nested failures
remain separate and no failed work is refunded. Pending intent is partial.
This is only an operation journal building block: it is not a native chronology,
array, RGB, model/source, whole-directory or full-episode strict consumer.
Fixtures use temporary files; no production execution or scientific admission.
