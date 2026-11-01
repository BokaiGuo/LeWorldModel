# E3 saved numeric row

Persists a complete HOST node row as bounded, ordered base64 chunks in the
unified append-only event chain. Saved audit reconstructs exact row bytes/SHA,
requires all referenced arrays and no extra files through the transport consumer,
and replays every numeric native IO/AD event, candidate, gradient and 27-axis
count with the existing saved backend. Chunking preserves the event size cap.

This is numerical-only fixture functionality, not a full E3 strict receipt.
It does not authenticate source, model, precision, physical state or RGB, and
does not support multiple node rows in the same journal yet. Do not treat it as
a production dispatcher or a science release. No real NN/physics executed.
