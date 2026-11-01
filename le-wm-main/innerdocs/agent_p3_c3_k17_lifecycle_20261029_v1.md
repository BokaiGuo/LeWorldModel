# K17 durable six-slot lifecycle

This independent lifecycle persists a fixed six-slot roster before startup,
then records startup and sequential slot intent/return/failure. A startup
failure preserves six `not_started` rows; a slot failure preserves subsequent
`not_started` rows. Native early-stop status may be returned by a callback and
does not trigger a retry. The consumer independently reconstructs the roster
from the journal and checks exact canonical report/completion hashes.

This is an executable lifecycle component tested with tempdisk fixtures, not a
physical adapter or production CLI. Callbacks are not authenticated: it does
not validate actual sources, reset/history, native actions, PNG pixels, resource
counts, wall, early-stop semantics or a real physical receipt. The report and
consumer explicitly keep physical provenance and release false. It must not be
used as the requested complete saved physical consumer.

Writes are exclusive, fsynced, capped by the independent 128 MiB store.
Startup/slot exceptions are saved when the store remains operational. A store
failure propagates; even a failure record cannot be guaranteed after I/O,
quota, disk-full or process interruption. This contract makes no equal-compute
or scientific claim.
