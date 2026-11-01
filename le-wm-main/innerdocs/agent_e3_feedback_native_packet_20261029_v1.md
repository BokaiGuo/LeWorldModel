# E3 native incremental packet primitive

Calls the frozen native dispatch, observation and unchanged native return
predicate in step order, retaining returned raw float32 bits, observations and
terminated/truncated events. A stop in the middle returns the actual short
prefix; it neither pads nor executes another step. Raised native failures are
not swallowed; frozen meter writes attempt/failure while earlier accepted steps
remain in its durable journal.

Root attachment separately verifies the source root hash using its original
goal before changing task goal; the task native7 and decoded RGB then must match
their own pinned expected values, without changing snapshot/pose/renderer.
Caller must verify the expected values and evidence themselves before calling.
This is not a full setup/encoder/launcher or independent saved consumer, and
the tests are fixtures, not a physical execution or production authority.
