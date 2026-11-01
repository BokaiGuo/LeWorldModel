# K17 native first5 episode seam — not production-ready

This implements reset/place, ten original warm actions, goal binding, 33 warm
captures and plan0 capture using the frozen H3 primitive functions. It checks
saved PNGs and invokes the frozen guard and paid replay bridge before emitting
the durable accepted guard. It dispatches only the exact original first5 and
stops on native termination. This function requires an independently bound
root/task/reference, a bounded meter, a compatible store, and a trustworthy
outer runner. It does not authenticate those arguments by itself.

There is deliberately no production CLI. Six-slot fixed denominator, source
binding/revalidation, actual resource caps, wall and failure publication,
complete independent saved consumer and full tempdisk end-to-end fixtures are
not yet implemented. Do not use this seam as physical authority or evidence.
