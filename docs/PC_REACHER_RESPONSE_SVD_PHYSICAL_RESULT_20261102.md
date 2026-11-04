# Response-SVD exhaustive common-tail physical panel

All 130 predeclared branches completed on the two previously exposed Reacher
roots: 13 fixed prefixes per root, each followed by five fixed common tails.
There were no stops or errors. The physical plan postcheck passed; the
saved-only consumer checked the fixed branch order, action bytes, per-step
actions, RGB hashes, joint distances, prefix consistency across tails and
the 130-branch denominator. It does not independently replay the simulator.

The model selector selected no prefix. The following are therefore
post-exposure descriptions of the *complete* candidate set, not
confirmation of a selected action or a P1 gate:

| Development root | Non-neutral prefixes with no immediate improvement vs initial | No immediate improvement vs parameter-zero | All four active tails have negative neutral-adjusted difference | All five absolute final distances no worse than neutral |
| --- | ---: | ---: | ---: | ---: |
| 20263160 | 9/12 | 9/12 | 0/12 | 3/12 |
| 20263161 | 11/12 | 6/12 | 3/12 | 6/12 |

No prefix on either root satisfies the three physical conditions jointly:
no immediate improvement relative to parameter-zero, no worse absolute final
distance for every tail, and strictly negative neutral-adjusted difference for
every active tail. These conditions are exploratory descriptions, not the
official P1 gate. Parameter-zero is a zero *model parameter* and is not
necessarily zero physical action. A negative adjusted difference compares
each candidate to its own parameter-zero tail and neutral prefix to its
parameter-zero tail; it does not by itself isolate a causal capability.

These results expose response variation and an operational failure of this
fixed bank/selector on two roots, not impossibility of the mechanism.
No per-prefix independently optimized suffix, fair strong baseline, selected
model witness or independent confirmation has been established.

- Physical plan SHA-256:
  `550f2f900a1ce617b0b79a4617875f8350d6b568d9ee5d08c1bad5cdabe65f4c`.
- Event journal SHA-256:
  `1deb2e7fe371e16f4c25439bbd77805407d1c719765561397b89d27c6a3369b8`.
- Saved analysis SHA-256:
  `c3e77d943e65d2bf7de5ba9f32f7fc5fb9299ad86626e5567b074ab1d1f8399f`.
