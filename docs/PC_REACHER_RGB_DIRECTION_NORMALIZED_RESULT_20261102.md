# Dimension-normalized RGB-direction CEM: development result

All 26 fixed prefixes on the two exposed roots completed. Each real cp5 RGB,
goal and action history was checked against both the L2 and the earlier
strong-direction feedback runs. The saved consumer verifies the full action
and step sequences, RGB timelines, absence of early stops, frozen model
hashes and declared CEM budget. It does not independently replay the
simulator.

| Root | Normalized better than L2 | Mean normalized − L2 | Normalized better than strong direction | Mean normalized − strong |
| --- | ---: | ---: | ---: | ---: |
| 20263160 | 4/13 | +0.015086 | 13/13 | −1.285461 |
| 20263161 | 0/13 | +0.076660 | 13/13 | −0.494803 |

Values are final wrapped joint-distance differences; negative is better.
The normalized suffix differs from the L2 suffix in every arm. Results are
within two shared development roots, not 26 independent outcomes. The new
weight was selected after observing the strong-direction failure. The large
improvement over the strong weight is consistent with a scale issue being
important, but this comparison does not establish a unique cause, generalize
to new roots, or show that direction weighting improves on L2.

The recorded objective `sum(e²)/d + 0.1*(e·u)²/d` is tested to have a
1.1:1 quadratic weight ratio between the direction and an orthogonal
coordinate, independent of latent dimension. The prior negative result is
preserved separately. The new rule uses actual cp5 RGB only through the
frozen encoder and does not see simulator outcomes when planning; it does
not predict future RGB or implement the original pixel diagnostic.

Instrumented totals over the 26 suffix replans: 104 encoder calls, 104
`model.predict` calls and 1,664 predicted samples. Synchronized suffix wall
times are recorded per arm in the journal. These counts do not cover
candidate-bank generation, environment work or the total method; they do
not certify fair total computation.

- Plan SHA-256:
  `d9e505458e5a825c90b3c1c619ca3a2530723f3e828f881afa9ba887dcea2bbe`
- Journal SHA-256:
  `2a69e3047366a19914a5f6c59a672a40bfe1569f0cc3346c8741f86a83d3b15b`
- Saved pairing SHA-256:
  `38347e5f7719bc25e45a0ff337a16ac5dead98517308137d0fb54098a89f655a`
- Targeted tests, including full saved-result recomputation: 13 passed.

P1 mechanism intervention, root/source independence, matched strong
baselines and the complete research programme remain unproved.
