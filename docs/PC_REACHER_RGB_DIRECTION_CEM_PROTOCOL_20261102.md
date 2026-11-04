# Real-RGB direction-conditioned CEM: development protocol

On the two exposed Reacher roots, execute each of the existing 13 sealed
response-SVD prefixes. At cp5 reobserve RGB and actual action history and
compute frozen-model planning coordinates for the current observation `z`
and goal `g`. Let `u=(z-g)/||z-g||`; reject nonfinite or zero directions.
Score each model-predicted suffix terminal `x` by

`mean((x-g)^2) + 0.1 * dot(x-g,u)^2`.

Use exactly the existing 16-sample, two-iteration, four-elite, root-seeded
two-packet CEM optimizer, then physically execute its ten-step suffix. The
isotropic term prevents free movement in directions orthogonal to `u`. The
constant 0.1 and this rule were selected after prior outcomes were opened;
this is exploratory, not pre-registered confirmation. This metric uses the
real RGB context through its frozen encoding, not future true RGB or simulator
state. It is not a pixel-distance predictor.

Keep all 26 prefixes and errors; compare against the archived same-prefix L2
feedback run only when cp5 RGB, history, goal and prefix match. Account for
the additional two current/goal encoder invocations per suffix call, separately
from the base optimizer's costs. Equal candidate counts do not certify equal
total compute. Existing diagnostic zero-control physical steps are not part
of the evaluated 15-step action chain. No extra training or model modification.
Development roots cannot establish independence, P1 or strong-baseline wins.
