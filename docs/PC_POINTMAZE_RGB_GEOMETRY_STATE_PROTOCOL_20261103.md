# PointMaze RGB geometry state interface: development protocol

The fixed top-down camera and static marker geometry used by the existing
native RGB bridge are environment metadata, not dynamic simulator state.
Query their actual legacy renderer parameters before evaluating any root,
derive the pixel-to-world pinhole projection, and detect the green marker
from RGB. Report calibration provenance: camera look-at `(3,3)` was chosen
using previously exposed root 1880. No state-labelled fitting or extra
learned policy, critic, readout, or adapter is allowed.

The current candidate panel starts at archive frame zero with no available
causal predecessor. Set initial velocity to zero as an explicit assumption;
do not infer velocity from future frames, outcomes, archive state, or goal.
Use the same previously fixed 384 candidates on roots 1884–1887 and compare
four separately labelled inputs: original raw-zero state; projected RGB
position with assumed zero velocity; privileged actual position with zero
velocity (diagnostic); privileged actual full state (diagnostic). Model
prediction may use only each arm's stated input, initial RGB, candidate
actions and goal RGB. Simulator states may be used only by the two explicitly
privileged diagnostic conditions and after selection for evaluation.

Save camera metadata, projected values, input hashes, model scores, selected
candidate indices, actual RGB ranking and position-only replay distances.
Test segmentation failure, camera projection direction and estimated state
contract. Existing archive-wide normalizer, development-selected camera,
already exposed roots and use of scene geometry prevent independent
confirmation; all conditions are development diagnostics, not P1 passage.
