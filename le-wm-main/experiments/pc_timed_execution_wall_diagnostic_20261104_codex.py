"""One-task exact-dynamics check of the common timed execution interface."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v17_e0_multistart_baseline_dev_20261104_codex as latest
from experiments.spectral_reachability import pc_timed_program_execution as timed
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = latest.ROOT
OUT = ROOT / "results/pc_timed_execution_wall_diagnostic_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_TIMED_EXECUTION_WALL_DIAGNOSTIC_PROTOCOL_20261104_CODEX.md"
TASK = {"start": [-.5, .317, -1.17], "goal": [.5, -.283]}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior = latest.prior
    old = prior.prior
    panels = (latest.tasks(), prior.tasks(), prior.v15.tasks(), latest.v16.tasks(),
              prior.fixed.tasks(), *prior.v15.source_panels())
    if any(TASK in panel for panel in panels):
        raise ValueError("task occurs in a checked panel")
    deps = seal.capture([Path(__file__), PROTOCOL, Path(timed.__file__),
                         Path(prior.compiler.__file__), Path(old.compiler.__file__),
                         Path(old.compiler_child.__file__), Path(latest.baseline.__file__),
                         Path(prior.v15.prior.cem.__file__), Path(old.exact.__file__),
                         Path(old.exact.wall.__file__), Path(old.exact.wall.physics.__file__),
                         Path(old.previous.independent.__code__.co_filename)])
    nominals = tuple(tuple(tuple(float(x) for x in point) for point in nominal)
                     for nominal in old.previous.prior.core.NOMINALS)
    config = old.compiler.Config(action_dim=2, segment_length=5, segments=3,
                                 model_packet_length=5, nominals=nominals,
                                 token_cap=1920, seed=2029311000)
    goal = np.asarray(TASK["goal"])

    def predict(actions):
        position = old.exact.endpoints(TASK, actions)
        return position, np.linalg.norm(position-goal, axis=1)

    OUT.mkdir()
    old.writer(OUT / "manifest.json", {"task": TASK, "dependencies": deps,
                                      "scope": "wiring diagnostic, not performance evidence"})
    rows, banks = [], []
    for arm in ("v17", "multistart_cem"):
        output = timed.execute(predict, goal, config, arm=arm, baseline_seed=2029311001)
        traces = output.pop("traces")
        banks.append(traces[0]["actions"])
        for index, (call, batch) in enumerate(zip(output["calls"], traces, strict=True)):
            actions = batch["actions"]
            path, _ = old.previous.independent(TASK["start"], actions)
            np.testing.assert_allclose(path[:, -1, :2], batch["features"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.linalg.norm(batch["features"]-goal, axis=1),
                                       batch["scores"], atol=1e-12, rtol=0)
            if call["prediction_tokens"] != len(actions) * actions.shape[1] // 5:
                raise ValueError(f"incorrect token count in batch {index}")
        selected = np.asarray(output["result"]["selected"], dtype=np.float32)
        if not any(batch["actions"].shape[1] == 15 and
                   np.any(np.all(batch["actions"] == selected, axis=(1, 2))) for batch in traces):
            raise ValueError("selected chain was not scored")
        trace_path = OUT / f"{arm}.npz"
        with trace_path.open("xb") as stream:
            np.savez_compressed(stream, **{f"{name}_{i}": batch[name]
                for i, batch in enumerate(traces) for name in ("actions", "features", "scores")})
        output.update({"arm": arm, "trace": trace_path.name, "trace_sha256": seal.digest(trace_path)})
        rows.append(output)
        print(f"{arm}: calls={len(output['calls'])}, tokens={output['resource']['prediction_tokens']}", flush=True)
    np.testing.assert_array_equal(*banks)
    seal.verify(deps)
    old.writer(OUT / "result.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"),
                                     "rows": rows, "audit": "same-environment geometry passed"})


if __name__ == "__main__":
    run()
