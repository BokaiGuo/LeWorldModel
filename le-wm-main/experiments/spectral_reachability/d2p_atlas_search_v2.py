"""Development-only bounded program search; no scientific admission implied."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .d2p_composition_witness_v2 import digest
from .d2p_local_compiler import LocalCompiler


@dataclass(frozen=True)
class AtlasConfig:
    depth: int = 2
    beam_width: int = 2
    random_bridges: int = 2
    seed: int = 0
    improvement_margin: float = 0.0
    nonprogress_tolerance: float = 0.0

    def __post_init__(self):
        for name in ("depth", "beam_width", "random_bridges", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ("seed", "random_bridges") else 1):
                raise ValueError("invalid atlas integer configuration")
        if self.depth < 2:
            raise ValueError("composition requires at least two packets")
        for name in ("improvement_margin", "nonprogress_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError("finite nonnegative comparison tolerances required")


class AtlasSearch:
    """Expand action-witnessed nodes; all scores are model-space, not outcomes."""

    def __init__(self, compiler, config):
        if not isinstance(compiler, LocalCompiler) or not isinstance(config, AtlasConfig):
            raise TypeError("local compiler and frozen atlas config required")
        self.compiler, self.config = compiler, config

    def _bank(self, context, rng):
        c = self.compiler
        progress = c.progress(context, packets=1)
        shape = (c.config.packet_steps, len(c.scale))
        bank = [("zero", np.zeros(shape)),
                ("progress", np.asarray(progress["actions"], dtype=float))]
        for i, raw in enumerate(progress["geometry"]["raw_spectral_directions"][:c.config.spectral_modes]):
            direction = np.asarray(raw, dtype=float).reshape(shape)
            direction /= max(float(np.max(np.abs(direction))), 1e-12)
            for sign in (-1, 1):
                proposal = sign * c.config.prefix_radius * direction
                if np.all(proposal >= c.low) and np.all(proposal <= c.high):
                    bank.append((f"spectral_{i}_{sign:+d}", proposal))
        for i in range(self.config.random_bridges):
            bank.append((f"random_{i}", rng.uniform(c.low, c.high, size=shape)))
        return bank

    def search(self, context):
        c, cfg = self.compiler, self.config
        rng = np.random.default_rng(cfg.seed)
        initial_counts = dict(c.counts)
        root_cost = c._cost(context)
        roots = []
        for name, actions in self._bank(context, rng):
            successor = c._call("rollout", context, actions)
            roots.append({"root_name": name, "root_cost": c._cost(successor),
                          "context": successor, "actions": actions.tolist(),
                          "edges": [{"name": name, "actions": actions.tolist(),
                                     "output_sha256": digest(successor)}]})
        greedy_index = min(range(len(roots)), key=lambda i: (roots[i]["root_cost"], i))
        complete = []
        # Each first-prefix branch gets the same beam width and depth; the
        # actual branch cost can differ when local spectral ranks differ.
        for root_index, root in enumerate(roots):
            beam = [root]
            for _ in range(1, cfg.depth):
                expanded = []
                for node in beam:
                    for name, actions in self._bank(node["context"], rng):
                        successor = c._call("rollout", node["context"], actions)
                        expanded.append({"root_name": node["root_name"],
                                         "root_cost": node["root_cost"],
                                         "context": successor,
                                         "actions": node["actions"] + actions.tolist(),
                                         "edges": node["edges"] + [
                                             {"name": name, "actions": actions.tolist(),
                                              "output_sha256": digest(successor)}]})
                beam = sorted(expanded, key=lambda node: c._cost(node["context"]))[:cfg.beam_width]
            best = beam[0]
            direct = c._call("rollout", context, np.asarray(best["actions"]))
            if digest(direct) != digest(best["context"]):
                raise ValueError("full action witness differs from split composition")
            complete.append({"root_index": root_index, "root_name": best["root_name"],
                             "prefix_cost": best["root_cost"], "terminal_cost": c._cost(direct),
                             "actions": best["actions"], "edges": best["edges"],
                             "output_context": direct, "witness_replayed": True})
        selected = min(range(len(complete)), key=lambda i: (complete[i]["terminal_cost"], i))
        control_cost = min(complete[0]["terminal_cost"], complete[greedy_index]["terminal_cost"])
        eligible = [i for i, item in enumerate(complete)
                    if item["root_name"] not in ("zero", "progress")
                    and item["prefix_cost"] >= root_cost - cfg.nonprogress_tolerance
                    and item["terminal_cost"] + cfg.improvement_margin < control_cost]
        qualified = min(eligible, key=lambda i: (complete[i]["terminal_cost"], i)) if eligible else None
        counts = {key: c.counts[key] - initial_counts[key] for key in c.counts}
        return {"schema": "d2p_atlas_search.v2", "configuration": vars(cfg).copy(),
                "root_cost": root_cost, "programs": complete, "selected_index": selected,
                "zero_index": 0, "same_bank_greedy_index": greedy_index,
                "qualified_reconfiguration_index": qualified,
                "qualification_scope": "model_prediction_only_not_physical_mechanism",
                "operation_counts": counts, "backend_counters": deepcopy(getattr(c.backend, "counters", {})),
                "actual_neural_cost_certified": False, "physical_mechanism_verified": False,
                "P1_admission": False}
