"""Recompute complete-horizon physical oracle and prefix-greedy counterfactual."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_two_packet_physical_oracle_20261102 as source

base, need, ROOT = source.base, source.need, source.ROOT
CONTRACT_SHA = "f826ea69ac9a4afb4d207a8980b281eea7262a3e8c20285b00788670c909c921"
REPORT_SHA = "0472a27f45dd5c5075c67a06b95edba274a2d8617fdc21e30e2b99df27237ef2"


def analyze():
    spec = base.read(ROOT / f"{source.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    report = base.read(ROOT / f"{source.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    need(spec == source.protocol() and report["contract_sha256"] == CONTRACT_SHA
         and not report["errors"] and len(report["rows"]) == 867, "sealed roster")
    summaries = []
    for prefix in spec["prefixes"]:
        rows = [r for r in report["rows"] if r["prefix"] == prefix["name"]]
        need(len(rows) == 289 and [(r["first"], r["second"]) for r in rows]
             == [(i, j) for i in range(17) for j in range(17)], "ordered candidate bank")
        need(all(r["status"] == "fixed_horizon_completed" and len(r["transitions"]) == 10 for r in rows),
             "incomplete/censored cannot enter reward oracle")
        prefix_rewards, terminal_rewards = np.empty(17), np.empty((17, 17))
        for i in range(17):
            first_packet = rows[i * 17]["transitions"][:5]
            for j in range(17):
                branch = rows[i * 17 + j]
                need(branch["transitions"][:5] == first_packet, "first packet varies with future action")
                need(all(t["step"] == k + 1 and not t["terminated"] and not t["truncated"]
                         and np.isfinite(t["reward"]) for k, t in enumerate(branch["transitions"])), "trajectory")
                terminal_rewards[i, j] = branch["transitions"][-1]["reward"]
            prefix_rewards[i] = first_packet[-1]["reward"]
        greedy = int(np.argmax(prefix_rewards))
        greedy_tail = int(np.argmax(terminal_rewards[greedy]))
        first, second = map(int, np.unravel_index(np.argmax(terminal_rewards), terminal_rewards.shape))
        summaries.append({
            "prefix": prefix["name"], "greedy_pair": [greedy, greedy_tail], "oracle_pair": [first, second],
            "greedy_first_reward": float(prefix_rewards[greedy]),
            "oracle_first_reward": float(prefix_rewards[first]),
            "greedy_terminal_reward": float(terminal_rewards[greedy, greedy_tail]),
            "oracle_terminal_reward": float(terminal_rewards[first, second]),
            "oracle_terminal_advantage": float(terminal_rewards[first, second] - terminal_rewards[greedy, greedy_tail]),
            "different_first_action": first != greedy,
            "first_reward_worse_for_oracle": bool(prefix_rewards[first] < prefix_rewards[greedy]),
            "early_stops": 0, "completed": len(rows),
            "oracle_final_state": rows[first * 17 + second]["transitions"][-1]["state"],
            "prefix_state": prefix["state_after_prefix"],
        })
    return {"schema": "p1_dev305_two_packet_oracle_saved_20261102.v1", "rows": summaries,
            "scope": "privileged fixed-bank physical oracle; known root; no RGB planning or independent confirmation",
            "p1_admission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
