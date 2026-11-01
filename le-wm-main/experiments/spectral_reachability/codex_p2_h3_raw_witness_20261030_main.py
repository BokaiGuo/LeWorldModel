"""New H3 exploratory runner recording actual pre-transition CUDA action bytes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p2_symmetric_h3_run_20261030_v2 as base
from . import codex_p2_h3_physical_preflight_20261030_v1 as raw_hook

NAME = "codex_p2_h3_raw_witness_20261030_main"
_source_bindings = base.source_bindings


class Backend(base.Backend):
    """Capture rollout packets before transition; derivative probes are unchanged."""

    def __init__(self, *args):
        super().__init__(*args)
        self.candidate_ordinal = 0

    def _terminal(self, context, u, capture):
        if not capture:
            return super()._terminal(context, u, capture)
        t = self.t
        state = t.tensor(context["latent_history"], dtype=t.float32, device=self.device)[None]
        history = t.tensor(context["action_history"], dtype=t.float32, device=self.device)[None]
        actions = t.tanh(u).reshape(3, 5, 2)
        ordinal = self.candidate_ordinal
        self.candidate_ordinal += 1
        contexts = []
        for i in range(3):
            raw_hook.save_cuda_raw_packet(
                actions[i], index=i,
                sink=lambda event: self.meter.emit(dict(candidate_ordinal=ordinal, **event)))
            self.emit(dict(kind="packet_attempt", index=self.counts["packet"], packet=i))
            self.counts["packet"] += 1
            state, history = self.transition(state, history, actions[i])
            self.clock()
            self.emit(dict(kind="packet_return", index=self.counts["packet"] - 1, packet=i))
            contexts.append(dict(context, latent_history=state[0].detach().cpu().tolist(),
                                 action_history=history[0].detach().cpu().tolist()))
        return state[0, -1], contexts


def configure():
    base.NAME = NAME
    base.Backend = Backend

    def bindings():
        manifest, protocol, cases, sources = _source_bindings()
        for path in (Path(__file__), Path(raw_hook.__file__)):
            path = path.resolve()
            sources[str(path)] = base.digest_file(path)
        return manifest, protocol, cases, sources
    base.source_bindings = bindings


def check_raw_events(events, rows):
    """Require exactly three ordered action records for each candidate in every arm."""
    winners = []
    for row in rows:
        root, arm = row["root"], row["arm"]
        sub = [x["event"] for x in events if x.get("root") == root and x.get("arm") == arm]
        pending = []
        candidates = []
        for pos, event in enumerate(sub):
            if event["kind"] == "actual_cuda_raw_packet_before_transition":
                ordinal = len(candidates)
                base.bridge.host.need(event["candidate_ordinal"] == ordinal
                                      and event["packet"] == len(pending)
                                      and event["dtype"] == "<f4" and event["shape"] == [5, 2],
                                      "raw packet identity/order")
                try:
                    raw = bytes.fromhex(event["raw_hex"])
                except (ValueError, TypeError) as exc:
                    raise ValueError("raw packet hex") from exc
                base.bridge.host.need(len(raw) == 40
                                      and hashlib.sha256(raw).hexdigest() == event["raw_sha256"],
                                      "raw packet bytes/hash")
                values = np.frombuffer(raw, dtype="<f4")
                base.bridge.host.need(np.isfinite(values).all() and np.all(np.abs(values) <= 1),
                                      "raw packet domain")
                base.bridge.host.need(pos + 1 < len(sub) and sub[pos + 1]["kind"] == "attempt"
                                      and sub[pos + 1]["resource"] == "packet",
                                      "raw packet must precede transition attempt")
                pending.append(event["raw_hex"])
            elif event["kind"] == "candidate":
                base.bridge.host.need(len(pending) == 3, "candidate missing raw packets")
                candidates.append((event["witness"], pending))
                pending = []
        base.bridge.host.need(not pending and len(candidates) == 11, "raw candidate census")
        matches = [p for w, p in candidates[:9] if w == row["winner"]]
        base.bridge.host.need(len(matches) >= 1 and all(p == matches[0] for p in matches),
                              "winner raw witness ambiguity")
        winners.append(dict(root=root, arm=arm, winner_tag=row["winner"]["tag"],
                            raw_float32_hex="".join(matches[0])))
    return winners


def audit(directory, sha):
    result = base.audit(directory, sha)
    path = Path(directory)
    report = base.bridge.host.frontier.read(path / "report.json")
    with (path / "events.jsonl").open() as stream:
        events = [json.loads(line)["event"] for line in stream]
    winners = check_raw_events(events, report["rows"])
    base.bridge.host.need(base.digest_file(path / "report.json") == sha
                          and base.digest_file(path / "events.jsonl") == report["journal_sha256"],
                          "source drift")
    return dict(**result, winner_raw_actions=winners, original_raw_bits_proved_for_new_run=True,
                historical_runs_repaired=False, physical_replay=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "audit"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha")
    parser.add_argument("--ack")
    args = parser.parse_args()
    configure()
    if args.mode == "prepare":
        result = {"sha256": base.prepare(args.output)}
    elif args.mode == "run":
        result = base.run(args.prepared, args.sha, args.output, args.ack)
        result = {"status": result["status"], "rows": len(result["rows"]), "error": result.get("error")}
    else:
        result = audit(args.output, args.sha)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
