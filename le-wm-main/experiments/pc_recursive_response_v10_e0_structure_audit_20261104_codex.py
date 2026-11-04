"""Independent reconstruction of v10 probe, candidate and retention structure."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_recursive_response_v10_e0_dev_20261104_codex import OUT, previous, seal


def main():
    destination = OUT / "structure_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((OUT / "manifest.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())
    seal.verify(manifest["dependencies"])
    if audit["result_sha256"] != previous.sha(OUT / "result.json"):
        raise ValueError("primary audit mismatch")
    checked_layers, checked_candidates = 0, 0
    for row in result["rows"]:
        if not row["arm"].startswith("v10"):
            continue
        if previous.sha(OUT / row["trace_file"]) != row["trace_sha256"]:
            raise ValueError("trace changed")
        result_plan = row["result"]
        config = result_plan["config"]
        dim, length = config["action_dim"], config["segment_length"]
        nominal = np.asarray(config["nominals"], np.float64)
        axis = np.eye(length*dim).reshape(-1, length, dim)
        beams = [{"chain": np.empty((0, dim), np.float32), "path": []}]
        cursor = 1
        with np.load(OUT / row["trace_file"], allow_pickle=False) as trace:
            for layer in result_plan["layers"]:
                candidate_chains, candidate_paths = [], []
                for parent_index, parent in enumerate(beams):
                    for nominal_index, base in enumerate(nominal):
                        record = layer["responses"][parent_index * len(nominal) + nominal_index]
                        if (record["parent"], record["nominal"]) != (parent_index, nominal_index):
                            raise ValueError("wrong response lineage")
                        probes = np.concatenate((base[None], base[None]+config["fd_step"]*axis,
                                                 base[None]-config["fd_step"]*axis))
                        expected = np.concatenate((np.broadcast_to(parent["chain"],
                            (len(probes), len(parent["chain"]), dim)), probes.astype(np.float32)), axis=1)
                        np.testing.assert_array_equal(trace[f"actions_{cursor}"], expected)
                        positions = trace[f"positions_{cursor}"]
                        n = len(axis)
                        matrix = ((positions[1:n+1] - positions[n+1:]) / (2*config["fd_step"])).T
                        u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
                        contrib = singular**2 / (singular**2 + config["damping"]) * (
                            u.T @ (np.asarray(row["task"]["goal"]) - positions[0]))**2
                        modes = np.argsort(-contrib[:int(np.sum(singular > 1e-10))],
                                           kind="stable")[:config["max_modes"]]
                        np.testing.assert_allclose(record["singular"], singular, rtol=0, atol=1e-12)
                        np.testing.assert_allclose(record["contributions"], contrib, rtol=0, atol=1e-12)
                        np.testing.assert_array_equal(record["selected_modes"], modes)
                        candidate_chains.append(np.concatenate((parent["chain"], base.astype(np.float32))))
                        candidate_paths.append(parent["path"] + [{"nominal": nominal_index, "mode": None, "scale": 0.}])
                        for mode in modes:
                            direction = vt[mode].reshape(length, dim)
                            for amplitude in config["amplitudes"]:
                                for sign in (-1, 1):
                                    segment = np.clip(base + sign * amplitude * direction / np.max(np.abs(direction)),
                                                      -1, 1).astype(np.float32)
                                    candidate_chains.append(np.concatenate((parent["chain"], segment)))
                                    candidate_paths.append(parent["path"] + [{"nominal": nominal_index, "mode": int(mode),
                                                                             "scale": float(sign*amplitude)}])
                        cursor += 1
                candidate_chains = np.stack(candidate_chains)
                candidates = layer["candidates"]
                np.testing.assert_array_equal(candidate_chains, trace[f"actions_{cursor}"])
                np.testing.assert_array_equal(candidate_chains, np.asarray([c["chain"] for c in candidates], np.float32))
                if candidate_paths != [c["path"] for c in candidates]:
                    raise ValueError("candidate provenance mismatch")
                values, scores = trace[f"positions_{cursor}"], trace[f"scores_{cursor}"]
                np.testing.assert_allclose(values, [c["feature"] for c in candidates], atol=1e-12, rtol=0)
                count = min(config["beam_width"], len(candidates))
                chosen = np.argsort(scores, kind="stable")[:min(config["greedy_slots"], count)].tolist()
                while len(chosen) < count:
                    dist = np.linalg.norm(values[:, None] - values[chosen][None], axis=2).min(axis=1)
                    dist[chosen] = -np.inf
                    chosen.append(int(np.argmax(dist)))
                if chosen != layer["retained"]:
                    raise ValueError("retention mismatch")
                beams = [{"chain": candidate_chains[i], "path": candidate_paths[i]} for i in chosen]
                checked_layers += 1
                checked_candidates += len(candidates)
                cursor += 1
    if checked_layers != 48:
        raise ValueError("wrong layer count")
    seal.verify(manifest["dependencies"])
    previous.write(destination, {"result_sha256": previous.sha(OUT / "result.json"),
        "primary_audit_sha256": previous.sha(OUT / "audit.json"),
        "audit_script_sha256": previous.sha(Path(__file__)), "layers": checked_layers,
        "candidates": checked_candidates,
        "scope": "same saved predictions and SVD implementation; independent construction and retention replay"})
    print(f"audited {checked_layers} layers and {checked_candidates} generated candidates")


if __name__ == "__main__":
    main()
