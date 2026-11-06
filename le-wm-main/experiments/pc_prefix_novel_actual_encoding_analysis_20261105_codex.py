"""Paired ordering and physical-success analysis for saved RGB scores."""
import json
from collections import Counter
from pathlib import Path


def sign(value):
    return 0 if abs(value) <= 1e-8 else 1 if value > 0 else -1


def main():
    path = (Path(__file__).resolve().parents[1] / "results" /
            "pc_prefix_novel_actual_encoding_native_pixels_v2_seed42_20261105_codex.json")
    data = json.loads(path.read_text())
    if len(data["records"]) != 50:
        raise ValueError("incomplete panel")
    counts = Counter()
    for row in data["records"]:
        if row["no_novel_candidate"]:
            counts["no_candidate"] += 1
            continue
        a, b = row["chains"]
        if [a["chain"], b["chain"]] != ["novel", "common"]:
            raise ValueError("chain order changed")
        counts["paired"] += 1
        prediction = sign(a["predicted_cost"] - b["predicted_cost"])
        actual = sign(a["actual_encoded_cost"] - b["actual_encoded_cost"])
        counts["prediction_actual_tie" if 0 in (prediction, actual)
               else "prediction_actual_agree" if prediction == actual else "prediction_actual_opposite"] += 1
        success = [a["first_success_step"] is not None, b["first_success_step"] is not None]
        if success[0] != success[1]:
            counts["physical_distinguishable"] += 1
            for name, ordering in (("predicted", prediction), ("actual", actual)):
                if ordering == 0:
                    counts[f"{name}_success_tie"] += 1
                else:
                    chosen = 0 if ordering < 0 else 1
                    counts[f"{name}_selected_success"] += int(success[chosen])
                    counts[f"{name}_selected_failure"] += int(not success[chosen])
    print(json.dumps({"counts": counts, "encoding_images": data["encoding_images"],
                      "model_sha256": data["model_sha256"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
