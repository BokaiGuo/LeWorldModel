import subprocess
import tempfile
from pathlib import Path


def test_no_theory_disables_all_theory_substages_without_creating_output():
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="lewm-dry-run-") as temp_dir:
        output = Path(temp_dir) / "planned"
        completed = subprocess.run(
            [
                "bash",
                "experiments/run_full_spectral_transfer.sh",
                "--output",
                str(output),
                "--dry-run",
                "--no-theory",
                "--no-empirical",
                "--no-adapter",
                "--no-eval",
                "--no-summary",
                "--no-evidence-summary",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )

    assert completed.returncode == 0, completed.stderr
    assert "theory_" not in completed.stdout
    assert not output.exists()
