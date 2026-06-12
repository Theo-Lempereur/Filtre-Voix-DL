"""Orchestrate the complete dataset preparation and generation pipeline.

This is the main command-line entry point for data work. It can open the GUI,
compile-check Python files, clean previous outputs, prepare clean chunks,
prepare noise chunks, generate noisy/clean pairs, and run the dataset health
check.
"""

from pathlib import Path
import argparse
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    """Run one subprocess pipeline step and fail fast with a readable step name.

    Args:
        name: Human-readable step name printed in the terminal.
        command: Subprocess command list executed from the project root.

    Returns:
        None.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero status.
    """
    print("\n" + "=" * 80)
    print(f"RUNNING: {name}")
    print("=" * 80)

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name}")

    print(f"[OK] {name}")


def main() -> None:
    """Parse CLI flags and orchestrate the selected dataset pipeline stages.

    Returns:
        None. Each selected stage runs as a subprocess and writes its own
        outputs, metadata, and logs.
    """
    parser = argparse.ArgumentParser(
        description="Run the complete speech enhancement dataset pipeline."
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Check Python syntax before running.",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the visual dataset configuration interface.",
    )

    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Clear chunks, generated data, metadata, and logs before running.",
    )

    parser.add_argument(
        "--reset-generated",
        action="store_true",
        help="Clear only generated data, metadata, and logs before running.",
    )

    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip prepare_clean.py.",
    )

    parser.add_argument(
        "--skip-noise",
        action="store_true",
        help="Skip prepare_noise.py.",
    )

    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip generate_noisy.py.",
    )

    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip check_dataset.py.",
    )

    args = parser.parse_args()

    python = sys.executable

    if args.gui:
        run_step(
            "Dataset configuration interface",
            [python, "scripts/config_gui.py"],
        )
        return

    print("=" * 80)
    print("FULL SPEECH ENHANCEMENT PIPELINE")
    print("=" * 80)
    print(f"Project: {PROJECT_ROOT}")
    print(f"Python : {python}")

    if args.compile:
        run_step(
            "Python syntax check",
            [python, "-m", "compileall", "scripts", "src"],
        )

    if args.reset_all:
        run_step(
            "Full output cleanup",
            [python, "scripts/clean_pipeline_outputs.py", "--all"],
        )

    if args.reset_generated:
        run_step(
            "Generated dataset + metadata + logs cleanup",
            [
                python,
                "scripts/clean_pipeline_outputs.py",
                "--generated",
                "--metadata",
                "--logs",
            ],
        )

    if not args.skip_clean:
        run_step(
            "Preprocessing CLEAN",
            [python, "scripts/prepare_clean.py"],
        )

    if not args.skip_noise:
        run_step(
            "Preprocessing NOISE",
            [python, "scripts/prepare_noise.py"],
        )

    if not args.skip_generate:
        run_step(
            "NOISY/CLEAN generation",
            [python, "scripts/generate_noisy.py"],
        )

    if not args.skip_check:
        run_step(
            "Dataset checkup",
            [python, "scripts/check_dataset.py"],
        )

    print("\n" + "=" * 80)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
