from pathlib import Path
import argparse
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    print("\n" + "=" * 80)
    print(f"LANCEMENT : {name}")
    print("=" * 80)

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Étape échouée : {name}")

    print(f"[OK] {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lance tout le pipeline dataset speech enhancement."
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Vérifie la syntaxe Python avant lancement.",
    )

    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Vide chunks, generated, metadata et logs avant lancement.",
    )

    parser.add_argument(
        "--reset-generated",
        action="store_true",
        help="Vide seulement generated, metadata et logs avant lancement.",
    )

    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Ne lance pas prepare_clean.py.",
    )

    parser.add_argument(
        "--skip-noise",
        action="store_true",
        help="Ne lance pas prepare_noise.py.",
    )

    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Ne lance pas generate_noisy.py.",
    )

    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Ne lance pas check_dataset.py.",
    )

    args = parser.parse_args()

    python = sys.executable

    print("=" * 80)
    print("PIPELINE COMPLET SPEECH ENHANCEMENT")
    print("=" * 80)
    print(f"Projet : {PROJECT_ROOT}")
    print(f"Python : {python}")

    if args.compile:
        run_step(
            "Check syntaxe Python",
            [python, "-m", "compileall", "scripts", "src"],
        )

    if args.reset_all:
        run_step(
            "Nettoyage complet des sorties",
            [python, "scripts/clean_pipeline_outputs.py", "--all"],
        )

    if args.reset_generated:
        run_step(
            "Nettoyage dataset généré + metadata + logs",
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
            "Génération NOISY/CLEAN",
            [python, "scripts/generate_noisy.py"],
        )

    if not args.skip_check:
        run_step(
            "Checkup dataset",
            [python, "scripts/check_dataset.py"],
        )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLET TERMINÉ AVEC SUCCÈS")
    print("=" * 80)


if __name__ == "__main__":
    main()