
"""
FOOTBALL PREDICTOR — SAFE MODEL TRAINING

Shared by direct PowerShell runs and the Streamlit dashboard.
Loads .env, configures UTF-8, acquires the shared training lock,
creates timestamped backups, validates artifacts and restores
previous artifacts if training fails.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# Load .env before importing project modules.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=False)
except ImportError:
    pass

# Configure UTF-8 before importing train.py.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(
                encoding="utf-8",
                errors="replace"
            )
        except (OSError, ValueError, TypeError):
            pass

import joblib

from config import SUPPORTED_LEAGUES
from train import main as training_main


LOCK_FILE = BASE_DIR / "training.lock"
BACKUP_ROOT = BASE_DIR / "training_backups"

ARTIFACTS = [
    "final_model.pkl",
    "dc_models.pkl",
    "corner_model.pkl",
    "corner_feature_cols.pkl",
    "card_model.pkl",
    "card_feature_cols.pkl",
    "league_config.pkl",
    "feature_cols.pkl",
    "processed_data.pkl",
    "team_mapping.pkl",
    "all_teams.pkl",
    "current_teams.pkl",
    "training_manifest.json",
]


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False

    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (OSError, ProcessLookupError):
        return False


def acquire_training_lock():
    """Acquire the shared lock, accepting a lock created by app.py."""

    parent_pid = os.getppid()

    for _ in range(3):
        try:
            fd = os.open(
                str(LOCK_FILE),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )

            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))

            return

        except FileExistsError:
            try:
                owner_pid = int(
                    LOCK_FILE.read_text(encoding="utf-8").strip()
                )
            except (OSError, ValueError):
                owner_pid = -1

            # Streamlit creates the lock immediately before launching us.
            if owner_pid in (parent_pid, os.getpid()):
                LOCK_FILE.write_text(
                    str(os.getpid()),
                    encoding="utf-8"
                )
                return

            if pid_is_running(owner_pid):
                raise RuntimeError(
                    f"Another training process is active (PID {owner_pid}). "
                    "Wait for it to finish."
                )

            # Remove stale lock and retry.
            try:
                LOCK_FILE.unlink(missing_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot remove stale training lock: {exc}"
                ) from exc

    raise RuntimeError(
        "Could not acquire the training lock."
    )


def release_training_lock():
    """Remove the lock only if this process owns it."""

    try:
        owner = LOCK_FILE.read_text(
            encoding="utf-8"
        ).strip()

        if owner == str(os.getpid()):
            LOCK_FILE.unlink(missing_ok=True)

    except OSError:
        pass


def create_backup() -> Path:
    """Create a timestamped backup without overwriting older backups."""

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    backup_dir = (
        BACKUP_ROOT / f"training_{stamp}"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    copied = []

    for filename in ARTIFACTS:
        source = BASE_DIR / filename

        if source.is_file():
            shutil.copy2(
                source,
                backup_dir / filename
            )
            copied.append(filename)

    manifest = {
        "pre_existing_artifacts": copied
    }

    (backup_dir / "backup_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print(f"Backup created: {backup_dir}")
    print(f"Existing artifacts backed up: {len(copied)}")

    return backup_dir


def restore_backup(backup_dir: Path):
    """Restore the exact pre-training artifact set."""

    manifest_path = backup_dir / "backup_manifest.json"

    backup_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    pre_existing = set(
        backup_manifest.get("pre_existing_artifacts", [])
    )

    restored = []
    removed = []

    for filename in ARTIFACTS:

        live_path = BASE_DIR / filename
        backup_path = backup_dir / filename

        if filename in pre_existing and backup_path.is_file():

            shutil.copy2(
                backup_path,
                live_path
            )

            restored.append(filename)

        elif filename not in pre_existing and live_path.exists():

            live_path.unlink()
            removed.append(filename)

    print(
        f"Rollback restored {len(restored)} existing artifacts."
    )

    if removed:
        print(
            "Rollback removed newly created artifacts: "
            + ", ".join(removed)
        )


def validate_artifacts() -> dict:
    """Validate the model artefacts generated by train.py."""

    missing = [
        filename
        for filename in ARTIFACTS
        if not (BASE_DIR / filename).is_file()
    ]

    if missing:
        raise RuntimeError(
            "Training did not create all required artifacts: "
            + ", ".join(missing)
        )

    loaded = {}

    for filename in ARTIFACTS:

        if filename.endswith(".pkl"):

            print(f"Validating {filename}...")

            loaded[filename] = joblib.load(
                BASE_DIR / filename
            )

    expected = set(SUPPORTED_LEAGUES)

    # Validate historical data.
    processed = loaded["processed_data.pkl"]

    if processed.empty or "League" not in processed.columns:
        raise RuntimeError(
            "processed_data.pkl is empty or has no League column."
        )

    historical = set(
        processed["League"].dropna().unique()
    )

    if historical != expected:
        raise RuntimeError(
            "Historical data coverage does not match the configured "
            "15 leagues. "
            f"Missing={sorted(expected - historical)}; "
            f"unexpected={sorted(historical - expected)}"
        )

    # Validate Dixon-Coles coverage.
    dc_models = loaded["dc_models.pkl"]

    if not isinstance(dc_models, dict):
        raise RuntimeError(
            "dc_models.pkl must contain a league-to-model dictionary."
        )

    if set(dc_models) != expected:
        raise RuntimeError(
            "Dixon-Coles model coverage mismatch. "
            f"Missing={sorted(expected - set(dc_models))}; "
            f"unexpected={sorted(set(dc_models) - expected)}"
        )

    # Validate the main ML model.
    final_model = loaded["final_model.pkl"]
    feature_cols = loaded["feature_cols.pkl"]

    if not isinstance(feature_cols, list) or not feature_cols:
        raise RuntimeError(
            "feature_cols.pkl is missing or invalid."
        )

    if not hasattr(final_model, "predict_proba"):
        raise RuntimeError(
            "final_model.pkl does not expose predict_proba()."
        )

    # Validate corner and card bundles.
    for filename in ("corner_model.pkl", "card_model.pkl"):

        bundle = loaded[filename]

        if (
            not isinstance(bundle, dict)
            or not isinstance(bundle.get("models"), dict)
        ):
            raise RuntimeError(
                f"{filename} is not a valid model bundle."
            )

        model_leagues = set(bundle["models"])

        if not model_leagues.issubset(expected):
            raise RuntimeError(
                f"{filename} contains unsupported leagues: "
                f"{sorted(model_leagues - expected)}"
            )

    # Validate current team mappings.
    current_teams = loaded["current_teams.pkl"]

    if not isinstance(current_teams, dict):
        raise RuntimeError(
            "current_teams.pkl must contain a dictionary."
        )

    teams_by_league = current_teams.get(
        "current_teams_by_league"
    )

    if not isinstance(teams_by_league, dict):
        raise RuntimeError(
            "current_teams.pkl has no current_teams_by_league mapping."
        )

    if set(teams_by_league) != expected:
        raise RuntimeError(
            "current_teams.pkl does not cover all 15 configured leagues."
        )

    # Validate training manifest.
    manifest_path = BASE_DIR / "training_manifest.json"

    with manifest_path.open(
        "r",
        encoding="utf-8"
    ) as handle:
        training_manifest = json.load(handle)

    if not training_manifest.get("training_timestamp"):
        raise RuntimeError(
            "training_manifest.json has no training timestamp."
        )

    manifest_matches = int(
        training_manifest.get(
            "dataset", {}
        ).get("match_count", 0)
    )

    if manifest_matches != len(processed):
        raise RuntimeError(
            "Training manifest match_count does not match "
            "processed_data.pkl."
        )

    if int(
        training_manifest.get("configured_leagues", 0)
    ) != len(expected):
        raise RuntimeError(
            "Training manifest does not report the expected 15 leagues."
        )

    print(
        "\nArtifact validation passed: "
        f"{len(historical)}/15 historical leagues, "
        f"{len(dc_models)}/15 Dixon-Coles models, "
        f"{len(loaded['corner_model.pkl']['models'])}/15 corner models, "
        f"{len(loaded['card_model.pkl']['models'])}/15 card models."
    )

    return training_manifest


def run_training(
    force_refresh: bool = True,
    n_seasons: int = 10
) -> dict:

    if n_seasons < 1:
        raise ValueError(
            "n_seasons must be at least 1."
        )

    acquire_training_lock()

    backup_dir = None

    try:

        print("\n" + "=" * 76)
        print("FOOTBALL PREDICTOR — SAFE MODEL TRAINING")
        print("=" * 76)

        # Back up existing models first.
        backup_dir = create_backup()

        # Run the original training pipeline.
        print("\nStarting training pipeline...")

        training_main(
            force_refresh=force_refresh,
            n_seasons=n_seasons
        )

        # Validate all new artifacts.
        print("\nValidating complete artifact set...")

        manifest = validate_artifacts()

        print("\n" + "=" * 76)
        print("TRAINING AND VALIDATION COMPLETED")
        print("=" * 76)

        print(
            f"Training timestamp: "
            f"{manifest.get('training_timestamp')}"
        )

        print(
            f"Historical matches: "
            f"{manifest.get('dataset', {}).get('match_count')}"
        )

        print(f"Backup retained at: {backup_dir}")

        return manifest

    except BaseException:

        # Roll back even after Ctrl+C.
        print(
            "\nTRAINING FAILED OR WAS INTERRUPTED — "
            "RESTORING PREVIOUS ARTIFACTS"
        )

        traceback.print_exc()

        if backup_dir is not None:

            try:
                restore_backup(backup_dir)

            except Exception:

                print(
                    "CRITICAL: automatic rollback failed. "
                    "Backup remains at:"
                )

                print(backup_dir)
                traceback.print_exc()

        raise

    finally:
        release_training_lock()


if __name__ == "__main__":

    force_refresh = "--no-force" not in sys.argv
    seasons = 10

    for arg in sys.argv:

        if arg.startswith("--seasons="):
            seasons = int(arg.split("=", 1)[1])

    run_training(
        force_refresh=force_refresh,
        n_seasons=seasons
    )
