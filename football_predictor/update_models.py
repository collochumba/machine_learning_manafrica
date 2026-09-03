"""Deliberate model update command. The normal Streamlit app never calls this."""
from pathlib import Path
from fixtures import backup_artifacts
from train import main

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    backup = backup_artifacts(BASE_DIR)
    print(f"Backed up {len(backup)} existing artifacts to backup_before_fix/")
    main(force_refresh=True, n_seasons=10)
