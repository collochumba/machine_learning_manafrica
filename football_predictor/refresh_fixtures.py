"""Command-line fixture refresher. Safe to run from Windows Task Scheduler."""
from pathlib import Path
import sys
from fixtures import refresh_fixture_cache

BASE_DIR = Path(__file__).resolve().parent
CACHE_FILE = BASE_DIR / "cache" / "latest_fixtures.pkl"

if __name__ == "__main__":
    cached, status = refresh_fixture_cache(CACHE_FILE, max_age_hours=6, force=True)
    if status.get("error"):
        print(f"WARNING: {status['error']}")
        if cached is not None:
            print("Using last known good fixture cache.")
        sys.exit(0 if cached is not None else 1)
    print(f"Fixtures refreshed successfully: {cached['fetched_at']}")
