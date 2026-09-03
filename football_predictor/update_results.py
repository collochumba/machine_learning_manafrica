"""Refresh historical/current-season result data without retraining models."""
from pathlib import Path
import joblib
from train import load_data_with_cache

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    df, report = load_data_with_cache(force_refresh=True, n_seasons=10)
    joblib.dump(df, BASE_DIR / "cache" / "latest_results.pkl")
    joblib.dump(report, BASE_DIR / "cache" / "latest_results_meta.pkl")
    loaded = sum(bool(v) for v in report.get("loaded", {}).values())
    print(f"Downloaded result data for {loaded} configured leagues; {len(df):,} rows.")
