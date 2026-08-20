"""
FIXTURE LOADER MODULE

Loads upcoming fixtures from football-data.co.uk (either the live
fixtures.csv/fixtures.xlsx download, or a user-uploaded copy of the same
file), filters down to the 5 supported leagues, and extracts usable 1X2
odds where available.

This module NEVER touches the training/model artifacts (final_model.pkl,
processed_data.pkl, cache/raw_data.pkl, cache/features.pkl, etc.) — it only
produces a list of upcoming-fixture dicts to feed into predict.py.
"""

import io
import time
import pandas as pd
import numpy as np

FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTURES_XLSX_URL = "https://www.football-data.co.uk/fixtures.xlsx"

# football-data.co.uk division codes -> our canonical league names.
# ONLY these five are supported; everything else (Championship=E1,
# League One=E2, League Two=E3, La Liga 2=SP2, Serie B=I2, Bundesliga
# 2=D2, Ligue 2=F2, Scottish/Belgian/Greek leagues, etc.) is excluded.
DIV_TO_LEAGUE = {
    'E0': 'Premier League',
    'SP1': 'La Liga',
    'I1': 'Serie A',
    'D1': 'Bundesliga',
    'F1': 'Ligue 1',
}

# Odds-column fallback chain, checked in order, for each of Home/Draw/Away.
# Bet365 first (most consistently populated by football-data.co.uk), then
# the market average, then the market max, as last resort.
ODDS_FALLBACK_CHAIN = {
    'Home': ['B365H', 'AvgH', 'MaxH', 'PSH'],
    'Draw': ['B365D', 'AvgD', 'MaxD', 'PSD'],
    'Away': ['B365A', 'AvgA', 'MaxA', 'PSA'],
}


def fetch_fixtures_bytes(url=FIXTURES_CSV_URL, timeout=15):
    """
    Download the raw fixture file bytes from football-data.co.uk.

    Requires outbound network access to football-data.co.uk, which is
    only available wherever this app is actually deployed/run — not
    guaranteed in every sandboxed environment.

    Returns:
        (content_bytes, error_message). error_message is None on success.
    """
    try:
        import requests
        resp = requests.get(url, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
        return resp.content, None
    except Exception as e:
        return None, f"Unable to download latest fixtures ({e})"


def parse_fixture_bytes(content, filename_hint=""):
    """
    Parse fixture file bytes into a raw DataFrame. Handles both the CSV and
    XLSX variants football-data.co.uk publishes.

    Returns:
        (df, error_message)
    """
    try:
        if filename_hint.lower().endswith('.xlsx') or filename_hint.lower().endswith('.xls'):
            df = pd.read_excel(io.BytesIO(content))
        else:
            # Try CSV; football-data.co.uk CSVs are typically latin-1 encoded.
            try:
                df = pd.read_csv(io.BytesIO(content), encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), encoding='latin1')
        return df, None
    except Exception as e:
        return None, f"Could not parse fixture file — format may have changed ({e})"


def _extract_odds(row):
    """
    Extract 1X2 odds for a fixture row using the fallback chain. Returns
    a dict {'Home': x, 'Draw': y, 'Away': z} with all three present, or
    None if any of the three couldn't be found — we require all three
    valid 1X2 odds before treating the fixture as having usable odds
    (matches the "don't calculate value/Kelly with partial odds" rule).
    """
    odds = {}
    for outcome, candidate_cols in ODDS_FALLBACK_CHAIN.items():
        value = None
        for col in candidate_cols:
            if col in row and pd.notna(row[col]):
                try:
                    v = float(row[col])
                    if v > 1.0:
                        value = v
                        break
                except (ValueError, TypeError):
                    continue
        odds[outcome] = value

    if all(v is not None for v in odds.values()):
        return odds
    return None  # partial or missing odds — treat as "odds unavailable"


def extract_supported_fixtures(raw_df):
    """
    Filter a raw fixture DataFrame down to the 5 supported leagues and
    extract the fields needed downstream (league, date, time, teams, odds).

    Does NOT do team-name resolution — that's team_normalization.py's job,
    kept separate so this module has no dependency on the trained model's
    team lists.

    Returns:
        supported: list of dicts with keys
            league, date, time, home_raw, away_raw, odds (dict or None)
        excluded_counts: dict of {div_code: count} for excluded rows
        total_rows: int, total rows in the raw file
    """

    if raw_df is None or len(raw_df) == 0:
        return [], {}, 0

    total_rows = len(raw_df)
    div_col = 'Div' if 'Div' in raw_df.columns else None
    home_col = 'HomeTeam' if 'HomeTeam' in raw_df.columns else None
    away_col = 'AwayTeam' if 'AwayTeam' in raw_df.columns else None
    date_col = 'Date' if 'Date' in raw_df.columns else None
    time_col = 'Time' if 'Time' in raw_df.columns else None

    if not (div_col and home_col and away_col):
        raise ValueError(
            "Fixture file is missing expected columns (Div/HomeTeam/AwayTeam). "
            "The football-data.co.uk format may have changed."
        )

    supported = []
    excluded_counts = {}

    for _, row in raw_df.iterrows():
        div = str(row[div_col]).strip() if pd.notna(row[div_col]) else ""

        if div not in DIV_TO_LEAGUE:
            excluded_counts[div] = excluded_counts.get(div, 0) + 1
            continue

        home_raw = str(row[home_col]).strip() if pd.notna(row[home_col]) else None
        away_raw = str(row[away_col]).strip() if pd.notna(row[away_col]) else None
        if not home_raw or not away_raw:
            continue  # malformed row, skip silently (not a supported-league count issue)

        date_val = row[date_col] if date_col and pd.notna(row.get(date_col, None)) else None
        time_val = row[time_col] if time_col and pd.notna(row.get(time_col, None)) else None

        odds = _extract_odds(row)

        supported.append({
            'league': DIV_TO_LEAGUE[div],
            'date': str(date_val) if date_val is not None else "TBD",
            'time': str(time_val) if time_val is not None else "TBD",
            'home_raw': home_raw,
            'away_raw': away_raw,
            'odds': odds,  # None if any of H/D/A odds unavailable
        })

    return supported, excluded_counts, total_rows


DIV_NAME_HINTS = {
    'E1': 'Championship', 'E2': 'League One', 'E3': 'League Two', 'EC': 'Conference',
    'SP2': 'La Liga 2', 'I2': 'Serie B', 'D2': 'Bundesliga 2', 'F2': 'Ligue 2',
    'SC0': 'Scottish Premiership', 'SC1': 'Scottish Championship',
    'B1': 'Belgium', 'G1': 'Greece', 'P1': 'Portugal', 'N1': 'Netherlands',
    'T1': 'Turkey',
}


def summarize_excluded(excluded_counts):
    """Turn {'E1': 24, 'E2': 10} into {'Championship': 24, 'League One': 10}
    for display, falling back to the raw code for anything not in our hint
    table (we don't guess — an unrecognised code is shown as-is)."""
    named = {}
    for div, count in excluded_counts.items():
        label = DIV_NAME_HINTS.get(div, div)
        named[label] = named.get(label, 0) + count
    return named
