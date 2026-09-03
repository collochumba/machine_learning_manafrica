"""
FIXTURES MODULE

Consolidates everything to do with upcoming/live fixtures and team-name
resolution:

  1. Fixture download + parsing         (from fixture_loader.py)
  2. Team-name normalization/resolution (from team_normalization.py)
  3. Artifact/model health + backup     (from data_manager.py)

This module NEVER touches the training/model artifacts on its own — it only
produces upcoming-fixture dicts (section 1), resolves raw fixture team names
against the trained model's current-season team lists (section 2), and
reports/backs up artifact files on request (section 3). train.py owns actual
model training.
"""

import io
import re
import time
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from difflib import get_close_matches

import joblib
import pandas as pd
import numpy as np

from config import DIV_TO_LEAGUE, LEAGUE_CONFIG

# ============================================================================
# 1. FIXTURE DOWNLOAD + PARSING  (from fixture_loader.py)
# ============================================================================

FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTURES_XLSX_URL = "https://www.football-data.co.uk/fixtures.xlsx"

# football-data.co.uk division codes -> our canonical league names.
# This is derived from the single authoritative 16-league configuration.

# Odds-column fallback chain, checked in order, for each of Home/Draw/Away.
# Bet365 first (most consistently populated by football-data.co.uk), then
# the market average, then the market max, as last resort.
ODDS_FALLBACK_CHAIN = {
    'Home': ['B365H', 'AvgH', 'MaxH', 'PSH'],
    'Draw': ['B365D', 'AvgD', 'MaxD', 'PSD'],
    'Away': ['B365A', 'AvgA', 'MaxA', 'PSA'],
}


def _requests_session(retries=4, backoff_factor=1.0):
    """Create a resilient HTTP session for football-data downloads."""
    import requests
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
        retry = Retry(
            total=retries, connect=retries, read=retries, status=retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({'GET'}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
    except Exception:
        adapter = HTTPAdapter(max_retries=retries)
    session = requests.Session()
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0 FootballPredictor/1.0'})
    return session


def fetch_fixtures_bytes(url=FIXTURES_CSV_URL, timeout=20, retries=4):
    """Download fixture bytes with retry/backoff and clear failure reporting."""
    try:
        with _requests_session(retries=retries) as session:
            resp = session.get(url, timeout=(10, timeout))
            resp.raise_for_status()
            if not resp.content:
                raise ValueError('empty response')
            return resp.content, None
    except Exception as e:
        return None, f"Unable to download latest fixtures after retries ({e})"


def refresh_fixture_cache(cache_file, max_age_hours=6, force=False):
    """Refresh the live fixture cache when stale, retaining last-good data on failure.

    Returns: (cache_dict_or_none, status_dict).
    """
    import joblib
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    cached = None
    if cache_path.exists():
        try:
            cached = joblib.load(cache_path)
        except Exception:
            cached = None

    fresh = False
    if cached and cached.get('fetched_at'):
        try:
            age_hours = (now - cached['fetched_at']).total_seconds() / 3600.0
            fresh = age_hours < max_age_hours
        except Exception:
            age_hours = None
    else:
        age_hours = None

    if fresh and not force:
        return cached, {'attempted': False, 'fresh': True, 'age_hours': age_hours, 'error': None}

    content, err = fetch_fixtures_bytes()
    if err:
        return cached, {'attempted': True, 'fresh': False, 'age_hours': age_hours, 'error': err, 'using_last_good': cached is not None}

    raw_df, perr = parse_fixture_bytes(content, 'fixtures.csv')
    if perr:
        return cached, {'attempted': True, 'fresh': False, 'age_hours': age_hours, 'error': perr, 'using_last_good': cached is not None}

    joblib.dump({'raw_df': raw_df, 'source': 'football-data.co.uk (automatic refresh)', 'fetched_at': now}, cache_path)
    return {'raw_df': raw_df, 'source': 'football-data.co.uk (automatic refresh)', 'fetched_at': now}, {'attempted': True, 'fresh': True, 'age_hours': 0.0, 'error': None}


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
        # Optional markets: only include them when the source actually
        # provides valid bookmaker columns. Never invent odds.
        optional = {}
        ou_candidates = [('Over2.5', ['B365>2.5', 'B365O25', 'Avg>2.5', 'AvgO25']), ('Under2.5', ['B365<2.5', 'B365U25', 'Avg<2.5', 'AvgU25'])]
        for key, cols in ou_candidates:
            for col in cols:
                if col in row and pd.notna(row[col]):
                    try:
                        v = float(row[col])
                        if v > 1.0:
                            optional[key] = v
                            break
                    except (ValueError, TypeError):
                        pass
        ah_map = {'Home': ['AHh', 'B365AHH', 'PAHH'], 'Away': ['AHa', 'B365AHA', 'PAHA']}
        ah = {}
        for side, cols in ah_map.items():
            for col in cols:
                if col in row and pd.notna(row[col]):
                    try:
                        v = float(row[col])
                        if v > 1.0:
                            ah[side] = v
                            break
                    except (ValueError, TypeError):
                        pass
        if optional:
            odds['OverUnder'] = optional
        if len(ah) == 2:
            odds['AsianHandicap'] = ah
        return odds
    return None  # partial or missing 1X2 odds — no value calculation


def _parse_fixture_date(date_val):
    """
    Parse football-data.co.uk's Date column (typically DD/MM/YY or
    DD/MM/YYYY) into a datetime, trying both. Returns None if unparseable
    — callers must not treat that as "definitely current", only as
    "couldn't verify", per the strict-rejection philosophy.
    """
    if date_val is None:
        return None
    s = str(date_val).strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def extract_supported_fixtures(raw_df, reference_date=None, max_past_days=2):
    """
    Filter a raw fixture DataFrame down to all configured supported leagues and
    extract the fields needed downstream (league, date, time, teams, odds).

    Does NOT do team-name resolution — that's team_normalization.py's job,
    kept separate so this module has no dependency on the trained model's
    team lists.

    Also excludes rows whose Date is more than `max_past_days` before
    `reference_date` (defaults to now): the "latest fixtures" workflow is
    for UPCOMING matches, so a stray historical row surviving in the
    downloaded/uploaded file (a stale cache, a manually-edited file, etc.)
    should not be silently treated as a live fixture. Rows with an
    unparseable date are kept (never silently dropped) but flagged.

    Returns:
        supported: list of dicts with keys
            league, date, time, home_raw, away_raw, odds (dict or None)
        excluded_counts: dict of {div_code: count} for excluded rows
        total_rows: int, total rows in the raw file
    """

    if raw_df is None or len(raw_df) == 0:
        return [], {}, 0

    reference_date = reference_date or datetime.now()
    cutoff = reference_date - timedelta(days=max_past_days)

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

        parsed_date = _parse_fixture_date(date_val)
        if parsed_date is not None and parsed_date < cutoff:
            # Genuinely stale/historical row — exclude, tracked separately
            # from the unsupported-division counts so the UI can tell them
            # apart (see summarize_excluded).
            excluded_counts['__historical__'] = excluded_counts.get('__historical__', 0) + 1
            continue

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
    'T1': 'Turkey', 'USA': 'MLS',
}


def summarize_excluded(excluded_counts):
    """Turn {'E1': 24, 'E2': 10} into {'Championship': 24, 'League One': 10}
    for display, falling back to the raw code for anything not in our hint
    table (we don't guess — an unrecognised code is shown as-is). The
    '__historical__' pseudo-code (past-dated rows) is labelled separately
    from unsupported-division exclusions since it's a different reason."""
    named = {}
    for div, count in excluded_counts.items():
        if div == '__historical__':
            label = 'Historical/past-dated (excluded from live fixtures)'
        else:
            label = DIV_NAME_HINTS.get(div, div)
        named[label] = named.get(label, 0) + count
    return named

# ============================================================================
# 2. TEAM NAME NORMALIZATION / RESOLUTION  (from team_normalization.py)
# ============================================================================

# ---------------------------------------------------------------------------
# Known aliases: football-data.co.uk / common media names -> canonical name
# (canonical names are whatever current_teams.pkl / all_teams.pkl use).
# Keys are matched case-insensitively after normalization, so casing here is
# just for readability.
# ---------------------------------------------------------------------------

ALIASES = {
    "Premier League": {
        "aston villa fc": "Aston Villa",
        "brighton & hove albion": "Brighton",
        "brighton and hove albion": "Brighton",
        "leeds united": "Leeds",
        "manchester city": "Man City",
        "manchester united": "Man United",
        "man utd": "Man United",
        "newcastle united": "Newcastle",
        "nottingham forest": "Nott'm Forest",
        "nottm forest": "Nott'm Forest",
        "nott'm forest": "Nott'm Forest",
        "tottenham hotspur": "Tottenham",
        "spurs": "Tottenham",
        "west ham united": "West Ham",
        "wolverhampton": "Wolves",
        "wolverhampton wanderers": "Wolves",
    },
    "La Liga": {
        "atletico madrid": "Ath Madrid",
        "atlético madrid": "Ath Madrid",
        "atletico de madrid": "Ath Madrid",
        "atl. madrid": "Ath Madrid",
        "atl madrid": "Ath Madrid",
        "athletic bilbao": "Ath Bilbao",
        "athletic club": "Ath Bilbao",
        "ath. bilbao": "Ath Bilbao",
        "real betis": "Betis",
        "celta vigo": "Celta",
        "espanyol": "Espanol",
        "rcd espanyol": "Espanol",
        "real betis balompie": "Betis",
        "getafe cf": "Getafe",
        "girona fc": "Girona",
        "ca osasuna": "Osasuna",
        "real sociedad": "Sociedad",
        "valencia cf": "Valencia",
        "rayo vallecano": "Vallecano",
        "villarreal cf": "Villarreal",
        "sevilla fc": "Sevilla",
    },
    "Serie A": {
        "atalanta bc": "Atalanta",
        "bologna fc": "Bologna",
        "cagliari calcio": "Cagliari",
        "como 1907": "Como",
        "us cremonese": "Cremonese",
        "acf fiorentina": "Fiorentina",
        "genoa cfc": "Genoa",
        "fc internazionale": "Inter",
        "inter milan": "Inter",
        "inter milano": "Inter",
        "internazionale": "Inter",
        "juventus fc": "Juventus",
        "ss lazio": "Lazio",
        "us lecce": "Lecce",
        "ac milan": "Milan",
        "ac monza": "Monza",
        "ssc napoli": "Napoli",
        "as roma": "Roma",
        "us sassuolo": "Sassuolo",
        "udinese calcio": "Udinese",
        "hellas verona": "Verona",
    },
    "Bundesliga": {
        "bayern": "Bayern Munich",
        "bayern munich": "Bayern Munich",
        "borussia dortmund": "Dortmund",
        "bvb": "Dortmund",
        "eintracht frankfurt": "Ein Frankfurt",
        "cologne": "FC Koln",
        "koln": "FC Koln",
        "1. fc koln": "FC Koln",
        "sc freiburg": "Freiburg",
        "hamburger sv": "Hamburg",
        "tsg 1899 hoffenheim": "Hoffenheim",
        "tsg hoffenheim": "Hoffenheim",
        "bayer 04 leverkusen": "Leverkusen",
        "bayer leverkusen": "Leverkusen",
        "borussia monchengladbach": "M'gladbach",
        "borussia mönchengladbach": "M'gladbach",
        "monchengladbach": "M'gladbach",
        "m'gladbach": "M'gladbach",
        "rb leipzig": "RB Leipzig",
        "fc st. pauli": "St Pauli",
        "st. pauli": "St Pauli",
        "1. fc union berlin": "Union Berlin",
        "fc union berlin": "Union Berlin",
        "sv werder bremen": "Werder Bremen",
        "vfl wolfsburg": "Wolfsburg",
    },
    "Ligue 1": {
        "angers sco": "Angers",
        "aj auxerre": "Auxerre",
        "stade brestois": "Brest",
        "havre ac": "Le Havre",
        "rc lens": "Lens",
        "losc lille": "Lille",
        "fc lorient": "Lorient",
        "olympique lyonnais": "Lyon",
        "ol": "Lyon",
        "olympique marseille": "Marseille",
        "om": "Marseille",
        "fc metz": "Metz",
        "as monaco": "Monaco",
        "fc nantes": "Nantes",
        "ogc nice": "Nice",
        "paris saint germain": "Paris SG",
        "paris saint-germain": "Paris SG",
        "psg": "Paris SG",
        "stade rennais": "Rennes",
        "rc strasbourg": "Strasbourg",
        "toulouse fc": "Toulouse",
        # Deliberately NOT included: "paris" alone (ambiguous between Paris
        # SG and Paris FC — must not be auto-resolved) and "le mans" (not a
        # current top-flight team in any supported league).
    },
}


def _normalize_key(name):
    """Lowercase, strip, collapse whitespace, drop punctuation for comparison."""
    s = name.strip().lower()
    s = re.sub(r"[^\w\s]", "", s)  # drop punctuation (periods, apostrophes, &, etc.)
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------------------------------------------------------
# SEASON ROLLOVER OVERRIDES
#
# ROOT CAUSE this table works around: train.py's get_current_season_teams()
# only trusts a season's team list once that season has accumulated
# min_matches_for_current (100) completed matches — a deliberate guard
# against building a ~4-team "current" list from a season that's just
# started. The side effect: for the first several weeks of every new
# season, current_teams_by_league silently falls back to LAST season's
# roster, even for leagues whose new-season data file loaded successfully
# (see current_teams.pkl's `latest_season_by_league`, which will show the
# PRIOR season for every league until enough matches accumulate).
#
# This table is the safe, auditable fix for that gap: a hand-verified,
# dated promotion/relegation delta (same philosophy as ALIASES above —
# explicit and sourced, never inferred/fuzzy-guessed). It is applied
# ADDITIVELY/SUBTRACTIVELY on top of whatever current_teams_by_league
# already contains; it never touches all_teams.pkl or the historical
# training data, and it never causes an unresolved name to be guessed —
# it only ever adds/removes exact, verified team names from the
# resolution pool.
#
# Verified via web search (Wikipedia / league sources) on 2026-08-22:
#   - Premier League: Coventry, Hull, Ipswich UP; West Ham, Burnley, Wolves DOWN
#   - La Liga: Santander, La Coruna, Malaga UP; Oviedo, Girona, Mallorca DOWN
#   - Serie A: Frosinone, Venezia, Monza UP; Cremonese, Verona, Pisa DOWN
#   - Bundesliga: Schalke 04, Elversberg, Paderborn UP; Wolfsburg, Heidenheim, St Pauli DOWN
#   - Ligue 1: Troyes, Le Mans UP; Metz, Nantes DOWN
#
# MAINTENANCE: update this table (and the SEASON constant) once a year,
# after each league's promotion/relegation is finalised for the upcoming
# season — normally in Jun/Jul before a new campaign starts in Aug.
# ---------------------------------------------------------------------------

SEASON_ROLLOVER_OVERRIDES = {
    "2627": {
        "Premier League": {
            "add": ["Coventry", "Hull", "Ipswich"],
            "remove": ["West Ham", "Burnley", "Wolves"],
        },
        "La Liga": {
            "add": ["Santander", "La Coruna", "Malaga"],
            "remove": ["Oviedo", "Girona", "Mallorca"],
        },
        "Serie A": {
            "add": ["Frosinone", "Venezia", "Monza"],
            "remove": ["Cremonese", "Verona", "Pisa"],
        },
        "Bundesliga": {
            "add": ["Schalke 04", "Elversberg", "Paderborn"],
            "remove": ["Wolfsburg", "Heidenheim", "St Pauli"],
        },
        "Ligue 1": {
            "add": ["Troyes", "Le Mans"],
            "remove": ["Metz", "Nantes"],
        },
    },
}


def apply_season_overrides(current_teams_by_league, target_season="2627"):
    """
    Apply the verified SEASON_ROLLOVER_OVERRIDES on top of whatever
    current_teams_by_league was loaded from current_teams.pkl, to correct
    for the season-rollover staleness described above.

    Additive/subtractive only — every name added or removed is an exact,
    hand-verified team name, never a guess. Names already present are
    left untouched (idempotent — safe to call every time the app loads).

    Args:
        current_teams_by_league: dict league -> list of team names, as
            loaded from current_teams.pkl.
        target_season: which season's override table to apply (defaults
            to the current/latest one maintained above).

    Returns:
        (corrected_by_league, changes): corrected_by_league is a NEW dict
        (input is not mutated); changes is a list of human-readable
        strings describing every addition/removal actually made, for
        transparency in the UI/logs.
    """
    overrides = SEASON_ROLLOVER_OVERRIDES.get(target_season, {})
    corrected = {league: list(teams) for league, teams in current_teams_by_league.items()}
    changes = []

    for league, delta in overrides.items():
        current_list = corrected.setdefault(league, [])
        for team in delta.get("add", []):
            if team not in current_list:
                current_list.append(team)
                changes.append(f"{league}: + {team} (promoted for {target_season})")
        for team in delta.get("remove", []):
            if team in current_list:
                current_list.remove(team)
                changes.append(f"{league}: − {team} (relegated out of {target_season})")
        corrected[league] = sorted(current_list)

    return corrected, changes


def resolve_team(raw_name, league, current_teams_by_league, all_teams=None):
    """
    Resolve a raw team name to a canonical current-season team name for the
    given league, using ONLY the strict hierarchy described in the module
    docstring. Never auto-substitutes an unrelated team.

    Args:
        raw_name: team name as it appeared in the fixture source
        league: canonical league name (e.g. 'Premier League')
        current_teams_by_league: dict league -> list of canonical current
            teams (from current_teams.pkl['current_teams_by_league'],
            ideally already passed through apply_season_overrides())
        all_teams: optional full historical team list, used to (a) detect
            a genuinely historical (not fuzzy) team name and (b) widen
            fuzzy SUGGESTIONS — never to resolve automatically.

    Returns:
        dict with:
            'input': raw_name
            'league': league
            'resolved': canonical name, or None if unresolved
            'method': one of 'exact', 'case_insensitive', 'normalized',
                      'alias', 'unresolved'
            'category': 'current' (resolved), 'historical' (an EXACT match
                exists in all_teams but not in this league's current list
                — a real club, just not currently in this league/season),
                or 'unknown' (no exact or close match anywhere)
            'suggestions': list of plausible canonical names (only
                populated when category=='unknown') — for DISPLAY ONLY,
                never auto-applied
            'message': ready-to-display, unambiguous status line
    """

    canonical_list = current_teams_by_league.get(league, [])
    raw_stripped = raw_name.strip()
    norm_key = _normalize_key(raw_stripped)

    def _resolved(resolved_name, method):
        return {'input': raw_name, 'league': league, 'resolved': resolved_name,
                'method': method, 'category': 'current', 'suggestions': [],
                'message': f"✅ {raw_name} → {resolved_name}" if resolved_name != raw_name
                           else f"✅ {resolved_name}"}

    # 1. Exact match
    if raw_stripped in canonical_list:
        return _resolved(raw_stripped, 'exact')

    # 2. Case-insensitive match
    lower_map = {c.lower(): c for c in canonical_list}
    if raw_stripped.lower() in lower_map:
        return _resolved(lower_map[raw_stripped.lower()], 'case_insensitive')

    # 3. Punctuation/whitespace-normalized match
    norm_map = {_normalize_key(c): c for c in canonical_list}
    if norm_key in norm_map:
        return _resolved(norm_map[norm_key], 'normalized')

    # 4. Explicit alias match (checked against the normalized key, so
    # "Nottingham Forest", "nottingham   forest", "Nottingham-Forest" etc.
    # all hit the same alias entry)
    league_aliases = ALIASES.get(league, {})
    alias_norm_map = {_normalize_key(k): v for k, v in league_aliases.items()}
    if norm_key in alias_norm_map:
        canonical = alias_norm_map[norm_key]
        # Only trust the alias if it actually IS a current team for this
        # league (guards against a stale alias pointing to a relegated team).
        if canonical in canonical_list:
            return _resolved(canonical, 'alias')

    # 5. Not currently valid for this league. Distinguish a genuinely
    # historical/unsupported club (exact name match in the wider
    # all_teams pool, e.g. Frosinone before this season's override was
    # applied) from a name with no real match at all — these need very
    # different messages, and conflating them produced the old
    # "❌ Unrecognised team: Ipswich / Suggested: Ipswich" contradiction.
    all_teams_norm_map = {_normalize_key(t): t for t in (all_teams or [])}
    if norm_key in all_teams_norm_map:
        historical_name = all_teams_norm_map[norm_key]
        return {
            'input': raw_name, 'league': league, 'resolved': None,
            'method': 'unresolved', 'category': 'historical', 'suggestions': [],
            'message': (
                f"⚠️ Historical/unsupported team: {historical_name} — "
                f"not part of the current {league} team set for this fixture."
            ),
        }

    # 6. Genuinely unknown — fuzzy suggestions are for a human to confirm
    # only; they are never auto-applied. Suggestions are drawn only from
    # the widened all_teams pool (case 5 above already handled anything
    # that's an EXACT historical hit, so remaining fuzzy hits are always
    # genuinely approximate, never a contradictory self-match).
    pool = sorted(set(canonical_list) | set(all_teams or []))
    suggestions = get_close_matches(raw_stripped, pool, n=3, cutoff=0.6)

    if suggestions:
        message = (
            f"⚠️ Unresolved team: {raw_name} — possible matches: "
            f"{', '.join(suggestions)}. No automatic substitution was made."
        )
    else:
        message = (
            f"⚠️ Unsupported team: {raw_name} — no valid current-season "
            f"match found in {league}. This fixture will not be predicted."
        )

    return {
        'input': raw_name, 'league': league, 'resolved': None,
        'method': 'unresolved', 'category': 'unknown',
        'suggestions': suggestions, 'message': message,
    }


def resolve_fixture(league, home_raw, away_raw, current_teams_by_league, all_teams=None, fixture_candidates=None):
    """
    Resolve both teams of a fixture. Returns a dict describing the fixture's
    validation status:

        'status': 'valid' | 'needs_review'
        'league', 'home_raw', 'away_raw'
        'home': resolve_team(...) result
        'away': resolve_team(...) result
    """

    # fixture_candidates is an optional, league-scoped pool of names taken
    # from the latest fixture source. It is deliberately passed as an
    # additional candidate source rather than being allowed to override the
    # strict current-team/alias resolution rules. This keeps old call sites
    # backward compatible while allowing newly promoted/current-season names
    # to be recognized before the historical training universe is updated.
    candidates_for_league = fixture_candidates
    if isinstance(fixture_candidates, dict):
        candidates_for_league = fixture_candidates.get(league, [])
    candidates_for_league = list(candidates_for_league or [])

    # First use the normal strict resolver. If the current-team metadata is
    # stale at the beginning of a season, retry only against the exact live
    # fixture candidates for this same league. No cross-league candidate is
    # ever considered.
    home_res = resolve_team(home_raw, league, current_teams_by_league, all_teams)
    away_res = resolve_team(away_raw, league, current_teams_by_league, all_teams)

    def _resolve_from_fixture_candidates(raw_name, existing):
        if existing.get('resolved'):
            return existing
        raw_key = _normalize_key(raw_name)
        candidate_map = {_normalize_key(str(c)): str(c) for c in candidates_for_league}
        candidate = candidate_map.get(raw_key)
        if candidate is None:
            return existing
        # Candidate names come from the selected league's own fixture rows.
        # We still reject an exact candidate that is explicitly known to
        # belong to a different current league.
        for other_league, teams in (current_teams_by_league or {}).items():
            if other_league != league and any(_normalize_key(candidate) == _normalize_key(str(t)) for t in (teams or [])):
                return existing
        return {
            'input': raw_name, 'league': league, 'resolved': candidate,
            'method': 'fixture_candidate', 'category': 'current',
            'suggestions': [], 'message': f'✅ {raw_name} → {candidate} (live fixture candidate)'
        }

    home_res = _resolve_from_fixture_candidates(home_raw, home_res)
    away_res = _resolve_from_fixture_candidates(away_raw, away_res)

    status = 'valid' if (home_res['resolved'] and away_res['resolved']) else 'needs_review'

    return {
        'status': status,
        'league': league,
        'home_raw': home_raw,
        'away_raw': away_raw,
        'home': home_res,
        'away': away_res,
    }

# ============================================================================
# 3. ARTIFACT / MODEL HEALTH + BACKUP  (from data_manager.py)
# ============================================================================

SUPPORTED_ARTIFACTS = [
    "final_model.pkl", "dc_models.pkl", "corner_model.pkl", "card_model.pkl",
    "feature_cols.pkl", "processed_data.pkl", "team_mapping.pkl", "all_teams.pkl",
    "current_teams.pkl",
]


def artifact_health(base_dir):
    base = Path(base_dir)
    out = {"files": {}, "leagues": {}}
    for name in SUPPORTED_ARTIFACTS:
        path = base / name
        out["files"][name] = {
            "exists": path.exists(),
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
        }

    def load_dict_leagues(name, key=None):
        path = base / name
        if not path.exists():
            return []
        try:
            obj = joblib.load(path)
            if key:
                obj = obj.get(key, {}) if isinstance(obj, dict) else {}
            return sorted(obj.keys()) if isinstance(obj, dict) else []
        except Exception:
            return []

    df_leagues = []
    p = base / "processed_data.pkl"
    if p.exists():
        try:
            df = joblib.load(p)
            if "League" in df.columns:
                df_leagues = sorted(df["League"].dropna().unique().tolist())
        except Exception:
            pass

    out["leagues"] = {
        "historical_data": df_leagues,
        "dixon_coles": load_dict_leagues("dc_models.pkl"),
        "corners": load_dict_leagues("corner_model.pkl", "models"),
        "cards": load_dict_leagues("card_model.pkl", "models"),
    }
    return out


def backup_artifacts(base_dir, destination=None):
    """Back up existing PKL artifacts before a deliberate retraining run."""
    base = Path(base_dir)
    destination = Path(destination or (base / "backup_before_fix"))
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in SUPPORTED_ARTIFACTS:
        src = base / name
        if src.exists():
            dst = destination / name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied
