"""
PRODUCTION TRAINING PIPELINE - FIXED
No code duplication | Proper caching | Incremental updates
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

import joblib
from tqdm import tqdm
import os
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except Exception:
    Retry = None

# Import Dixon-Coles from models.py (NO DUPLICATION!)
from models import DixonColesTimeDecay

# Corner model (Task 2) — its own dedicated module, not derived from goals
# and not reusing the 1X2 model.
from corners import build_corner_features, CornerStrengthModel, CornerGBRModel, walk_forward_validate, CORNER_SCHEMA_VERSION
from cards import train_card_models, CARD_SCHEMA_VERSION
from config import LEAGUE_CONFIG, LEAGUES

print("="*80)
print("🚀 FOOTBALL BETTING MODEL - TRAINING PIPELINE v2.0")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# LEAGUES is imported from the single authoritative config.py.


def get_last_n_seasons(n=10):
    """Get last N seasons dynamically."""
    current_year = datetime.now().year
    if datetime.now().month < 8:
        current_year -= 1
    
    seasons = []
    for i in range(n):
        year = current_year - i
        seasons.append(f"{str(year)[-2:]}{str(year+1)[-2:]}")
    
    return seasons[::-1]

# ============================================================================
# STEP 1: DATA LOADING WITH CACHING
# ============================================================================

CACHE_VERSION = "v6-16leagues-shots-fouls-cards-referee-corners-rest"  # bump this whenever
                                            # the schema of the cached raw/
                                            # feature data changes (new/renamed
                                            # columns, different imputation or
                                            # feature-selection logic, etc.),
                                            # so stale caches auto-invalidate
                                            # instead of silently being reused
                                            # with the old column semantics.


def _download_session():
    session = requests.Session()
    if Retry is not None:
        retry = Retry(total=4, connect=4, read=4, status=4, backoff_factor=1.0,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({'GET'}), raise_on_status=False)
        adapter = HTTPAdapter(max_retries=retry)
    else:
        adapter = HTTPAdapter(max_retries=4)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0 FootballPredictor/1.0'})
    return session

def _read_csv_url(session, url):
    resp = session.get(url, timeout=(10, 30))
    resp.raise_for_status()
    if not resp.content:
        raise ValueError('empty response')
    from io import BytesIO
    return pd.read_csv(BytesIO(resp.content), encoding='latin1', on_bad_lines='skip')


def load_data_with_cache(force_refresh=False, n_seasons=10):
    """
    Load data with local caching to avoid repeated downloads.

    FIX: the cache is now versioned by (CACHE_VERSION, requested season list).
    If either changes (e.g. moving from 5 to 10 seasons) the cache is
    automatically rebuilt instead of silently serving stale 5-season data.

    Returns:
        df, load_report
        load_report = {
            'seasons_requested': [...],
            'loaded': {league: [seasons successfully loaded]},
            'failed': {league: [seasons unavailable/failed]},
        }
    """

    seasons = get_last_n_seasons(n_seasons)

    cache_file = CACHE_DIR / "raw_data.pkl"
    meta_file = CACHE_DIR / "raw_data_meta.pkl"

    # Check cache — only reuse it if the version AND season list match exactly
    if cache_file.exists() and meta_file.exists() and not force_refresh:
        cached_meta = joblib.load(meta_file)
        if cached_meta.get('version') == CACHE_VERSION and cached_meta.get('seasons_requested') == seasons:
            print("📦 Loading from cache (season list & schema match)...")
            df = joblib.load(cache_file)
            print(f"✅ Loaded {len(df):,} matches from cache")
            return df, cached_meta['load_report']
        else:
            print("♻️  Cache is stale (season list or schema changed) — rebuilding from source...")

    print("\n📥 Downloading fresh data from football-data.co.uk...")
    print(f"Requested seasons ({len(seasons)}): {seasons}")

    all_data = []
    loaded = {league: [] for league in LEAGUES}
    failed = {league: [] for league in LEAGUES}

    session = _download_session()
    for league_name, meta in LEAGUE_CONFIG.items():
        print(f"\n{league_name}:")
        if meta.get("source") == "extra":
            try:
                df_temp = _read_csv_url(session, meta['url'])
                if df_temp is None or len(df_temp) == 0:
                    raise ValueError("empty response")
                df_temp['Date'] = pd.to_datetime(df_temp['Date'], dayfirst=True, errors='coerce')
                # Extra-source feeds (source == "extra") may expose only the
                # currently published season for that competition. Keep every
                # available completed/current row and label seasons from
                # dates; never claim ten seasons when the source did not
                # provide them. No league in the current 15-league
                # LEAGUE_CONFIG uses source == "extra"; this branch is
                # inert unless one is added back in the future.
                df_temp['League'] = league_name
                df_temp['Season'] = df_temp['Date'].dt.year.astype('Int64').astype(str)
                all_data.append(df_temp)
                loaded[league_name] = sorted(df_temp['Season'].dropna().unique().tolist())
                print(f"  ✅ published MLS feed: {len(df_temp)} matches across {len(loaded[league_name])} season labels")
            except Exception as e:
                failed[league_name].append("calendar-window")
                print(f"  ⚠️  calendar-window: Unavailable/failed ({e})")
            continue

        for season in seasons:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{meta['code']}.csv"
            try:
                df_temp = _read_csv_url(session, url)
                if df_temp is None or len(df_temp) == 0:
                    raise ValueError("empty response")
                df_temp['League'] = league_name
                df_temp['Season'] = season
                all_data.append(df_temp)
                loaded[league_name].append(season)
                print(f"  ✅ {season}: {len(df_temp)} matches")
            except Exception as e:
                failed[league_name].append(season)
                print(f"  ⚠️  {season}: Unavailable/failed ({e})")

    if not all_data:
        raise RuntimeError(
            "No season data could be loaded for any league. "
            "Check network access to football-data.co.uk and the season codes."
        )

    df = pd.concat(all_data, ignore_index=True)
    
    # Process
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    required = ['FTHG', 'FTAG', 'FTR', 'HomeTeam', 'AwayTeam']
    df = df.dropna(subset=required)
    
    df['FTHG'] = df['FTHG'].astype(int)
    df['FTAG'] = df['FTAG'].astype(int)
    df['Outcome'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    df['DaysSinceMatch'] = (df['Date'].max() - df['Date']).dt.days
    
    # Preserve raw observed targets BEFORE any feature imputation. These
    # columns are the only legal targets for the dedicated corner/card models.
    # Missing historical values may be imputed for PRE-MATCH FEATURES, but
    # must never become synthetic training targets.
    for _target_col in ['HC', 'AC', 'HY', 'AY', 'HR', 'AR']:
        if _target_col in df.columns:
            df[f'{_target_col}_observed'] = pd.to_numeric(df[_target_col], errors='coerce')
        else:
            df[f'{_target_col}_observed'] = np.nan

    # Ensure shot/corner columns.
    # FIX (leakage): the previous version imputed missing values using
    # df.groupby('League')[col].transform('mean') — a mean computed over the
    # ENTIRE league history, including matches that happen AFTER the match
    # being imputed. That leaks future information into a training row.
    #
    # Time-safe hierarchy used instead (each step uses ONLY matches strictly
    # before the current one — df is already sorted chronologically by Date
    # at this point):
    #   1. This team's own last recorded value for that stat, in that role
    #      (HomeTeam for H* columns, AwayTeam for A* columns)
    #   2. The league's running (expanding) historical mean up to that point
    #   3. The global running (expanding) historical mean up to that point
    #   4. A FIXED neutral fallback, computed ONCE from only the earliest
    #      chronological slice of the dataset (a "burn-in" baseline) —
    #      NOT from the full dataset. Using `frame[col].median()` over the
    #      whole dataset would leak future information into the very
    #      earliest rows, which is exactly the kind of leakage step 1-3
    #      are designed to avoid; the fallback must not be exempt from that
    #      same rule. Only ever hit for the first few rows of the whole
    #      dataset, where no prior history exists yet at all.
    _ROLE_COL = {
        'HS': 'HomeTeam', 'HST': 'HomeTeam', 'HC': 'HomeTeam',
        'AS': 'AwayTeam', 'AST': 'AwayTeam', 'AC': 'AwayTeam',
        'HF': 'HomeTeam', 'AF': 'AwayTeam',
        'HY': 'HomeTeam', 'AY': 'AwayTeam',
        'HR': 'HomeTeam', 'AR': 'AwayTeam',
    }
    _BURN_IN_MIN_ROWS = 20
    _BURN_IN_FRACTION = 0.05

    def _time_safe_impute(frame, col, role_col):
        team_prev = frame.groupby(['League', role_col])[col].transform(lambda s: s.shift(1).ffill())
        league_running = frame.groupby('League')[col].transform(lambda s: s.shift(1).expanding().mean())
        global_running = frame[col].shift(1).expanding().mean()

        burn_in_n = max(_BURN_IN_MIN_ROWS, int(len(frame) * _BURN_IN_FRACTION))
        fixed_fallback = frame[col].iloc[:burn_in_n].median()
        if pd.isna(fixed_fallback):
            fixed_fallback = 0.0

        return (
            frame[col]
            .fillna(team_prev)
            .fillna(league_running)
            .fillna(global_running)
            .fillna(fixed_fallback)
            .fillna(0)
        )

    imputed_pct = {}
    for col in ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df[f'{col}_available'] = df[col].notna().astype(int)
        imputed_pct[col] = float((~df[col].notna()).mean() * 100)

        df[col] = _time_safe_impute(df, col, _ROLE_COL[col])

    print("\n📉 Missing-statistic rates (time-safe imputed, no future leakage):")
    for col, pct in imputed_pct.items():
        print(f"  • {col}: {pct:.1f}% imputed")

    load_report = {
        'seasons_requested': seasons,
        'loaded': loaded,
        'failed': failed,
        'status_by_league': {
            league: ('loaded' if loaded.get(league) else ('failed' if failed.get(league) else 'unavailable'))
            for league in LEAGUES
        },
    }

    print("\n📊 Season load summary:")
    print(f"Configured leagues: {len(LEAGUES)}")
    print(f"Actually loaded leagues: {sum(bool(v) for v in loaded.values())}")
    print(f"Failed/unavailable leagues: {sum(not bool(v) for v in loaded.values())}")
    print(f"Failed/unavailable leagues: {sum(not bool(v) for v in loaded.values())}")
    for league in LEAGUES:
        print(f"  • {league}: loaded {loaded[league]} | failed/unavailable {failed[league]}")
    
    # Cache it (data + metadata, so a later run can validate reuse)
    joblib.dump(df, cache_file, compress=3)
    joblib.dump({'version': CACHE_VERSION, 'seasons_requested': seasons, 'load_report': load_report}, meta_file)
    print(f"\n💾 Cached to {cache_file}")
    
    return df, load_report

def compute_elo_ratings(df, k=20, base_rating=1500):
    """
    Compute ELO ratings with proper time ordering.

    FIX (cross-league contamination): ratings were previously keyed only by
    team name, so a name shared across leagues (or a promoted/relegated team
    that changes competitions) would incorrectly carry its rating over from
    an unrelated league. Ratings are now keyed by (League, Team), so each
    league's Elo pool is fully independent, while still being computed
    chronologically using only information available before each match.
    """
    
    print("\n🏆 Computing ELO ratings...")
    
    df = df.sort_values(['League', 'Date']).reset_index(drop=True)
    ratings = {}  # key: (League, Team)
    home_elo = []
    away_elo = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ELO computation"):
        league = row['League']
        home = row['HomeTeam']
        away = row['AwayTeam']

        hkey = (league, home)
        akey = (league, away)
        
        if hkey not in ratings:
            ratings[hkey] = base_rating
        if akey not in ratings:
            ratings[akey] = base_rating
        
        h_elo = ratings[hkey]
        a_elo = ratings[akey]
        
        home_elo.append(h_elo)
        away_elo.append(a_elo)
        
        expected = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        result = 1.0 if row['FTR'] == 'H' else (0.5 if row['FTR'] == 'D' else 0.0)
        
        ratings[hkey] = h_elo + k * (result - expected)
        ratings[akey] = a_elo + k * ((1 - result) - (1 - expected))
    
    df['ELO_home'] = home_elo
    df['ELO_away'] = away_elo
    df['ELO_diff'] = df['ELO_home'] - df['ELO_away']
    
    n_teams_total = len(set(k[1] for k in ratings))
    print(f"✅ ELO computed for {len(ratings)} (league, team) pairs across {n_teams_total} distinct team names, {df['League'].nunique()} leagues — no cross-league contamination")
    
    return df

# ============================================================================
# STEP 3: FEATURE ENGINEERING WITH CACHING
# ============================================================================

def create_features_with_cache(df, force_refresh=False):
    """Create features with caching to avoid recomputation."""
    
    cache_file = CACHE_DIR / "features.pkl"
    meta_file = CACHE_DIR / "features_meta.pkl"

    # Version/size fingerprint so a stale (e.g. 5-season) feature cache can't
    # silently be reused after the underlying data changed.
    current_fingerprint = {'version': CACHE_VERSION, 'n_rows': len(df), 'seasons': sorted(df['Season'].unique().tolist())}

    if cache_file.exists() and meta_file.exists() and not force_refresh:
        cached_fp = joblib.load(meta_file)
        if cached_fp == current_fingerprint:
            print("\n📦 Loading features from cache (schema & data match)...")
            df, feature_cols = joblib.load(cache_file)
            print(f"✅ Loaded {len(feature_cols)} features from cache")
            return df, feature_cols
        else:
            print("\n♻️  Feature cache is stale (underlying data changed) — recomputing...")
    
    print("\n🔧 Creating features...")
    
    df = df.sort_values(['League', 'Date']).reset_index(drop=True)
    
    # Goals features
    print("  • Goals rolling averages...")
    for window in [5, 10]:
        df[f'HGS_L{window}'] = df.groupby(['League', 'HomeTeam'])['FTHG'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'HGC_L{window}'] = df.groupby(['League', 'HomeTeam'])['FTAG'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'AGS_L{window}'] = df.groupby(['League', 'AwayTeam'])['FTAG'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'AGC_L{window}'] = df.groupby(['League', 'AwayTeam'])['FTHG'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    
    # Shots
    print("  • Shots rolling averages...")
    df['HS_L5'] = df.groupby(['League', 'HomeTeam'])['HS'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['AS_L5'] = df.groupby(['League', 'AwayTeam'])['AS'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['HST_L5'] = df.groupby(['League', 'HomeTeam'])['HST'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['AST_L5'] = df.groupby(['League', 'AwayTeam'])['AST'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    
    # Corners
    print("  • Corners rolling averages...")
    df['HC_L5'] = df.groupby(['League', 'HomeTeam'])['HC'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['AC_L5'] = df.groupby(['League', 'AwayTeam'])['AC'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Shots AGAINST (defense) — a team's shots-conceded rate, role-specific
    # exactly like HGC_L5/AGC_L5 for goals: the home team's shots-conceded
    # figure in a given match IS the away team's HS/AS/AST value in that
    # same row, so this is a same-row lookup, not a separate join.
    print("  • Shot/shots-on-target-against rolling averages...")
    df['HSC_L5'] = df.groupby(['League', 'HomeTeam'])['AS'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['ASC_L5'] = df.groupby(['League', 'AwayTeam'])['HS'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['HSTC_L5'] = df.groupby(['League', 'HomeTeam'])['AST'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['ASTC_L5'] = df.groupby(['League', 'AwayTeam'])['HST'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Shots-on-target percentage (rolling numerator/denominator, both
    # already shift(1)'d above, so the ratio itself is time-safe; guarded
    # against a zero/near-zero denominator rather than dividing directly).
    print("  • Shots-on-target percentage...")
    df['HSTPct_L5'] = np.where(df['HS_L5'] > 0.5, df['HST_L5'] / df['HS_L5'].replace(0, np.nan), np.nan)
    df['ASTPct_L5'] = np.where(df['AS_L5'] > 0.5, df['AST_L5'] / df['AS_L5'].replace(0, np.nan), np.nan)

    # Fouls
    print("  • Fouls rolling averages...")
    df['HF_L5'] = df.groupby(['League', 'HomeTeam'])['HF'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['AF_L5'] = df.groupby(['League', 'AwayTeam'])['AF'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Cards (yellow/red rolling rates; red cards are rare, so this is a
    # rolling RATE over recent matches rather than a raw count — smoother
    # and less prone to a single fluke match dominating the feature)
    print("  • Cards rolling averages (yellow, red)...")
    df['HY_L5'] = df.groupby(['League', 'HomeTeam'])['HY'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['AY_L5'] = df.groupby(['League', 'AwayTeam'])['AY'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['HR_L10'] = df.groupby(['League', 'HomeTeam'])['HR'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    df['AR_L10'] = df.groupby(['League', 'AwayTeam'])['AR'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    # Card points (standard weighting: yellow=1, red=3), rolling. Built via
    # a temporary combined column + the same groupby/shift/rolling pattern
    # as every other feature here (avoids a groupby().apply() re-index).
    df['_HCardWeighted'] = df['HY'] + 3 * df['HR']
    df['_ACardWeighted'] = df['AY'] + 3 * df['AR']
    df['HCardPts_L5'] = df.groupby(['League', 'HomeTeam'])['_HCardWeighted'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['ACardPts_L5'] = df.groupby(['League', 'AwayTeam'])['_ACardWeighted'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df = df.drop(columns=['_HCardWeighted', '_ACardWeighted'])

    # Referee historical stats (HISTORICAL-ONLY: for a match refereed by X,
    # only matches refereed by X strictly BEFORE this match's date are used
    # — same shift(1)+expanding pattern as everything else, just grouped by
    # Referee instead of team. 'Referee' is missing for a large share of
    # rows in this dataset (older seasons / smaller leagues rarely record
    # it) — those rows, and any referee with too few prior matches, fall
    # back to the league-wide rolling average computed the same way.
    print("  • Referee historical averages (fouls, cards)...")
    if 'Referee' not in df.columns:
        df['Referee'] = np.nan
    _MIN_REF_MATCHES = 15
    ref_sort = df.sort_values('Date')
    ref_total_fouls = (ref_sort['HF'] + ref_sort['AF'])
    ref_total_cards = (ref_sort['HY'] + ref_sort['AY'] + 3 * (ref_sort['HR'] + ref_sort['AR']))

    ref_group_n = ref_sort.groupby('Referee').cumcount()  # matches seen so far for this ref (time-ordered)
    ref_sort = ref_sort.assign(_TotalFouls=ref_total_fouls, _TotalCards=ref_total_cards)
    ref_fouls_roll = ref_sort.groupby('Referee')['_TotalFouls'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    ref_cards_roll = ref_sort.groupby('Referee')['_TotalCards'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    league_fouls_roll = ref_sort.groupby('League')['_TotalFouls'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    league_cards_roll = ref_sort.groupby('League')['_TotalCards'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    enough_history = ref_group_n >= _MIN_REF_MATCHES
    ref_fouls_final = ref_fouls_roll.where(enough_history & ref_sort['Referee'].notna(), league_fouls_roll)
    ref_cards_final = ref_cards_roll.where(enough_history & ref_sort['Referee'].notna(), league_cards_roll)

    df['RefFouls_hist'] = ref_fouls_final.reindex(df.index)
    df['RefCards_hist'] = ref_cards_final.reindex(df.index)

    # Data-availability reliability features.
    # FIX: the raw *_available flags (0/1 for whether HS/AST/etc. was
    # actually recorded for a given match) can't be used directly as ML
    # features, because at prediction time for a future fixture we don't yet
    # know whether that match's stats will be recorded — using the raw flag
    # would work fine in training but be unconstructible at prediction time,
    # breaking train/predict feature parity. Instead we roll each team's
    # historical availability rate (same shift(1)+rolling pattern as every
    # other feature here), which IS knowable in advance and lets the model
    # learn "this team/league's rolling stats tend to be reliable vs. often
    # imputed" as a genuine predictive signal.
    print("  • Stats-availability reliability (rolling)...")
    df['HS_avail_L5'] = df.groupby(['League', 'HomeTeam'])['HS_available'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['HST_avail_L5'] = df.groupby(['League', 'HomeTeam'])['HST_available'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['HC_avail_L5'] = df.groupby(['League', 'HomeTeam'])['HC_available'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['AS_avail_L5'] = df.groupby(['League', 'AwayTeam'])['AS_available'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['AST_avail_L5'] = df.groupby(['League', 'AwayTeam'])['AST_available'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['AC_avail_L5'] = df.groupby(['League', 'AwayTeam'])['AC_available'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    
    # Form (points-per-game over last 5, NOT a raw 0-15 sum — a sum
    # conflates "played more games" with "played well", and isn't
    # comparable across teams with different numbers of recent fixtures.
    # Named *PPG explicitly so the scale is unambiguous.)
    print("  • Form features (PPG)...")
    df['HP'] = (df['FTR'] == 'H') * 3 + (df['FTR'] == 'D') * 1
    df['AP'] = (df['FTR'] == 'A') * 3 + (df['FTR'] == 'D') * 1
    
    df['HFormPPG'] = df.groupby(['League', 'HomeTeam'])['HP'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['AFormPPG'] = df.groupby(['League', 'AwayTeam'])['AP'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    
    # Rest / schedule congestion features
    # IMPORTANT: these are strictly pre-match. For each team, the previous
    # match date is shifted by one before the difference is calculated, so
    # the current match date/result can never contribute to its own rest value.
    print("  • Rest / schedule congestion features...")
    home_prev_date = df.groupby(['League', 'HomeTeam'])['Date'].transform(
        lambda x: x.shift(1)
    )
    away_prev_date = df.groupby(['League', 'AwayTeam'])['Date'].transform(
        lambda x: x.shift(1)
    )

    df['HomeRest'] = (df['Date'] - home_prev_date).dt.days
    df['AwayRest'] = (df['Date'] - away_prev_date).dt.days

    # Burn-in / promoted-team fallback: if a team has no previous same-role
    # fixture, use the league-level historical median rest observed in prior
    # rows, then a fixed neutral fallback. This keeps the feature constructible
    # without borrowing a future match or an away-role row.
    def _safe_rest_fallback(series):
        prior = series.shift(1)
        running_median = prior.expanding().median()
        burn_in_n = max(20, int(len(series) * 0.05))
        fixed = series.iloc[:burn_in_n].median()
        if pd.isna(fixed):
            fixed = 7.0
        return running_median.fillna(fixed)

    home_rest_fallback = df.groupby('League')['HomeRest'].transform(_safe_rest_fallback)
    away_rest_fallback = df.groupby('League')['AwayRest'].transform(_safe_rest_fallback)
    df['HomeRest'] = df['HomeRest'].fillna(home_rest_fallback).fillna(7.0).clip(lower=0)
    df['AwayRest'] = df['AwayRest'].fillna(away_rest_fallback).fillna(7.0).clip(lower=0)
    df['RestDiff'] = df['HomeRest'] - df['AwayRest']

    # Cap extreme gaps so one postponed/rescheduled match does not dominate
    # the model. The underlying values remain available in processed_data.pkl.
    df['HomeRest'] = df['HomeRest'].clip(0, 30)
    df['AwayRest'] = df['AwayRest'].clip(0, 30)
    df['RestDiff'] = df['HomeRest'] - df['AwayRest']

    # Matchup differentials
    print("  • Matchup differentials...")
    df['AttackDiff'] = df['HGS_L5'] - df['AGC_L5']
    df['DefenseDiff'] = df['HGC_L5'] - df['AGS_L5']
    df['ShotDiff'] = df['HS_L5'] - df['AS_L5']
    df['ShotTargetDiff'] = df['HST_L5'] - df['AST_L5']
    df['CornerDiff'] = df['HC_L5'] - df['AC_L5']
    df['ShotConcededDiff'] = df['HSC_L5'] - df['ASC_L5']
    df['FoulDiff'] = df['HF_L5'] - df['AF_L5']
    df['CardPtsDiff'] = df['HCardPts_L5'] - df['ACardPts_L5']
    
    # League dummies
    print("  • League dummies...")
    league_dummies = pd.get_dummies(df['League'], prefix='Lg')
    df = pd.concat([df, league_dummies], axis=1)
    
    # Explicit feature allowlist.
    # FIX: the previous approach selected feature_cols by substring pattern
    # ('_L' in c, 'Form' in c, etc.), which meant any future column that
    # happened to match one of those substrings would silently become a
    # model input with no review. This is now an explicit, named list —
    # adding a new engineered column requires deliberately adding it here.
    _ROLLING_WINDOWS_GOALS = [5, 10]
    feature_cols = []

    for window in _ROLLING_WINDOWS_GOALS:
        feature_cols += [f'HGS_L{window}', f'HGC_L{window}', f'AGS_L{window}', f'AGC_L{window}']

    feature_cols += ['HS_L5', 'AS_L5', 'HST_L5', 'AST_L5', 'HC_L5', 'AC_L5']

    feature_cols += [
        'HS_avail_L5', 'AS_avail_L5',
        'HST_avail_L5', 'AST_avail_L5',
        'HC_avail_L5', 'AC_avail_L5',
    ]

    feature_cols += ['HFormPPG', 'AFormPPG']

    # SHOT FEATURES (against/defense + on-target %)
    feature_cols += ['HSC_L5', 'ASC_L5', 'HSTC_L5', 'ASTC_L5', 'HSTPct_L5', 'ASTPct_L5']

    # FOUL FEATURES
    feature_cols += ['HF_L5', 'AF_L5']

    # CARD FEATURES
    feature_cols += ['HY_L5', 'AY_L5', 'HR_L10', 'AR_L10', 'HCardPts_L5', 'ACardPts_L5']

    # REFEREE FEATURES (historical-only; league fallback baked in upstream)
    feature_cols += ['RefFouls_hist', 'RefCards_hist']

    # REST / SCHEDULE FEATURES
    feature_cols += ['HomeRest', 'AwayRest', 'RestDiff']

    # MATCHUP DIFFERENTIAL FEATURES
    feature_cols += [
        'AttackDiff', 'DefenseDiff', 'ShotDiff', 'ShotTargetDiff', 'CornerDiff',
        'ShotConcededDiff', 'FoulDiff', 'CardPtsDiff',
    ]

    # ELO FEATURES
    feature_cols += ['ELO_home', 'ELO_away', 'ELO_diff']

    # OFFSIDE FEATURES: not added. This dataset (football-data.co.uk, the
    # only source `load_data_with_cache` pulls from) does not include HO/AO
    # columns for any of the 5 supported leagues/seasons currently loaded —
    # confirmed by inspecting processed_data.pkl. Per rule 12 (do not
    # fabricate data), no offside feature is created. If a future data
    # source adds HO/AO, mirror the fouls-feature pattern above.

    # League dummies are the one dynamic part — the set of leagues is a
    # deliberate runtime choice (LEAGUES dict above), not an accidental
    # pattern match, so it's fine for this part to stay derived from data.
    feature_cols += sorted(c for c in df.columns if c.startswith('Lg_'))

    missing_expected = [c for c in feature_cols if c not in df.columns]
    if missing_expected:
        raise RuntimeError(
            f"Expected feature columns were not created by the feature "
            f"engineering steps above: {missing_expected}. This means a "
            f"column was renamed/removed without updating the allowlist."
        )
    
    df = df.dropna(subset=feature_cols)
    
    # Cache it (+ fingerprint so it auto-invalidates on future data/schema changes)
    joblib.dump((df, feature_cols), cache_file, compress=3)
    joblib.dump(current_fingerprint, meta_file)
    print(f"\n💾 Cached features to {cache_file}")
    
    print(f"\n✅ Created {len(feature_cols)} features (includes {sum('avail_L5' in c for c in feature_cols)} availability-reliability features)")
    print(f"📊 {len(df):,} matches after engineering")
    
    return df, feature_cols

# ============================================================================
# STEP 4: TRAIN ML MODEL
# ============================================================================

def train_ml_model(df, feature_cols):
    """Train XGBoost with calibration."""
    
    print("\n🤖 Training ML model...")
    
    X = df[feature_cols].fillna(0).values
    y = df['Outcome'].values
    
    # CRITICAL VALIDATION
    assert X.shape[1] == len(feature_cols), f"Feature mismatch! Expected {len(feature_cols)}, got {X.shape[1]}"
    print(f"✅ Feature validation passed: {X.shape[1]} features")
    
    base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    tscv = TimeSeriesSplit(n_splits=3)

    # Walk-forward (out-of-fold) evaluation. This is now the PRIMARY,
    # headline performance metric — not the in-sample number computed later.
    #
    # FIX (de-nested calibration): the previous version ran
    # CalibratedClassifierCV(base_model, cv=3) — a second, independent CV —
    # INSIDE each outer walk-forward fold. That's cross-validation nested
    # inside cross-validation: not a leakage bug, but it muddies what the
    # OOF number represents and multiplies training cost for no benefit.
    # Replaced with a single chronological split per fold: fit the base
    # model on the first ~80% of that fold's training block, calibrate on
    # the remaining ~20% (a plain holdout, cv='prefit'), then score the
    # fold's test block. One level of validation, cleanly interpretable.
    #
    # NOTE: sklearn's cross_val_predict requires every sample to appear in
    # exactly one test fold, which TimeSeriesSplit does not guarantee (the
    # first training block is never used as a test fold, by design — there's
    # no "past" to train on for it). So we manually walk the folds and only
    # score the rows that were genuinely held out at some point.
    from sklearn.base import clone

    def _fit_calibrated_on_prefit(fitted_estimator, X_calib, y_calib):
        """
        Wrap an ALREADY-FITTED estimator in CalibratedClassifierCV for a
        holdout-based calibration fit, compatible across sklearn versions:
        sklearn >= 1.6 removed the cv='prefit' string in favor of wrapping
        the fitted estimator in sklearn.frozen.FrozenEstimator; older
        versions don't have FrozenEstimator at all. Try the modern path
        first, fall back to the legacy string for older installs.
        """
        try:
            from sklearn.frozen import FrozenEstimator
            calib = CalibratedClassifierCV(FrozenEstimator(fitted_estimator), method='isotonic')
        except ImportError:
            calib = CalibratedClassifierCV(fitted_estimator, method='isotonic', cv='prefit')
        calib.fit(X_calib, y_calib)
        return calib

    print("  • Computing out-of-fold performance (walk-forward, single-level calibration)...")
    oof_proba = np.full((len(y), 3), np.nan)
    scored_mask = np.zeros(len(y), dtype=bool)

    for train_idx, test_idx in tscv.split(X):
        split_point = int(len(train_idx) * 0.8)
        fit_idx, calib_idx = train_idx[:split_point], train_idx[split_point:]
        if len(calib_idx) < 10:
            # Fold's training block too small to hold out a calibration
            # slice meaningfully — fit and calibrate on the same data for
            # this fold only (degrades gracefully; only affects early folds).
            fit_idx, calib_idx = train_idx, train_idx

        fold_base = clone(base_model)
        fold_base.fit(X[fit_idx], y[fit_idx])
        fold_model = _fit_calibrated_on_prefit(fold_base, X[calib_idx], y[calib_idx])

        oof_proba[test_idx] = fold_model.predict_proba(X[test_idx])
        scored_mask[test_idx] = True

    if scored_mask.sum() > 0:
        y_scored = y[scored_mask]
        p_scored = oof_proba[scored_mask]

        oof_ll = float(log_loss(y_scored, p_scored))
        oof_acc = float((p_scored.argmax(axis=1) == y_scored).mean())
        y_onehot = np.eye(3)[y_scored]
        oof_brier = float(np.mean(np.sum((p_scored - y_onehot) ** 2, axis=1)))

        print(f"  • Walk-forward log-loss:  {oof_ll:.4f}  ({scored_mask.sum()}/{len(y)} rows)  <-- PRIMARY METRIC")
        print(f"  • Walk-forward accuracy:  {oof_acc:.4f}")
        print(f"  • Walk-forward Brier:     {oof_brier:.4f}")
    else:
        oof_ll = oof_acc = oof_brier = None
        print("  • Not enough data for walk-forward validation — skipping OOF metrics")

    final_model = CalibratedClassifierCV(base_model, method='isotonic', cv=tscv)
    final_model.fit(X, y)
    
    y_pred = final_model.predict_proba(X)
    ll = log_loss(y, y_pred)
    
    print(f"✅ ML Model trained")
    print(f"  • In-sample log-loss:     {ll:.4f}  (diagnostic only — DO NOT use to judge live performance)")
    print(f"  • Walk-forward log-loss:  {f'{oof_ll:.4f}' if oof_ll is not None else 'n/a'}  (this is the number that matters)")

    # Stash for reporting elsewhere (e.g. app "Guide"/"Statistics" tab)
    final_model.oof_log_loss_ = oof_ll
    final_model.oof_accuracy_ = oof_acc
    final_model.oof_brier_ = oof_brier
    final_model.in_sample_log_loss_ = ll

    return final_model

# ============================================================================
# STEP 5: TRAIN DIXON-COLES (IMPORT FROM models.py!)
# ============================================================================

def train_dixon_coles(df):
    """Train Dixon-Coles per league using imported class.

    A non-converged optimizer result must never be silently reported as a
    production-ready model. For each league:
      1. Run the normal fit (DixonColesTimeDecay.fit() already makes its own
         two-attempt internal optimization pass — see models.py).
      2. Check res.success (exposed as `converged_`) explicitly.
      3. If it did not converge, retry using the same model implementation
         with a different `xi` (time-decay) option — a legitimate, existing
         constructor option, not a different algorithm — since retrying
         with identical inputs to a deterministic optimizer reproduces the
         same non-converged result.
      4. Check convergence again.
      5. If still non-converged, deploy the model anyway (so
         `ensemble_prediction()` in models.py — which already checks
         `converged_` at predict time — can safely fall back per-fixture)
         but record it as non-converged in `dc_diagnostics` rather than
         reporting it as a normal trained model.

    Returns:
        dc_models: {league: DixonColesTimeDecay}  (includes non-converged
            leagues, so callers relying on `len(dc_models)` for coverage
            still see every configured league that had data)
        dc_diagnostics: {league: 'converged' | 'non_converged'}
    """

    print("\n🎯 Training Dixon-Coles models...")

    dc_models = {}
    dc_diagnostics = {}

    RETRY_XI = 0.008  # distinct decay rate for the retry-only pass

    for league in df['League'].unique():
        print(f"\n  Training {league}...")

        # Use imported class - NO DUPLICATION
        dc_model = DixonColesTimeDecay(xi=0.002)
        dc_model.fit(df, league=league)

        if not dc_model.converged_:
            print(f"    ⚠️ Dixon-Coles did not converge for {league} "
                  f"({dc_model.optimizer_message_ or 'no message'}); retrying "
                  f"with xi={RETRY_XI}")
            retry_model = DixonColesTimeDecay(xi=RETRY_XI)
            retry_model.fit(df, league=league)
            if retry_model.converged_:
                dc_model = retry_model
                print(f"    ✅ Converged on retry")
            else:
                print(f"    ❌ Still did not converge for {league} after retry "
                      f"({retry_model.optimizer_message_ or 'no message'}); "
                      f"deploying as non-converged (predict-time code falls "
                      f"back to a safe per-fixture average — see "
                      f"ensemble_prediction() in models.py)")
                dc_model = retry_model

        dc_diagnostics[league] = 'converged' if dc_model.converged_ else 'non_converged'
        dc_models[league] = dc_model

        print(f"    ✅ Teams: {len(dc_model.teams)}")
        print(f"    ✅ Home adv: {dc_model.home_adv:.3f}")
        print(f"    ✅ Converged: {dc_model.converged_}")

    n_converged = sum(1 for status in dc_diagnostics.values() if status == 'converged')
    n_total = len(dc_models)
    print(f"\n✅ Dixon-Coles: {n_converged}/{n_total} leagues converged "
          f"({n_total - n_converged} deployed non-converged, fallback-safe at predict time)")
    if n_total - n_converged:
        non_converged = sorted(lg for lg, s in dc_diagnostics.items() if s == 'non_converged')
        print(f"  ⚠️ Non-converged leagues: {non_converged}")

    return dc_models, dc_diagnostics

# ============================================================================
# STEP 5b: TRAIN DEDICATED CORNER MODELS (Task 2)
# ============================================================================

def train_corner_models(df):
    """
    Fit a dedicated per-league corner-strength model (Negative-Binomial team
    strength — chosen by default; see rationale below) and report the
    walk-forward comparison against Poisson-strength and GBR alternatives
    that justified that choice.

    Returns:
        corner_models: {league: CornerStrengthModel}
        corner_feature_cols: list[str] (used by predict.py for the
            GBR-style features shown in diagnostics; the strength model
            itself does not consume these — it fits its own attack/defense
            parameters directly from HC/AC, mirroring Dixon-Coles)
        corner_validation: {league: walk_forward_validate(...) results}
    """

    print("\n⚽ Training dedicated corner models...")

    df_c, corner_feature_cols = build_corner_features(df)

    corner_models = {}
    corner_validation = {}

    for league in df_c['League'].unique():
        print(f"\n  Corner model — {league}...")

        val = walk_forward_validate(df_c, corner_feature_cols, league)
        corner_validation[league] = val
        recommended = val.get('_recommended')
        print(f"    Walk-forward recommendation: {recommended}")
        for name, metrics in val.items():
            if name == '_recommended' or 'error' in metrics:
                continue
            print(f"      {name:18s} MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
                  f"PoissonDev={metrics['poisson_deviance']:.3f}  (n={metrics['n']})")

        # Fit the final model on ALL available data for this league using
        # whichever family the walk-forward validation preferred between
        # the two strength-model variants; if GBR was recommended we still
        # deploy the negbinom strength model (it consistently scored
        # within noise of GBR across leagues in validation and — unlike
        # the GBR features — degrades gracefully to a league-average
        # fallback for teams with almost no history, matching the
        # promoted-team handling already required for the 1X2 model).
        # This choice, and the numbers that justify it, are reported above
        # rather than silently swapped in.
        if recommended == 'gbr_poisson':
            data = df_c[df_c['League'] == league].sort_values('Date').dropna(subset=corner_feature_cols + ['HC_observed', 'AC_observed'] if 'HC_observed' in df_c.columns else corner_feature_cols + ['HC', 'AC'])
            hc_t = 'HC_observed' if 'HC_observed' in data.columns else 'HC'
            ac_t = 'AC_observed' if 'AC_observed' in data.columns else 'AC'
            X_all = data[corner_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            gbr = CornerGBRModel().fit(X_all, data[hc_t].values, data[ac_t].values, corner_feature_cols, history_df=df_c)
            gbr.model_family = 'gbr_poisson'
            corner_models[league] = gbr
            print(f"    ✅ Deployed: GBR Poisson model (walk-forward selected)")
        else:
            dist = 'negbinom' if recommended != 'poisson_strength' else 'poisson'
            model = CornerStrengthModel(distribution=dist).fit(df_c, league)
            if not model.converged_:
                # BUG FIX: the previous fallback always retried with
                # distribution='poisson', even when `dist` was ALREADY
                # 'poisson' — retrying the exact same distribution against
                # the same data with the same deterministic L-BFGS-B
                # optimizer reproduces the identical non-converged result
                # every time, so leagues where walk-forward recommended
                # poisson_strength could never actually benefit from a
                # fallback attempt. The correct fallback is the OTHER
                # distribution family (poisson<->negbinom) — a genuinely
                # different likelihood surface, not a no-op retry. This is
                # purely an optimization/robustness fix; it does not change
                # which family is preferred when the first attempt DOES
                # converge, and does not touch either model's mathematics.
                other_dist = 'poisson' if dist == 'negbinom' else 'negbinom'
                print(f"    ⚠️ {dist} model did not converge for {league}; "
                      f"trying {other_dist} fallback")
                fallback = CornerStrengthModel(distribution=other_dist).fit(df_c, league)
                if not fallback.converged_:
                    print(f"    ❌ No converged corner strength model for {league} "
                          f"(tried {dist} and {other_dist}); skipping deployment")
                    corner_validation[league]['deployment_status'] = 'failed_nonconvergence'
                    continue
                model = fallback
                dist = other_dist
            model.model_family = f'{dist}_strength'
            corner_models[league] = model
            print(f"    ✅ Deployed: {dist} strength model (home_adv={model.home_adv:.3f}, alpha={model.alpha:.3f}, converged={model.converged_})")

    print(f"\n✅ Trained {len(corner_models)} corner models")

    return corner_models, corner_feature_cols, corner_validation


# ============================================================================
# STEP 6: CREATE TEAM MAPPINGS
# ============================================================================

def create_team_mappings(df):
    """Create comprehensive team name mappings."""
    
    print("\n🔤 Creating team mappings...")
    
    all_teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
    
    team_mapping = {}
    for team in all_teams:
        team_mapping[team] = team
        team_mapping[team.lower()] = team
    
    # Manual variations — covers common user-input variants and full club names
    manual = {
        # ── Premier League ──────────────────────────────────────────────────
        'brighton': 'Brighton',
        'brighton & hove albion': 'Brighton',
        'brighton and hove albion': 'Brighton',
        'newcastle': 'Newcastle',
        'newcastle united': 'Newcastle',
        'west ham': 'West Ham',
        'west ham united': 'West Ham',
        'man united': 'Man United',
        'manchester united': 'Man United',
        'man utd': 'Man United',
        'man city': 'Man City',
        'manchester city': 'Man City',
        'nottm forest': 'Nott\'m Forest',
        'nottingham forest': 'Nott\'m Forest',
        'wolverhampton': 'Wolves',
        'wolverhampton wanderers': 'Wolves',
        'tottenham hotspur': 'Tottenham',
        'spurs': 'Tottenham',
        'leicester city': 'Leicester',
        'leeds united': 'Leeds',
        'aston villa fc': 'Aston Villa',
        'sheffield united': 'Sheffield United',
        'sheffield utd': 'Sheffield United',

        # ── La Liga ──────────────────────────────────────────────────────────
        'atletico madrid': 'Ath Madrid',
        'atlético madrid': 'Ath Madrid',
        'atletico de madrid': 'Ath Madrid',
        'real atletico': 'Ath Madrid',
        'athletic bilbao': 'Ath Bilbao',
        'athletic club': 'Ath Bilbao',
        'deportivo alaves': 'Alaves',
        'deportivo alavés': 'Alaves',
        'alaves': 'Alaves',
        'real betis': 'Betis',
        'real betis balompie': 'Betis',
        'real sociedad': 'Real Sociedad',
        'ca osasuna': 'Osasuna',
        'rcd espanyol': 'Espanol',
        'espanyol': 'Espanol',
        'rcd mallorca': 'Mallorca',
        'ud las palmas': 'Las Palmas',
        'girona fc': 'Girona',
        'getafe cf': 'Getafe',
        'villarreal cf': 'Villarreal',
        'sevilla fc': 'Sevilla',
        'valencia cf': 'Valencia',

        # ── Bundesliga ───────────────────────────────────────────────────────
        'borussia dortmund': 'Dortmund',
        'bvb': 'Dortmund',
        'bayer leverkusen': 'Leverkusen',
        'bayer 04 leverkusen': 'Leverkusen',
        'eintracht frankfurt': 'Ein Frankfurt',
        'rb leipzig': 'RB Leipzig',
        'rasenballsport leipzig': 'RB Leipzig',
        'borussia monchengladbach': 'M\'gladbach',
        'borussia mönchengladbach': 'M\'gladbach',
        'monchengladbach': 'M\'gladbach',
        'vfl wolfsburg': 'Wolfsburg',
        'fc st. pauli': 'St Pauli',
        'st. pauli': 'St Pauli',
        'fc augsburg': 'Augsburg',
        'sc freiburg': 'Freiburg',
        'tsg hoffenheim': 'Hoffenheim',
        'tsg 1899 hoffenheim': 'Hoffenheim',
        'fc union berlin': 'Union Berlin',
        '1. fc union berlin': 'Union Berlin',
        'vfl bochum': 'Bochum',
        'sv werder bremen': 'Werder Bremen',
        'hamburger sv': 'Hamburg',
        'fortuna dusseldorf': 'Fortuna Dusseldorf',
        'fc koln': 'FC Koln',
        'cologne': 'FC Koln',
        'koln': 'FC Koln',
        '1. fc heidenheim': 'Heidenheim',
        'heidenheim': 'Heidenheim',
        'holstein kiel': 'Kiel',
        'mainz': 'Mainz',
        'fsv mainz 05': 'Mainz',
        '1. fsv mainz 05': 'Mainz',
        'bayer leverkusen': 'Leverkusen',

        # ── Serie A ──────────────────────────────────────────────────────────
        'cagliari calcio': 'Cagliari',
        'hellas verona': 'Verona',
        'verona': 'Verona',
        'udinese calcio': 'Udinese',
        'ac milan': 'Milan',
        'as roma': 'Roma',
        'ss lazio': 'Lazio',
        'juventus fc': 'Juventus',
        'fc internazionale': 'Inter',
        'inter milan': 'Inter',
        'internazionale': 'Inter',
        'inter': 'Inter',
        'atalanta bc': 'Atalanta',
        'ssc napoli': 'Napoli',
        'acf fiorentina': 'Fiorentina',
        'fiorentina': 'Fiorentina',
        'torino fc': 'Torino',
        'bologna fc': 'Bologna',
        'us sassuolo': 'Sassuolo',
        'genoa cfc': 'Genoa',
        'us lecce': 'Lecce',
        'empoli fc': 'Empoli',
        'us salernitana': 'Salernitana',
        'ac monza': 'Monza',
        'frosinone calcio': 'Frosinone',
        'us cremonese': 'Cremonese',

        # ── Ligue 1 ──────────────────────────────────────────────────────────
        'angers sco': 'Angers',
        'olympique marseille': 'Marseille',
        'om': 'Marseille',
        'olympique lyonnais': 'Lyon',
        'ol': 'Lyon',
        'paris saint-germain': 'Paris SG',
        'paris saint germain': 'Paris SG',
        'psg': 'Paris SG',
        'stade rennais': 'Rennes',
        'stade brestois': 'Brest',
        'brest': 'Brest',
        'rc lens': 'Lens',
        'losc lille': 'Lille',
        'lille': 'Lille',
        'ogc nice': 'Nice',
        'as monaco': 'Monaco',
        'monaco': 'Monaco',
        'fc nantes': 'Nantes',
        'toulouse fc': 'Toulouse',
        'montpellier hsc': 'Montpellier',
        'rc strasbourg': 'Strasbourg',
        'havre ac': 'Le Havre',
        'le havre': 'Le Havre',
        'clermont foot': 'Clermont',
        'fc metz': 'Metz',
        'fc lorient': 'Lorient',
        'auxerre': 'Auxerre',
        'aj auxerre': 'Auxerre',
    }
    
    team_mapping.update(manual)
    
    print(f"✅ Mappings for {len(all_teams)} teams")
    
    return team_mapping, all_teams

# ============================================================================
# STEP 6b: CURRENT-SEASON TEAM UNIVERSE (distinct from historical all_teams)
# ============================================================================

def get_current_season_teams(df, load_report, min_matches_for_current=1):
    """Build current team lists from the newest actually loaded season.

    The training dataset is authoritative for historical membership. At the
    beginning of a season the roster may be incomplete because only a few
    matches have finished; the live fixture loader supplies exact same-league
    fixture candidates at prediction time, so no fuzzy or cross-league team
    substitution is required.
    """
    current_teams_by_league = {}
    latest_season_by_league = {}
    partial_season_by_league = {}
    for league in LEAGUES:
        loaded_seasons = load_report.get('loaded', {}).get(league, [])
        ldf = df[df['League'] == league]
        if not loaded_seasons or ldf.empty:
            current_teams_by_league[league] = []
            latest_season_by_league[league] = None
            partial_season_by_league[league] = False
            continue
        # `loaded_seasons` is a source-status list, not a guarantee that the
        # normalized dataframe contains rows for that exact season label.
        # This matters when a source publishes no rows for a requested season
        # (for example Bundesliga can lag one season behind other leagues).
        # Choose the newest season that actually has usable rows.
        season_sizes = (
            ldf.groupby('Season', dropna=True)
               .size()
               .sort_index()
        )
        usable_seasons = [season for season in loaded_seasons
                          if season in season_sizes.index and int(season_sizes.loc[season]) >= min_matches_for_current]
        if not usable_seasons:
            current_teams_by_league[league] = []
            latest_season_by_league[league] = None
            partial_season_by_league[league] = False
            print(f"  ⚠️ {league}: no usable loaded season found for current-team universe")
            continue

        chosen = usable_seasons[-1]
        subset = ldf[ldf['Season'] == chosen]
        teams = sorted(set(subset['HomeTeam'].dropna()) | set(subset['AwayTeam'].dropna()))
        current_teams_by_league[league] = teams
        latest_season_by_league[league] = chosen
        partial_season_by_league[league] = len(subset) < 30
        if partial_season_by_league[league]:
            print(f"  ℹ️ {league}: current season {chosen} is partial ({len(subset)} matches); live fixture candidates will complete the roster at prediction time")
    return current_teams_by_league, latest_season_by_league

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(force_refresh=False, n_seasons=10):
    """Main training pipeline with proper caching. Production default is 10 seasons."""
    if n_seasons != 10:
        print(f"⚠️  Non-production season count requested: {n_seasons}. Production default remains 10.")
    
    # Load data (now returns a load_report: which seasons succeeded/failed per league)
    df, load_report = load_data_with_cache(force_refresh=force_refresh, n_seasons=n_seasons)
    
    # ELO (now scoped per (League, Team) — no cross-league contamination)
    df = compute_elo_ratings(df)
    
    # Features (now includes time-safe imputation + availability-reliability features)
    df, feature_cols = create_features_with_cache(df, force_refresh=force_refresh)
    
    # Train ML (now reports genuine out-of-fold log-loss)
    final_model = train_ml_model(df, feature_cols)
    
    # Train DC
    dc_models, dc_diagnostics = train_dixon_coles(df)

    # Train dedicated corner models (Task 2)
    corner_models, corner_feature_cols, corner_validation = train_corner_models(df)

    # Train dedicated card models (Task 3).
    card_models, card_feature_cols, card_validation = train_card_models(df)

    # Team mappings — historical universe (everything the trained model knows about)
    team_mapping, all_teams = create_team_mappings(df)

    # Current-season team universe — distinct from historical `all_teams`
    current_teams_by_league, latest_season_by_league = get_current_season_teams(df, load_report)
    
    # Save models
    print("\n💾 Saving models...")
    
    joblib.dump(final_model, 'final_model.pkl', compress=3)
    print("  ✅ final_model.pkl")
    
    joblib.dump(dc_models, 'dc_models.pkl', compress=3)
    print("  ✅ dc_models.pkl")

    joblib.dump({'models': corner_models, 'schema_version': CORNER_SCHEMA_VERSION,
                 'validation': corner_validation}, 'corner_model.pkl', compress=3)
    print("  ✅ corner_model.pkl  (new artifact)")

    joblib.dump(corner_feature_cols, 'corner_feature_cols.pkl')
    print("  ✅ corner_feature_cols.pkl  (new artifact)")

    joblib.dump({'models': card_models, 'schema_version': CARD_SCHEMA_VERSION,
                 'validation': card_validation}, 'card_model.pkl', compress=3)
    print("  ✅ card_model.pkl  (new artifact)")

    joblib.dump(card_feature_cols, 'card_feature_cols.pkl')
    print("  ✅ card_feature_cols.pkl  (new artifact)")

    joblib.dump(LEAGUE_CONFIG, 'league_config.pkl', compress=3)
    print("  ✅ league_config.pkl")

    joblib.dump(feature_cols, 'feature_cols.pkl')
    print("  ✅ feature_cols.pkl")
    
    joblib.dump(df, 'processed_data.pkl', compress=3)
    print("  ✅ processed_data.pkl")
    
    joblib.dump(team_mapping, 'team_mapping.pkl')
    print("  ✅ team_mapping.pkl")
    
    # Save all teams for validation
    joblib.dump(all_teams, 'all_teams.pkl')
    print("  ✅ all_teams.pkl")

    # Save current-season team metadata (new artifact)
    current_teams_meta = {
        'current_teams_by_league': current_teams_by_league,
        'latest_season_by_league': latest_season_by_league,
        'seasons_requested': load_report['seasons_requested'],
        'seasons_loaded': load_report['loaded'],
        'seasons_failed': load_report['failed'],
    }
    joblib.dump(current_teams_meta, 'current_teams.pkl')
    print("  ✅ current_teams.pkl")
    
    # Summary
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)

    oof_ll = getattr(final_model, 'oof_log_loss_', None)
    in_sample_ll = log_loss(df['Outcome'], final_model.predict_proba(df[feature_cols].fillna(0)))

    print(f"""
📊 SUMMARY:
  • Seasons requested: {n_seasons} ({load_report['seasons_requested'][0]} .. {load_report['seasons_requested'][-1]})
  • Matches (final training set): {len(df):,}
  • Historical teams (all_teams): {len(all_teams)}
  • Features: {len(feature_cols)}
  • Leagues: {len(dc_models)}
  • In-sample log-loss: {in_sample_ll:.4f}  (optimistic, do not trust for live performance)
  • Out-of-fold log-loss: {f'{oof_ll:.4f}' if oof_ll is not None else 'n/a'}  (realistic estimate)

📅 SEASON LOAD REPORT (per league):""")
    for league in load_report['loaded']:
        loaded = load_report['loaded'][league]
        failed = load_report['failed'][league]
        latest = latest_season_by_league.get(league)
        n_current_teams = len(current_teams_by_league.get(league, []))
        print(f"  • {league}: {len(loaded)}/{n_seasons} seasons loaded, {len(failed)} unavailable "
              f"({failed if failed else 'none'}) | current season: {latest} | current teams: {n_current_teams}")

    n_dc_converged = sum(1 for s in dc_diagnostics.values() if s == 'converged')
    n_dc_non_converged = len(dc_models) - n_dc_converged

    print("\n📐 ARTIFACT LEAGUE COVERAGE:")
    print(f"  • Configured leagues: {len(LEAGUES)}")
    print(f"  • Historical-data leagues: {df['League'].nunique()}")
    print(f"  • ML model leagues: {df['League'].nunique()}")
    print(f"  • Dixon-Coles leagues (deployed): {len(dc_models)}")
    print(f"  • Dixon-Coles leagues (converged): {n_dc_converged}")
    print(f"  • Dixon-Coles leagues (non-converged, fallback-safe): {n_dc_non_converged}")
    print(f"  • Corner model leagues: {len(corner_models)}")
    print(f"  • Card model leagues: {len(card_models)}")
    missing_dc = sorted(set(LEAGUES) - set(dc_models))
    non_converged_dc = sorted(lg for lg, s in dc_diagnostics.items() if s == 'non_converged')
    missing_corner = sorted(set(LEAGUES) - set(corner_models))
    missing_card = sorted(set(LEAGUES) - set(card_models))
    if missing_dc: print(f"  ⚠️ Missing DC entirely: {missing_dc}")
    if non_converged_dc: print(f"  ⚠️ DC deployed but non-converged: {non_converged_dc}")
    if missing_corner: print(f"  ⚠️ Missing corners: {missing_corner}")
    if missing_card: print(f"  ⚠️ Missing cards: {missing_card}")

    # ------------------------------------------------------------------
    # Training manifest — a single source of truth for what this run
    # actually produced. app.py should read this instead of re-deriving
    # coverage numbers from raw artifact contents on every rerun, and it
    # gives a durable, timestamped record of convergence/retry/fallback
    # decisions instead of only ever printing them to a training log that
    # gets discarded.
    # ------------------------------------------------------------------
    import json

    n_card_converged = sum(
        1 for v in card_validation.values()
        if v.get('deployment_status') not in ('failed_nonconvergence', 'missing')
    ) if isinstance(card_validation, dict) else len(card_models)

    manifest = {
        'training_timestamp': datetime.now().isoformat(),
        'dataset': {
            'match_count': int(len(df)),
            'date_range': [
                str(df['Date'].min().date()), str(df['Date'].max().date())
            ] if 'Date' in df.columns else None,
            'seasons_requested': load_report['seasons_requested'],
        },
        'configured_leagues': len(LEAGUES),
        'historical_data_leagues': int(df['League'].nunique()),
        'ml_models': {
            'loaded': int(df['League'].nunique()),
            'out_of_fold_log_loss': oof_ll,
        },
        'dixon_coles': {
            'total_deployed': len(dc_models),
            'converged': n_dc_converged,
            'non_converged': non_converged_dc,
            'missing': missing_dc,
        },
        'corners': {
            'loaded': len(corner_models),
            'missing': missing_corner,
            'schema_version': CORNER_SCHEMA_VERSION,
            'deployment_status_by_league': {
                lg: v.get('deployment_status', 'deployed')
                for lg, v in (corner_validation or {}).items()
            },
        },
        'cards': {
            'loaded': len(card_models),
            'converged_or_deployed': n_card_converged,
            'missing': missing_card,
            'schema_version': CARD_SCHEMA_VERSION,
            'deployment_status_by_league': {
                lg: v.get('deployment_status', 'deployed')
                for lg, v in (card_validation or {}).items()
            } if isinstance(card_validation, dict) else {},
        },
        'current_teams': {
            'season_used_by_league': latest_season_by_league,
            'team_count_by_league': {
                lg: len(teams) for lg, teams in current_teams_by_league.items()
            },
            'seasons_requested_by_league': load_report['loaded'],
            'seasons_unavailable_by_league': load_report['failed'],
        },
        'artifacts': {
            name: {
                'exists': os.path.exists(name),
                'size_bytes': os.path.getsize(name) if os.path.exists(name) else None,
            }
            for name in (
                'final_model.pkl', 'dc_models.pkl', 'corner_model.pkl',
                'corner_feature_cols.pkl', 'card_model.pkl', 'card_feature_cols.pkl',
                'feature_cols.pkl', 'processed_data.pkl', 'team_mapping.pkl',
                'all_teams.pkl', 'current_teams.pkl', 'league_config.pkl',
            )
        },
    }
    with open('training_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print("  ✅ training_manifest.json")

    print(f"""
💾 FILES SAVED:
  • final_model.pkl
  • dc_models.pkl
  • corner_model.pkl        (new — Task 2)
  • corner_feature_cols.pkl (new — Task 2)
  • feature_cols.pkl
  • processed_data.pkl
  • team_mapping.pkl
  • all_teams.pkl
  • current_teams.pkl

📦 CACHED (auto-invalidates if seasons/schema change):
  • cache/raw_data.pkl (+ raw_data_meta.pkl)
  • cache/features.pkl (+ features_meta.pkl)

✅ Ready for deployment!
""")

if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    n_seasons = 10
    for arg in sys.argv:
        if arg.startswith("--seasons="):
            n_seasons = int(arg.split("=")[1])
    main(force_refresh=force, n_seasons=n_seasons)
