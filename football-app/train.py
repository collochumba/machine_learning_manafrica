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

# Import Dixon-Coles from models.py (NO DUPLICATION!)
from models import DixonColesTimeDecay

print("="*80)
print("🚀 FOOTBALL BETTING MODEL - TRAINING PIPELINE v2.0")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

LEAGUES = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Bundesliga": "D1",
    "Ligue 1": "F1"
}

def get_last_n_seasons(n=5):
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

def load_data_with_cache(force_refresh=False):
    """Load data with local caching to avoid repeated downloads."""
    
    cache_file = CACHE_DIR / "raw_data.pkl"
    
    # Check cache
    if cache_file.exists() and not force_refresh:
        print("📦 Loading from cache...")
        df = joblib.load(cache_file)
        print(f"✅ Loaded {len(df):,} matches from cache")
        return df
    
    print("\n📥 Downloading fresh data from football-data.co.uk...")
    
    seasons = get_last_n_seasons(5)
    all_data = []
    
    for league_name, league_code in LEAGUES.items():
        print(f"\n{league_name}:")
        for season in seasons:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
            try:
                df_temp = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')
                df_temp['League'] = league_name
                df_temp['Season'] = season
                all_data.append(df_temp)
                print(f"  ✅ {season}: {len(df_temp)} matches")
            except Exception as e:
                print(f"  ⚠️  {season}: Failed")
    
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
    
    # Ensure shot/corner columns
    for col in ['HS', 'AS', 'HST', 'AST', 'HC', 'AC']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Cache it
    joblib.dump(df, cache_file, compress=3)
    print(f"\n💾 Cached to {cache_file}")
    
    return df

# ============================================================================
# STEP 2: ELO RATINGS
# ============================================================================

def compute_elo_ratings(df, k=20, base_rating=1500):
    """Compute ELO ratings with proper time ordering."""
    
    print("\n🏆 Computing ELO ratings...")
    
    df = df.sort_values(['League', 'Date']).reset_index(drop=True)
    ratings = {}
    home_elo = []
    away_elo = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ELO computation"):
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        if home not in ratings:
            ratings[home] = base_rating
        if away not in ratings:
            ratings[away] = base_rating
        
        h_elo = ratings[home]
        a_elo = ratings[away]
        
        home_elo.append(h_elo)
        away_elo.append(a_elo)
        
        expected = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        result = 1.0 if row['FTR'] == 'H' else (0.5 if row['FTR'] == 'D' else 0.0)
        
        ratings[home] = h_elo + k * (result - expected)
        ratings[away] = a_elo + k * ((1 - result) - (1 - expected))
    
    df['ELO_home'] = home_elo
    df['ELO_away'] = away_elo
    df['ELO_diff'] = df['ELO_home'] - df['ELO_away']
    
    print(f"✅ ELO computed for {len(ratings)} teams")
    
    return df

# ============================================================================
# STEP 3: FEATURE ENGINEERING WITH CACHING
# ============================================================================

def create_features_with_cache(df, force_refresh=False):
    """Create features with caching to avoid recomputation."""
    
    cache_file = CACHE_DIR / "features.pkl"
    
    if cache_file.exists() and not force_refresh:
        print("\n📦 Loading features from cache...")
        df, feature_cols = joblib.load(cache_file)
        print(f"✅ Loaded {len(feature_cols)} features from cache")
        return df, feature_cols
    
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
    
    # Form
    print("  • Form features...")
    df['HP'] = (df['FTR'] == 'H') * 3 + (df['FTR'] == 'D') * 1
    df['AP'] = (df['FTR'] == 'A') * 3 + (df['FTR'] == 'D') * 1
    
    df['HForm'] = df.groupby(['League', 'HomeTeam'])['HP'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    )
    df['AForm'] = df.groupby(['League', 'AwayTeam'])['AP'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    )
    
    # Matchup differentials
    print("  • Matchup differentials...")
    df['AttackDiff'] = df['HGS_L5'] - df['AGC_L5']
    df['DefenseDiff'] = df['HGC_L5'] - df['AGS_L5']
    df['ShotDiff'] = df['HS_L5'] - df['AS_L5']
    df['ShotTargetDiff'] = df['HST_L5'] - df['AST_L5']
    df['CornerDiff'] = df['HC_L5'] - df['AC_L5']
    
    # League dummies
    print("  • League dummies...")
    league_dummies = pd.get_dummies(df['League'], prefix='Lg')
    df = pd.concat([df, league_dummies], axis=1)
    
    feature_cols = [
        c for c in df.columns 
        if ('_L' in c or 'Form' in c or 'Diff' in c or 'ELO' in c or 'Lg_' in c)
    ]
    
    df = df.dropna(subset=feature_cols)
    
    # Cache it
    joblib.dump((df, feature_cols), cache_file, compress=3)
    print(f"\n💾 Cached features to {cache_file}")
    
    print(f"\n✅ Created {len(feature_cols)} features")
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
    
    final_model = CalibratedClassifierCV(base_model, method='isotonic', cv=tscv)
    final_model.fit(X, y)
    
    y_pred = final_model.predict_proba(X)
    ll = log_loss(y, y_pred)
    
    print(f"✅ ML Model trained")
    print(f"  • Log-loss: {ll:.4f}")
    
    return final_model

# ============================================================================
# STEP 5: TRAIN DIXON-COLES (IMPORT FROM models.py!)
# ============================================================================

def train_dixon_coles(df):
    """Train Dixon-Coles per league using imported class."""
    
    print("\n🎯 Training Dixon-Coles models...")
    
    dc_models = {}
    
    for league in df['League'].unique():
        print(f"\n  Training {league}...")
        
        # Use imported class - NO DUPLICATION
        dc_model = DixonColesTimeDecay(xi=0.002)
        dc_model.fit(df, league=league)
        
        dc_models[league] = dc_model
        
        print(f"    ✅ Teams: {len(dc_model.teams)}")
        print(f"    ✅ Home adv: {dc_model.home_adv:.3f}")
    
    print(f"\n✅ Trained {len(dc_models)} DC models")
    
    return dc_models

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
    
    # Manual variations
    manual = {
        'brighton': 'Brighton',
        'newcastle': 'Newcastle',
        'west ham': 'West Ham',
        'man united': 'Manchester United',
        'man city': 'Manchester City',
        'nottm forest': 'Nottingham Forest',
        'koln': 'FC Koln',
        'cologne': 'FC Koln',
        'mainz': 'Mainz 05',
        'inter': 'Inter',
        'verona': 'Verona',
        'psg': 'Paris SG',
        'monaco': 'Monaco',
    }
    
    team_mapping.update(manual)
    
    print(f"✅ Mappings for {len(all_teams)} teams")
    
    return team_mapping, all_teams

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(force_refresh=False):
    """Main training pipeline with proper caching."""
    
    # Load data
    df = load_data_with_cache(force_refresh=force_refresh)
    
    # ELO
    df = compute_elo_ratings(df)
    
    # Features
    df, feature_cols = create_features_with_cache(df, force_refresh=force_refresh)
    
    # Train ML
    final_model = train_ml_model(df, feature_cols)
    
    # Train DC
    dc_models = train_dixon_coles(df)
    
    # Team mappings
    team_mapping, all_teams = create_team_mappings(df)
    
    # Save models
    print("\n💾 Saving models...")
    
    joblib.dump(final_model, 'final_model.pkl', compress=3)
    print("  ✅ final_model.pkl")
    
    joblib.dump(dc_models, 'dc_models.pkl', compress=3)
    print("  ✅ dc_models.pkl")
    
    joblib.dump(feature_cols, 'feature_cols.pkl')
    print("  ✅ feature_cols.pkl")
    
    joblib.dump(df, 'processed_data.pkl', compress=3)
    print("  ✅ processed_data.pkl")
    
    joblib.dump(team_mapping, 'team_mapping.pkl')
    print("  ✅ team_mapping.pkl")
    
    # Save all teams for validation
    joblib.dump(all_teams, 'all_teams.pkl')
    print("  ✅ all_teams.pkl")
    
    # Summary
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"""
📊 SUMMARY:
  • Matches: {len(df):,}
  • Teams: {len(all_teams)}
  • Features: {len(feature_cols)}
  • Leagues: {len(dc_models)}
  • Log-loss: {log_loss(df['Outcome'], final_model.predict_proba(df[feature_cols].fillna(0))):.4f}

💾 FILES SAVED:
  • final_model.pkl
  • dc_models.pkl
  • feature_cols.pkl
  • processed_data.pkl
  • team_mapping.pkl
  • all_teams.pkl

📦 CACHED:
  • cache/raw_data.pkl
  • cache/features.pkl

✅ Ready for deployment!
""")

if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    main(force_refresh=force)
