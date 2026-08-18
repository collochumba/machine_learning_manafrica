"""
PREDICTION MODULE - FIXED
Proper validation | Fallback strategies | All markets utilized
"""

import pandas as pd
import numpy as np
from models import (
    ensemble_prediction,
    build_market_probabilities,
    calculate_value,
    find_value_bets,
    calculate_kelly_stake,
    calculate_confidence_score
)


def normalize_team_name(team_name, team_mapping, all_teams, league, df):
    """
    Normalize team name with user feedback.

    IMPROVED:
    - Returns suggestion if fuzzy match
    - Shows all possible matches
    - Allows manual override
    """

    # Direct match
    if team_name in all_teams:
        return team_name, 1.0, None

    # Mapping match
    if team_name in team_mapping:
        return team_mapping[team_name], 0.9, None

    # Case-insensitive
    lower = team_name.lower()
    if lower in team_mapping:
        return team_mapping[lower], 0.9, None

    # League-specific teams
    league_teams = set(df[df['League'] == league]['HomeTeam'].unique())

    if team_name in league_teams:
        return team_name, 1.0, None

    # Fuzzy match with user notification
    from difflib import get_close_matches
    matches = get_close_matches(team_name, league_teams, n=3, cutoff=0.6)

    if matches:
        # Return best match + suggestions
        return matches[0], 0.7, {
            'original': team_name,
            'suggestions': matches,
            'warning': f"'{team_name}' not found. Using '{matches[0]}'. Other options: {matches[1:]}"
        }

    # No match found
    return None, 0.0, {
        'original': team_name,
        'error': f"Team '{team_name}' not found in {league}",
        'available': sorted(list(league_teams))[:10]
    }


# Feature-column classification used by get_fixture_features below.
# 'home' columns describe the HOME team's own recent home-match record and
# should be sourced from that team's most recent match AS HOME TEAM.
# 'away' columns describe the AWAY team's own recent away-match record and
# should be sourced from that team's most recent match AS AWAY TEAM.
# 'diff' columns are recomputed from the resolved home/away values rather
# than copied from either source row, since they compare the two teams
# directly and a copied value would reflect the wrong pairing.
_HOME_PREFIXES = ('HGS_', 'HGC_', 'HS_', 'HST_', 'HC_', 'HFormPPG')
_AWAY_PREFIXES = ('AGS_', 'AGC_', 'AS_', 'AST_', 'AC_', 'AFormPPG')
_DIFF_COLS = {
    'AttackDiff': ('HGS_L5', 'AGC_L5'),
    'DefenseDiff': ('HGC_L5', 'AGS_L5'),
    'ShotDiff': ('HS_L5', 'AS_L5'),
    'ShotTargetDiff': ('HST_L5', 'AST_L5'),
    'CornerDiff': ('HC_L5', 'AC_L5'),
}


def get_fixture_features(df, feature_cols, league, home, away):
    """
    Build a feature vector for a SPECIFIC upcoming fixture (home vs away),
    rather than grabbing the single most recent row involving either team.

    CRITICAL FIX (replaces the old get_latest_features):
    The old approach pulled the last row where either team appeared in
    ANY role (home or away, against ANY opponent) and used that row's
    features wholesale. Because feature columns are role-specific
    (e.g. 'HGS_L5' = the row's home team's scoring form), that row's
    values often did not correspond to `home` playing at home and `away`
    playing away at all — e.g. it could be a row for "Away vs Someone
    Else" where "Away" was actually the home team of that historical row.

    The fix: pull `home`'s own most recent HOME-role features, and
    `away`'s own most recent AWAY-role features, independently, then
    recompute the matchup-differential columns and set the league
    dummies for the target league. This mirrors exactly what train.py
    computes per-row, just assembled for a hypothetical fixture rather
    than read off an existing one.
    """

    league_df = df[df['League'] == league].sort_values('Date')

    if len(league_df) == 0:
        raise ValueError(f"No historical data at all for league: {league}")

    home_rows = league_df[league_df['HomeTeam'] == home]
    away_rows = league_df[league_df['AwayTeam'] == away]

    missing_data_flag = False
    home_row = None
    away_row = None

    if len(home_rows) > 0:
        home_valid = home_rows.dropna(subset=[c for c in feature_cols if c.startswith(_HOME_PREFIXES) or c == 'ELO_home'])
        home_row = home_valid.iloc[-1] if len(home_valid) > 0 else home_rows.iloc[-1]
    else:
        # No home-role history for this team in this league at all (e.g. a
        # newly promoted side). FIX: we deliberately do NOT fall back to an
        # away-role row for this team here — doing so would put away-context
        # stats (how this team performs on the road) into home-context
        # feature slots, which is exactly the row-mismatch bug this function
        # exists to prevent. Instead, a league-wide baseline is used below
        # for every home-context column, and the prediction is flagged.
        missing_data_flag = True

    if len(away_rows) > 0:
        away_valid = away_rows.dropna(subset=[c for c in feature_cols if c.startswith(_AWAY_PREFIXES) or c == 'ELO_away'])
        away_row = away_valid.iloc[-1] if len(away_valid) > 0 else away_rows.iloc[-1]
    else:
        missing_data_flag = True

    def _league_baseline(col):
        """League-wide average for `col`, used only when a team has no
        role-specific history at all. Returns NaN (not 0) if the league
        itself has no data for this column, so downstream NaN handling
        applies uniformly."""
        if col not in league_df.columns:
            return np.nan
        val = league_df[col].mean()
        return val if val == val else np.nan

    values = {}

    for col in feature_cols:
        if col == 'ELO_home':
            values[col] = home_row.get('ELO_home', np.nan) if home_row is not None else _league_baseline('ELO_home')
        elif col == 'ELO_away':
            values[col] = away_row.get('ELO_away', np.nan) if away_row is not None else _league_baseline('ELO_away')
        elif col == 'ELO_diff':
            continue  # recomputed below
        elif col in _DIFF_COLS:
            continue  # recomputed below
        elif col.startswith('Lg_'):
            values[col] = 1.0 if col == f'Lg_{league}' else 0.0
        elif col.startswith(_HOME_PREFIXES):
            values[col] = home_row.get(col, np.nan) if home_row is not None else _league_baseline(col)
        elif col.startswith(_AWAY_PREFIXES):
            values[col] = away_row.get(col, np.nan) if away_row is not None else _league_baseline(col)
        else:
            # Unrecognized column pattern: best-effort, prefer home row,
            # then away row, then league baseline.
            if home_row is not None and col in home_row and home_row.get(col, np.nan) == home_row.get(col, np.nan):
                values[col] = home_row.get(col)
            elif away_row is not None and col in away_row and away_row.get(col, np.nan) == away_row.get(col, np.nan):
                values[col] = away_row.get(col)
            else:
                values[col] = _league_baseline(col)

    # Recompute differential columns from the resolved values so they
    # actually compare `home` against `away`, not two mismatched rows.
    for diff_col, (a, b) in _DIFF_COLS.items():
        if diff_col in feature_cols:
            va = values.get(a, np.nan)
            vb = values.get(b, np.nan)
            values[diff_col] = (va if va == va else 0) - (vb if vb == vb else 0)

    if 'ELO_diff' in feature_cols:
        eh = values.get('ELO_home', np.nan)
        ea = values.get('ELO_away', np.nan)
        values['ELO_diff'] = (eh if eh == eh else 1500) - (ea if ea == ea else 1500)

    features = np.array([
        values.get(c, 0.0) if values.get(c, np.nan) == values.get(c, np.nan) else 0.0
        for c in feature_cols
    ], dtype=float)

    assert len(features) == len(feature_cols), \
        f"Feature count mismatch! Expected {len(feature_cols)}, got {len(features)}"

    return features, missing_data_flag


def predict_with_fallback(
    fixture,
    final_model,
    dc_models,
    feature_cols,
    df,
    team_mapping,
    all_teams,
    use_fallback=True
):
    """
    Predict with fallback for missing teams.

    CRITICAL FIX:
    - If team not found in DC → use league average parameters
    - No crash on unknown teams
    """

    league = fixture['league']
    home = fixture['home']
    away = fixture['away']

    # Normalize teams
    home_norm, home_conf, home_info = normalize_team_name(home, team_mapping, all_teams, league, df)
    away_norm, away_conf, away_info = normalize_team_name(away, team_mapping, all_teams, league, df)

    # Collect warnings/errors
    warnings = []

    if home_info and 'warning' in home_info:
        warnings.append(home_info['warning'])
    if away_info and 'warning' in away_info:
        warnings.append(away_info['warning'])

    if not home_norm or not away_norm:
        error_msg = ""
        if not home_norm and home_info:
            error_msg += home_info.get('error', f"Team not found: {home}")
        if not away_norm and away_info:
            error_msg += " | " + away_info.get('error', f"Team not found: {away}")

        raise ValueError(error_msg)

    # Check league
    if league not in dc_models:
        raise ValueError(f"League not supported: {league}")

    # Get fixture-specific features (FIXED: no longer a mismatched last row)
    features, low_data = get_fixture_features(df, feature_cols, league, home_norm, away_norm)
    if low_data:
        warnings.append(
            f"Limited home/away-specific history for {home_norm} or {away_norm}; "
            f"features may be less reliable."
        )

    # Try prediction with fallback
    try:
        # Standard prediction
        probs, dc_pred = ensemble_prediction(
            final_model, dc_models, league, home_norm, away_norm, features
        )
        used_fallback = False

    except ValueError as e:
        # Team not in Dixon-Coles model
        if not use_fallback:
            raise

        # FALLBACK: Use league average
        warnings.append(f"Using league average for unknown team")

        league_data = df[df['League'] == league].tail(100)

        avg_home_prob = (league_data['FTR'] == 'H').mean()
        avg_draw_prob = (league_data['FTR'] == 'D').mean()
        avg_away_prob = (league_data['FTR'] == 'A').mean()

        probs = np.array([avg_home_prob, avg_draw_prob, avg_away_prob])

        # Estimate over/under 2.5 from the league's actual recent scoring
        # rate instead of assuming a flat 50/50 split.
        total_goals = league_data['FTHG'] + league_data['FTAG']
        prob_over25 = float((total_goals > 2.5).mean()) if len(league_data) > 0 else 0.5
        prob_over25 = prob_over25 if prob_over25 == prob_over25 else 0.5  # guard NaN

        dc_pred = {
            'lambda_home': league_data['FTHG'].mean(),
            'lambda_away': league_data['FTAG'].mean(),
            'exp_goals': league_data['FTHG'].mean() + league_data['FTAG'].mean(),
            'prob_over_25': prob_over25,
            'prob_under_25': 1 - prob_over25,
            'score_matrix': None
        }

        used_fallback = True

    # Build market probabilities
    market_probs = build_market_probabilities(probs, dc_pred)

    # Calculate confidence
    confidence = calculate_confidence_score(probs)

    # Prepare result
    result = {
        'league': league,
        'home': home_norm,
        'away': away_norm,
        'prob_home': float(probs[0]),
        'prob_draw': float(probs[1]),
        'prob_away': float(probs[2]),
        'lambda_home': dc_pred['lambda_home'],
        'lambda_away': dc_pred['lambda_away'],
        'exp_goals': dc_pred['exp_goals'],
        'confidence': confidence,
        'market_probs': market_probs,
        'value_bets': [],
        'warnings': warnings,
        'used_fallback': used_fallback
    }

    # Calculate value if odds provided
    if 'odds' in fixture:
        all_values = calculate_value(market_probs, fixture['odds'])
        result['all_bets'] = all_values

    return result


def predict_multiple_fixtures(
    fixtures,
    final_model,
    dc_models,
    feature_cols,
    df,
    team_mapping,
    all_teams,
    min_prob=0.45,
    min_ev=0.03
):
    """
    Predict multiple fixtures with proper error handling.

    Returns:
        results, errors, warnings_collected
        (NOTE: this order is errors-then-warnings; callers must unpack in
        this order. See app.py FIX.)
    """

    results = []
    errors = []
    warnings_collected = []

    for i, fixture in enumerate(fixtures, 1):
        try:
            result = predict_with_fallback(
                fixture,
                final_model,
                dc_models,
                feature_cols,
                df,
                team_mapping,
                all_teams,
                use_fallback=True
            )

            # Collect warnings
            if result['warnings']:
                warnings_collected.extend(result['warnings'])

            # Find value bets
            if 'all_bets' in result:
                value_bets = find_value_bets(
                    result['all_bets'],
                    min_prob=min_prob,
                    min_ev=min_ev
                )
                result['value_bets'] = value_bets

                # Add Kelly stakes
                for bet in value_bets:
                    bet['kelly_stake'] = calculate_kelly_stake(bet['prob'], bet['odds'])

            results.append(result)

        except Exception as e:
            errors.append({
                'fixture': f"{fixture['home']} vs {fixture['away']}",
                'error': str(e)
            })

    return results, errors, warnings_collected


def generate_summary_stats(results):
    """Generate summary statistics."""

    total_matches = len(results)

    matches_with_value = sum(1 for r in results if len(r['value_bets']) > 0)

    total_value_bets = sum(len(r['value_bets']) for r in results)

    if total_value_bets > 0:
        avg_ev = np.mean([bet['ev'] for r in results for bet in r['value_bets']])
        avg_prob = np.mean([bet['prob'] for r in results for bet in r['value_bets']])
        avg_odds = np.mean([bet['odds'] for r in results for bet in r['value_bets']])
    else:
        avg_ev = 0
        avg_prob = 0
        avg_odds = 0

    confidences = [r['confidence'] for r in results]
    avg_confidence = np.mean(confidences)

    exp_goals = [r['exp_goals'] for r in results]
    avg_exp_goals = np.mean(exp_goals)

    # Count fallback usage
    fallback_count = sum(1 for r in results if r.get('used_fallback', False))

    summary = {
        'total_matches': total_matches,
        'matches_with_value': matches_with_value,
        'total_value_bets': total_value_bets,
        'avg_ev': avg_ev,
        'avg_prob': avg_prob,
        'avg_odds': avg_odds,
        'avg_confidence': avg_confidence,
        'avg_exp_goals': avg_exp_goals,
        'hit_rate': matches_with_value / total_matches if total_matches > 0 else 0,
        'fallback_used': fallback_count
    }

    return summary


def rank_top_value_bets(results, n=7):
    """
    Rank ALL value bets across matches.

    IMPROVED:
    - Groups by market type
    - Shows best per market
    - Cross-match ranking
    """

    all_bets = []

    for result in results:
        for bet in result['value_bets']:
            all_bets.append({
                'match': f"{result['home']} vs {result['away']}",
                'league': result['league'],
                'market': bet['market'],
                'prob': bet['prob'],
                'odds': bet['odds'],
                'edge': bet['edge'],
                'ev': bet['ev'],
                'kelly_stake': bet['kelly_stake'],
                'exp_goals': result['exp_goals'],
                'confidence': result['confidence']
            })

    # Sort by EV
    all_bets = sorted(all_bets, key=lambda x: x['ev'], reverse=True)

    return all_bets[:n]


def group_bets_by_market(results):
    """
    Group value bets by market type.

    NEW FUNCTION:
    - Shows best opportunities per market
    - Helps identify market-specific edges
    """

    markets = {}

    for result in results:
        for bet in result['value_bets']:
            market_type = bet['market']

            if market_type not in markets:
                markets[market_type] = []

            markets[market_type].append({
                'match': f"{result['home']} vs {result['away']}",
                'league': result['league'],
                'prob': bet['prob'],
                'odds': bet['odds'],
                'ev': bet['ev'],
                'kelly': bet['kelly_stake']
            })

    # Sort each market by EV
    for market in markets:
        markets[market] = sorted(markets[market], key=lambda x: x['ev'], reverse=True)

    return markets


def simulate_bankroll(top_bets, initial_bankroll=1000):
    """Simulate bankroll with Kelly criterion."""

    bankroll = initial_bankroll
    total_staked = 0

    bets_placed = []

    for bet in top_bets:
        stake_pct = bet['kelly_stake']
        stake_amount = bankroll * stake_pct

        total_staked += stake_amount

        bets_placed.append({
            'match': bet['match'],
            'market': bet['market'],
            'stake': stake_amount,
            'odds': bet['odds'],
            'prob': bet['prob'],
            'ev': bet['ev']
        })

    expected_profit = sum(b['stake'] * b['ev'] for b in bets_placed)
    expected_bankroll = bankroll + expected_profit
    expected_roi = (expected_profit / total_staked * 100) if total_staked > 0 else 0

    return {
        'initial_bankroll': initial_bankroll,
        'total_staked': total_staked,
        'expected_profit': expected_profit,
        'expected_bankroll': expected_bankroll,
        'expected_roi': expected_roi,
        'num_bets': len(bets_placed),
        'bets': bets_placed
    }
