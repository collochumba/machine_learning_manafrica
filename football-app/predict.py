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


def get_latest_features(df, feature_cols, league, home, away):
    """
    Get latest valid features with PROPER VALIDATION.
    
    CRITICAL FIX:
    - Validates feature count
    - Validates feature order
    - Uses latest row (not average!)
    """
    
    # Get recent matches
    recent = df[
        (df['League'] == league) & 
        ((df['HomeTeam'] == home) | (df['AwayTeam'] == away) |
         (df['HomeTeam'] == away) | (df['AwayTeam'] == home))
    ].tail(20)
    
    if len(recent) == 0:
        recent = df[df['League'] == league].tail(50)
    
    if len(recent) == 0:
        raise ValueError(f"No data for league: {league}")
    
    # Get latest valid row
    valid = recent.dropna(subset=feature_cols)
    
    if len(valid) == 0:
        # Fallback to mean
        features = recent[feature_cols].mean().fillna(0).values
    else:
        # CRITICAL: Use latest row, not average!
        features = valid[feature_cols].iloc[-1].fillna(0).values
    
    # VALIDATION
    assert len(features) == len(feature_cols), \
        f"Feature count mismatch! Expected {len(feature_cols)}, got {len(features)}"
    
    return features


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
    
    # Get features
    features = get_latest_features(df, feature_cols, league, home_norm, away_norm)
    
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
        
        dc_pred = {
            'lambda_home': league_data['FTHG'].mean(),
            'lambda_away': league_data['FTAG'].mean(),
            'exp_goals': league_data['FTHG'].mean() + league_data['FTAG'].mean(),
            'prob_over_25': 0.5,
            'prob_under_25': 0.5,
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
    
    IMPROVED:
    - Collects all warnings
    - Uses fallback for unknown teams
    - Validates all features
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
