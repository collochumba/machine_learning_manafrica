"""
MODELS MODULE
Complete implementation of Dixon-Coles, ensemble, and market probability generation
"""

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from scipy.special import softmax


class DixonColesTimeDecay:
    """
    Dixon-Coles model with time decay weighting.
    
    Full implementation with:
    - Poisson-based goal modeling
    - Home advantage parameter
    - Low-score correction (rho)
    - Time decay weighting
    - Maximum likelihood estimation
    """
    
    def __init__(self, xi=0.002, max_goals=10):
        """
        Args:
            xi: Time decay parameter (higher = more recent weight)
            max_goals: Maximum goals to consider in calculations
        """
        self.xi = xi
        self.max_goals = max_goals
        self.teams = None
        self.attack = None
        self.defence = None
        self.home_adv = None
        self.rho = None
    
    def fit(self, df, league=None):
        """
        Fit Dixon-Coles model using MLE.
        
        Args:
            df: DataFrame with match history
            league: League to fit (None = all data)
        
        Returns:
            self
        """
        
        if league:
            data = df[df['League'] == league].copy()
        else:
            data = df.copy()
        
        data = data.sort_values('Date')
        
        self.teams = sorted(set(data['HomeTeam']) | set(data['AwayTeam']))
        n = len(self.teams)
        
        team_to_idx = {team: i for i, team in enumerate(self.teams)}
        
        home_idx = data['HomeTeam'].map(team_to_idx).values
        away_idx = data['AwayTeam'].map(team_to_idx).values
        hg = data['FTHG'].values
        ag = data['FTAG'].values
        days = data['DaysSinceMatch'].values
        
        weights = np.exp(-self.xi * days)
        
        def nll(params):
            """Negative log-likelihood with Dixon-Coles correction."""
            att = params[:n]
            deff = params[n:2*n]
            home = params[2*n]
            rho = params[2*n+1]
            
            att = att - np.mean(att)
            
            lh = np.exp(home + att[home_idx] - deff[away_idx])
            la = np.exp(att[away_idx] - deff[home_idx])
            
            p = poisson.pmf(hg, lh) * poisson.pmf(ag, la)
            
            corr = np.ones_like(p)
            mask00 = (hg == 0) & (ag == 0)
            mask01 = (hg == 0) & (ag == 1)
            mask10 = (hg == 1) & (ag == 0)
            mask11 = (hg == 1) & (ag == 1)
            
            corr[mask00] = 1 - lh[mask00] * la[mask00] * rho
            corr[mask01] = 1 + lh[mask01] * rho
            corr[mask10] = 1 + la[mask10] * rho
            corr[mask11] = 1 - rho
            
            p *= corr
            ll = np.sum(weights * np.log(np.maximum(p, 1e-12)))
            
            return -ll
        
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [0]])
        bounds = [(-3,3)]*(2*n) + [(0,0.5), (-0.1,0.1)]
        
        res = minimize(nll, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200})
        
        params = res.x
        att = params[:n] - np.mean(params[:n])
        deff = params[n:2*n]
        
        self.attack = dict(zip(self.teams, att))
        self.defence = dict(zip(self.teams, deff))
        self.home_adv = params[2*n]
        self.rho = params[2*n+1]
        
        return self
    
    def predict(self, home, away):
        """
        Generate full match prediction.
        
        Returns:
            Dictionary with probabilities, expected goals, score matrix
        """
        
        if home not in self.attack or away not in self.attack:
            raise ValueError(f"Team not found: {home} or {away}")
        
        lh = np.exp(self.home_adv + self.attack[home] - self.defence[away])
        la = np.exp(self.attack[away] - self.defence[home])
        
        max_g = self.max_goals
        
        home_probs = poisson.pmf(range(max_g+1), lh)
        away_probs = poisson.pmf(range(max_g+1), la)
        
        score_matrix = np.outer(home_probs, away_probs)
        
        score_matrix[0,0] *= (1 - lh*la*self.rho)
        score_matrix[0,1] *= (1 + lh*self.rho)
        score_matrix[1,0] *= (1 + la*self.rho)
        score_matrix[1,1] *= (1 - self.rho)
        
        prob_home = np.tril(score_matrix, -1).sum()
        prob_draw = np.trace(score_matrix)
        prob_away = np.triu(score_matrix, 1).sum()
        
        total_goals = np.add.outer(range(max_g+1), range(max_g+1))
        prob_over25 = score_matrix[total_goals > 2.5].sum()
        prob_under25 = 1 - prob_over25
        
        return {
            'lambda_home': float(lh),
            'lambda_away': float(la),
            'prob_home': float(prob_home),
            'prob_draw': float(prob_draw),
            'prob_away': float(prob_away),
            'prob_over_25': float(prob_over25),
            'prob_under_25': float(prob_under25),
            'exp_goals': float(lh + la),
            'score_matrix': score_matrix
        }


def ensemble_prediction(final_model, dc_models, league, home, away, features, dc_weight=0.6):
    """
    Ensemble prediction via log-odds pooling.
    
    Combines:
    - Dixon-Coles (Poisson-based)
    - XGBoost (ML-based)
    
    Args:
        final_model: Trained ML model
        dc_models: Dict of Dixon-Coles models by league
        league: League name
        home: Home team
        away: Away team
        features: Feature vector for ML
        dc_weight: Weight for Dixon-Coles (default 0.6)
    
    Returns:
        probs: [home, draw, away] probabilities
        dc_pred: Full Dixon-Coles prediction dict
    """
    
    # Dixon-Coles prediction
    dc_pred = dc_models[league].predict(home, away)
    
    dc_probs = np.array([
        dc_pred['prob_home'],
        dc_pred['prob_draw'],
        dc_pred['prob_away']
    ])
    
    # ML prediction
    ml_probs = final_model.predict_proba(features.reshape(1, -1))[0]
    
    # Safety clipping
    dc_probs = np.clip(dc_probs, 1e-9, 1 - 1e-9)
    ml_probs = np.clip(ml_probs, 1e-9, 1 - 1e-9)
    
    # Log-odds pooling
    dc_log = np.log(dc_probs)
    ml_log = np.log(ml_probs)
    
    combined_log = dc_weight * dc_log + (1 - dc_weight) * ml_log
    
    # Normalize
    probs = softmax(combined_log)
    
    return probs, dc_pred


def build_market_probabilities(probs, dc_pred):
    """
    Build probabilities for all betting markets.
    
    Markets:
    - 1X2 (Home/Draw/Away)
    - Over/Under 2.5
    - Double Chance (1X, X2, 12)
    - Draw No Bet (DNB Home, DNB Away)
    - Asian Handicap (estimated from lambda difference)
    
    Args:
        probs: [home, draw, away] probabilities from ensemble
        dc_pred: Dixon-Coles prediction dict
    
    Returns:
        Dictionary of all market probabilities
    """
    
    markets = {}
    
    # 1X2
    markets['Home'] = float(probs[0])
    markets['Draw'] = float(probs[1])
    markets['Away'] = float(probs[2])
    
    # Over/Under 2.5
    markets['Over 2.5'] = dc_pred['prob_over_25']
    markets['Under 2.5'] = dc_pred['prob_under_25']
    
    # Double Chance
    markets['1X'] = probs[0] + probs[1]  # Home or Draw
    markets['X2'] = probs[1] + probs[2]  # Draw or Away
    markets['12'] = probs[0] + probs[2]  # Home or Away (no draw)
    
    # Draw No Bet
    home_dnb = probs[0] / (probs[0] + probs[2])
    away_dnb = probs[2] / (probs[0] + probs[2])
    markets['DNB Home'] = float(home_dnb)
    markets['DNB Away'] = float(away_dnb)
    
    # Asian Handicap (estimated from lambda difference)
    lambda_diff = dc_pred['lambda_home'] - dc_pred['lambda_away']
    
    # Sigmoid transformation for handicap coverage
    ah_home = 1 / (1 + np.exp(-lambda_diff))
    ah_away = 1 - ah_home
    
    markets['AH Home'] = float(ah_home)
    markets['AH Away'] = float(ah_away)
    
    return markets


def calculate_value(market_probs, bookmaker_odds):
    """
    Calculate betting value (EV and edge) for each market.
    
    Args:
        market_probs: Dict of market probabilities
        bookmaker_odds: Dict of bookmaker odds
    
    Returns:
        List of dicts with market, prob, odds, edge, EV
    """
    
    value_bets = []
    
    for market, model_prob in market_probs.items():
        if market not in bookmaker_odds:
            continue
        
        odds = bookmaker_odds[market]
        
        if odds is None or odds <= 1.0:
            continue
        
        # Calculate implied probability (with margin)
        implied_prob = 1 / odds
        
        # Edge = model probability - implied probability
        edge = model_prob - implied_prob
        
        # Expected Value = (prob * odds) - 1
        ev = (model_prob * odds) - 1
        
        value_bets.append({
            'market': market,
            'prob': model_prob,
            'odds': odds,
            'edge': edge,
            'ev': ev
        })
    
    return value_bets


def find_value_bets(value_bets, min_prob=0.45, min_ev=0.03):
    """
    Filter for genuine value bets.

    Args:
        value_bets: List of all bets with EV
        min_prob: Minimum probability threshold
        min_ev: Minimum EV threshold

    Returns:
        Filtered list sorted by EV
    """

    filtered = [
        bet for bet in value_bets
        if bet['prob'] >= min_prob
        and bet['ev'] >= min_ev
        and bet['ev'] <= 0.50       # Hard cap: EV > 50% almost certainly indicates model error
        and bet['prob'] <= 0.95     # Reject near-certainty claims — model overconfidence
        and bet['odds'] >= 1.20     # Ignore non-meaningful odds
    ]

    # Sort by EV descending
    filtered = sorted(filtered, key=lambda x: x['ev'], reverse=True)

    return filtered


def calculate_kelly_stake(prob, odds, fraction=0.125):
    """
    Calculate Kelly criterion stake.

    Kelly = (prob * (odds - 1) - (1 - prob)) / (odds - 1)

    Args:
        prob: Win probability
        odds: Bookmaker odds
        fraction: Kelly fraction (default 0.125 = eighth Kelly — conservative
                  until model is fully backtested and calibrated)

    Returns:
        Recommended stake as fraction of bankroll
    """

    if odds <= 1.0:
        return 0.0

    kelly = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
    kelly = max(0, kelly)  # No negative stakes

    # Apply fractional Kelly
    kelly *= fraction

    # Cap at 2% of bankroll per bet (safety guardrail)
    kelly = min(kelly, 0.02)

    return kelly


def calculate_confidence_score(probs):
    """
    Calculate prediction confidence.

    Confidence is scaled relative to a uniform distribution (1/3 each).
    A score of 0 = maximally uncertain (all outcomes equal),
    a score of 1 = maximally confident (one outcome probability = 1).

    Args:
        probs: Array of [home, draw, away] probabilities

    Returns:
        Confidence score (0 = uncertain, 1 = very confident)
    """

    sorted_probs = sorted(probs, reverse=True)
    # Scale: 0 when max_prob = 1/3 (uniform), 1 when max_prob = 1
    confidence = (sorted_probs[0] - 1/3) * 1.5
    confidence = max(0.0, min(1.0, float(confidence)))

    return confidence
