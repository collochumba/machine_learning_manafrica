"""
BETTING MODULE

All betting economics, kept separate from the prediction mathematics in
models.py:

  - No-vig (overround-free) pricing
  - Expected value calculation, with push-aware settlement for DNB / whole-
    line Asian Handicap / corner handicap markets
  - Value-bet filtering
  - Fractional-Kelly staking
  - Cross-fixture ranking, market grouping, and bankroll simulation
    (moved here from the old predict.py)
  - Closing-line-value (CLV) tracking infrastructure (from clv_tracker.py) —
    stores opening/prediction/closing odds without claiming CLV exists until
    both prediction-time and closing prices are recorded.

Nothing here decides WHAT a fixture's probabilities are (that's models.py,
corners.py, cards.py) — only what a bet on those probabilities is worth.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path

def no_vig_probabilities(odds):
    """
    Convert bookmaker odds for a market group (e.g. Home/Draw/Away, or
    Over/Under) into overround-free ("no-vig") probabilities.

    Args:
        odds: Dict mapping outcome name -> decimal odds, for outcomes that
              are mutually exclusive and collectively exhaustive (e.g.
              {'Home': 2.1, 'Draw': 3.4, 'Away': 3.5})

    Returns:
        Dict mapping outcome name -> no-vig probability, plus the overround
        under the key '_overround' (e.g. 0.06 means a 6% margin).
    """

    implied = {k: 1.0 / v for k, v in odds.items() if v and v > 1.0}

    if not implied:
        return {}

    total = sum(implied.values())

    no_vig = {k: v / total for k, v in implied.items()}
    no_vig['_overround'] = total - 1.0

    return no_vig


def calculate_value(market_probs, bookmaker_odds):
    """Calculate market value with no-vig pricing and push-aware settlement.

    DNB and whole-number Asian Handicap markets are not treated as ordinary
    binary bets: draw/push outcomes refund the stake and therefore reduce
    the expected return.
    """
    value_bets = []
    if not bookmaker_odds:
        return value_bets
    if all(isinstance(v, (int, float)) for v in bookmaker_odds.values()):
        groups = {'_all': bookmaker_odds}
    else:
        groups = bookmaker_odds
    for group_name, group_odds in groups.items():
        if not isinstance(group_odds, dict):
            continue
        novig = no_vig_probabilities({k:v for k,v in group_odds.items() if isinstance(v,(int,float)) and v > 1.0})
        for market, odds in group_odds.items():
            if market not in market_probs or odds is None or odds <= 1.0:
                continue
            model_prob = float(market_probs[market])
            market_prob = float(novig.get(market, 1.0 / odds))
            push_prob = 0.0
            settlement = 'win_lose'
            if market.startswith('DNB '):
                push_prob = float(market_probs.get('Draw', 0.0))
                settlement = 'push_on_draw'
            elif market.startswith('AH Home ') or market.startswith('AH Away '):
                label = market.split(' ', 2)[2]
                push_prob = float(market_probs.get(f'AH Push {label}', 0.0))
                settlement = 'push_on_whole_line' if push_prob > 0 else 'win_lose'
            elif market.startswith('Corner Hcp '):
                # Corner handicap markets currently expose half-lines only, so
                # there is no push probability unless a future whole line is added.
                settlement = 'win_lose'
            if push_prob > 0:
                nonpush = max(0.0, 1.0 - push_prob)
                ev = nonpush * (model_prob * odds - 1.0)
            else:
                ev = model_prob * odds - 1.0
            edge = model_prob - market_prob
            value_bets.append({'market': market, 'prob': model_prob, 'odds': float(odds),
                               'market_prob': market_prob, 'edge': edge, 'ev': float(ev),
                               'push_prob': push_prob, 'settlement': settlement})
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


# ============================================================================
# CROSS-FIXTURE REPORTING (moved here from the old predict.py)
# ============================================================================

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

# ============================================================================
# CLOSING-LINE-VALUE (CLV) TRACKING  (from clv_tracker.py)
# ============================================================================

@dataclass
class CLVRecord:
    fixture_id: str
    market: str
    opening_odds: float | None = None
    prediction_odds: float | None = None
    closing_odds: float | None = None
    created_at: str = ''

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()

def calculate_clv(prediction_odds, closing_odds):
    if not prediction_odds or not closing_odds or prediction_odds <= 1 or closing_odds <= 1:
        return None
    # Positive CLV means the price taken was better than the closing price.
    return float((1.0 / closing_odds) - (1.0 / prediction_odds)) * -1.0

def save_records(records, path):
    Path(path).write_text(json.dumps([asdict(r) for r in records], indent=2), encoding='utf-8')

def load_records(path):
    p=Path(path)
    if not p.exists(): return []
    return [CLVRecord(**x) for x in json.loads(p.read_text(encoding='utf-8'))]
