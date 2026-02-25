"""
INSTITUTIONAL BETTING OPTIMIZER
Professional Kelly Criterion implementation with safety constraints

Features:
- Full and fractional Kelly calculation
- Expected value computation
- Value bet identification
- Arbitrage detection
- Stake sizing with constraints
- Performance tracking
- Risk-adjusted position sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BettingConstraints:
    """Constraints for bet sizing."""
    max_kelly_fraction: float = 0.25  # Quarter Kelly
    max_bet_pct: float = 0.05  # 5% max per bet
    min_edge: float = 0.03  # 3% minimum edge
    min_prob: float = 0.05  # Don't bet below 5%
    max_prob: float = 0.95  # Don't bet above 95%
    min_odds: float = 1.01  # Minimum odds to accept


class BettingOptimizer:
    """
    Professional betting optimizer using Kelly Criterion.
    
    Implements proper Kelly sizing with institutional-grade
    safety constraints and risk management.
    """
    
    def __init__(
        self,
        bankroll: float = 10000.0,
        constraints: Optional[BettingConstraints] = None
    ):
        """
        Args:
            bankroll: Total betting bankroll
            constraints: Betting constraints (uses defaults if None)
        """
        self.bankroll = bankroll
        self.constraints = constraints or BettingConstraints()
    
    def kelly_criterion(
        self,
        prob: float,
        odds: float,
        full_kelly: bool = False
    ) -> float:
        """
        Calculate Kelly Criterion stake fraction.
        
        Kelly formula: f* = (bp - q) / b
        where:
            b = odds - 1 (net odds)
            p = win probability
            q = 1 - p (loss probability)
            f* = optimal fraction of bankroll to bet
        
        Args:
            prob: Win probability (0-1)
            odds: Decimal odds
            full_kelly: Use full Kelly (default: fractional Kelly)
        
        Returns:
            Stake fraction (0-1)
        """
        # Validate inputs
        if prob <= self.constraints.min_prob or prob >= self.constraints.max_prob:
            return 0.0
        
        if odds < self.constraints.min_odds:
            return 0.0
        
        # Calculate Kelly
        b = odds - 1
        q = 1 - prob
        
        kelly = (prob * b - q) / b
        
        # Never bet on negative edge
        if kelly <= 0:
            return 0.0
        
        # Apply fractional Kelly (safety)
        if not full_kelly:
            kelly *= self.constraints.max_kelly_fraction
        
        # Cap at maximum bet size
        kelly = min(kelly, self.constraints.max_bet_pct)
        
        return kelly
    
    def kelly_stake(
        self,
        prob: float,
        odds: float,
        full_kelly: bool = False
    ) -> float:
        """
        Calculate Kelly stake in dollars.
        
        Args:
            prob: Win probability
            odds: Decimal odds
            full_kelly: Use full Kelly
        
        Returns:
            Stake amount in dollars
        """
        kelly_frac = self.kelly_criterion(prob, odds, full_kelly)
        stake = kelly_frac * self.bankroll
        
        return stake
    
    def calculate_value(
        self,
        model_prob: float,
        bookmaker_odds: float
    ) -> Dict:
        """
        Calculate expected value and betting metrics.
        
        Args:
            model_prob: Model's probability estimate
            bookmaker_odds: Bookmaker's decimal odds
        
        Returns:
            Dictionary with comprehensive value analysis
        """
        # Implied probability from odds
        if bookmaker_odds <= 0:
            return {
                'has_value': False,
                'expected_value': -1.0,
                'edge': -1.0,
                'kelly_fraction': 0.0,
                'stake': 0.0
            }
        
        implied_prob = 1 / bookmaker_odds
        
        # Edge calculation
        edge = model_prob - implied_prob
        
        # Expected value per unit bet
        ev = model_prob * (bookmaker_odds - 1) - (1 - model_prob)
        ev_pct = ev * 100
        
        # Kelly stake
        kelly_frac = self.kelly_criterion(model_prob, bookmaker_odds)
        stake = kelly_frac * self.bankroll
        
        # Value determination
        has_value = (
            ev > 0 and
            edge >= self.constraints.min_edge and
            kelly_frac > 0
        )
        
        # Expected profit
        expected_profit = stake * ev if has_value else 0
        
        return {
            'model_prob': model_prob,
            'implied_prob': implied_prob,
            'bookmaker_odds': bookmaker_odds,
            'edge': edge,
            'edge_percentage': edge * 100,
            'expected_value': ev,
            'ev_percentage': ev_pct,
            'kelly_fraction': kelly_frac,
            'kelly_stake_amount': stake,
            'expected_profit': expected_profit,
            'has_value': has_value,
            'roi_if_bet': ev_pct
        }
    
    def find_value_bets(
        self,
        predictions: Dict[str, float],
        odds: Dict[str, float],
        min_edge: Optional[float] = None
    ) -> List[Dict]:
        """
        Find all value bets from predictions and odds.
        
        Args:
            predictions: Dict of {market: probability}
            odds: Dict of {market: decimal_odds}
            min_edge: Minimum edge required (uses default if None)
        
        Returns:
            List of value bets, sorted by expected value
        """
        min_edge = min_edge or self.constraints.min_edge
        
        value_bets = []
        
        for market in predictions:
            if market not in odds:
                continue
            
            value = self.calculate_value(predictions[market], odds[market])
            
            if value['has_value'] and value['edge'] >= min_edge:
                value['market'] = market
                value_bets.append(value)
        
        # Sort by expected value (descending)
        value_bets = sorted(
            value_bets,
            key=lambda x: x['expected_value'],
            reverse=True
        )
        
        return value_bets
    
    def calculate_arbitrage(
        self,
        odds_list: List[float],
        markets: Optional[List[str]] = None
    ) -> Dict:
        """
        Detect arbitrage opportunity.
        
        Arbitrage exists when sum of implied probabilities < 1.0
        
        Args:
            odds_list: List of odds for all outcomes
            markets: Market names (optional)
        
        Returns:
            Dictionary with arbitrage analysis
        """
        if not odds_list:
            return {'is_arbitrage': False}
        
        # Calculate implied probabilities
        implied_probs = [1/odds for odds in odds_list if odds > 0]
        
        if not implied_probs:
            return {'is_arbitrage': False}
        
        total_prob = sum(implied_probs)
        
        # Arbitrage exists if total < 1
        is_arb = total_prob < 1.0
        
        if is_arb:
            # Calculate profit percentage
            arb_pct = ((1 / total_prob) - 1) * 100
            
            # Optimal stake allocation
            total_stake = 100  # Assume $100 total
            stakes = [(imp_prob / total_prob) * total_stake for imp_prob in implied_probs]
            
            # Expected profit
            profit = total_stake * (1 / total_prob - 1)
        else:
            arb_pct = 0
            stakes = [0] * len(implied_probs)
            profit = 0
        
        result = {
            'is_arbitrage': is_arb,
            'total_implied_prob': total_prob,
            'arbitrage_percentage': arb_pct,
            'expected_profit_pct': arb_pct,
            'optimal_stakes': stakes,
            'implied_probabilities': implied_probs
        }
        
        if markets:
            result['markets'] = markets
        
        return result
    
    def compare_bookmakers(
        self,
        model_prob: float,
        bookmaker_odds: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Compare value across multiple bookmakers.
        
        Args:
            model_prob: Model probability
            bookmaker_odds: Dict of {bookmaker_name: odds}
        
        Returns:
            DataFrame with comparison
        """
        comparisons = []
        
        for bookmaker, odds in bookmaker_odds.items():
            value = self.calculate_value(model_prob, odds)
            value['bookmaker'] = bookmaker
            comparisons.append(value)
        
        df = pd.DataFrame(comparisons)
        
        # Sort by expected value
        df = df.sort_values('expected_value', ascending=False)
        
        return df


class BettingTracker:
    """
    Track betting performance over time.
    
    Records all bets and calculates performance metrics.
    """
    
    def __init__(self):
        self.bets: List[Dict] = []
    
    def add_bet(
        self,
        stake: float,
        odds: float,
        won: bool,
        market: str = '',
        date: Optional[str] = None,
        model_prob: Optional[float] = None
    ):
        """
        Record a bet.
        
        Args:
            stake: Stake amount
            odds: Decimal odds
            won: Whether bet won
            market: Market type
            date: Bet date
            model_prob: Model's probability
        """
        profit = stake * (odds - 1) if won else -stake
        
        self.bets.append({
            'date': date or pd.Timestamp.now().date(),
            'stake': stake,
            'odds': odds,
            'won': won,
            'profit': profit,
            'market': market,
            'model_prob': model_prob
        })
    
    def get_statistics(self) -> Dict:
        """
        Calculate performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.bets:
            return {'n_bets': 0}
        
        df = pd.DataFrame(self.bets)
        
        total_staked = df['stake'].sum()
        total_profit = df['profit'].sum()
        n_bets = len(df)
        n_wins = df['won'].sum()
        
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        win_rate = (n_wins / n_bets * 100)
        
        avg_odds = df['odds'].mean()
        avg_stake = df['stake'].mean()
        
        biggest_win = df['profit'].max()
        biggest_loss = df['profit'].min()
        
        # Winning vs losing bets
        winners = df[df['won'] == True]
        losers = df[df['won'] == False]
        
        avg_win = winners['profit'].mean() if len(winners) > 0 else 0
        avg_loss = losers['profit'].mean() if len(losers) > 0 else 0
        
        # Profit factor
        total_wins = winners['profit'].sum() if len(winners) > 0 else 0
        total_losses = abs(losers['profit'].sum()) if len(losers) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        return {
            'n_bets': n_bets,
            'n_wins': int(n_wins),
            'n_losses': n_bets - int(n_wins),
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi_pct': roi,
            'win_rate_pct': win_rate,
            'avg_odds': avg_odds,
            'avg_stake': avg_stake,
            'biggest_win': biggest_win,
            'biggest_loss': biggest_loss,
            'avg_winning_bet': avg_win,
            'avg_losing_bet': avg_loss,
            'profit_factor': profit_factor
        }
    
    def export_bets(self, filepath: str = 'bets_history.csv'):
        """Export bet history to CSV."""
        if not self.bets:
            print("No bets to export")
            return
        
        df = pd.DataFrame(self.bets)
        df.to_csv(filepath, index=False)
        print(f"Exported {len(df)} bets to {filepath}")


if __name__ == "__main__":
    print("="*70)
    print("INSTITUTIONAL BETTING OPTIMIZER")
    print("="*70)
    
    optimizer = BettingOptimizer(bankroll=10000)
    
    # Example
    value = optimizer.calculate_value(model_prob=0.55, bookmaker_odds=2.0)
    
    print("\nValue calculation:")
    print(f"  Model prob: {value['model_prob']:.1%}")
    print(f"  Implied prob: {value['implied_prob']:.1%}")
    print(f"  Edge: {value['edge_percentage']:.2f}%")
    print(f"  EV: {value['ev_percentage']:.2f}%")
    print(f"  Kelly stake: ${value['kelly_stake_amount']:.2f}")
    print(f"  Has value: {value['has_value']}")
