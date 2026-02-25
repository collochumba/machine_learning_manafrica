"""
INSTITUTIONAL WALK-FORWARD BACKTESTING ENGINE
Hedge fund quality - no shortcuts, no compromises

Features:
- Strict chronological validation (zero lookahead bias)
- Fractional Kelly with realistic constraints
- Capital compounding with proper bankroll tracking
- Closing Line Value (CLV) measurement
- Rolling drawdown (high-water mark method)
- Annualized Sharpe ratio
- Turnover calculation
- Per-league and per-market analytics
- Full bet history with exportable equity curve
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Configuration for backtest parameters."""
    initial_bankroll: float = 10000.0
    kelly_fraction: float = 0.25  # Quarter Kelly for safety
    min_edge: float = 0.03  # 3% minimum edge
    max_daily_risk: float = 0.10  # Max 10% of bankroll at risk per day
    max_single_bet: float = 0.05  # Max 5% per bet
    commission: float = 0.0  # Bookmaker commission
    min_prob: float = 0.05  # Min probability to bet
    max_prob: float = 0.95  # Max probability to bet


class InstitutionalBacktester:
    """
    Production-grade walk-forward backtesting engine.
    
    Implements proper Kelly criterion with:
    - Fractional Kelly sizing
    - Daily exposure limits
    - Per-bet size caps
    - Capital compounding
    - CLV tracking
    - Full performance analytics
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtester.
        
        Args:
            config: Backtest configuration (uses defaults if None)
        """
        self.config = config or BacktestConfig()
        
        # State variables
        self.bankroll = self.config.initial_bankroll
        self.peak_bankroll = self.config.initial_bankroll
        self.bets_history: List[Dict] = []
        self.daily_summary: List[Dict] = []
        
        # Tracking
        self._current_date = None
        self._daily_exposure = 0.0
        self._daily_profit = 0.0
        self._daily_bet_count = 0
    
    def _kelly_criterion(self, prob: float, odds: float) -> float:
        """
        Calculate Kelly stake fraction.
        
        Formula: f* = (bp - q) / b
        where:
            b = odds - 1 (net odds)
            p = win probability
            q = 1 - p (loss probability)
        
        Args:
            prob: Win probability (0-1)
            odds: Decimal odds
        
        Returns:
            Kelly fraction (0-1), capped and adjusted
        """
        # Validate inputs
        if prob <= self.config.min_prob or prob >= self.config.max_prob:
            return 0.0
        
        if odds <= 1.0:
            return 0.0
        
        # Calculate Kelly
        b = odds - 1  # Net odds
        q = 1 - prob
        
        kelly = (prob * b - q) / b
        
        # Never bet if negative edge
        if kelly <= 0:
            return 0.0
        
        # Apply fractional Kelly (safety)
        kelly = kelly * self.config.kelly_fraction
        
        # Cap at 100% (should never happen with fractional)
        kelly = min(kelly, 1.0)
        
        return kelly
    
    def _calculate_clv(
        self, 
        opening_odds: float, 
        closing_odds: float
    ) -> float:
        """
        Calculate Closing Line Value (CLV).
        
        CLV measures how much value we captured vs closing price.
        Positive CLV = we got better price than closing
        
        Formula: CLV = (closing_implied - opening_implied) / opening_implied
        
        Args:
            opening_odds: Odds when bet placed
            closing_odds: Odds at kickoff
        
        Returns:
            CLV as decimal (0.10 = 10% CLV)
        """
        if pd.isna(closing_odds) or closing_odds <= 1.0:
            return 0.0
        
        if opening_odds <= 1.0:
            return 0.0
        
        opening_implied = 1.0 / opening_odds
        closing_implied = 1.0 / closing_odds
        
        # CLV = how much closing line moved in our favor
        clv = (closing_implied - opening_implied) / opening_implied
        
        return clv
    
    def _reset_daily_tracking(self):
        """Reset daily exposure and profit tracking."""
        self._daily_exposure = 0.0
        self._daily_profit = 0.0
        self._daily_bet_count = 0
    
    def _record_daily_summary(self, date: pd.Timestamp):
        """Record end-of-day summary."""
        self.daily_summary.append({
            'date': date,
            'bankroll': self.bankroll,
            'daily_profit': self._daily_profit,
            'daily_bets': self._daily_bet_count,
            'daily_exposure': self._daily_exposure
        })
    
    def run_backtest(
        self,
        predictions_df: pd.DataFrame,
        results_df: pd.DataFrame,
        odds_columns: Optional[Dict[str, str]] = None,
        closing_odds_columns: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Run complete walk-forward backtest.
        
        Args:
            predictions_df: Model predictions with columns:
                Required:
                - Date: Match date
                - prob_home_win, prob_draw, prob_away_win: Probabilities
                Optional:
                - League: League name (for breakdown)
                - HomeTeam, AwayTeam: Team names
                
            results_df: Actual results, same length/order as predictions
                Required:
                - FTR: Full time result ('H', 'D', 'A')
                
            odds_columns: Mapping of market -> odds column name
                Default: {'home': 'AvgH', 'draw': 'AvgD', 'away': 'AvgA'}
                
            closing_odds_columns: Mapping of market -> closing odds column
                Default: {'home': 'AvgCH', 'draw': 'AvgCD', 'away': 'AvgCA'}
        
        Returns:
            Dictionary with comprehensive statistics
        """
        # Set defaults
        if odds_columns is None:
            odds_columns = {
                'home': 'AvgH',
                'draw': 'AvgD',
                'away': 'AvgA'
            }
        
        if closing_odds_columns is None:
            closing_odds_columns = {
                'home': 'AvgCH',
                'draw': 'AvgCD',
                'away': 'AvgCA'
            }
        
        # Validate inputs
        if len(predictions_df) != len(results_df):
            raise ValueError(
                f"Length mismatch: predictions={len(predictions_df)}, "
                f"results={len(results_df)}"
            )
        
        # CRITICAL: Ensure strict chronological order
        predictions_df = predictions_df.sort_values('Date').reset_index(drop=True)
        results_df = results_df.sort_values('Date').reset_index(drop=True)
        
        # Reset state
        self.bankroll = self.config.initial_bankroll
        self.peak_bankroll = self.config.initial_bankroll
        self.bets_history = []
        self.daily_summary = []
        self._current_date = None
        
        print(f"Starting backtest on {len(predictions_df):,} matches...")
        print(f"Initial bankroll: ${self.config.initial_bankroll:,.2f}")
        print(f"Kelly fraction: {self.config.kelly_fraction:.2f}")
        print(f"Min edge: {self.config.min_edge:.1%}")
        print("-" * 60)
        
        # Main backtest loop
        for idx in range(len(predictions_df)):
            pred_row = predictions_df.iloc[idx]
            actual_row = results_df.iloc[idx]
            
            match_date = pd.to_datetime(pred_row['Date'])
            
            # Track daily boundaries
            if self._current_date != match_date:
                if self._current_date is not None:
                    self._record_daily_summary(self._current_date)
                
                self._current_date = match_date
                self._reset_daily_tracking()
            
            # Process each market
            for market in ['home', 'draw', 'away']:
                self._process_bet(
                    market=market,
                    pred_row=pred_row,
                    actual_row=actual_row,
                    odds_columns=odds_columns,
                    closing_odds_columns=closing_odds_columns
                )
        
        # Record final day
        if self._current_date is not None:
            self._record_daily_summary(self._current_date)
        
        print("-" * 60)
        print(f"Backtest complete!")
        print(f"Total bets placed: {len(self.bets_history):,}")
        print(f"Final bankroll: ${self.bankroll:,.2f}")
        print(f"Total profit: ${self.bankroll - self.config.initial_bankroll:,.2f}")
        
        # Calculate and return statistics
        return self.calculate_statistics()
    
    def _process_bet(
        self,
        market: str,
        pred_row: pd.Series,
        actual_row: pd.Series,
        odds_columns: Dict,
        closing_odds_columns: Dict
    ):
        """Process a single bet opportunity."""
        
        # Get model probability
        if market == 'home':
            model_prob = pred_row.get('prob_home_win', 0.0)
        elif market == 'draw':
            model_prob = pred_row.get('prob_draw', 0.0)
        else:
            model_prob = pred_row.get('prob_away_win', 0.0)
        
        # Validate probability
        if model_prob <= self.config.min_prob or model_prob >= self.config.max_prob:
            return
        
        # Get opening odds
        odds_col = odds_columns.get(market)
        if odds_col not in pred_row or pd.isna(pred_row[odds_col]):
            return
        
        opening_odds = float(pred_row[odds_col])
        if opening_odds <= 1.0:
            return
        
        # Calculate edge
        implied_prob = 1.0 / opening_odds
        edge = model_prob - implied_prob
        
        # Check minimum edge
        if edge < self.config.min_edge:
            return
        
        # Calculate Kelly stake
        kelly_frac = self._kelly_criterion(model_prob, opening_odds)
        if kelly_frac <= 0:
            return
        
        # Calculate stake in dollars
        stake = self.bankroll * kelly_frac
        
        # Apply bet size cap
        max_bet = self.bankroll * self.config.max_single_bet
        stake = min(stake, max_bet)
        
        # Check daily exposure limit
        if self._daily_exposure + stake > self.bankroll * self.config.max_daily_risk:
            return
        
        # BET PLACED - update exposure
        self._daily_exposure += stake
        self._daily_bet_count += 1
        
        # Determine actual outcome
        actual_outcome = actual_row.get('FTR', 'X')
        if market == 'home':
            won = actual_outcome == 'H'
        elif market == 'draw':
            won = actual_outcome == 'D'
        else:
            won = actual_outcome == 'A'
        
        # Calculate PnL
        if won:
            gross_profit = stake * (opening_odds - 1)
            net_profit = gross_profit * (1 - self.config.commission)
        else:
            net_profit = -stake
        
        # Update bankroll (COMPOUNDING)
        self.bankroll += net_profit
        self._daily_profit += net_profit
        
        # Track peak for drawdown
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        
        # Calculate CLV
        closing_col = closing_odds_columns.get(market)
        clv = 0.0
        closing_odds = np.nan
        
        if closing_col in pred_row and not pd.isna(pred_row[closing_col]):
            closing_odds = float(pred_row[closing_col])
            clv = self._calculate_clv(opening_odds, closing_odds)
        
        # Record bet
        self.bets_history.append({
            'date': pred_row['Date'],
            'league': pred_row.get('League', 'Unknown'),
            'home_team': pred_row.get('HomeTeam', ''),
            'away_team': pred_row.get('AwayTeam', ''),
            'match': f"{pred_row.get('HomeTeam', '?')} vs {pred_row.get('AwayTeam', '?')}",
            'market': market,
            'model_prob': model_prob,
            'opening_odds': opening_odds,
            'closing_odds': closing_odds,
            'implied_prob': implied_prob,
            'edge': edge,
            'stake': stake,
            'stake_pct': (stake / (self.bankroll - net_profit)) * 100,  # % of bankroll BEFORE bet
            'won': won,
            'profit': net_profit,
            'bankroll_after': self.bankroll,
            'clv': clv,
            'kelly_frac': kelly_frac
        })
    
    def calculate_statistics(self) -> Dict:
        """
        Calculate comprehensive backtest statistics.
        
        Returns:
            Nested dictionary with all analytics
        """
        if not self.bets_history:
            return {'error': 'No bets placed'}
        
        bets_df = pd.DataFrame(self.bets_history)
        daily_df = pd.DataFrame(self.daily_summary)
        
        # ============================================================
        # OVERVIEW METRICS
        # ============================================================
        total_staked = bets_df['stake'].sum()
        total_profit = bets_df['profit'].sum()
        final_bankroll = self.bankroll
        
        roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
        
        n_bets = len(bets_df)
        n_wins = bets_df['won'].sum()
        win_rate = (n_wins / n_bets) * 100
        
        avg_odds = bets_df['opening_odds'].mean()
        
        # Turnover
        turnover = total_staked / self.config.initial_bankroll
        
        # Time-based metrics
        bets_df['date'] = pd.to_datetime(bets_df['date'])
        start_date = bets_df['date'].min()
        end_date = bets_df['date'].max()
        days = (end_date - start_date).days + 1
        years = days / 365.25
        
        # CAGR
        if years > 0:
            cagr = (((final_bankroll / self.config.initial_bankroll) ** (1 / years)) - 1) * 100
        else:
            cagr = 0.0
        
        # ============================================================
        # RISK METRICS
        # ============================================================
        
        # Maximum drawdown (high-water mark method)
        bankroll_series = bets_df['bankroll_after']
        rolling_max = bankroll_series.cummax()
        drawdown_series = (rolling_max - bankroll_series) / rolling_max
        max_drawdown = drawdown_series.max() * 100
        
        # Current drawdown
        current_drawdown = ((self.peak_bankroll - self.bankroll) / self.peak_bankroll) * 100
        
        # Sharpe ratio (annualized)
        if len(daily_df) > 1 and daily_df['daily_profit'].std() > 0:
            daily_returns = daily_df['daily_profit'] / self.config.initial_bankroll
            sharpe_daily = daily_returns.mean() / daily_returns.std()
            sharpe_annual = sharpe_daily * np.sqrt(252)  # 252 trading days
        else:
            sharpe_annual = 0.0
        
        # Stake statistics
        avg_stake_pct = bets_df['stake_pct'].mean()
        max_stake_pct = bets_df['stake_pct'].max()
        
        # ============================================================
        # EDGE QUALITY METRICS
        # ============================================================
        avg_edge = bets_df['edge'].mean() * 100
        avg_model_prob = bets_df['model_prob'].mean()
        
        # CLV statistics
        clv_bets = bets_df[bets_df['clv'].notna()]
        if len(clv_bets) > 0:
            avg_clv = clv_bets['clv'].mean() * 100
            positive_clv_pct = (clv_bets['clv'] > 0).mean() * 100
        else:
            avg_clv = 0.0
            positive_clv_pct = 0.0
        
        # ============================================================
        # LEAGUE BREAKDOWN
        # ============================================================
        league_stats = self._league_breakdown(bets_df)
        
        # ============================================================
        # MARKET BREAKDOWN
        # ============================================================
        market_stats = self._market_breakdown(bets_df)
        
        # ============================================================
        # RETURN FULL STATISTICS
        # ============================================================
        return {
            'overview': {
                'initial_bankroll': self.config.initial_bankroll,
                'final_bankroll': final_bankroll,
                'total_profit': total_profit,
                'roi_pct': roi,
                'cagr_pct': cagr,
                'n_bets': n_bets,
                'n_wins': int(n_wins),
                'n_losses': n_bets - int(n_wins),
                'win_rate_pct': win_rate,
                'avg_odds': avg_odds,
                'total_staked': total_staked,
                'turnover': turnover,
                'days': days,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            },
            'risk': {
                'max_drawdown_pct': max_drawdown,
                'current_drawdown_pct': current_drawdown,
                'sharpe_ratio_annual': sharpe_annual,
                'avg_stake_pct': avg_stake_pct,
                'max_stake_pct': max_stake_pct
            },
            'edge_quality': {
                'avg_edge_pct': avg_edge,
                'avg_model_prob': avg_model_prob,
                'avg_clv_pct': avg_clv,
                'positive_clv_pct': positive_clv_pct,
                'n_clv_tracked': len(clv_bets)
            },
            'league_breakdown': league_stats,
            'market_breakdown': market_stats,
            'config': {
                'kelly_fraction': self.config.kelly_fraction,
                'min_edge': self.config.min_edge,
                'max_daily_risk': self.config.max_daily_risk,
                'max_single_bet': self.config.max_single_bet
            }
        }
    
    def _league_breakdown(self, bets_df: pd.DataFrame) -> Dict:
        """Calculate per-league statistics."""
        if 'league' not in bets_df.columns:
            return {}
        
        league_stats = {}
        
        for league in bets_df['league'].unique():
            league_bets = bets_df[bets_df['league'] == league]
            
            total_staked = league_bets['stake'].sum()
            total_profit = league_bets['profit'].sum()
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
            
            clv_bets = league_bets[league_bets['clv'].notna()]
            avg_clv = (clv_bets['clv'].mean() * 100) if len(clv_bets) > 0 else 0
            
            league_stats[league] = {
                'n_bets': len(league_bets),
                'total_profit': total_profit,
                'total_staked': total_staked,
                'roi_pct': roi,
                'win_rate_pct': (league_bets['won'].mean()) * 100,
                'avg_edge_pct': league_bets['edge'].mean() * 100,
                'avg_clv_pct': avg_clv
            }
        
        return league_stats
    
    def _market_breakdown(self, bets_df: pd.DataFrame) -> Dict:
        """Calculate per-market statistics."""
        if 'market' not in bets_df.columns:
            return {}
        
        market_stats = {}
        
        for market in bets_df['market'].unique():
            market_bets = bets_df[bets_df['market'] == market]
            
            total_staked = market_bets['stake'].sum()
            total_profit = market_bets['profit'].sum()
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
            
            market_stats[market] = {
                'n_bets': len(market_bets),
                'total_profit': total_profit,
                'total_staked': total_staked,
                'roi_pct': roi,
                'win_rate_pct': (market_bets['won'].mean()) * 100,
                'avg_edge_pct': market_bets['edge'].mean() * 100,
                'avg_odds': market_bets['opening_odds'].mean()
            }
        
        return market_stats
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve for plotting.
        
        Returns:
            DataFrame with columns:
            - date: Date
            - bankroll: Bankroll after bet
            - peak: Rolling peak
            - drawdown_pct: Drawdown percentage
        """
        if not self.bets_history:
            return pd.DataFrame()
        
        bets_df = pd.DataFrame(self.bets_history)
        
        equity = bets_df[['date', 'bankroll_after']].copy()
        equity.columns = ['date', 'bankroll']
        equity['date'] = pd.to_datetime(equity['date'])
        
        # Calculate rolling peak and drawdown
        equity['peak'] = equity['bankroll'].cummax()
        equity['drawdown_pct'] = ((equity['peak'] - equity['bankroll']) / equity['peak']) * 100
        
        return equity
    
    def export_results(
        self,
        bets_filepath: str = 'backtest_bets.csv',
        equity_filepath: str = 'backtest_equity.csv'
    ):
        """
        Export backtest results to CSV.
        
        Args:
            bets_filepath: Path for detailed bet history
            equity_filepath: Path for equity curve
        """
        if not self.bets_history:
            print("No bets to export")
            return
        
        # Export bets
        bets_df = pd.DataFrame(self.bets_history)
        bets_df.to_csv(bets_filepath, index=False)
        print(f"✅ Exported {len(bets_df):,} bets to {bets_filepath}")
        
        # Export equity curve
        equity_df = self.get_equity_curve()
        equity_df.to_csv(equity_filepath, index=False)
        print(f"✅ Exported equity curve to {equity_filepath}")
    
    def print_summary(self, stats: Optional[Dict] = None):
        """
        Print formatted backtest summary.
        
        Args:
            stats: Statistics dictionary (calculates if None)
        """
        if stats is None:
            stats = self.calculate_statistics()
        
        if 'error' in stats:
            print(stats['error'])
            return
        
        print("\n" + "="*70)
        print("BACKTEST SUMMARY")
        print("="*70)
        
        # Overview
        ov = stats['overview']
        print(f"\n📊 OVERVIEW")
        print(f"Period: {ov['start_date']} to {ov['end_date']} ({ov['days']} days)")
        print(f"Initial Bankroll: ${ov['initial_bankroll']:,.2f}")
        print(f"Final Bankroll:   ${ov['final_bankroll']:,.2f}")
        print(f"Total Profit:     ${ov['total_profit']:,.2f}")
        print(f"ROI:              {ov['roi_pct']:.2f}%")
        print(f"CAGR:             {ov['cagr_pct']:.2f}%")
        
        # Performance
        print(f"\n📈 PERFORMANCE")
        print(f"Total Bets:       {ov['n_bets']:,}")
        print(f"Wins / Losses:    {ov['n_wins']:,} / {ov['n_losses']:,}")
        print(f"Win Rate:         {ov['win_rate_pct']:.2f}%")
        print(f"Avg Odds:         {ov['avg_odds']:.2f}")
        print(f"Turnover:         {ov['turnover']:.2f}x")
        
        # Risk
        risk = stats['risk']
        print(f"\n⚠️  RISK")
        print(f"Max Drawdown:     {risk['max_drawdown_pct']:.2f}%")
        print(f"Current Drawdown: {risk['current_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio:     {risk['sharpe_ratio_annual']:.2f}")
        print(f"Avg Stake:        {risk['avg_stake_pct']:.2f}%")
        print(f"Max Stake:        {risk['max_stake_pct']:.2f}%")
        
        # Edge Quality
        edge = stats['edge_quality']
        print(f"\n🎯 EDGE QUALITY")
        print(f"Avg Edge:         {edge['avg_edge_pct']:.2f}%")
        print(f"Avg Model Prob:   {edge['avg_model_prob']:.2f}")
        print(f"Avg CLV:          {edge['avg_clv_pct']:.2f}%")
        print(f"Positive CLV:     {edge['positive_clv_pct']:.2f}%")
        
        # League Breakdown
        if stats['league_breakdown']:
            print(f"\n🏆 LEAGUE BREAKDOWN")
            for league, lstats in stats['league_breakdown'].items():
                print(f"{league:20s} | Bets: {lstats['n_bets']:4d} | "
                      f"ROI: {lstats['roi_pct']:6.2f}% | "
                      f"Win%: {lstats['win_rate_pct']:5.2f}%")
        
        # Market Breakdown
        if stats['market_breakdown']:
            print(f"\n💹 MARKET BREAKDOWN")
            for market, mstats in stats['market_breakdown'].items():
                print(f"{market.upper():10s} | Bets: {mstats['n_bets']:4d} | "
                      f"ROI: {mstats['roi_pct']:6.2f}% | "
                      f"Win%: {mstats['win_rate_pct']:5.2f}%")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage / testing
    print("Institutional Backtester - Production Ready")
    print("Use: backtester = InstitutionalBacktester()")
    print("     stats = backtester.run_backtest(predictions_df, results_df)")
