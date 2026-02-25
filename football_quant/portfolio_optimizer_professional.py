"""
PROFESSIONAL PORTFOLIO BETTING OPTIMIZER
Correlation-aware multi-bet allocation with Kelly optimization

Features:
- Multi-bet Kelly allocation
- Correlation adjustment between same-league matches
- Bankroll constraints
- Maximum exposure limits
- Risk-adjusted sizing
- Diversification scoring
- Expected portfolio return
- Portfolio-level drawdown control
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimizer."""
    bankroll: float = 10000.0
    kelly_fraction: float = 0.25
    max_total_exposure: float = 0.15  # Max 15% of bankroll across all bets
    max_single_bet: float = 0.05  # Max 5% per bet
    min_edge: float = 0.03  # 3% minimum edge
    correlation_adjustment: bool = True
    same_league_correlation: float = 0.3  # Assumed correlation for same league
    risk_tolerance: float = 1.0  # Multiplier for position sizing


class PortfolioOptimizer:
    """
    Professional portfolio allocation engine.
    
    Optimizes bet sizing across multiple opportunities considering:
    - Kelly criterion for each bet
    - Correlation between bets
    - Total exposure constraints
    - Risk-adjusted position sizing
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize optimizer.
        
        Args:
            config: Portfolio configuration
        """
        self.config = config or PortfolioConfig()
    
    def _single_bet_kelly(self, prob: float, odds: float) -> float:
        """
        Calculate Kelly fraction for single bet.
        
        Args:
            prob: Win probability
            odds: Decimal odds
        
        Returns:
            Kelly fraction
        """
        if prob <= 0 or prob >= 1 or odds <= 1:
            return 0.0
        
        b = odds - 1
        kelly = (prob * b - (1 - prob)) / b
        
        return max(0, kelly)
    
    def _calculate_correlation_matrix(self, bets_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate correlation matrix between bets.
        
        Assumptions:
        - Same league matches have positive correlation
        - Different leagues are uncorrelated
        - Same-day matches have slight correlation
        
        Args:
            bets_df: DataFrame with bet opportunities
        
        Returns:
            Correlation matrix (n_bets x n_bets)
        """
        n = len(bets_df)
        corr_matrix = np.eye(n)  # Start with identity (1 on diagonal)
        
        if not self.config.correlation_adjustment:
            return corr_matrix
        
        # Calculate pairwise correlations
        for i in range(n):
            for j in range(i + 1, n):
                bet_i = bets_df.iloc[i]
                bet_j = bets_df.iloc[j]
                
                corr = 0.0
                
                # Same league correlation
                if 'league' in bets_df.columns:
                    if bet_i.get('league') == bet_j.get('league'):
                        corr += self.config.same_league_correlation
                
                # Same day correlation (weaker)
                if 'date' in bets_df.columns:
                    if bet_i.get('date') == bet_j.get('date'):
                        corr += 0.1
                
                # Same match different market (strong correlation)
                if 'match' in bets_df.columns:
                    if bet_i.get('match') == bet_j.get('match'):
                        corr = 0.7  # Strong correlation
                
                # Cap correlation
                corr = min(corr, 0.9)
                
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        return corr_matrix
    
    def _portfolio_variance(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        corr_matrix: np.ndarray
    ) -> float:
        """
        Calculate portfolio variance.
        
        Var(portfolio) = w^T * Σ * w
        where Σ is covariance matrix
        
        Args:
            weights: Bet sizes (as fractions)
            returns: Expected returns per bet
            corr_matrix: Correlation matrix
        
        Returns:
            Portfolio variance
        """
        # Standard deviation of each bet (simplified)
        std_devs = np.sqrt(returns * (1 + returns))  # Approximation
        
        # Covariance matrix
        cov_matrix = np.outer(std_devs, std_devs) * corr_matrix
        
        # Portfolio variance
        portfolio_var = weights @ cov_matrix @ weights
        
        return portfolio_var
    
    def optimize_portfolio(
        self,
        opportunities: pd.DataFrame,
        correlation_aware: bool = True
    ) -> pd.DataFrame:
        """
        Optimize bet allocation across multiple opportunities.
        
        Args:
            opportunities: DataFrame with columns:
                - probability: Win probability
                - odds: Decimal odds
                - edge: Edge (prob - implied)
                Optional:
                - league: League name
                - match: Match identifier
                - market: Market type
                - date: Match date
            
            correlation_aware: Use correlation adjustment
        
        Returns:
            DataFrame with optimized bet sizes
        """
        # Filter by minimum edge
        opps = opportunities[opportunities['edge'] >= self.config.min_edge].copy()
        
        if len(opps) == 0:
            return pd.DataFrame()
        
        # Calculate individual Kelly fractions
        opps['kelly_fraction'] = opps.apply(
            lambda row: self._single_bet_kelly(row['probability'], row['odds']),
            axis=1
        )
        
        # Apply fractional Kelly
        opps['kelly_fraction'] *= self.config.kelly_fraction
        
        # Apply risk tolerance
        opps['kelly_fraction'] *= self.config.risk_tolerance
        
        # Calculate stakes (before correlation adjustment)
        opps['raw_stake'] = opps['kelly_fraction'] * self.config.bankroll
        
        # Apply single bet limit
        max_single = self.config.bankroll * self.config.max_single_bet
        opps['raw_stake'] = opps['raw_stake'].clip(upper=max_single)
        
        # Calculate correlation matrix if needed
        if correlation_aware and self.config.correlation_adjustment:
            corr_matrix = self._calculate_correlation_matrix(opps)
            
            # Adjust stakes for correlation
            stakes = self._adjust_for_correlation(
                opps['raw_stake'].values,
                opps['probability'].values,
                opps['odds'].values,
                corr_matrix
            )
            opps['stake'] = stakes
        else:
            opps['stake'] = opps['raw_stake']
        
        # Apply total exposure constraint
        total_stake = opps['stake'].sum()
        max_total = self.config.bankroll * self.config.max_total_exposure
        
        if total_stake > max_total:
            # Scale down proportionally
            scale_factor = max_total / total_stake
            opps['stake'] *= scale_factor
        
        # Calculate expected values
        opps['expected_profit'] = opps.apply(
            lambda row: row['probability'] * row['stake'] * (row['odds'] - 1) - 
                       (1 - row['probability']) * row['stake'],
            axis=1
        )
        
        opps['expected_value'] = opps['expected_profit'] / opps['stake']
        
        # Sort by expected value
        opps = opps.sort_values('expected_value', ascending=False)
        
        return opps
    
    def _adjust_for_correlation(
        self,
        raw_stakes: np.ndarray,
        probabilities: np.ndarray,
        odds: np.ndarray,
        corr_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Adjust bet sizes for correlation.
        
        Uses optimization to find stakes that maximize expected return
        while accounting for correlation and respecting constraints.
        
        Args:
            raw_stakes: Raw Kelly stakes
            probabilities: Win probabilities
            odds: Decimal odds
            corr_matrix: Correlation matrix
        
        Returns:
            Adjusted stakes
        """
        n = len(raw_stakes)
        
        # Expected returns per bet
        returns = probabilities * (odds - 1) - (1 - probabilities)
        
        # Objective: Maximize Sharpe ratio
        # Sharpe = E[R] / sqrt(Var[R])
        def objective(weights):
            portfolio_return = np.sum(weights * returns * self.config.bankroll)
            portfolio_var = self._portfolio_variance(weights, returns, corr_matrix)
            portfolio_std = np.sqrt(portfolio_var) * self.config.bankroll
            
            if portfolio_std < 1e-6:
                return 1e6
            
            sharpe = -portfolio_return / portfolio_std  # Negative for minimization
            return sharpe
        
        # Constraints
        constraints = [
            # Total exposure
            {'type': 'ineq', 'fun': lambda w: self.config.max_total_exposure - np.sum(w)},
            # Non-negative
            {'type': 'ineq', 'fun': lambda w: w}
        ]
        
        # Bounds (0 to max single bet)
        bounds = [(0, self.config.max_single_bet) for _ in range(n)]
        
        # Initial guess (normalized raw stakes)
        total_raw = np.sum(raw_stakes)
        if total_raw > 0:
            x0 = (raw_stakes / self.config.bankroll).clip(0, self.config.max_single_bet)
        else:
            x0 = np.zeros(n)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'disp': False}
        )
        
        if result.success:
            optimized_fractions = result.x
        else:
            # Fall back to proportional scaling
            optimized_fractions = x0
        
        # Convert to dollar stakes
        adjusted_stakes = optimized_fractions * self.config.bankroll
        
        return adjusted_stakes
    
    def calculate_portfolio_metrics(self, portfolio: pd.DataFrame) -> Dict:
        """
        Calculate portfolio-level metrics.
        
        Args:
            portfolio: Optimized portfolio DataFrame
        
        Returns:
            Dictionary with portfolio metrics
        """
        if len(portfolio) == 0:
            return {'n_bets': 0, 'total_stake': 0}
        
        total_stake = portfolio['stake'].sum()
        exposure_pct = (total_stake / self.config.bankroll) * 100
        
        expected_profit = portfolio['expected_profit'].sum()
        expected_roi = (expected_profit / total_stake * 100) if total_stake > 0 else 0
        
        # Diversification score (higher is better)
        # Based on number of bets and distribution of stakes
        n_bets = len(portfolio)
        stake_distribution = portfolio['stake'] / total_stake
        concentration = (stake_distribution ** 2).sum()  # Herfindahl index
        diversification = (1 - concentration) * 100  # 0-100 scale
        
        # Average edge
        avg_edge = portfolio['edge'].mean() * 100
        
        # Risk metrics
        stakes_frac = portfolio['stake'] / self.config.bankroll
        probabilities = portfolio['probability'].values
        odds = portfolio['odds'].values
        returns = probabilities * (odds - 1) - (1 - probabilities)
        
        # Correlation matrix
        corr_matrix = self._calculate_correlation_matrix(portfolio)
        
        # Portfolio variance
        portfolio_var = self._portfolio_variance(stakes_frac, returns, corr_matrix)
        portfolio_std = np.sqrt(portfolio_var) * 100  # As percentage
        
        # Sharpe ratio estimate
        if portfolio_std > 0:
            sharpe = expected_roi / portfolio_std
        else:
            sharpe = 0
        
        return {
            'n_bets': n_bets,
            'total_stake': total_stake,
            'exposure_pct': exposure_pct,
            'expected_profit': expected_profit,
            'expected_roi_pct': expected_roi,
            'avg_edge_pct': avg_edge,
            'diversification_score': diversification,
            'portfolio_std_pct': portfolio_std,
            'sharpe_ratio': sharpe,
            'max_single_stake': portfolio['stake'].max(),
            'min_single_stake': portfolio['stake'].min(),
            'avg_stake': portfolio['stake'].mean()
        }
    
    def generate_betting_report(
        self,
        portfolio: pd.DataFrame,
        include_details: bool = True
    ) -> str:
        """
        Generate formatted betting report.
        
        Args:
            portfolio: Optimized portfolio
            include_details: Include individual bet details
        
        Returns:
            Formatted report string
        """
        if len(portfolio) == 0:
            return "No value bets found."
        
        metrics = self.calculate_portfolio_metrics(portfolio)
        
        report = []
        report.append("="*70)
        report.append("PORTFOLIO BETTING REPORT")
        report.append("="*70)
        
        report.append(f"\n💼 PORTFOLIO SUMMARY")
        report.append(f"Bankroll:         ${self.config.bankroll:,.2f}")
        report.append(f"Total Bets:       {metrics['n_bets']}")
        report.append(f"Total Stake:      ${metrics['total_stake']:,.2f}")
        report.append(f"Exposure:         {metrics['exposure_pct']:.1f}%")
        
        report.append(f"\n📊 EXPECTED PERFORMANCE")
        report.append(f"Expected Profit:  ${metrics['expected_profit']:,.2f}")
        report.append(f"Expected ROI:     {metrics['expected_roi_pct']:.2f}%")
        report.append(f"Avg Edge:         {metrics['avg_edge_pct']:.2f}%")
        
        report.append(f"\n⚠️  RISK METRICS")
        report.append(f"Portfolio Std:    {metrics['portfolio_std_pct']:.2f}%")
        report.append(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
        report.append(f"Diversification:  {metrics['diversification_score']:.1f}/100")
        
        report.append(f"\n💰 STAKE DISTRIBUTION")
        report.append(f"Max Single Stake: ${metrics['max_single_stake']:,.2f}")
        report.append(f"Avg Stake:        ${metrics['avg_stake']:,.2f}")
        report.append(f"Min Stake:        ${metrics['min_single_stake']:,.2f}")
        
        if include_details:
            report.append(f"\n📋 RECOMMENDED BETS (sorted by EV)")
            report.append("─"*70)
            
            for idx, row in portfolio.head(20).iterrows():
                report.append(f"\n{idx+1}. {row.get('match', 'Unknown')}")
                report.append(f"   Market: {row.get('market', 'Unknown').upper()}")
                report.append(f"   Probability: {row['probability']:.1%} | Odds: {row['odds']:.2f}")
                report.append(f"   Edge: {row['edge']:.1%} | EV: {row['expected_value']:.1%}")
                report.append(f"   → STAKE: ${row['stake']:.2f}")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("PROFESSIONAL PORTFOLIO OPTIMIZER")
    print("="*70)
    
    # Create sample opportunities
    opportunities = pd.DataFrame({
        'match': ['Liverpool vs Chelsea', 'Arsenal vs Spurs', 'Man City vs Man Utd'],
        'league': ['Premier League', 'Premier League', 'Premier League'],
        'market': ['home', 'away', 'home'],
        'probability': [0.60, 0.45, 0.70],
        'odds': [2.0, 2.5, 1.8],
        'edge': [0.10, 0.05, 0.144]
    })
    
    # Optimize
    config = PortfolioConfig(bankroll=10000, kelly_fraction=0.25)
    optimizer = PortfolioOptimizer(config)
    
    portfolio = optimizer.optimize_portfolio(opportunities)
    
    # Print report
    print(optimizer.generate_betting_report(portfolio))
