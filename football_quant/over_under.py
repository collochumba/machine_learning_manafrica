"""
INSTITUTIONAL OVER/UNDER PRICING ENGINE
Exact Poisson probability calculation for total goals markets

Features:
- Exact convolution (no Monte Carlo)
- All standard lines (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
- Corner markets (8.5, 9.5, 10.5, 11.5, 12.5)
- Cards markets
- Custom threshold support
- Fair odds calculation
- Expected value computation
"""

import numpy as np
from scipy.stats import poisson
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class OverUnderPredictor:
    """
    Professional Over/Under market predictor.
    
    Uses exact Poisson convolution to calculate probabilities
    for total goals/corners/cards markets.
    """
    
    def __init__(self, max_goals: int = 15):
        """
        Args:
            max_goals: Maximum goals to consider (15 covers 99.99%+)
        """
        self.max_goals = max_goals
        
        # Standard thresholds
        self.goals_thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        self.corners_thresholds = [8.5, 9.5, 10.5, 11.5, 12.5]
        self.cards_thresholds = [3.5, 4.5, 5.5, 6.5]
    
    def predict_from_lambdas(
        self,
        lambda_home: float,
        lambda_away: float,
        thresholds: Optional[List[float]] = None
    ) -> Dict:
        """
        Calculate over/under probabilities from Poisson parameters.
        
        Uses exact convolution of two independent Poisson distributions.
        
        Args:
            lambda_home: Expected home goals
            lambda_away: Expected away goals
            thresholds: Custom thresholds (uses defaults if None)
        
        Returns:
            Dictionary with probabilities for each threshold
        """
        thresholds = thresholds or self.goals_thresholds
        
        # Calculate probability distribution for total goals
        total_probs = self._calculate_total_distribution(lambda_home, lambda_away)
        
        results = {}
        
        for threshold in thresholds:
            # P(Total <= threshold)
            prob_under = self._probability_at_most(total_probs, threshold)
            prob_over = 1 - prob_under
            
            # Fair odds
            fair_over_odds = 1 / prob_over if prob_over > 0 else 999.0
            fair_under_odds = 1 / prob_under if prob_under > 0 else 999.0
            
            results[f'over_{threshold}'] = prob_over
            results[f'under_{threshold}'] = prob_under
            results[f'fair_over_{threshold}_odds'] = fair_over_odds
            results[f'fair_under_{threshold}_odds'] = fair_under_odds
        
        # Add expected total
        expected_total = lambda_home + lambda_away
        results['expected_total'] = expected_total
        
        return results
    
    def _calculate_total_distribution(
        self,
        lambda_1: float,
        lambda_2: float
    ) -> np.ndarray:
        """
        Calculate probability distribution for sum of two Poisson variables.
        
        P(X+Y = k) = sum_{i=0}^{k} P(X=i) * P(Y=k-i)
        
        Args:
            lambda_1: First Poisson parameter
            lambda_2: Second Poisson parameter
        
        Returns:
            Array where index k = P(Total = k)
        """
        max_total = 2 * self.max_goals
        total_probs = np.zeros(max_total)
        
        # Convolution
        for total in range(max_total):
            prob = 0.0
            for i in range(min(total + 1, self.max_goals)):
                j = total - i
                if j < self.max_goals:
                    prob += poisson.pmf(i, lambda_1) * poisson.pmf(j, lambda_2)
            total_probs[total] = prob
        
        return total_probs
    
    def _probability_at_most(
        self,
        distribution: np.ndarray,
        threshold: float
    ) -> float:
        """
        Calculate P(X <= threshold).
        
        Args:
            distribution: Probability distribution
            threshold: Upper bound
        
        Returns:
            Cumulative probability
        """
        # Include all values <= threshold
        max_value = int(np.floor(threshold))
        prob = np.sum(distribution[:max_value + 1])
        
        return prob
    
    def predict_corners(
        self,
        avg_home_corners: float,
        avg_away_corners: float,
        thresholds: Optional[List[float]] = None
    ) -> Dict:
        """
        Predict corner over/unders.
        
        Args:
            avg_home_corners: Expected home corners
            avg_away_corners: Expected away corners
            thresholds: Custom thresholds
        
        Returns:
            Dictionary with corner probabilities
        """
        thresholds = thresholds or self.corners_thresholds
        
        return self.predict_from_lambdas(
            avg_home_corners,
            avg_away_corners,
            thresholds=thresholds
        )
    
    def predict_cards(
        self,
        avg_home_cards: float,
        avg_away_cards: float,
        thresholds: Optional[List[float]] = None
    ) -> Dict:
        """
        Predict cards over/unders.
        
        Args:
            avg_home_cards: Expected home cards
            avg_away_cards: Expected away cards
            thresholds: Custom thresholds
        
        Returns:
            Dictionary with cards probabilities
        """
        thresholds = thresholds or self.cards_thresholds
        
        return self.predict_from_lambdas(
            avg_home_cards,
            avg_away_cards,
            thresholds=thresholds
        )
    
    def calculate_expected_value(
        self,
        lambda_home: float,
        lambda_away: float,
        threshold: float,
        bookmaker_over_odds: float,
        bookmaker_under_odds: float,
        side: str = 'over'
    ) -> Dict:
        """
        Calculate expected value for over/under bet.
        
        Args:
            lambda_home: Expected home goals
            lambda_away: Expected away goals
            threshold: O/U line
            bookmaker_over_odds: Bookmaker's over odds
            bookmaker_under_odds: Bookmaker's under odds
            side: 'over' or 'under'
        
        Returns:
            Dictionary with EV analysis
        """
        # Get probabilities
        result = self.predict_from_lambdas(lambda_home, lambda_away, [threshold])
        
        if side.lower() == 'over':
            model_prob = result[f'over_{threshold}']
            bookmaker_odds = bookmaker_over_odds
            fair_odds = result[f'fair_over_{threshold}_odds']
        else:
            model_prob = result[f'under_{threshold}']
            bookmaker_odds = bookmaker_under_odds
            fair_odds = result[f'fair_under_{threshold}_odds']
        
        # EV calculation
        ev = model_prob * (bookmaker_odds - 1) - (1 - model_prob)
        ev_pct = ev * 100
        
        # Implied probability
        bookmaker_implied = 1 / bookmaker_odds if bookmaker_odds > 0 else 0
        
        # Edge
        edge = model_prob - bookmaker_implied
        edge_pct = edge * 100
        
        return {
            'side': side,
            'threshold': threshold,
            'model_prob': model_prob,
            'bookmaker_odds': bookmaker_odds,
            'bookmaker_implied': bookmaker_implied,
            'fair_odds': fair_odds,
            'expected_value': ev,
            'ev_percentage': ev_pct,
            'edge': edge,
            'edge_percentage': edge_pct,
            'has_value': ev > 0 and edge > 0.03
        }
    
    def find_fair_line(
        self,
        lambda_home: float,
        lambda_away: float,
        target_prob: float = 0.5
    ) -> float:
        """
        Find the O/U line where over has target probability.
        
        Uses binary search.
        
        Args:
            lambda_home: Expected home goals
            lambda_away: Expected away goals
            target_prob: Target probability for over (default 0.5)
        
        Returns:
            Fair O/U line
        """
        total_dist = self._calculate_total_distribution(lambda_home, lambda_away)
        
        # Binary search
        low = 0.5
        high = 10.5
        tolerance = 0.01
        
        for _ in range(50):
            mid = (low + high) / 2
            
            prob_under = self._probability_at_most(total_dist, mid)
            prob_over = 1 - prob_under
            
            if abs(prob_over - target_prob) < tolerance:
                return round(mid * 2) / 2  # Round to nearest 0.5
            
            if prob_over > target_prob:
                # Over too likely, increase line
                low = mid
            else:
                # Over not likely enough, decrease line
                high = mid
        
        return round(((low + high) / 2) * 2) / 2


if __name__ == "__main__":
    print("="*70)
    print("INSTITUTIONAL OVER/UNDER PRICING ENGINE")
    print("="*70)
    
    predictor = OverUnderPredictor()
    
    # Example
    lambda_home = 1.5
    lambda_away = 1.2
    
    result = predictor.predict_from_lambdas(lambda_home, lambda_away)
    
    print(f"\nExpected total: {result['expected_total']:.2f}")
    print(f"\nOver/Under probabilities:")
    for threshold in [0.5, 1.5, 2.5, 3.5]:
        print(f"  O/U {threshold}: {result[f'over_{threshold}']:.1%} / {result[f'under_{threshold}']:.1%}")
    
    # Fair line
    fair_line = predictor.find_fair_line(lambda_home, lambda_away)
    print(f"\nFair 50/50 line: {fair_line}")
