"""
PROFESSIONAL ASIAN HANDICAP PRICING ENGINE
Exact Poisson convolution - no Monte Carlo shortcuts

Features:
- Exact probability calculation via Poisson convolution
- Quarter-line support (-0.25, -0.75, -1.25, -1.75, -2.25, -2.75)
- Half-line support (-0.5, -1.5, -2.5)
- Full-line support (-1, -2, -3) with push handling
- Fair odds calculation
- Expected value computation
- Reverse engineering: odds -> implied probability
- Integration with Dixon-Coles outputs (lambda_home, lambda_away)
"""

import numpy as np
from scipy.stats import poisson
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class LineType(Enum):
    """Asian Handicap line types."""
    FULL = "full"      # -1, -2, -3 (push possible)
    HALF = "half"      # -0.5, -1.5, -2.5 (no push)
    QUARTER = "quarter"  # -0.25, -0.75, -1.25 (split stake)


@dataclass
class AsianHandicapResult:
    """Result of AH calculation."""
    handicap: float
    line_type: LineType
    prob_home_cover: float
    prob_push: float
    prob_away_cover: float
    fair_home_odds: float
    fair_away_odds: float
    expected_home_value: float  # Per unit stake
    expected_away_value: float


class ProfessionalAsianHandicap:
    """
    Institutional-grade Asian Handicap pricing engine.
    
    Uses exact Poisson convolution for probability calculation.
    No Monte Carlo approximation - mathematically precise.
    """
    
    def __init__(self, max_goals: int = 15):
        """
        Initialize AH engine.
        
        Args:
            max_goals: Maximum goals to consider (15 covers >99.99% of cases)
        """
        self.max_goals = max_goals
    
    @staticmethod
    def determine_line_type(handicap: float) -> LineType:
        """
        Determine the type of Asian Handicap line.
        
        Args:
            handicap: Handicap value (e.g., -0.5, -0.75, -1.0)
        
        Returns:
            LineType enum
        """
        # Get fractional part
        frac = abs(handicap) - int(abs(handicap))
        
        if abs(frac) < 1e-9:  # Essentially zero
            return LineType.FULL
        elif abs(frac - 0.5) < 1e-9:
            return LineType.HALF
        elif abs(frac - 0.25) < 1e-9 or abs(frac - 0.75) < 1e-9:
            return LineType.QUARTER
        else:
            raise ValueError(f"Invalid handicap: {handicap}. Must be multiple of 0.25")
    
    def calculate_goal_difference_probabilities(
        self,
        lambda_home: float,
        lambda_away: float
    ) -> np.ndarray:
        """
        Calculate exact probability matrix for all goal combinations.
        
        Uses Poisson distribution for each team independently,
        then computes joint probabilities.
        
        Args:
            lambda_home: Expected home goals (Poisson parameter)
            lambda_away: Expected away goals (Poisson parameter)
        
        Returns:
            2D array where element [h, a] = P(home=h, away=a)
        """
        prob_matrix = np.zeros((self.max_goals, self.max_goals))
        
        for h in range(self.max_goals):
            for a in range(self.max_goals):
                prob_matrix[h, a] = (
                    poisson.pmf(h, lambda_home) * 
                    poisson.pmf(a, lambda_away)
                )
        
        return prob_matrix
    
    def calculate_full_line(
        self,
        lambda_home: float,
        lambda_away: float,
        handicap: float
    ) -> Tuple[float, float, float]:
        """
        Calculate probabilities for FULL line (e.g., -1, -2).
        
        Full lines have three outcomes:
        - Home covers: (home + handicap) > away
        - Push: (home + handicap) == away
        - Away covers: (home + handicap) < away
        
        Args:
            lambda_home: Expected home goals
            lambda_away: Expected away goals
            handicap: Handicap (e.g., -1.0, -2.0)
        
        Returns:
            (prob_home_cover, prob_push, prob_away_cover)
        """
        prob_matrix = self.calculate_goal_difference_probabilities(lambda_home, lambda_away)
        
        prob_home = 0.0
        prob_push = 0.0
        prob_away = 0.0
        
        for h in range(self.max_goals):
            for a in range(self.max_goals):
                adjusted_home = h + handicap
                
                if adjusted_home > a:
                    prob_home += prob_matrix[h, a]
                elif abs(adjusted_home - a) < 1e-9:  # Equal (push)
                    prob_push += prob_matrix[h, a]
                else:
                    prob_away += prob_matrix[h, a]
        
        return prob_home, prob_push, prob_away
    
    def calculate_half_line(
        self,
        lambda_home: float,
        lambda_away: float,
        handicap: float
    ) -> Tuple[float, float, float]:
        """
        Calculate probabilities for HALF line (e.g., -0.5, -1.5).
        
        Half lines have only two outcomes (no push possible):
        - Home covers: (home + handicap) > away
        - Away covers: (home + handicap) < away
        
        Args:
            lambda_home: Expected home goals
            lambda_away: Expected away goals
            handicap: Handicap (e.g., -0.5, -1.5)
        
        Returns:
            (prob_home_cover, 0.0, prob_away_cover)
        """
        prob_matrix = self.calculate_goal_difference_probabilities(lambda_home, lambda_away)
        
        prob_home = 0.0
        prob_away = 0.0
        
        for h in range(self.max_goals):
            for a in range(self.max_goals):
                adjusted_home = h + handicap
                
                if adjusted_home > a:
                    prob_home += prob_matrix[h, a]
                else:
                    prob_away += prob_matrix[h, a]
        
        return prob_home, 0.0, prob_away
    
    def calculate_quarter_line(
        self,
        lambda_home: float,
        lambda_away: float,
        handicap: float
    ) -> Tuple[float, float, float]:
        """
        Calculate probabilities for QUARTER line (e.g., -0.25, -0.75).
        
        Quarter lines split the stake between two half-lines:
        Example: -0.25 = 50% on 0.0 + 50% on -0.5
        
        Outcomes:
        - Full win: Both halves win
        - Half win: One wins, one pushes
        - Half loss: One loses, one pushes
        - Full loss: Both halves lose
        
        We convert to effective probabilities:
        - prob_home = P(full win) + 0.5 * P(half win)
        - prob_push = P(half win) + P(half loss) (effective push)
        - prob_away = P(full loss) + 0.5 * P(half loss)
        
        Args:
            lambda_home: Expected home goals
            lambda_away: Expected away goals
            handicap: Quarter handicap (e.g., -0.25, -0.75)
        
        Returns:
            (prob_home_cover, prob_push, prob_away_cover)
        """
        # Quarter line = average of two adjacent half/full lines
        # Example: -0.25 = average(-0.5, 0.0)
        #          -0.75 = average(-1.0, -0.5)
        
        if handicap % 1 == 0.25:  # e.g., -0.25, -1.25
            line1 = int(handicap)  # Full line (0, -1, -2)
            line2 = handicap - 0.25  # Half line (-0.5, -1.5)
        else:  # 0.75, -1.75
            line1 = handicap + 0.25  # Half line (-0.5, -1.5)
            line2 = int(handicap - 0.5)  # Full line (-1, -2)
        
        # Calculate both components
        if abs(line1 - int(line1)) < 1e-9:  # Line 1 is full
            p1_home, p1_push, p1_away = self.calculate_full_line(lambda_home, lambda_away, line1)
        else:  # Line 1 is half
            p1_home, p1_push, p1_away = self.calculate_half_line(lambda_home, lambda_away, line1)
        
        if abs(line2 - int(line2)) < 1e-9:  # Line 2 is full
            p2_home, p2_push, p2_away = self.calculate_full_line(lambda_home, lambda_away, line2)
        else:  # Line 2 is half
            p2_home, p2_push, p2_away = self.calculate_half_line(lambda_home, lambda_away, line2)
        
        # Combine (equal weight to each half)
        # Effective probability accounting for split stake
        prob_home = 0.5 * (p1_home + p2_home)
        prob_push = 0.5 * (p1_push + p2_push)
        prob_away = 0.5 * (p1_away + p2_away)
        
        return prob_home, prob_push, prob_away
    
    def calculate_handicap(
        self,
        lambda_home: float,
        lambda_away: float,
        handicap: float
    ) -> AsianHandicapResult:
        """
        Calculate complete Asian Handicap pricing.
        
        Args:
            lambda_home: Expected home goals (from Dixon-Coles)
            lambda_away: Expected away goals (from Dixon-Coles)
            handicap: Handicap line (negative favors away, positive favors home)
        
        Returns:
            AsianHandicapResult with all probabilities and fair odds
        """
        # Determine line type
        line_type = self.determine_line_type(handicap)
        
        # Calculate probabilities based on line type
        if line_type == LineType.FULL:
            prob_home, prob_push, prob_away = self.calculate_full_line(
                lambda_home, lambda_away, handicap
            )
        elif line_type == LineType.HALF:
            prob_home, prob_push, prob_away = self.calculate_half_line(
                lambda_home, lambda_away, handicap
            )
        else:  # QUARTER
            prob_home, prob_push, prob_away = self.calculate_quarter_line(
                lambda_home, lambda_away, handicap
            )
        
        # Calculate fair odds
        # Fair odds = 1 / (probability of winning)
        # For lines with push, adjust for push probability
        
        if prob_home > 0:
            # Expected return = prob_win * (odds - 1) + prob_push * 0 - prob_lose * 1
            # Set expected return = 0 for fair odds
            # fair_odds = 1 / prob_home (simplified for binary outcomes)
            # For push: fair_odds = 1 / (prob_home + 0.5 * prob_push)
            effective_home_prob = prob_home + 0.5 * prob_push
            fair_home_odds = 1.0 / effective_home_prob if effective_home_prob > 0 else 999.0
        else:
            fair_home_odds = 999.0
        
        if prob_away > 0:
            effective_away_prob = prob_away + 0.5 * prob_push
            fair_away_odds = 1.0 / effective_away_prob if effective_away_prob > 0 else 999.0
        else:
            fair_away_odds = 999.0
        
        # Expected value per unit stake (at fair odds, EV = 0)
        exp_home_value = prob_home * (fair_home_odds - 1) - prob_away
        exp_away_value = prob_away * (fair_away_odds - 1) - prob_home
        
        return AsianHandicapResult(
            handicap=handicap,
            line_type=line_type,
            prob_home_cover=prob_home,
            prob_push=prob_push,
            prob_away_cover=prob_away,
            fair_home_odds=fair_home_odds,
            fair_away_odds=fair_away_odds,
            expected_home_value=exp_home_value,
            expected_away_value=exp_away_value
        )
    
    def find_fair_handicap(
        self,
        lambda_home: float,
        lambda_away: float,
        target_prob: float = 0.5
    ) -> float:
        """
        Find the Asian Handicap line where home has target probability.
        
        Uses binary search to find fair line.
        
        Args:
            lambda_home: Expected home goals
            lambda_away: Expected away goals
            target_prob: Target probability for home (default 0.5 for even line)
        
        Returns:
            Fair handicap line (rounded to nearest 0.25)
        """
        # Binary search bounds
        low = -5.0
        high = 5.0
        tolerance = 0.01
        
        for _ in range(50):  # Max iterations
            mid = (low + high) / 2
            
            # Round to nearest 0.25
            mid_rounded = round(mid * 4) / 4
            
            result = self.calculate_handicap(lambda_home, lambda_away, mid_rounded)
            prob_home = result.prob_home_cover + 0.5 * result.prob_push
            
            if abs(prob_home - target_prob) < tolerance:
                return mid_rounded
            
            if prob_home > target_prob:
                # Home too strong, increase handicap (more negative)
                high = mid
            else:
                # Home too weak, decrease handicap (less negative)
                low = mid
        
        return round(((low + high) / 2) * 4) / 4
    
    def calculate_expected_value(
        self,
        lambda_home: float,
        lambda_away: float,
        handicap: float,
        bookmaker_odds_home: float,
        bookmaker_odds_away: float,
        side: str = 'home'
    ) -> Dict:
        """
        Calculate expected value of betting on AH line.
        
        Args:
            lambda_home: Expected home goals
            lambda_away: Expected away goals
            handicap: Asian Handicap line
            bookmaker_odds_home: Bookmaker's odds for home
            bookmaker_odds_away: Bookmaker's odds for away
            side: 'home' or 'away'
        
        Returns:
            Dictionary with EV analysis
        """
        result = self.calculate_handicap(lambda_home, lambda_away, handicap)
        
        if side.lower() == 'home':
            model_prob = result.prob_home_cover + 0.5 * result.prob_push
            bookmaker_odds = bookmaker_odds_home
            fair_odds = result.fair_home_odds
        else:
            model_prob = result.prob_away_cover + 0.5 * result.prob_push
            bookmaker_odds = bookmaker_odds_away
            fair_odds = result.fair_away_odds
        
        # Expected value per unit stake
        ev = model_prob * (bookmaker_odds - 1) - (1 - model_prob)
        ev_pct = ev * 100
        
        # Implied probability
        bookmaker_implied = 1 / bookmaker_odds
        
        # Edge
        edge = model_prob - bookmaker_implied
        edge_pct = edge * 100
        
        return {
            'side': side,
            'handicap': handicap,
            'model_prob': model_prob,
            'bookmaker_odds': bookmaker_odds,
            'bookmaker_implied': bookmaker_implied,
            'fair_odds': fair_odds,
            'expected_value': ev,
            'ev_percentage': ev_pct,
            'edge': edge,
            'edge_percentage': edge_pct,
            'has_value': ev > 0 and edge > 0.03  # 3% edge threshold
        }
    
    def extract_implied_probability(
        self,
        handicap: float,
        odds_home: float,
        odds_away: float
    ) -> Dict:
        """
        Extract implied probabilities from Asian Handicap odds.
        
        Args:
            handicap: AH line
            odds_home: Odds for home
            odds_away: Odds for away
        
        Returns:
            Dictionary with implied probabilities and margin
        """
        implied_home = 1 / odds_home
        implied_away = 1 / odds_away
        
        # Total probability (includes bookmaker margin)
        total_prob = implied_home + implied_away
        
        # Margin
        margin = total_prob - 1.0
        margin_pct = margin * 100
        
        # Fair probabilities (removing margin proportionally)
        fair_home = implied_home / total_prob
        fair_away = implied_away / total_prob
        
        return {
            'handicap': handicap,
            'implied_home': implied_home,
            'implied_away': implied_away,
            'total_prob': total_prob,
            'margin': margin,
            'margin_percentage': margin_pct,
            'fair_home': fair_home,
            'fair_away': fair_away
        }


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("PROFESSIONAL ASIAN HANDICAP ENGINE")
    print("="*70)
    
    ah = ProfessionalAsianHandicap()
    
    # Example: Strong home team
    lambda_home = 2.0
    lambda_away = 1.0
    
    print(f"\nScenario: Home λ={lambda_home}, Away λ={lambda_away}")
    print("-"*70)
    
    # Test different handicap lines
    for handicap in [-0.25, -0.5, -0.75, -1.0, -1.25, -1.5]:
        result = ah.calculate_handicap(lambda_home, lambda_away, handicap)
        
        print(f"\nHandicap: {handicap:+.2f} ({result.line_type.value})")
        print(f"  Home cover: {result.prob_home_cover:.1%}")
        print(f"  Push:       {result.prob_push:.1%}")
        print(f"  Away cover: {result.prob_away_cover:.1%}")
        print(f"  Fair odds:  Home {result.fair_home_odds:.2f} | Away {result.fair_away_odds:.2f}")
    
    # Find fair line
    fair_line = ah.find_fair_handicap(lambda_home, lambda_away)
    print(f"\nFair 50/50 line: {fair_line:+.2f}")
    
    print("\n" + "="*70)
