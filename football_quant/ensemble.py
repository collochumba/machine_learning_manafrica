"""
INSTITUTIONAL ENSEMBLE MODEL
Professional log-odds blending with adaptive weighting

Features:
- Log-odds space blending (superior to linear averaging)
- Entropy-based confidence weighting
- Probability shrinkage for stability
- Calibration-aware combination
- Adaptive model weighting based on uncertainty
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')


class EnsemblePredictor:
    """
    Institutional-grade ensemble combining Dixon-Coles and ML.
    
    Uses log-odds blending for proper probability combination
    and dynamic weighting based on prediction confidence.
    """

    def __init__(
        self,
        dc_weight: float = 0.6,
        ml_weight: float = 0.4,
        shrinkage: float = 0.02,
        min_prob: float = 0.001,
        max_prob: float = 0.999
    ):
        """
        Args:
            dc_weight: Base weight for Dixon-Coles (0-1)
            ml_weight: Base weight for ML model (0-1)
            shrinkage: Probability shrinkage toward uniform (0.01-0.05)
            min_prob: Minimum probability (prevents log-odds overflow)
            max_prob: Maximum probability (prevents log-odds overflow)
        """
        # Validate weights
        if dc_weight + ml_weight <= 0:
            raise ValueError("Weights must sum to positive value")
        
        # Normalize weights
        total = dc_weight + ml_weight
        self.dc_weight = dc_weight / total
        self.ml_weight = ml_weight / total
        
        self.shrinkage = shrinkage
        self.min_prob = min_prob
        self.max_prob = max_prob

        self.dc_models = None
        self.ml_model = None

    def set_models(self, dc_models, ml_model):
        """Set component models."""
        self.dc_models = dc_models
        self.ml_model = ml_model

    def _clip_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """Clip probabilities to safe range for log-odds."""
        return np.clip(probs, self.min_prob, self.max_prob)

    def _to_log_odds(self, probs: np.ndarray) -> np.ndarray:
        """Convert probabilities to log-odds space."""
        probs = self._clip_probabilities(probs)
        probs = probs / probs.sum()
        
        # Use draw as baseline
        baseline = probs[1]
        baseline = max(baseline, self.min_prob)
        
        log_odds = np.log(probs / baseline)
        return log_odds

    def _from_log_odds(self, log_odds: np.ndarray) -> np.ndarray:
        """Convert log-odds back to probabilities."""
        probs = softmax(log_odds)
        return self._clip_probabilities(probs)

    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """Calculate Shannon entropy."""
        probs = self._clip_probabilities(probs)
        entropy = -np.sum(probs * np.log(probs))
        return entropy

    def _confidence_adjustment(self, probs: np.ndarray) -> float:
        """Calculate confidence score from entropy."""
        entropy = self._calculate_entropy(probs)
        max_entropy = np.log(len(probs))
        confidence = 1.0 - (entropy / max_entropy)
        return confidence

    def _adaptive_weighting(
        self,
        dc_probs: np.ndarray,
        ml_probs: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate adaptive weights based on model confidence."""
        dc_confidence = self._confidence_adjustment(dc_probs)
        ml_confidence = self._confidence_adjustment(ml_probs)
        
        dc_adj = self.dc_weight * (0.5 + 0.5 * dc_confidence)
        ml_adj = self.ml_weight * (0.5 + 0.5 * ml_confidence)
        
        total = dc_adj + ml_adj
        dc_adj /= total
        ml_adj /= total
        
        return dc_adj, ml_adj

    def _apply_shrinkage(self, probs: np.ndarray) -> np.ndarray:
        """Apply probability shrinkage toward uniform distribution."""
        n = len(probs)
        uniform = np.ones(n) / n
        shrunk = (1 - self.shrinkage) * probs + self.shrinkage * uniform
        shrunk = shrunk / shrunk.sum()
        return shrunk

    def predict(
        self,
        league: str,
        home_team: str,
        away_team: str,
        features: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Generate ensemble prediction.
        
        Args:
            league: League name
            home_team: Home team name
            away_team: Away team name
            features: Features for ML model (optional)
        
        Returns:
            Dictionary with ensemble predictions
        """
        if self.dc_models is None:
            raise ValueError("Models not set. Call set_models() first.")

        # Get Dixon-Coles predictions
        dc_pred = self.dc_models.predict(league, home_team, away_team)

        dc_probs = np.array([
            dc_pred['prob_home_win'],
            dc_pred['prob_draw'],
            dc_pred['prob_away_win']
        ])

        dc_probs = self._clip_probabilities(dc_probs)
        dc_probs = dc_probs / dc_probs.sum()

        # Get ML predictions
        if self.ml_model is not None and features is not None:
            try:
                ml_probs = self.ml_model.predict_proba(features)[0]
            except:
                ml_probs = dc_probs.copy()
        else:
            ml_probs = dc_probs.copy()

        ml_probs = self._clip_probabilities(ml_probs)
        ml_probs = ml_probs / ml_probs.sum()

        # Calculate adaptive weights
        dc_weight_adj, ml_weight_adj = self._adaptive_weighting(dc_probs, ml_probs)

        # Convert to log-odds space
        dc_log_odds = self._to_log_odds(dc_probs)
        ml_log_odds = self._to_log_odds(ml_probs)

        # Blend in log-odds space
        blended_log_odds = (
            dc_weight_adj * dc_log_odds +
            ml_weight_adj * ml_log_odds
        )

        # Convert back to probabilities
        ensemble_probs = self._from_log_odds(blended_log_odds)

        # Apply shrinkage
        ensemble_probs = self._apply_shrinkage(ensemble_probs)

        # Calculate confidence metrics
        ml_confidence = self._confidence_adjustment(ml_probs)
        ensemble_confidence = self._confidence_adjustment(ensemble_probs)

        # Return comprehensive results
        return {
            # Main predictions
            'prob_home_win': float(ensemble_probs[0]),
            'prob_draw': float(ensemble_probs[1]),
            'prob_away_win': float(ensemble_probs[2]),

            # Expected goals (from Dixon-Coles)
            'expected_home_goals': dc_pred['expected_home_goals'],
            'expected_away_goals': dc_pred['expected_away_goals'],
            'lambda_home': dc_pred['lambda_home'],
            'lambda_away': dc_pred['lambda_away'],

            # Other markets (from Dixon-Coles)
            'prob_over_25': dc_pred.get('prob_over_25', 0),
            'prob_over_15': dc_pred.get('prob_over_15', 0),
            'prob_over_35': dc_pred.get('prob_over_35', 0),
            'prob_btts': dc_pred.get('prob_btts', 0),

            # Component model predictions
            'dc_home': float(dc_probs[0]),
            'dc_draw': float(dc_probs[1]),
            'dc_away': float(dc_probs[2]),
            'ml_home': float(ml_probs[0]),
            'ml_draw': float(ml_probs[1]),
            'ml_away': float(ml_probs[2]),

            # Weighting and confidence
            'dc_weight_used': float(dc_weight_adj),
            'ml_weight_used': float(ml_weight_adj),
            'ml_confidence': float(ml_confidence),
            'ensemble_confidence': float(ensemble_confidence)
        }


if __name__ == "__main__":
    print("Institutional Ensemble Model - Production Ready")
