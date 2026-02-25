"""
INSTITUTIONAL DIXON-COLES MODEL
Professional statistical model with time decay and rho correction

Reference:
    Dixon, M. J., & Coles, S. G. (1997). Modelling Association Football Scores
    and Inefficiencies in the Football Betting Market. Applied Statistics.

Features:
- Rho correction for low-score dependency
- Exponential time decay
- L-BFGS-B optimization
- Monte Carlo probability estimation
- Model persistence (save/load)
- Multi-league support
- Team strength rankings
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import Dict, Optional, List
import pickle
import warnings
warnings.filterwarnings('ignore')


class DixonColesModel:
    """
    Professional Dixon-Coles implementation.
    
    Models match outcomes using Poisson distribution with:
    - Attack and defence strengths per team
    - Home advantage parameter
    - Rho correction for score correlation
    - Time decay for recent form emphasis
    """
    
    def __init__(self, xi: float = 0.002):
        """
        Initialize model.
        
        Args:
            xi: Time decay parameter (0.001-0.01)
                Lower = slower decay (longer memory)
                Higher = faster decay (emphasize recent matches)
                Default 0.002 ≈ half-life of ~350 days
        """
        self.xi = xi
        
        # Model parameters
        self.params: Optional[np.ndarray] = None
        self.teams: Optional[List[str]] = None
        self.attack: Optional[Dict[str, float]] = None
        self.defence: Optional[Dict[str, float]] = None
        self.home_adv: Optional[float] = None
        self.rho: Optional[float] = None
        
        # Metadata
        self.n_matches_fitted: int = 0
        self.convergence_success: bool = False
    
    def rho_correction(
        self,
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        lambda_away: float,
        rho: float
    ) -> float:
        """
        Dixon-Coles rho correction for low-score dependency.
        
        Corrects Poisson independence assumption for:
        - 0-0: Draw bias
        - 1-0, 0-1: Close game correlation
        - 1-1: Draw bias
        
        Args:
            home_goals: Home team goals scored
            away_goals: Away team goals scored
            lambda_home: Expected home goals
            lambda_away: Expected away goals
            rho: Correlation parameter
        
        Returns:
            Correction multiplier (typically 0.8-1.2)
        """
        if home_goals == 0 and away_goals == 0:
            return 1 - lambda_home * lambda_away * rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + lambda_home * rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + lambda_away * rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - rho
        else:
            return 1.0
    
    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        home_teams: np.ndarray,
        away_teams: np.ndarray,
        time_weights: np.ndarray
    ) -> float:
        """
        Calculate negative log-likelihood for optimization.
        
        This is the objective function we minimize.
        
        Args:
            params: Parameter vector [attack, defence, home_adv, rho]
            home_goals: Array of home goals
            away_goals: Array of away goals
            home_teams: Array of home team names
            away_teams: Array of away team names
            time_weights: Exponential time weights
        
        Returns:
            Negative log-likelihood
        """
        n_teams = len(self.teams)
        
        # Unpack parameters
        attack = dict(zip(self.teams, params[:n_teams]))
        defence = dict(zip(self.teams, params[n_teams:2*n_teams]))
        home_adv = params[2*n_teams]
        rho = params[2*n_teams + 1]
        
        # Calculate log-likelihood
        log_lik = 0.0
        
        for hg, ag, ht, at, weight in zip(
            home_goals, away_goals, home_teams, away_teams, time_weights
        ):
            # Expected goals (Poisson parameters)
            lambda_home = np.exp(home_adv + attack[ht] - defence[at])
            lambda_away = np.exp(attack[at] - defence[ht])
            
            # Joint probability (independent Poisson)
            prob = poisson.pmf(hg, lambda_home) * poisson.pmf(ag, lambda_away)
            
            # Apply rho correction
            prob *= self.rho_correction(hg, ag, lambda_home, lambda_away, rho)
            
            # Weighted log-likelihood (time decay)
            log_lik += weight * np.log(max(prob, 1e-10))
        
        return -log_lik
    
    def fit(
        self,
        df: pd.DataFrame,
        league: Optional[str] = None,
        verbose: bool = True
    ) -> 'DixonColesModel':
        """
        Fit Dixon-Coles model to data.
        
        Args:
            df: DataFrame with columns:
                - FTHG: Full time home goals
                - FTAG: Full time away goals
                - HomeTeam: Home team name
                - AwayTeam: Away team name
                - DaysSinceMatch: Days since match (for time decay)
                Optional:
                - League: League name
            league: Filter to specific league
            verbose: Print fitting progress
        
        Returns:
            self (fitted model)
        """
        # Filter by league
        if league:
            data = df[df['League'] == league].copy()
        else:
            data = df.copy()
        
        if len(data) == 0:
            raise ValueError(f"No data for league: {league}")
        
        # Get unique teams
        self.teams = sorted(
            set(data['HomeTeam'].unique()) | set(data['AwayTeam'].unique())
        )
        n_teams = len(self.teams)
        self.n_matches_fitted = len(data)
        
        if verbose:
            print(f"  Fitting {n_teams} teams, {len(data)} matches")
        
        # Time weights (exponential decay)
        if 'DaysSinceMatch' in data.columns:
            time_weights = np.exp(-self.xi * data['DaysSinceMatch'].values)
        else:
            time_weights = np.ones(len(data))
        
        # Initial parameters
        # Start with: neutral attack/defence, small home advantage, no correlation
        x0 = np.concatenate([
            np.zeros(n_teams),      # attack strengths
            np.zeros(n_teams),      # defence strengths
            [0.2],                  # home advantage
            [0.0]                   # rho
        ])
        
        # Optimize using L-BFGS-B
        result = minimize(
            self._negative_log_likelihood,
            x0,
            args=(
                data['FTHG'].values,
                data['FTAG'].values,
                data['HomeTeam'].values,
                data['AwayTeam'].values,
                time_weights
            ),
            method='L-BFGS-B',
            options={'maxiter': 200, 'disp': False}
        )
        
        self.convergence_success = result.success
        
        if not result.success and verbose:
            print(f"  Warning: Optimization did not converge")
        
        # Store parameters
        self.params = result.x
        self.attack = dict(zip(self.teams, result.x[:n_teams]))
        self.defence = dict(zip(self.teams, result.x[n_teams:2*n_teams]))
        self.home_adv = result.x[2*n_teams]
        self.rho = result.x[2*n_teams + 1]
        
        if verbose:
            print(f"  Home advantage: {self.home_adv:.3f}")
            print(f"  Rho: {self.rho:.4f}")
        
        return self
    
    def predict_match_probs(
        self,
        home_team: str,
        away_team: str,
        n_sims: int = 20000
    ) -> Dict:
        """
        Predict match outcome probabilities using Monte Carlo.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            n_sims: Number of Monte Carlo simulations
        
        Returns:
            Dictionary with predictions
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if home_team not in self.teams or away_team not in self.teams:
            raise ValueError(
                f"Teams not in model. Available: {', '.join(self.teams[:5])}..."
            )
        
        # Calculate expected goals (lambdas)
        lambda_home = np.exp(
            self.home_adv + self.attack[home_team] - self.defence[away_team]
        )
        lambda_away = np.exp(
            self.attack[away_team] - self.defence[home_team]
        )
        
        # Monte Carlo simulation
        np.random.seed(42)  # Reproducibility
        home_goals = poisson.rvs(lambda_home, size=n_sims)
        away_goals = poisson.rvs(lambda_away, size=n_sims)
        
        # Calculate probabilities
        prob_home = np.sum(home_goals > away_goals) / n_sims
        prob_draw = np.sum(home_goals == away_goals) / n_sims
        prob_away = np.sum(home_goals < away_goals) / n_sims
        
        # Over/Under markets
        total_goals = home_goals + away_goals
        prob_over_15 = np.sum(total_goals > 1.5) / n_sims
        prob_over_25 = np.sum(total_goals > 2.5) / n_sims
        prob_over_35 = np.sum(total_goals > 3.5) / n_sims
        
        # Both Teams To Score
        prob_btts = np.sum((home_goals > 0) & (away_goals > 0)) / n_sims
        
        return {
            'lambda_home': float(lambda_home),
            'lambda_away': float(lambda_away),
            'expected_home_goals': float(lambda_home),
            'expected_away_goals': float(lambda_away),
            'expected_total_goals': float(lambda_home + lambda_away),
            'prob_home_win': float(prob_home),
            'prob_draw': float(prob_draw),
            'prob_away_win': float(prob_away),
            'prob_over_15': float(prob_over_15),
            'prob_over_25': float(prob_over_25),
            'prob_over_35': float(prob_over_35),
            'prob_btts': float(prob_btts),
            'prob_btts_no': float(1 - prob_btts)
        }
    
    def get_team_strengths(self) -> pd.DataFrame:
        """
        Get team strength rankings.
        
        Returns:
            DataFrame sorted by net strength (attack - defence)
        """
        if self.params is None:
            raise ValueError("Model not fitted")
        
        df = pd.DataFrame({
            'Team': self.teams,
            'Attack': [self.attack[team] for team in self.teams],
            'Defence': [self.defence[team] for team in self.teams]
        })
        
        df['NetStrength'] = df['Attack'] - df['Defence']
        df = df.sort_values('NetStrength', ascending=False).reset_index(drop=True)
        df.index = df.index + 1  # 1-indexed ranking
        
        return df
    
    def save_model(self, filepath: str):
        """Save fitted model to disk."""
        if self.params is None:
            raise ValueError("No model to save")
        
        model_data = {
            'params': self.params,
            'teams': self.teams,
            'attack': self.attack,
            'defence': self.defence,
            'home_adv': self.home_adv,
            'rho': self.rho,
            'xi': self.xi,
            'n_matches_fitted': self.n_matches_fitted,
            'convergence_success': self.convergence_success
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load fitted model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.params = model_data['params']
        self.teams = model_data['teams']
        self.attack = model_data['attack']
        self.defence = model_data['defence']
        self.home_adv = model_data['home_adv']
        self.rho = model_data['rho']
        self.xi = model_data['xi']
        self.n_matches_fitted = model_data.get('n_matches_fitted', 0)
        self.convergence_success = model_data.get('convergence_success', True)


class MultiLeagueDixonColes:
    """
    Manage Dixon-Coles models for multiple leagues.
    
    Each league gets its own model to account for
    league-specific characteristics.
    """
    
    def __init__(self, xi: float = 0.002):
        """
        Args:
            xi: Time decay parameter for all models
        """
        self.xi = xi
        self.models: Dict[str, DixonColesModel] = {}
    
    def fit_all(
        self,
        df: pd.DataFrame,
        leagues: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, DixonColesModel]:
        """
        Fit Dixon-Coles model for each league.
        
        Args:
            df: DataFrame with match data
            leagues: List of leagues (None = all in data)
            verbose: Print progress
        
        Returns:
            Dictionary of {league: model}
        """
        if leagues is None:
            leagues = df['League'].unique()
        
        if verbose:
            print("\n🎯 Training Dixon-Coles models...")
        
        for league in leagues:
            if verbose:
                print(f"\n{league}:")
            
            try:
                model = DixonColesModel(xi=self.xi)
                model.fit(df, league=league, verbose=verbose)
                self.models[league] = model
            except Exception as e:
                if verbose:
                    print(f"  ✗ Error: {str(e)[:50]}")
        
        if verbose:
            print(f"\n✅ Trained {len(self.models)} league models")
        
        return self.models
    
    def predict(
        self,
        league: str,
        home_team: str,
        away_team: str,
        n_sims: int = 20000
    ) -> Dict:
        """
        Predict match using appropriate league model.
        
        Args:
            league: League name
            home_team: Home team name
            away_team: Away team name
            n_sims: Monte Carlo simulations
        
        Returns:
            Prediction dictionary
        """
        if league not in self.models:
            raise ValueError(
                f"No model for {league}. Available: {list(self.models.keys())}"
            )
        
        return self.models[league].predict_match_probs(
            home_team, away_team, n_sims
        )
    
    def get_all_team_strengths(self) -> pd.DataFrame:
        """
        Get team strengths across all leagues.
        
        Returns:
            DataFrame with all team rankings
        """
        all_strengths = []
        
        for league, model in self.models.items():
            strengths = model.get_team_strengths()
            strengths['League'] = league
            all_strengths.append(strengths)
        
        return pd.concat(all_strengths, ignore_index=True)


if __name__ == "__main__":
    print("="*70)
    print("INSTITUTIONAL DIXON-COLES MODEL")
    print("="*70)
    print("\nProduction-ready statistical football model")
    print("Features: Rho correction, time decay, multi-league support")
