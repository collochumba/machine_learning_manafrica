"""
PRODUCTION INFERENCE ENGINE
Complete training/inference separation with model persistence

Features:
- Separate training and inference modes
- Model save/load (pickle + joblib)
- Daily prediction pipeline
- Fixture ingestion from football-data.co.uk
- Historical data management
- Model versioning
- Production-ready error handling
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from data_loader import FootballDataLoader, generate_season_codes
from feature_engineering import FeatureEngineer
from dixon_coles import MultiLeagueDixonColes
from ml_models import FootballMLModel
from ensemble import EnsemblePredictor
from fixtures_loader import FixturesLoader
from asian_handicap_professional import ProfessionalAsianHandicap


class InferenceEngine:
    """
    Production inference engine with complete training/inference separation.
    
    Training Mode:
    - Load historical data
    - Engineer features
    - Train Dixon-Coles models
    - Train ML models
    - Save models to disk
    
    Inference Mode:
    - Load pre-trained models
    - Load fixtures
    - Generate features for new matches
    - Predict outcomes
    - No training data leakage
    """
    
    def __init__(self, models_dir: str = 'models', data_dir: str = 'data'):
        """
        Initialize inference engine.
        
        Args:
            models_dir: Directory to save/load models
            data_dir: Directory for data cache
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Model components
        self.dc_models: Optional[MultiLeagueDixonColes] = None
        self.ml_model: Optional[FootballMLModel] = None
        self.ensemble: Optional[EnsemblePredictor] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.ah_engine = ProfessionalAsianHandicap()
        
        # Data
        self.training_data: Optional[pd.DataFrame] = None
        self.featured_data: Optional[pd.DataFrame] = None
        
        # Metadata
        self.model_version: Optional[str] = None
        self.trained_leagues: Optional[List[str]] = None
    
    # ================================================================
    # TRAINING MODE
    # ================================================================
    
    def train_models(
        self,
        leagues: List[str],
        n_seasons: int = 3,
        max_h2h: int = 0,
        verbose: bool = True
    ) -> Dict:
        """
        Complete training pipeline.
        
        Args:
            leagues: List of leagues to train on
            n_seasons: Number of historical seasons
            max_h2h: Max H2H matches to include (0 = skip for speed)
            verbose: Print progress
        
        Returns:
            Training metrics dictionary
        """
        if verbose:
            print("\n" + "="*70)
            print("🎓 TRAINING MODE - Building Models")
            print("="*70)
        
        # Step 1: Load data
        if verbose:
            print("\n[1/4] Loading historical data...")
        
        current_year = datetime.now().year
        season_codes, season_labels = generate_season_codes(
            current_year - n_seasons, current_year
        )
        
        loader = FootballDataLoader()
        self.training_data = loader.load_all_data(
            season_codes,
            season_labels,
            league_subset=leagues,
            verbose=verbose
        )
        self.training_data = loader.clean_data(self.training_data)
        
        # Step 2: Feature engineering
        if verbose:
            print("\n[2/4] Engineering features...")
        
        self.feature_engineer = FeatureEngineer(self.training_data)
        self.featured_data = self.feature_engineer.create_all_features(
            rolling_windows=[5, 10],
            form_window=5,
            max_h2h=max_h2h,
            include_streaks=True
        )
        
        # Step 3: Train Dixon-Coles
        if verbose:
            print("\n[3/4] Training Dixon-Coles models...")
        
        self.dc_models = MultiLeagueDixonColes(xi=0.002)
        self.dc_models.fit_all(self.featured_data, leagues=leagues, verbose=verbose)
        
        # Step 4: Train ML model
        if verbose:
            print("\n[4/4] Training XGBoost model...")
        
        self.ml_model = FootballMLModel(model_type='1x2', task='classification')
        ml_metrics = self.ml_model.train(self.featured_data, test_size=0.2)
        
        if verbose:
            print(f"   Accuracy: {ml_metrics['accuracy']:.3f}")
            print(f"   Log Loss: {ml_metrics['log_loss']:.4f}")
        
        # Step 5: Create ensemble
        self.ensemble = EnsemblePredictor(dc_weight=0.6, ml_weight=0.4)
        self.ensemble.set_models(self.dc_models, self.ml_model)
        
        # Store metadata
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trained_leagues = leagues
        
        if verbose:
            print("\n✅ Training complete!")
            print(f"   Model version: {self.model_version}")
            print(f"   Trained leagues: {', '.join(leagues)}")
        
        return {
            'model_version': self.model_version,
            'trained_leagues': leagues,
            'n_matches': len(self.featured_data),
            'ml_accuracy': ml_metrics['accuracy'],
            'ml_log_loss': ml_metrics['log_loss']
        }
    
    def save_models(self, version_tag: Optional[str] = None):
        """
        Save all trained models to disk.
        
        Args:
            version_tag: Optional tag (uses timestamp if None)
        """
        if self.dc_models is None or self.ml_model is None:
            raise ValueError("No models to save. Train models first.")
        
        tag = version_tag or self.model_version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n💾 Saving models (version: {tag})...")
        
        # Save Dixon-Coles
        dc_path = self.models_dir / f"dixon_coles_{tag}.pkl"
        with open(dc_path, 'wb') as f:
            pickle.dump(self.dc_models, f)
        print(f"   ✓ Dixon-Coles → {dc_path}")
        
        # Save ML model
        ml_path = self.models_dir / f"ml_model_{tag}.pkl"
        with open(ml_path, 'wb') as f:
            pickle.dump(self.ml_model, f)
        print(f"   ✓ ML Model → {ml_path}")
        
        # Save ensemble
        ensemble_path = self.models_dir / f"ensemble_{tag}.pkl"
        with open(ensemble_path, 'wb') as f:
            pickle.dump(self.ensemble, f)
        print(f"   ✓ Ensemble → {ensemble_path}")
        
        # Save metadata
        metadata = {
            'version': tag,
            'trained_date': datetime.now().isoformat(),
            'leagues': self.trained_leagues,
            'n_training_matches': len(self.featured_data) if self.featured_data is not None else 0
        }
        
        metadata_path = self.models_dir / f"metadata_{tag}.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"   ✓ Metadata → {metadata_path}")
        
        print(f"\n✅ Models saved successfully!")
    
    # ================================================================
    # INFERENCE MODE
    # ================================================================
    
    def load_models(self, version_tag: Optional[str] = None):
        """
        Load pre-trained models from disk.
        
        Args:
            version_tag: Specific version to load (loads latest if None)
        """
        print(f"\n📂 Loading models...")
        
        if version_tag is None:
            # Find latest version
            dc_files = list(self.models_dir.glob("dixon_coles_*.pkl"))
            if not dc_files:
                raise FileNotFoundError(f"No models found in {self.models_dir}")
            
            latest_file = max(dc_files, key=lambda p: p.stat().st_mtime)
            version_tag = latest_file.stem.replace("dixon_coles_", "")
        
        # Load Dixon-Coles
        dc_path = self.models_dir / f"dixon_coles_{version_tag}.pkl"
        with open(dc_path, 'rb') as f:
            self.dc_models = pickle.load(f)
        print(f"   ✓ Dixon-Coles loaded")
        
        # Load ML model
        ml_path = self.models_dir / f"ml_model_{version_tag}.pkl"
        with open(ml_path, 'rb') as f:
            self.ml_model = pickle.load(f)
        print(f"   ✓ ML Model loaded")
        
        # Load ensemble
        ensemble_path = self.models_dir / f"ensemble_{version_tag}.pkl"
        with open(ensemble_path, 'rb') as f:
            self.ensemble = pickle.load(f)
        print(f"   ✓ Ensemble loaded")
        
        # Load metadata
        metadata_path = self.models_dir / f"metadata_{version_tag}.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.model_version = metadata['version']
            self.trained_leagues = metadata['leagues']
            print(f"   ✓ Metadata loaded")
            print(f"\n   Model version: {self.model_version}")
            print(f"   Trained on: {metadata.get('trained_date', 'Unknown')}")
            print(f"   Leagues: {', '.join(self.trained_leagues)}")
        
        print(f"\n✅ Models loaded successfully!")
    
    def predict_today(
        self,
        leagues: Optional[List[str]] = None,
        min_edge: float = 0.03
    ) -> pd.DataFrame:
        """
        Predict today's fixtures.
        
        Args:
            leagues: Leagues to predict (uses trained leagues if None)
            min_edge: Minimum edge for value bets
        
        Returns:
            DataFrame with predictions
        """
        if self.ensemble is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        leagues = leagues or self.trained_leagues
        
        # Load today's fixtures
        fixture_loader = FixturesLoader()
        fixtures = fixture_loader.load_upcoming_fixtures()
        
        if fixtures.empty:
            print("⚠️  No fixtures today")
            return pd.DataFrame()
        
        # Filter by leagues
        fixtures = fixtures[fixtures['League'].isin(leagues)]
        
        if fixtures.empty:
            print(f"⚠️  No fixtures in {leagues}")
            return pd.DataFrame()
        
        print(f"\n🔮 Predicting {len(fixtures)} matches...")
        
        # Generate predictions
        predictions = []
        
        for idx, row in fixtures.iterrows():
            try:
                pred = self._predict_single_match(
                    league=row['League'],
                    home_team=row['HomeTeam'],
                    away_team=row['AwayTeam'],
                    fixture_row=row
                )
                predictions.append(pred)
            except Exception as e:
                print(f"   ✗ Error: {row['HomeTeam']} vs {row['AwayTeam']} - {str(e)[:50]}")
                continue
        
        if not predictions:
            return pd.DataFrame()
        
        predictions_df = pd.DataFrame(predictions)
        
        # Filter by edge
        predictions_df = predictions_df[predictions_df['best_edge'] >= min_edge]
        
        print(f"✅ Generated {len(predictions_df)} value bets")
        
        return predictions_df
    
    def predict_week(
        self,
        leagues: Optional[List[str]] = None,
        days_ahead: int = 7,
        min_edge: float = 0.02
    ) -> pd.DataFrame:
        """
        Predict upcoming week's fixtures.
        
        Args:
            leagues: Leagues to predict
            days_ahead: Number of days to look ahead
            min_edge: Minimum edge
        
        Returns:
            DataFrame with predictions
        """
        if self.ensemble is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        leagues = leagues or self.trained_leagues
        
        # Load fixtures
        fixture_loader = FixturesLoader()
        fixtures = fixture_loader.load_next_n_days(days_ahead)
        
        if fixtures.empty:
            print(f"⚠️  No fixtures in next {days_ahead} days")
            return pd.DataFrame()
        
        # Filter by leagues
        fixtures = fixtures[fixtures['League'].isin(leagues)]
        
        print(f"\n🔮 Predicting {len(fixtures)} matches...")
        
        predictions = []
        
        for idx, row in fixtures.iterrows():
            try:
                pred = self._predict_single_match(
                    league=row['League'],
                    home_team=row['HomeTeam'],
                    away_team=row['AwayTeam'],
                    fixture_row=row
                )
                predictions.append(pred)
            except Exception as e:
                continue
        
        if not predictions:
            return pd.DataFrame()
        
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df[predictions_df['best_edge'] >= min_edge]
        
        print(f"✅ Generated {len(predictions_df)} value bets")
        
        return predictions_df
    
    def _predict_single_match(
        self,
        league: str,
        home_team: str,
        away_team: str,
        fixture_row: pd.Series
    ) -> Dict:
        """Predict a single match."""
        
        # Get ensemble prediction (this uses Dixon-Coles + ML)
        # For now, we'll use DC directly since we need lambda values
        dc_pred = self.dc_models.predict(league, home_team, away_team)
        
        # Asian Handicap
        ah_result = self.ah_engine.calculate_handicap(
            dc_pred['lambda_home'],
            dc_pred['lambda_away'],
            handicap=-0.5  # Example line
        )
        
        # Find best value bet
        markets = {
            'home': (dc_pred['prob_home_win'], fixture_row.get('AvgH')),
            'draw': (dc_pred['prob_draw'], fixture_row.get('AvgD')),
            'away': (dc_pred['prob_away_win'], fixture_row.get('AvgA'))
        }
        
        best_edge = 0
        best_market = None
        best_prob = 0
        best_odds = 0
        
        for market, (prob, odds) in markets.items():
            if pd.isna(odds) or odds <= 1:
                continue
            
            implied = 1 / odds
            edge = prob - implied
            
            if edge > best_edge:
                best_edge = edge
                best_market = market
                best_prob = prob
                best_odds = odds
        
        # Kelly stake
        if best_edge > 0 and best_odds > 1:
            kelly_frac = (best_prob * (best_odds - 1) - (1 - best_prob)) / (best_odds - 1)
            kelly_stake = max(0, kelly_frac * 0.25 * 10000)  # Assuming 10k bankroll
        else:
            kelly_stake = 0
        
        return {
            'date': fixture_row.get('Date'),
            'league': league,
            'home_team': home_team,
            'away_team': away_team,
            'prob_home': dc_pred['prob_home_win'],
            'prob_draw': dc_pred['prob_draw'],
            'prob_away': dc_pred['prob_away_win'],
            'exp_goals_home': dc_pred['expected_home_goals'],
            'exp_goals_away': dc_pred['expected_away_goals'],
            'ah_line': ah_result.handicap,
            'best_market': best_market or 'none',
            'best_prob': best_prob,
            'best_odds': best_odds,
            'best_edge': best_edge,
            'kelly_stake': kelly_stake
        }
    
    def load_historical_for_backtest(self) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for backtesting.
        
        Returns:
            Dictionary with 'predictions' and 'results' DataFrames
        """
        # This is a placeholder - in production, you'd load pre-generated predictions
        # For now, return empty
        return {
            'predictions': pd.DataFrame(),
            'results': pd.DataFrame()
        }


if __name__ == "__main__":
    print("="*70)
    print("PRODUCTION INFERENCE ENGINE")
    print("="*70)
    
    # Example: Training mode
    engine = InferenceEngine()
    
    print("\nMode 1: Training")
    print("  engine.train_models(['Premier League'], n_seasons=2)")
    print("  engine.save_models()")
    
    print("\nMode 2: Inference")
    print("  engine.load_models()")
    print("  predictions = engine.predict_today()")
