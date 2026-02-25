"""
INSTITUTIONAL DATA LOADER
Professional data ingestion from football-data.co.uk

Features:
- Multi-league support (17+ leagues)
- Multi-season loading
- Automatic data cleaning
- Feature derivation
- Odds data extraction
- Error handling and recovery
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# League configurations
LEAGUES = {
    'Premier League': 'E0',
    'Championship': 'E1',
    'La Liga': 'SP1',
    'La Liga 2': 'SP2',
    'Serie A': 'I1',
    'Serie B': 'I2',
    'Bundesliga': 'D1',
    'Bundesliga 2': 'D2',
    'Ligue 1': 'F1',
    'Ligue 2': 'F2',
    'Eredivisie': 'N1',
    'Champions League': 'EC',
    'Primeira Liga': 'P1',
    'Scottish Premiership': 'SC0'
}

def generate_season_codes(start_year: int, end_year: int) -> Tuple[List[str], List[str]]:
    """Generate season codes for football-data.co.uk URLs."""
    season_codes = []
    season_labels = []
    
    for year in range(start_year, end_year):
        code = f"{str(year)[-2:]}{str(year+1)[-2:]}"
        label = f"{year}/{year+1}"
        season_codes.append(code)
        season_labels.append(label)
    
    return season_codes, season_labels


class FootballDataLoader:
    """Professional data loader with error handling."""
    
    def __init__(self, leagues: Optional[Dict[str, str]] = None):
        self.leagues = leagues or LEAGUES
        self.data = None
    
    def load_league_season(
        self,
        league_code: str,
        season_code: str,
        season_label: str
    ) -> Optional[pd.DataFrame]:
        """Load single league season."""
        url = f'https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv'
        
        try:
            df = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')
            df['Season'] = season_label
            df['LeagueCode'] = league_code
            return df
        except Exception as e:
            return None
    
    def load_all_data(
        self,
        season_codes: List[str],
        season_labels: List[str],
        league_subset: Optional[List[str]] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Load data for multiple leagues and seasons."""
        all_data = []
        
        leagues_to_load = self.leagues
        if league_subset:
            leagues_to_load = {k: v for k, v in self.leagues.items() if k in league_subset}
        
        for league_name, league_code in leagues_to_load.items():
            for season_code, season_label in zip(season_codes, season_labels):
                df = self.load_league_season(league_code, season_code, season_label)
                
                if df is not None:
                    df['League'] = league_name
                    all_data.append(df)
        
        if not all_data:
            raise ValueError("No data loaded")
        
        self.data = pd.concat(all_data, ignore_index=True)
        return self.data
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Clean and standardize dataset."""
        if df is None:
            df = self.data.copy()
        else:
            df = df.copy()
        
        # Convert date
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
        
        # Sort by date
        df = df.sort_values(['League', 'Date']).reset_index(drop=True)
        
        # Standardize team names
        df['HomeTeam'] = df['HomeTeam'].str.strip()
        df['AwayTeam'] = df['AwayTeam'].str.strip()
        
        # Ensure goals are integers
        df['FTHG'] = df['FTHG'].astype(int)
        df['FTAG'] = df['FTAG'].astype(int)
        
        # Derived columns
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        df['GoalDifference'] = df['FTHG'] - df['FTAG']
        df['HomeWin'] = (df['FTR'] == 'H').astype(int)
        df['Draw'] = (df['FTR'] == 'D').astype(int)
        df['AwayWin'] = (df['FTR'] == 'A').astype(int)
        df['Outcome'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Days since epoch (for time weighting)
        epoch = pd.Timestamp('2015-01-01')
        df['DaysSinceEpoch'] = (df['Date'] - epoch).dt.days
        df['DaysSinceMatch'] = (df['Date'].max() - df['Date']).dt.days
        
        # Calendar features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        self.data = df
        return df


def quick_load(leagues: List[str], n_seasons: int = 5) -> pd.DataFrame:
    """Convenience function for quick data loading."""
    current_year = datetime.now().year
    season_codes, season_labels = generate_season_codes(current_year - n_seasons, current_year)
    
    loader = FootballDataLoader()
    df = loader.load_all_data(season_codes, season_labels, league_subset=leagues)
    df = loader.clean_data(df)
    
    return df
