"""INSTITUTIONAL FIXTURES LOADER - football-data.co.uk integration"""
import pandas as pd
from datetime import datetime, timedelta

class FixturesLoader:
    def __init__(self, leagues=None):
        self.leagues = leagues or {
            'Premier League': 'E0',
            'La Liga': 'SP1',
            'Serie A': 'I1',
            'Bundesliga': 'D1',
            'Ligue 1': 'F1'
        }
    
    def load_upcoming_fixtures(self, target_date=None):
        if target_date is None:
            target_date = datetime.today().strftime("%d/%m/%y")
        
        all_fixtures = []
        for league_name, league_code in self.leagues.items():
            url = "https://www.football-data.co.uk/fixtures.csv"
            
            try:
                df = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')
                df = df[(df['Div'] == league_code) & (df['Date'] == target_date)].copy()
                
                if len(df) > 0:
                    df['League'] = league_name
                    all_fixtures.append(df)
            except:
                continue
        
        return pd.concat(all_fixtures, ignore_index=True) if all_fixtures else pd.DataFrame()
    
    def load_next_n_days(self, n_days=7):
        all_fixtures = []
        for i in range(n_days):
            date = (datetime.today() + timedelta(days=i)).strftime("%d/%m/%y")
            fixtures = self.load_upcoming_fixtures(target_date=date)
            if not fixtures.empty:
                all_fixtures.append(fixtures)
        
        return pd.concat(all_fixtures, ignore_index=True) if all_fixtures else pd.DataFrame()
    
    def prepare_odds_dict(self, fixture_row):
        odds = {}
        if 'AvgH' in fixture_row and pd.notna(fixture_row['AvgH']):
            odds['home'] = fixture_row['AvgH']
            odds['draw'] = fixture_row['AvgD']
            odds['away'] = fixture_row['AvgA']
        
        if 'Avg>2.5' in fixture_row and pd.notna(fixture_row['Avg>2.5']):
            odds['over_25'] = fixture_row['Avg>2.5']
            odds['under_25'] = fixture_row['Avg<2.5']
        
        return odds
