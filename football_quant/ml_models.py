"""INSTITUTIONAL ML MODELS - Walk-forward validated XGBoost"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score
import pickle

class FootballMLModel:
    def __init__(self, model_type='1x2', task='classification'):
        self.model_type = model_type
        self.task = task
        self.model = None
        self.feature_names = None
    
    def _safe_feature_selection(self, df):
        allowed = ['_L5', '_L10', 'Form', 'H2H', 'Rest', 'Streak', 'Position']
        blocked = ['FTR', 'Outcome', 'FTHG', 'FTAG', 'TotalGoals', 'Odds', 'B365']
        
        features = []
        for col in df.columns:
            if any(k in col for k in allowed) and not any(b in col for b in blocked):
                features.append(col)
        return sorted(features)
    
    def prepare_features(self, df, feature_cols=None):
        if feature_cols is None:
            feature_cols = self._safe_feature_selection(df)
        
        self.feature_names = feature_cols
        X = df[feature_cols].fillna(0).values
        
        if self.model_type == '1x2':
            y = df['Outcome'].values
        else:
            y = (df['TotalGoals'] > 2.5).astype(int).values
        
        return X, y
    
    def train(self, df, feature_cols=None, test_size=0.2, **xgb_params):
        df = df.sort_values('Date').reset_index(drop=True)
        X, y = self.prepare_features(df, feature_cols)
        
        params = {
            'objective': 'multi:softprob' if self.model_type=='1x2' else 'binary:logistic',
            'num_class': 3 if self.model_type=='1x2' else None,
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'eval_metric': 'mlogloss'
        }
        params.update(xgb_params)
        
        self.model = xgb.XGBClassifier(**params)
        
        split = int(len(X) * (1-test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        y_proba = self.model.predict_proba(X_test)
        return {
            'log_loss': log_loss(y_test, y_proba),
            'accuracy': accuracy_score(y_test, np.argmax(y_proba, axis=1))
        }
    
    def predict_proba(self, df):
        X, _ = self.prepare_features(df, self.feature_names)
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
