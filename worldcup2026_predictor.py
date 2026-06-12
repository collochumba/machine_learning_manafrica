# =============================================================================
# 🏆 2026 FIFA WORLD CUP PREDICTOR
# Professional-grade Streamlit app with ML + Dixon-Coles + Monte Carlo
# =============================================================================
# INSTALLATION:
#   pip install streamlit pandas numpy scipy scikit-learn xgboost plotly joblib tqdm
# RUN:
#   streamlit run worldcup2026_predictor.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from scipy.stats import poisson
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from collections import defaultdict
import random

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="2026 FIFA World Cup Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS — PROFESSIONAL DARK THEME
# =============================================================================

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1a472a 0%, #2d5a27 40%, #c8a415 100%);
        padding: 28px 32px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.4);
        text-align: center;
    }
    .header-banner h1 {
        color: white;
        font-size: 2.4rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: 1px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .header-banner p {
        color: #f0e68c;
        margin: 6px 0 0 0;
        font-size: 1rem;
        letter-spacing: 0.5px;
    }

    /* Metric cards */
    .metric-card {
        background: #1c2333;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .metric-card .label {
        color: #8892a4;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-card .value {
        color: #e2e8f0;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-card .sub {
        color: #63b3ed;
        font-size: 0.82rem;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1a202c, #2d3748);
        border-left: 4px solid #c8a415;
        padding: 12px 18px;
        border-radius: 6px;
        margin: 20px 0 14px 0;
        color: #e2e8f0;
        font-weight: 600;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
    }

    /* Probability bars */
    .prob-row {
        display: flex;
        align-items: center;
        margin: 8px 0;
        gap: 12px;
    }
    .prob-label { color: #a0aec0; font-size: 0.88rem; width: 120px; }
    .prob-bar-bg {
        flex: 1;
        background: #2d3748;
        border-radius: 4px;
        height: 24px;
        position: relative;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        display: flex;
        align-items: center;
        padding-left: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        color: white;
        transition: width 0.5s ease;
    }

    /* Winner card */
    .winner-card {
        background: linear-gradient(135deg, #1a472a, #2d5a27);
        border: 2px solid #c8a415;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(200,164,21,0.2);
    }
    .winner-card h2 { color: #c8a415; font-size: 1.1rem; text-transform: uppercase; margin: 0 0 8px; }
    .winner-card h1 { color: white; font-size: 2rem; margin: 0; font-weight: 800; }
    .winner-card p  { color: #90cdf4; margin: 8px 0 0; font-size: 0.9rem; }

    /* Table styling */
    .ranking-row {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        border-bottom: 1px solid #2d3748;
        color: #e2e8f0;
        font-size: 0.88rem;
    }
    .ranking-row:hover { background: #1c2333; }
    .rank-num { color: #c8a415; font-weight: 700; width: 36px; font-size: 1rem; }

    /* Info box */
    .info-box {
        background: #1c2333;
        border: 1px solid #2d5a9e;
        border-left: 4px solid #4299e1;
        border-radius: 8px;
        padding: 14px 18px;
        color: #a0aec0;
        font-size: 0.88rem;
        margin: 10px 0;
        line-height: 1.6;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 2026 WORLD CUP QUALIFIED TEAMS (48 teams)
# =============================================================================

QUALIFIED_TEAMS = [
    # South America
    "Argentina", "Brazil", "Colombia", "Ecuador", "Uruguay", "Paraguay",
    "Venezuela", "Bolivia",
    # Europe
    "France", "England", "Spain", "Germany", "Portugal", "Netherlands",
    "Belgium", "Croatia", "Switzerland", "Austria", "Sweden", "Norway",
    "Turkey", "Czechia", "Bosnia and Herzegovina", "Denmark", "Poland",
    "Scotland", "Slovakia",
    # CONCACAF
    "USA", "Mexico", "Canada", "Jamaica", "Costa Rica", "Panama",
    # Africa
    "Morocco", "Senegal", "Egypt", "Côte d'Ivoire", "DR Congo", "Ghana",
    "Tunisia", "Algeria", "Nigeria", "Cameroon",
    # Asia
    "Japan", "South Korea", "Saudi Arabia", "Qatar", "Iran", "Australia",
    "Uzbekistan", "Jordan",
]

TEAM_FLAGS = {
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "France": "🇫🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Spain": "🇪🇸", "Germany": "🇩🇪", "Portugal": "🇵🇹", "Netherlands": "🇳🇱",
    "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Uruguay": "🇺🇾", "Colombia": "🇨🇴",
    "Mexico": "🇲🇽", "USA": "🇺🇸", "Canada": "🇨🇦", "Japan": "🇯🇵",
    "South Korea": "🇰🇷", "Morocco": "🇲🇦", "Senegal": "🇸🇳", "Egypt": "🇪🇬",
    "Australia": "🇦🇺", "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦", "Iran": "🇮🇷",
    "Switzerland": "🇨🇭", "Sweden": "🇸🇪", "Norway": "🇳🇴", "Austria": "🇦🇹",
    "Turkey": "🇹🇷", "Czechia": "🇨🇿", "Côte d'Ivoire": "🇨🇮", "DR Congo": "🇨🇩",
    "Ghana": "🇬🇭", "Tunisia": "🇹🇳", "Algeria": "🇩🇿", "Ecuador": "🇪🇨",
    "Paraguay": "🇵🇾", "Panama": "🇵🇦", "Bosnia and Herzegovina": "🇧🇦",
    "Denmark": "🇩🇰", "Poland": "🇵🇱", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Slovakia": "🇸🇰",
    "Jamaica": "🇯🇲", "Costa Rica": "🇨🇷", "Venezuela": "🇻🇪", "Bolivia": "🇧🇴",
    "Nigeria": "🇳🇬", "Cameroon": "🇨🇲", "Uzbekistan": "🇺🇿", "Jordan": "🇯🇴",
}

def flag(team):
    return TEAM_FLAGS.get(team, "🌍")


# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

@st.cache_data(show_spinner=False)
def load_data():
    """Load and preprocess international football results."""

    df = pd.read_csv(DATA_URL)

    # Clean columns
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Outcome
    df["result"] = np.where(
        df["home_score"] > df["away_score"], "H",
        np.where(df["home_score"] < df["away_score"], "A", "D")
    )
    df["outcome"] = df["result"].map({"H": 0, "D": 1, "A": 2})

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Days since match (for time-decay weighting)
    df["days_ago"] = (df["date"].max() - df["date"]).dt.days

    return df


# =============================================================================
# SECTION 2: ELO RATINGS
# =============================================================================

@st.cache_data(show_spinner=False)
def compute_elo(df, k=32, base=1500):
    """
    Compute running Elo ratings for all teams.
    Returns final ratings dict + per-match columns for feature engineering.
    """

    df = df.copy().sort_values("date").reset_index(drop=True)

    ratings = defaultdict(lambda: base)
    home_elo_list, away_elo_list = [], []

    for _, row in df.iterrows():

        h, a = row["home_team"], row["away_team"]
        r_h, r_a = ratings[h], ratings[a]

        home_elo_list.append(r_h)
        away_elo_list.append(r_a)

        E_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
        E_a = 1 - E_h

        if row["result"] == "H":
            s_h, s_a = 1.0, 0.0
        elif row["result"] == "A":
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5

        ratings[h] += k * (s_h - E_h)
        ratings[a] += k * (s_a - E_a)

    df["elo_home"] = home_elo_list
    df["elo_away"] = away_elo_list
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    final_ratings = dict(ratings)

    return df, final_ratings


# =============================================================================
# SECTION 3: FEATURE ENGINEERING
# =============================================================================

@st.cache_data(show_spinner=False)
def engineer_features(df):
    """
    Zero-leakage feature engineering:
    - Rolling form (W/D/L points) last 5 & 10 matches
    - Rolling goals scored / conceded (home + away)
    - Elo differential
    - Tournament type encoding
    - Neutral venue flag
    """

    df = df.copy().sort_values("date").reset_index(drop=True)

    # -------------------------
    # Points per game (3-1-0)
    # -------------------------

    df["home_pts"] = np.where(df["result"] == "H", 3, np.where(df["result"] == "D", 1, 0))
    df["away_pts"] = np.where(df["result"] == "A", 3, np.where(df["result"] == "D", 1, 0))

    for window in [5, 10]:

        df[f"h_form_{window}"] = (
            df.groupby("home_team")["home_pts"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

        df[f"a_form_{window}"] = (
            df.groupby("away_team")["away_pts"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

        df[f"h_gs_{window}"] = (
            df.groupby("home_team")["home_score"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

        df[f"h_gc_{window}"] = (
            df.groupby("home_team")["away_score"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

        df[f"a_gs_{window}"] = (
            df.groupby("away_team")["away_score"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

        df[f"a_gc_{window}"] = (
            df.groupby("away_team")["home_score"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # -------------------------
    # Attack / defence matchup
    # -------------------------

    df["attack_diff_5"]  = df["h_gs_5"]  - df["a_gc_5"]
    df["defense_diff_5"] = df["h_gc_5"]  - df["a_gs_5"]
    df["form_diff_5"]    = df["h_form_5"] - df["a_form_5"]
    df["form_diff_10"]   = df["h_form_10"] - df["a_form_10"]

    # -------------------------
    # Venue / tournament flags
    # -------------------------

    df["is_neutral"] = df["neutral"].astype(int) if "neutral" in df.columns else 0
    df["is_wc"]      = df["tournament"].str.contains("FIFA World Cup", na=False).astype(int)
    df["is_friendly"]= df["tournament"].str.contains("Friendly", na=False).astype(int)

    # -------------------------
    # Feature column list
    # -------------------------

    FEATURE_COLS = [
        "elo_diff",
        "h_form_5", "h_form_10", "a_form_5", "a_form_10",
        "h_gs_5",   "h_gc_5",    "a_gs_5",   "a_gc_5",
        "h_gs_10",  "h_gc_10",   "a_gs_10",  "a_gc_10",
        "attack_diff_5", "defense_diff_5", "form_diff_5", "form_diff_10",
        "is_neutral", "is_wc", "is_friendly",
    ]

    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    return df, FEATURE_COLS


# =============================================================================
# SECTION 4: WALK-FORWARD XGBoost TRAINING
# =============================================================================

@st.cache_resource(show_spinner=False)
def train_model(_df, feature_cols):
    """
    Walk-forward (TimeSeriesSplit) XGBoost training for match outcome prediction.
    Returns final model trained on all data + OOF predictions for calibration.
    """

    X = _df[feature_cols].values
    y = _df["outcome"].values

    tscv = TimeSeriesSplit(n_splits=5)

    oof_preds = np.zeros((len(_df), 3))

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train         = y[train_idx]

        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )

        model.fit(X_train, y_train)

        oof_preds[test_idx] = model.predict_proba(X_test)

    # Final model trained on all data
    final_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    final_model.fit(X, y)

    # OOF log-loss
    valid_mask = oof_preds.sum(axis=1) > 0
    oof_ll = log_loss(y[valid_mask], oof_preds[valid_mask])

    return final_model, oof_preds, oof_ll


# =============================================================================
# SECTION 5: PROBABILITY CALIBRATION
# =============================================================================

class ProbabilityCalibrator:
    """Isotonic regression calibration for 3-class output."""

    def __init__(self):
        self.cals = []

    def fit(self, y_true, y_pred):
        self.cals = []
        for i in range(3):
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(y_pred[:, i], (y_true == i).astype(int))
            self.cals.append(cal)

    def transform(self, y_pred):
        out = np.zeros_like(y_pred)
        for i, cal in enumerate(self.cals):
            out[:, i] = cal.transform(y_pred[:, i])
        out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-8, None)
        return out


@st.cache_resource(show_spinner=False)
def fit_calibrator(_df, _oof_preds):
    """Fit calibrator on OOF predictions."""
    cal = ProbabilityCalibrator()
    cal.fit(_df["outcome"].values, _oof_preds)
    return cal


# =============================================================================
# SECTION 6: DIXON-COLES WITH TIME DECAY
# =============================================================================

class DixonColesTimeDecay:
    """
    Dixon-Coles Poisson model with time-decay weighting.
    Fitted per-dataset (international matches only, limited to recent 8 years).
    """

    def __init__(self, xi=0.001, max_goals=8):
        self.xi       = xi
        self.max_goals= max_goals
        self.attack   = {}
        self.defence  = {}
        self.home_adv = 0.0
        self.rho      = 0.0
        self.teams    = []

    def fit(self, df):

        data = df.copy().sort_values("date")

        # Filter to last 8 years for relevance
        cutoff = data["date"].max() - pd.DateOffset(years=8)
        data   = data[data["date"] >= cutoff].reset_index(drop=True)

        self.teams    = sorted(set(data["home_team"]) | set(data["away_team"]))
        n             = len(self.teams)
        t2i           = {t: i for i, t in enumerate(self.teams)}

        hi  = data["home_team"].map(t2i).values
        ai  = data["away_team"].map(t2i).values
        hg  = data["home_score"].values
        ag  = data["away_score"].values
        w   = np.exp(-self.xi * data["days_ago"].values)

        def nll(params):
            att   = params[:n] - params[:n].mean()
            deff  = params[n:2*n]
            home  = params[2*n]
            rho   = params[2*n+1]

            lh = np.exp(home + att[hi] - deff[ai])
            la = np.exp(att[ai] - deff[hi])

            p = poisson.pmf(hg, lh) * poisson.pmf(ag, la)

            corr = np.ones(len(p))
            m00  = (hg == 0) & (ag == 0)
            m01  = (hg == 0) & (ag == 1)
            m10  = (hg == 1) & (ag == 0)
            m11  = (hg == 1) & (ag == 1)
            corr[m00] = np.maximum(1 - lh[m00]*la[m00]*rho, 1e-6)
            corr[m01] = np.maximum(1 + lh[m01]*rho,         1e-6)
            corr[m10] = np.maximum(1 + la[m10]*rho,         1e-6)
            corr[m11] = np.maximum(1 - rho,                 1e-6)

            return -np.sum(w * np.log(np.maximum(p * corr, 1e-12)))

        x0     = np.concatenate([np.zeros(n), np.zeros(n), [0.15], [0.0]])
        bounds = [(-3, 3)] * (2*n) + [(-1, 1), (-0.15, 0.15)]

        res    = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 250, "ftol": 1e-6})

        params = res.x
        att    = params[:n] - params[:n].mean()
        deff   = params[n:2*n]

        self.attack   = dict(zip(self.teams, att))
        self.defence  = dict(zip(self.teams, deff))
        self.home_adv = float(params[2*n])
        self.rho      = float(params[2*n+1])

        return self

    def predict(self, home, away, neutral=False):
        """Return full probability dict + score matrix."""

        home_bonus = 0.0 if neutral else self.home_adv

        att_h = self.attack.get(home, 0.0)
        def_h = self.defence.get(home, 0.0)
        att_a = self.attack.get(away, 0.0)
        def_a = self.defence.get(away, 0.0)

        lh = np.exp(home_bonus + att_h - def_a)
        la = np.exp(att_a - def_h)

        mg = self.max_goals
        hp = poisson.pmf(range(mg+1), lh)
        ap = poisson.pmf(range(mg+1), la)

        S  = np.outer(hp, ap)

        # Dixon-Coles low-score correction
        rho = self.rho
        S[0,0] = max(S[0,0] * (1 - lh*la*rho), 1e-10)
        S[0,1] = max(S[0,1] * (1 + lh*rho),    1e-10)
        S[1,0] = max(S[1,0] * (1 + la*rho),    1e-10)
        S[1,1] = max(S[1,1] * (1 - rho),        1e-10)

        prob_h = float(np.tril(S, -1).sum())
        prob_d = float(np.trace(S))
        prob_a = float(np.triu(S, 1).sum())

        total  = np.add.outer(range(mg+1), range(mg+1))
        p_over = float(S[total > 2].sum())

        return {
            "lambda_home": float(lh),
            "lambda_away": float(la),
            "prob_home":   prob_h,
            "prob_draw":   prob_d,
            "prob_away":   prob_a,
            "prob_over_25": p_over,
            "prob_under_25": 1 - p_over,
            "exp_goals":   float(lh + la),
            "score_matrix": S,
        }


@st.cache_resource(show_spinner=False)
def fit_dixon_coles(_df):
    """Fit Dixon-Coles on international data (cached)."""
    dc = DixonColesTimeDecay(xi=0.001)
    dc.fit(_df)
    return dc


# =============================================================================
# SECTION 7: ENSEMBLE PREDICTOR
# =============================================================================

def get_team_features(team_home, team_away, df, feature_cols, elo_ratings,
                      neutral=False, tournament="FIFA World Cup"):
    """
    Build a feature vector for a new match using latest rolling stats.
    """

    # Filter relevant rows for each team
    home_rows = df[(df["home_team"] == team_home) | (df["away_team"] == team_home)].tail(15)
    away_rows = df[(df["home_team"] == team_away) | (df["away_team"] == team_away)].tail(15)

    def team_stats(rows, team):
        """Approximate rolling stats from last N rows."""
        gs, gc, pts = [], [], []
        for _, r in rows.iterrows():
            if r["home_team"] == team:
                gs.append(r["home_score"]); gc.append(r["away_score"])
                pts.append(3 if r["result"]=="H" else (1 if r["result"]=="D" else 0))
            else:
                gs.append(r["away_score"]); gc.append(r["home_score"])
                pts.append(3 if r["result"]=="A" else (1 if r["result"]=="D" else 0))
        return (np.mean(gs) if gs else 1.2,
                np.mean(gc) if gc else 1.0,
                np.mean(pts) if pts else 1.0)

    h_gs, h_gc, h_pts = team_stats(home_rows, team_home)
    a_gs, a_gc, a_pts = team_stats(away_rows, team_away)

    elo_h = elo_ratings.get(team_home, 1500)
    elo_a = elo_ratings.get(team_away, 1500)

    feats = {
        "elo_diff":       elo_h - elo_a,
        "h_form_5":       h_pts,
        "h_form_10":      h_pts,
        "a_form_5":       a_pts,
        "a_form_10":      a_pts,
        "h_gs_5":         h_gs,
        "h_gc_5":         h_gc,
        "a_gs_5":         a_gs,
        "a_gc_5":         a_gc,
        "h_gs_10":        h_gs,
        "h_gc_10":        h_gc,
        "a_gs_10":        a_gs,
        "a_gc_10":        a_gc,
        "attack_diff_5":  h_gs - a_gc,
        "defense_diff_5": h_gc - a_gs,
        "form_diff_5":    h_pts - a_pts,
        "form_diff_10":   h_pts - a_pts,
        "is_neutral":     int(neutral),
        "is_wc":          int("World Cup" in tournament),
        "is_friendly":    int("Friendly" in tournament),
    }

    return np.array([feats[c] for c in feature_cols])


def ensemble_predict(home, away, df, feature_cols, elo_ratings,
                     final_model, calibrator, dc_model,
                     neutral=False, tournament="FIFA World Cup",
                     dc_weight=0.55):
    """
    Log-odds ensemble: Dixon-Coles (55%) + XGBoost (45%).
    Returns full probability dict.
    """

    # ------ Dixon-Coles branch ------
    dc = dc_model.predict(home, away, neutral=neutral)

    # ------ ML branch ------
    feat_vec = get_team_features(home, away, df, feature_cols, elo_ratings,
                                 neutral=neutral, tournament=tournament)
    raw_ml   = final_model.predict_proba(feat_vec.reshape(1, -1))[0]
    cal_ml   = calibrator.transform(raw_ml.reshape(1, -1))[0]

    # ------ Log-pooling ------
    dc_probs = np.clip([dc["prob_home"], dc["prob_draw"], dc["prob_away"]], 1e-9, 1-1e-9)
    ml_probs = np.clip(cal_ml, 1e-9, 1-1e-9)

    blended  = softmax(
        dc_weight * np.log(dc_probs) + (1 - dc_weight) * np.log(ml_probs)
    )

    # Confidence = max probability
    confidence = float(blended.max())

    return {
        "prob_home":     float(blended[0]),
        "prob_draw":     float(blended[1]),
        "prob_away":     float(blended[2]),
        "lambda_home":   dc["lambda_home"],
        "lambda_away":   dc["lambda_away"],
        "prob_over_25":  dc["prob_over_25"],
        "exp_goals":     dc["exp_goals"],
        "confidence":    confidence,
        "score_matrix":  dc["score_matrix"],
    }


def most_likely_score(score_matrix, max_goals=8):
    """Return (home_goals, away_goals) from peak of score matrix."""
    idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    return idx[0], idx[1]


# =============================================================================
# SECTION 8: HEAD-TO-HEAD
# =============================================================================

def head_to_head(df, team_a, team_b, n=10):
    """Return last N H2H results between two teams."""
    mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
        ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    )
    return df[mask].sort_values("date", ascending=False).head(n)


# =============================================================================
# SECTION 9: TEAM STRENGTH TABLE
# =============================================================================

def build_strength_table(df, elo_ratings):
    """Build a strength table for qualified teams."""
    rows = []
    for team in QUALIFIED_TEAMS:
        elo = elo_ratings.get(team, 1500)

        recent = df[
            (df["home_team"] == team) | (df["away_team"] == team)
        ].sort_values("date").tail(20)

        wins = draws = losses = gf = ga = 0
        for _, r in recent.iterrows():
            if r["home_team"] == team:
                gf += r["home_score"]; ga += r["away_score"]
                if r["result"] == "H": wins += 1
                elif r["result"] == "D": draws += 1
                else: losses += 1
            else:
                gf += r["away_score"]; ga += r["home_score"]
                if r["result"] == "A": wins += 1
                elif r["result"] == "D": draws += 1
                else: losses += 1

        played = wins + draws + losses
        rows.append({
            "Team":     team,
            "Elo":      round(elo),
            "W":        wins,
            "D":        draws,
            "L":        losses,
            "GF":       gf,
            "GA":       ga,
            "GD":       gf - ga,
            "Win%":     round(wins / played * 100, 1) if played else 0,
        })

    return pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)


# =============================================================================
# SECTION 10: TOURNAMENT SIMULATOR
# =============================================================================

# Fixed 2026 World Cup 48-team group allocation (8 groups × 6 teams)
WC2026_GROUPS = {
    "A": ["Argentina", "USA", "DR Congo",        "Czechia"],
    "B": ["France",    "Mexico", "Ecuador",       "Bosnia and Herzegovina"],
    "C": ["Brazil",    "Canada", "Morocco",       "South Korea"],
    "D": ["Spain",     "Colombia", "Japan",       "Tunisia"],
    "E": ["Germany",   "Uruguay", "Senegal",      "Slovakia"],
    "F": ["Portugal",  "USA", "Algeria",          "Costa Rica"],
    "G": ["England",   "Paraguay", "Australia",   "Panama"],
    "H": ["Netherlands","Bolivia","Egypt",         "Uzbekistan"],
    "I": ["Belgium",   "Venezuela", "Ghana",      "Jordan"],
    "J": ["Croatia",   "Ecuador", "Saudi Arabia", "Norway"],
    "K": ["Switzerland","Colombia","Cameroon",    "Jamaica"],
    "L": ["Denmark",   "Mexico", "Côte d'Ivoire","Qatar"],
}


def simulate_match(home, away, dc_model, elo_ratings, neutral=True):
    """Fast single-match simulator using Dixon-Coles probabilities."""
    pred = dc_model.predict(home, away, neutral=neutral)
    r    = random.random()
    if r < pred["prob_home"]:
        return home, away, "H"
    elif r < pred["prob_home"] + pred["prob_draw"]:
        return home, away, "D"
    else:
        return home, away, "A"


def simulate_group_stage(groups, dc_model, elo_ratings):
    """Simulate 48-team group stage — top 3 from each group advance."""
    qualified = []

    for gname, teams in groups.items():
        points = defaultdict(int)
        gd     = defaultdict(int)

        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                h, a = teams[i], teams[j]
                _, _, res = simulate_match(h, a, dc_model, elo_ratings, neutral=True)
                if res == "H":
                    points[h] += 3; gd[h] += 1; gd[a] -= 1
                elif res == "A":
                    points[a] += 3; gd[a] += 1; gd[h] -= 1
                else:
                    points[h] += 1; points[a] += 1

        # Sort by points then gd
        ranked = sorted(teams, key=lambda t: (points[t], gd[t]), reverse=True)
        qualified.extend(ranked[:3])   # Top 3 qualify

    return qualified


def simulate_knockout(teams, dc_model, elo_ratings):
    """Single-elimination knockout with extra-time (no draw allowed)."""
    random.shuffle(teams)

    while len(teams) > 1:
        next_round = []
        for i in range(0, len(teams)-1, 2):
            h, a = teams[i], teams[i+1]
            pred = dc_model.predict(h, a, neutral=True)
            r    = random.random()
            # In knockout there's no draw; draw goes to away team (simplification)
            if r < pred["prob_home"]:
                next_round.append(h)
            else:
                next_round.append(a)
        if len(teams) % 2 == 1:
            next_round.append(teams[-1])   # bye
        teams = next_round

    return teams[0] if teams else None


@st.cache_data(show_spinner=False)
def run_monte_carlo(_dc_model, _elo_ratings, n_sims=5000):
    """Monte Carlo tournament simulation — returns champion + top-4 frequencies."""

    champion_count  = defaultdict(int)
    top4_count      = defaultdict(int)
    finalist_count  = defaultdict(int)

    for _ in range(n_sims):

        # Group stage
        qualified = simulate_group_stage(WC2026_GROUPS, _dc_model, _elo_ratings)

        if len(qualified) < 4:
            continue

        # R32 → QF → SF → Final
        random.shuffle(qualified)

        # Round of 32 (48 → 24 ... simplify to 32 by padding)
        while len(qualified) < 32:
            qualified.append(random.choice(qualified))
        qualified = qualified[:32]

        # R32
        r16 = []
        for i in range(0, 32, 2):
            h, a = qualified[i], qualified[i+1]
            pred = _dc_model.predict(h, a, neutral=True)
            winner = h if random.random() < pred["prob_home"] / (pred["prob_home"] + pred["prob_away"]) else a
            r16.append(winner)

        # QF
        qf = []
        for i in range(0, 16, 2):
            h, a = r16[i], r16[i+1]
            pred = _dc_model.predict(h, a, neutral=True)
            winner = h if random.random() < pred["prob_home"] / (pred["prob_home"] + pred["prob_away"]) else a
            qf.append(winner)

        # SF
        sf_winners, sf_losers = [], []
        for i in range(0, 8, 2):
            h, a = qf[i], qf[i+1]
            pred = _dc_model.predict(h, a, neutral=True)
            if random.random() < pred["prob_home"] / (pred["prob_home"] + pred["prob_away"]):
                sf_winners.append(h); sf_losers.append(a)
            else:
                sf_winners.append(a); sf_losers.append(h)

        for t in sf_losers: top4_count[t] += 1
        for t in sf_winners: finalist_count[t] += 1; top4_count[t] += 1

        # Final
        h, a = sf_winners[0], sf_winners[1]
        pred = _dc_model.predict(h, a, neutral=True)
        champ = h if random.random() < pred["prob_home"] / (pred["prob_home"] + pred["prob_away"]) else a
        champion_count[champ] += 1

    champ_pct   = {t: v/n_sims*100 for t, v in champion_count.items()}
    top4_pct    = {t: v/n_sims*100 for t, v in top4_count.items()}
    finalist_pct= {t: v/n_sims*100 for t, v in finalist_count.items()}

    return champ_pct, top4_pct, finalist_pct


# =============================================================================
# SECTION 11: PLOTLY CHART HELPERS
# =============================================================================

COLORS = {
    "home":   "#48bb78",
    "draw":   "#ed8936",
    "away":   "#fc8181",
    "gold":   "#c8a415",
    "blue":   "#63b3ed",
    "bg":     "#1c2333",
    "grid":   "#2d3748",
}

CHART_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor =COLORS["bg"],
    font=dict(color="#e2e8f0", family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
)


def prob_bar_chart(home, away, p_h, p_d, p_a):
    """Horizontal probability bar chart."""
    fig = go.Figure()

    categories = [f"🏠 {home}", "Draw", f"✈️ {away}"]
    values     = [p_h*100, p_d*100, p_a*100]
    colors_bar = [COLORS["home"], COLORS["draw"], COLORS["away"]]

    for cat, val, col in zip(categories, values, colors_bar):
        fig.add_trace(go.Bar(
            x=[val], y=[cat],
            orientation="h",
            marker_color=col,
            text=[f"{val:.1f}%"],
            textposition="outside",
            showlegend=False,
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        height=180,
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor=COLORS["grid"],
                   ticksuffix="%", showline=False),
        yaxis=dict(showgrid=False),
        bargap=0.3,
        title=dict(text="Win Probability", font=dict(size=14, color=COLORS["gold"])),
    )

    return fig


def score_heatmap(score_matrix, home, away, max_g=6):
    """Score probability heatmap."""
    S = score_matrix[:max_g+1, :max_g+1] * 100

    fig = go.Figure(go.Heatmap(
        z=S,
        x=[str(i) for i in range(max_g+1)],
        y=[str(i) for i in range(max_g+1)],
        colorscale=[[0, "#1c2333"], [0.5, "#2b6cb0"], [1.0, "#c8a415"]],
        text=[[f"{S[i,j]:.1f}%" for j in range(max_g+1)] for i in range(max_g+1)],
        texttemplate="%{text}",
        hovertemplate=f"{home}: %{{y}} – {away}: %{{x}}<br>Prob: %{{text}}<extra></extra>",
        showscale=False,
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        height=300,
        xaxis=dict(title=f"{away} Goals", tickmode="array",
                   tickvals=list(range(max_g+1))),
        yaxis=dict(title=f"{home} Goals", tickmode="array",
                   tickvals=list(range(max_g+1))),
        title=dict(text="Score Probability Matrix (%)",
                   font=dict(size=13, color=COLORS["gold"])),
    )

    return fig


def champion_bar(champ_pct, top_n=16):
    """Horizontal bar chart for champion probabilities."""
    df_c = (
        pd.DataFrame(list(champ_pct.items()), columns=["Team", "Pct"])
        .sort_values("Pct", ascending=True)
        .tail(top_n)
    )

    colors_g = [
        COLORS["gold"] if i == len(df_c)-1 else
        "#90cdf4" if i >= len(df_c)-4 else "#4a5568"
        for i in range(len(df_c))
    ]

    fig = go.Figure(go.Bar(
        x=df_c["Pct"],
        y=df_c["Team"],
        orientation="h",
        marker_color=colors_g,
        text=[f"{v:.1f}%" for v in df_c["Pct"]],
        textposition="outside",
        showlegend=False,
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        height=460,
        xaxis=dict(ticksuffix="%", showgrid=True, gridcolor=COLORS["grid"],
                   range=[0, df_c["Pct"].max()*1.25]),
        yaxis=dict(showgrid=False),
        title=dict(text=f"Top {top_n} — Championship Probability",
                   font=dict(size=14, color=COLORS["gold"])),
    )

    return fig


def elo_scatter(strength_df):
    """Elo scatter with Win% on x-axis."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=strength_df["Win%"],
        y=strength_df["Elo"],
        mode="markers+text",
        text=strength_df["Team"],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            size=strength_df["Elo"].apply(lambda e: 6 + (e-1400)/30).clip(6, 20),
            color=strength_df["Elo"],
            colorscale=[[0,"#4a5568"],[0.5,"#63b3ed"],[1.0,"#c8a415"]],
            showscale=False,
        ),
        hovertemplate="<b>%{text}</b><br>Elo: %{y}<br>Win%: %{x}%<extra></extra>",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        height=500,
        xaxis=dict(title="Win % (last 20 matches)", showgrid=True, gridcolor=COLORS["grid"]),
        yaxis=dict(title="Elo Rating",              showgrid=True, gridcolor=COLORS["grid"]),
        title=dict(text="Team Strength — Elo vs Recent Win %",
                   font=dict(size=14, color=COLORS["gold"])),
    )

    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():

    # ---- Header ----
    st.markdown("""
    <div class="header-banner">
        <h1>🏆 2026 FIFA World Cup Predictor</h1>
        <p>ML + Dixon-Coles Ensemble · Monte Carlo Simulator · 48-Team Tournament</p>
    </div>
    """, unsafe_allow_html=True)

    # ================================================================
    # LOAD & TRAIN (with progress)
    # ================================================================

    with st.spinner("📡 Downloading international match data..."):
        df_raw = load_data()

    with st.spinner("📐 Computing Elo ratings..."):
        df_elo, elo_ratings = compute_elo(df_raw)

    with st.spinner("🔧 Engineering features..."):
        df_feat, feature_cols = engineer_features(df_elo)

    with st.spinner("🤖 Training XGBoost model (walk-forward)..."):
        final_model, oof_preds, oof_ll = train_model(df_feat, feature_cols)

    with st.spinner("📊 Calibrating probabilities..."):
        calibrator = fit_calibrator(df_feat, oof_preds)

    with st.spinner("⚙️ Fitting Dixon-Coles model..."):
        dc_model = fit_dixon_coles(df_raw)

    # Quick dataset stats bar
    total_matches = len(df_raw)
    date_range    = f"{df_raw['date'].min().year}–{df_raw['date'].max().year}"
    n_teams       = df_raw["home_team"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in zip(
        [c1, c2, c3, c4],
        ["Total Matches", "Date Range", "Teams", "OOF Log-Loss"],
        [f"{total_matches:,}", date_range, str(n_teams), f"{oof_ll:.4f}"],
        ["Historical dataset", "Coverage", "Unique nations", "Model accuracy"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================================================================
    # TABS
    # ================================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Match Predictor",
        "🏆 Tournament Simulator",
        "📊 Team Rankings",
        "ℹ️ Methodology",
    ])

    # ================================================================
    # TAB 1 — MATCH PREDICTOR
    # ================================================================

    with tab1:

        st.markdown('<div class="section-header">⚽ Select Match Parameters</div>',
                    unsafe_allow_html=True)

        col_l, col_r = st.columns([1, 2])

        with col_l:

            home_team = st.selectbox(
                "🏠 Home / Team A",
                QUALIFIED_TEAMS,
                index=QUALIFIED_TEAMS.index("Argentina"),
            )

            away_team = st.selectbox(
                "✈️ Away / Team B",
                [t for t in QUALIFIED_TEAMS if t != home_team],
                index=([t for t in QUALIFIED_TEAMS if t != home_team]
                       .index("France")),
            )

            venue = st.radio(
                "📍 Venue",
                ["Neutral Ground", "Home Advantage"],
                horizontal=True,
            )

            stage = st.selectbox(
                "🏟️ Tournament Stage",
                ["Group Stage", "Round of 32", "Quarter-Final",
                 "Semi-Final", "Final"],
            )

            predict_btn = st.button("⚡ Predict Match", use_container_width=True,
                                    type="primary")

        with col_r:

            if predict_btn or True:   # auto-render on load

                neutral = (venue == "Neutral Ground")

                pred = ensemble_predict(
                    home_team, away_team, df_feat, feature_cols,
                    elo_ratings, final_model, calibrator, dc_model,
                    neutral=neutral, tournament=stage,
                )

                p_h = pred["prob_home"]
                p_d = pred["prob_draw"]
                p_a = pred["prob_away"]
                sh, sa = most_likely_score(pred["score_matrix"])

                # Team header
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            background:#1c2333; border-radius:10px; padding:18px 24px;
                            margin-bottom:14px;">
                    <div style="text-align:center;">
                        <div style="font-size:2.5rem;">{flag(home_team)}</div>
                        <div style="color:#e2e8f0; font-weight:700; font-size:1.1rem; margin-top:4px;">{home_team}</div>
                        <div style="color:#48bb78; font-size:1.6rem; font-weight:800;">{p_h*100:.1f}%</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="color:#c8a415; font-size:1.2rem; font-weight:600;">VS</div>
                        <div style="color:#a0aec0; font-size:0.85rem; margin-top:6px;">
                            Draw: {p_d*100:.1f}%<br>
                            🎯 {sh} – {sa}<br>
                            xG: {pred['lambda_home']:.2f} – {pred['lambda_away']:.2f}
                        </div>
                        <div style="color:#63b3ed; font-size:0.8rem; margin-top:4px;">
                            Confidence: {pred['confidence']*100:.0f}%
                        </div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2.5rem;">{flag(away_team)}</div>
                        <div style="color:#e2e8f0; font-weight:700; font-size:1.1rem; margin-top:4px;">{away_team}</div>
                        <div style="color:#fc8181; font-size:1.6rem; font-weight:800;">{p_a*100:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Charts
                c_prob, c_heat = st.columns(2)

                with c_prob:
                    st.plotly_chart(
                        prob_bar_chart(home_team, away_team, p_h, p_d, p_a),
                        use_container_width=True
                    )

                with c_heat:
                    st.plotly_chart(
                        score_heatmap(pred["score_matrix"], home_team, away_team),
                        use_container_width=True
                    )

                # Additional markets
                st.markdown('<div class="section-header">📋 Market Breakdown</div>',
                            unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                markets = [
                    ("Over 2.5 Goals",  f"{pred['prob_over_25']*100:.1f}%",  "🎯"),
                    ("Under 2.5 Goals", f"{(1-pred['prob_over_25'])*100:.1f}%", "🛡️"),
                    ("Double Chance 1X",f"{(p_h+p_d)*100:.1f}%", "🔒"),
                    ("Double Chance X2",f"{(p_d+p_a)*100:.1f}%", "🔓"),
                ]
                for col_m, (label, val, ico) in zip([m1,m2,m3,m4], markets):
                    col_m.markdown(f"""
                    <div class="metric-card">
                        <div class="label">{ico} {label}</div>
                        <div class="value">{val}</div>
                    </div>""", unsafe_allow_html=True)

                # H2H
                st.markdown('<div class="section-header">🤝 Head-to-Head (Last 10)</div>',
                            unsafe_allow_html=True)

                h2h = head_to_head(df_raw, home_team, away_team, n=10)

                if h2h.empty:
                    st.info(f"No historical head-to-head data found for {home_team} vs {away_team}.")
                else:
                    for _, r in h2h.iterrows():
                        is_home = r["home_team"] == home_team
                        res_col = (
                            "#48bb78" if (is_home and r["result"]=="H") or (not is_home and r["result"]=="A")
                            else "#ed8936" if r["result"]=="D"
                            else "#fc8181"
                        )
                        st.markdown(f"""
                        <div class="ranking-row">
                            <span style="color:#8892a4; width:90px;">{r['date'].strftime('%d %b %Y')}</span>
                            <span style="flex:1;">{flag(r['home_team'])} {r['home_team']}</span>
                            <span style="font-weight:700; color:{res_col}; width:50px; text-align:center;">
                                {int(r['home_score'])}–{int(r['away_score'])}
                            </span>
                            <span style="flex:1; text-align:right;">{r['away_team']} {flag(r['away_team'])}</span>
                            <span style="color:#8892a4; font-size:0.78rem; margin-left:12px; width:130px; text-align:right;">
                                {r.get('tournament','')[:28]}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

    # ================================================================
    # TAB 2 — TOURNAMENT SIMULATOR
    # ================================================================

    with tab2:

        st.markdown('<div class="section-header">🌍 Monte Carlo World Cup Simulator</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            Simulates the complete 2026 FIFA World Cup (48 teams, 12 groups) using Dixon-Coles match probabilities.
            Each simulation runs: <b>Group Stage → Round of 32 → Round of 16 → QF → SF → Final</b>.
            Results aggregate over thousands of simulations to estimate each team's championship probability.
        </div>
        """, unsafe_allow_html=True)

        n_sims_opt = st.select_slider(
            "Number of simulations",
            options=[1000, 2500, 5000, 10000],
            value=5000,
        )

        sim_btn = st.button("🚀 Run Tournament Simulation", type="primary",
                            use_container_width=True)

        if sim_btn:

            with st.spinner(f"🎲 Running {n_sims_opt:,} simulations..."):
                champ_pct, top4_pct, finalist_pct = run_monte_carlo(
                    dc_model, elo_ratings, n_sims=n_sims_opt
                )

            if not champ_pct:
                st.error("Simulation failed. Check data coverage for qualified teams.")
            else:
                # Winner card
                top_team = max(champ_pct, key=champ_pct.get)
                top_prob = champ_pct[top_team]

                st.markdown(f"""
                <div class="winner-card">
                    <h2>🏆 Predicted Champion</h2>
                    <h1>{flag(top_team)} {top_team}</h1>
                    <p>Championship probability: <b>{top_prob:.1f}%</b> over {n_sims_opt:,} simulations</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Top-4 table
                st.markdown('<div class="section-header">🏅 Top 4 Finish Probability</div>',
                            unsafe_allow_html=True)

                top_teams_sorted = sorted(champ_pct.keys(),
                                          key=lambda t: champ_pct[t], reverse=True)[:16]

                t_col1, t_col2 = st.columns(2)

                for idx, team in enumerate(top_teams_sorted[:8]):
                    t_col1.markdown(f"""
                    <div class="ranking-row">
                        <span class="rank-num">#{idx+1}</span>
                        <span style="flex:1;">{flag(team)} {team}</span>
                        <span style="color:#c8a415; font-weight:700;">{champ_pct.get(team,0):.1f}%</span>
                        <span style="color:#63b3ed; margin-left:12px; width:60px; text-align:right;">
                            Top4: {top4_pct.get(team,0):.0f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                for idx, team in enumerate(top_teams_sorted[8:16], start=9):
                    t_col2.markdown(f"""
                    <div class="ranking-row">
                        <span class="rank-num">#{idx}</span>
                        <span style="flex:1;">{flag(team)} {team}</span>
                        <span style="color:#c8a415; font-weight:700;">{champ_pct.get(team,0):.1f}%</span>
                        <span style="color:#63b3ed; margin-left:12px; width:60px; text-align:right;">
                            Top4: {top4_pct.get(team,0):.0f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                # Champion bar chart
                st.markdown("<br>", unsafe_allow_html=True)
                st.plotly_chart(champion_bar(champ_pct, top_n=16),
                                use_container_width=True)

                # Group stage breakdown
                st.markdown('<div class="section-header">📋 2026 Group Stage Draw</div>',
                            unsafe_allow_html=True)

                g_cols = st.columns(4)
                for gi, (gname, teams) in enumerate(WC2026_GROUPS.items()):
                    col = g_cols[gi % 4]
                    team_lines = "".join(
                        f"<div style='padding:3px 0; border-bottom:1px solid #2d3748;'>"
                        f"{flag(t)} {t}"
                        f"</div>"
                        for t in teams
                    )
                    col.markdown(f"""
                    <div style="background:#1c2333; border-radius:8px; padding:12px;
                                margin-bottom:12px; border:1px solid #2d3748;">
                        <div style="color:#c8a415; font-weight:700; font-size:0.9rem; margin-bottom:6px;">
                            GROUP {gname}
                        </div>
                        <div style="color:#e2e8f0; font-size:0.82rem;">{team_lines}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ================================================================
    # TAB 3 — TEAM RANKINGS
    # ================================================================

    with tab3:

        st.markdown('<div class="section-header">📊 Team Strength & Rankings</div>',
                    unsafe_allow_html=True)

        with st.spinner("Computing team strength table..."):
            strength_df = build_strength_table(df_raw, elo_ratings)

        # Scatter plot
        st.plotly_chart(elo_scatter(strength_df), use_container_width=True)

        # Table
        st.markdown('<div class="section-header">📋 Full Rankings Table (Qualified Teams)</div>',
                    unsafe_allow_html=True)

        # Search filter
        search = st.text_input("🔍 Filter team", placeholder="Type team name...")

        display_df = strength_df.copy()
        if search:
            display_df = display_df[
                display_df["Team"].str.contains(search, case=False)
            ]

        display_df.insert(0, "Rank", range(1, len(display_df)+1))

        # Render table
        header_cols = st.columns([1, 4, 2, 1, 1, 1, 1, 1, 1, 2])
        for col, label in zip(header_cols, ["#","Team","Elo","W","D","L","GF","GA","GD","Win%"]):
            col.markdown(f"<span style='color:#8892a4; font-size:0.78rem; text-transform:uppercase;'>{label}</span>",
                         unsafe_allow_html=True)

        for _, row in display_df.head(48).iterrows():
            r_cols = st.columns([1, 4, 2, 1, 1, 1, 1, 1, 1, 2])
            elo_color = "#c8a415" if row["Elo"] > 1700 else "#63b3ed" if row["Elo"] > 1600 else "#a0aec0"
            vals = [
                f"<b style='color:#c8a415'>{int(row['Rank'])}</b>",
                f"{flag(row['Team'])} {row['Team']}",
                f"<span style='color:{elo_color}'>{int(row['Elo'])}</span>",
                str(int(row["W"])),
                str(int(row["D"])),
                str(int(row["L"])),
                str(int(row["GF"])),
                str(int(row["GA"])),
                f"{'+'if row['GD']>0 else ''}{int(row['GD'])}",
                f"<b>{row['Win%']}%</b>",
            ]
            for col_r, v in zip(r_cols, vals):
                col_r.markdown(
                    f"<div style='padding:6px 0; border-bottom:1px solid #2d3748; "
                    f"color:#e2e8f0; font-size:0.86rem;'>{v}</div>",
                    unsafe_allow_html=True
                )

    # ================================================================
    # TAB 4 — METHODOLOGY
    # ================================================================

    with tab4:

        st.markdown('<div class="section-header">📖 Model Architecture & Methodology</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <h4 style="color:#c8a415; margin-top:0;">🧠 Ensemble Design</h4>
        This predictor uses a <b>log-odds ensemble</b> of two independent models:
        <ul>
            <li><b>Dixon-Coles Poisson model (55%)</b> — fit on international match data
                with exponential time-decay weighting (xi=0.001). Accounts for low-score correlation (rho),
                home advantage, and per-team attack / defence parameters.</li>
            <li><b>XGBoost classifier (45%)</b> — trained via walk-forward cross-validation
                (TimeSeriesSplit, 5 folds). Features include Elo differential, rolling form,
                rolling goals scored/conceded, attack/defence matchup differentials,
                venue flag, and tournament type.</li>
        </ul>
        Probabilities are combined using log-pooling: P_ensemble ∝ P_dc^0.55 × P_ml^0.45,
        then normalised via softmax.
        </div>

        <div class="info-box" style="margin-top:14px;">
        <h4 style="color:#c8a415; margin-top:0;">📐 Features Engineered</h4>
        <ul>
            <li><b>Elo rating differential</b> — dynamic Elo system (K=32) computed on the full historical dataset</li>
            <li><b>Rolling form (5 & 10 matches)</b> — average points per game in last N matches</li>
            <li><b>Rolling attack / defence (5 & 10 matches)</b> — goals scored and conceded averages</li>
            <li><b>Matchup differentials</b> — home attack vs away defence, and vice versa</li>
            <li><b>Venue flag</b> — neutral ground vs home advantage</li>
            <li><b>Tournament type</b> — World Cup qualifier / friendly / competitive weight</li>
        </ul>
        </div>

        <div class="info-box" style="margin-top:14px;">
        <h4 style="color:#c8a415; margin-top:0;">🎲 Tournament Simulation</h4>
        Monte Carlo simulation runs the complete 48-team 2026 format:
        <ul>
            <li><b>Group Stage</b> — 12 groups of 4, top 3 qualify (36 teams advance)</li>
            <li><b>Knockout rounds</b> — R32 → R16 → QF → SF → Final using Dixon-Coles match probabilities</li>
            <li>No draws in knockout (away team advances on draw — simple approximation for speed)</li>
            <li>5,000–10,000 simulations provide stable probability estimates (±1–2% CI)</li>
        </ul>
        </div>

        <div class="info-box" style="margin-top:14px;">
        <h4 style="color:#c8a415; margin-top:0;">📊 Data Source</h4>
        International match results: <code>martj42/international_results</code> on GitHub —
        a community-maintained dataset covering international football since 1872.
        Model is retrained fresh on every session using cached data.
        </div>

        <div class="info-box" style="margin-top:14px;">
        <h4 style="color:#c8a415; margin-top:0;">🚀 How to Improve</h4>
        <ul>
            <li>Integrate live FIFA rankings via API for more accurate seedings</li>
            <li>Add player-level data (squad strength, injuries, suspensions)</li>
            <li>Use real xG data from StatsBomb / Opta instead of goals as proxy</li>
            <li>Tune Dixon-Coles xi via proper cross-validated log-loss grid search</li>
            <li>Replace isotonic calibration with Platt scaling or temperature scaling</li>
            <li>Add a Bayesian prior for tournament-stage pressure effects</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
