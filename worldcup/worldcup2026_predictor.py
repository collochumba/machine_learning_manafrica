# =============================================================================
# 🏆 2026 FIFA WORLD CUP PREDICTOR
# Professional-grade Streamlit app with ML + Dixon-Coles + Monte Carlo
# =============================================================================
# INSTALLATION:
#   pip install streamlit pandas numpy scipy scikit-learn xgboost plotly
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
# PAGE CONFIG  (must be first Streamlit call)
# =============================================================================

st.set_page_config(
    page_title="2026 FIFA World Cup Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THEME SYSTEM
# =============================================================================
# We manage our own light / dark CSS variables so the app looks great on
# BOTH Streamlit Cloud's default light theme AND dark theme, and lets the
# user override with the sidebar toggle.

# Persist choice across reruns
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"   # default

DARK = {
    "app_bg":        "#0e1117",
    "card_bg":       "#1c2333",
    "card_border":   "#2d3748",
    "header_text":   "#ffffff",
    "body_text":     "#e2e8f0",
    "muted_text":    "#8892a4",
    "accent":        "#c8a415",
    "accent2":       "#63b3ed",
    "green":         "#48bb78",
    "orange":        "#ed8936",
    "red":           "#fc8181",
    "row_border":    "#2d3748",
    "chart_bg":      "#1c2333",
    "chart_grid":    "#2d3748",
    "chart_font":    "#e2e8f0",
    "info_bg":       "#1c2333",
    "info_border":   "#2d5a9e",
    "info_left":     "#4299e1",
    "header_grad":   "linear-gradient(135deg, #1a472a 0%, #2d5a27 40%, #c8a415 100%)",
    "sidebar_grad":  "linear-gradient(135deg,#1a472a,#2d5a27)",
    "section_grad":  "linear-gradient(90deg, #1a202c, #2d3748)",
    "section_left":  "#c8a415",
    "winner_grad":   "linear-gradient(135deg, #1a472a, #2d5a27)",
    "winner_border": "#c8a415",
}

LIGHT = {
    "app_bg":        "#f7f9fc",
    "card_bg":       "#ffffff",
    "card_border":   "#d1d9e6",
    "header_text":   "#ffffff",
    "body_text":     "#1a202c",
    "muted_text":    "#4a5568",
    "accent":        "#9a7a0a",
    "accent2":       "#2b6cb0",
    "green":         "#276749",
    "orange":        "#c05621",
    "red":           "#c53030",
    "row_border":    "#e2e8f0",
    "chart_bg":      "#ffffff",
    "chart_grid":    "#e2e8f0",
    "chart_font":    "#1a202c",
    "info_bg":       "#ebf8ff",
    "info_border":   "#bee3f8",
    "info_left":     "#2b6cb0",
    "header_grad":   "linear-gradient(135deg, #1a472a 0%, #276749 40%, #9a7a0a 100%)",
    "sidebar_grad":  "linear-gradient(135deg,#1a472a,#276749)",
    "section_grad":  "linear-gradient(90deg, #edf2f7, #e2e8f0)",
    "section_left":  "#9a7a0a",
    "winner_grad":   "linear-gradient(135deg, #c6f6d5, #9ae6b4)",
    "winner_border": "#9a7a0a",
}


def T():
    """Return current theme dict."""
    return DARK if st.session_state["theme"] == "dark" else LIGHT


def inject_css(t):
    st.markdown(f"""
    <style>
        /* ---- App shell ---- */
        .stApp, .stApp > div {{
            background-color: {t['app_bg']} !important;
        }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] > div {{
            background-color: {t['card_bg']} !important;
            border-right: 1px solid {t['card_border']};
        }}

        /* ---- All text ---- */
        .stApp, .stApp p, .stApp li, .stApp span, .stApp div,
        .stMarkdown, label, .stSelectbox label, .stRadio label {{
            color: {t['body_text']} !important;
        }}

        /* ---- Inputs ---- */
        .stSelectbox > div > div,
        .stTextInput > div > div > input {{
            background-color: {t['card_bg']} !important;
            color: {t['body_text']} !important;
            border-color: {t['card_border']} !important;
        }}

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {t['card_bg']};
            border-bottom: 2px solid {t['card_border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {t['muted_text']} !important;
        }}
        .stTabs [aria-selected="true"] {{
            color: {t['accent']} !important;
            border-bottom: 2px solid {t['accent']} !important;
        }}

        /* ---- Metric card ---- */
        .metric-card {{
            background: {t['card_bg']};
            border: 1px solid {t['card_border']};
            border-radius: 10px;
            padding: 18px 22px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        .metric-card .label {{
            color: {t['muted_text']};
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }}
        .metric-card .value {{
            color: {t['body_text']};
            font-size: 1.6rem;
            font-weight: 700;
        }}
        .metric-card .sub {{
            color: {t['accent2']};
            font-size: 0.82rem;
            margin-top: 4px;
        }}

        /* ---- Header banner ---- */
        .header-banner {{
            background: {t['header_grad']};
            padding: 28px 32px;
            border-radius: 14px;
            margin-bottom: 24px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.15);
            text-align: center;
        }}
        .header-banner h1 {{
            color: {t['header_text']};
            font-size: 2.4rem;
            margin: 0;
            font-weight: 800;
            letter-spacing: 1px;
            text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        .header-banner p {{
            color: #f0e68c;
            margin: 6px 0 0 0;
            font-size: 1rem;
        }}

        /* ---- Section headers ---- */
        .section-header {{
            background: {t['section_grad']};
            border-left: 4px solid {t['section_left']};
            padding: 12px 18px;
            border-radius: 6px;
            margin: 20px 0 14px 0;
            color: {t['body_text']};
            font-weight: 600;
            font-size: 1.05rem;
        }}

        /* ---- Ranking rows ---- */
        .ranking-row {{
            display: flex;
            align-items: center;
            padding: 8px 12px;
            border-bottom: 1px solid {t['row_border']};
            color: {t['body_text']};
            font-size: 0.88rem;
        }}
        .ranking-row:hover {{ background: {t['card_bg']}; }}
        .rank-num {{ color: {t['accent']}; font-weight: 700; width: 36px; font-size: 1rem; }}

        /* ---- Info box ---- */
        .info-box {{
            background: {t['info_bg']};
            border: 1px solid {t['info_border']};
            border-left: 4px solid {t['info_left']};
            border-radius: 8px;
            padding: 14px 18px;
            color: {t['body_text']};
            font-size: 0.88rem;
            margin: 10px 0;
            line-height: 1.6;
        }}
        .info-box h4 {{ color: {t['accent']} !important; margin-top: 0; }}

        /* ---- Winner card ---- */
        .winner-card {{
            background: {t['winner_grad']};
            border: 2px solid {t['winner_border']};
            border-radius: 14px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        .winner-card h2 {{ color: {t['accent']}; font-size: 1.1rem; text-transform: uppercase; margin: 0 0 8px; }}
        .winner-card h1 {{ color: {t['body_text']}; font-size: 2rem; margin: 0; font-weight: 800; }}
        .winner-card p  {{ color: {t['accent2']}; margin: 8px 0 0; font-size: 0.9rem; }}

        /* ---- Hide Streamlit chrome ---- */
        #MainMenu  {{ visibility: hidden; }}
        footer     {{ visibility: hidden; }}
        header     {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# 2026 WORLD CUP — 48 QUALIFIED TEAMS
# =============================================================================

QUALIFIED_TEAMS = [
    # South America (6)
    "Argentina", "Brazil", "Colombia", "Ecuador", "Uruguay", "Paraguay",
    # Europe (16)
    "France", "England", "Spain", "Germany", "Portugal", "Netherlands",
    "Belgium", "Croatia", "Switzerland", "Austria", "Denmark", "Poland",
    "Turkey", "Czechia", "Bosnia and Herzegovina", "Scotland",
    # CONCACAF (6)
    "USA", "Mexico", "Canada", "Jamaica", "Costa Rica", "Panama",
    # Africa (9)
    "Morocco", "Senegal", "Egypt", "Côte d'Ivoire", "DR Congo", "Ghana",
    "Tunisia", "Nigeria", "Cameroon",
    # Asia / AFC (8)
    "Japan", "South Korea", "Saudi Arabia", "Qatar", "Iran", "Australia",
    "Uzbekistan", "Jordan",
    # Intercontinental play-off (3 — illustrative)
    "New Zealand", "Venezuela", "Algeria",
]

TEAM_FLAGS = {
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "France": "🇫🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Spain": "🇪🇸", "Germany": "🇩🇪", "Portugal": "🇵🇹", "Netherlands": "🇳🇱",
    "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Uruguay": "🇺🇾", "Colombia": "🇨🇴",
    "Mexico": "🇲🇽", "USA": "🇺🇸", "Canada": "🇨🇦", "Japan": "🇯🇵",
    "South Korea": "🇰🇷", "Morocco": "🇲🇦", "Senegal": "🇸🇳", "Egypt": "🇪🇬",
    "Australia": "🇦🇺", "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦", "Iran": "🇮🇷",
    "Switzerland": "🇨🇭", "Austria": "🇦🇹", "Turkey": "🇹🇷", "Czechia": "🇨🇿",
    "Côte d'Ivoire": "🇨🇮", "DR Congo": "🇨🇩", "Ghana": "🇬🇭", "Tunisia": "🇹🇳",
    "Algeria": "🇩🇿", "Ecuador": "🇪🇨", "Paraguay": "🇵🇾", "Panama": "🇵🇦",
    "Bosnia and Herzegovina": "🇧🇦", "Denmark": "🇩🇰", "Poland": "🇵🇱",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Jamaica": "🇯🇲", "Costa Rica": "🇨🇷",
    "Venezuela": "🇻🇪", "Nigeria": "🇳🇬", "Cameroon": "🇨🇲",
    "Uzbekistan": "🇺🇿", "Jordan": "🇯🇴", "New Zealand": "🇳🇿",
}

def flag(team):
    return TEAM_FLAGS.get(team, "🌍")


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_URL)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date","home_team","away_team","home_score","away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["result"] = np.where(
        df["home_score"] > df["away_score"], "H",
        np.where(df["home_score"] < df["away_score"], "A", "D")
    )
    df["outcome"] = df["result"].map({"H": 0, "D": 1, "A": 2})
    df = df.sort_values("date").reset_index(drop=True)
    df["days_ago"] = (df["date"].max() - df["date"]).dt.days
    return df


# =============================================================================
# SECTION 2 — ELO RATINGS
# =============================================================================

@st.cache_data(show_spinner=False)
def compute_elo(df, k=32, base=1500):
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
        if   row["result"] == "H": s_h, s_a = 1.0, 0.0
        elif row["result"] == "A": s_h, s_a = 0.0, 1.0
        else:                       s_h, s_a = 0.5, 0.5

        ratings[h] += k * (s_h - E_h)
        ratings[a] += k * (s_a - E_a)

    df["elo_home"] = home_elo_list
    df["elo_away"] = away_elo_list
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df, dict(ratings)


# =============================================================================
# SECTION 3 — FEATURE ENGINEERING
# =============================================================================

@st.cache_data(show_spinner=False)
def engineer_features(df):
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["home_pts"] = np.where(df["result"]=="H", 3, np.where(df["result"]=="D", 1, 0))
    df["away_pts"] = np.where(df["result"]=="A", 3, np.where(df["result"]=="D", 1, 0))

    for w in [5, 10]:
        df[f"h_form_{w}"] = df.groupby("home_team")["home_pts"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"a_form_{w}"] = df.groupby("away_team")["away_pts"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"h_gs_{w}"]   = df.groupby("home_team")["home_score"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"h_gc_{w}"]   = df.groupby("home_team")["away_score"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"a_gs_{w}"]   = df.groupby("away_team")["away_score"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"a_gc_{w}"]   = df.groupby("away_team")["home_score"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())

    df["attack_diff_5"]  = df["h_gs_5"]  - df["a_gc_5"]
    df["defense_diff_5"] = df["h_gc_5"]  - df["a_gs_5"]
    df["form_diff_5"]    = df["h_form_5"] - df["a_form_5"]
    df["form_diff_10"]   = df["h_form_10"] - df["a_form_10"]
    df["is_neutral"]     = df["neutral"].astype(int) if "neutral" in df.columns else 0
    df["is_wc"]          = df["tournament"].str.contains("FIFA World Cup", na=False).astype(int)
    df["is_friendly"]    = df["tournament"].str.contains("Friendly", na=False).astype(int)

    FEATURE_COLS = [
        "elo_diff",
        "h_form_5","h_form_10","a_form_5","a_form_10",
        "h_gs_5","h_gc_5","a_gs_5","a_gc_5",
        "h_gs_10","h_gc_10","a_gs_10","a_gc_10",
        "attack_diff_5","defense_diff_5","form_diff_5","form_diff_10",
        "is_neutral","is_wc","is_friendly",
    ]
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df, FEATURE_COLS


# =============================================================================
# SECTION 4 — XGBoost WALK-FORWARD TRAINING
# =============================================================================

@st.cache_resource(show_spinner=False)
def train_model(_df, feature_cols, fast_mode=False):
    # Accept both list and tuple (tuple needed for st.cache_resource hashing)
    cols = list(feature_cols)
    X = _df[cols].values
    y = _df["outcome"].values
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros((len(_df), 3))

    hp = (dict(n_estimators=100, max_depth=4, learning_rate=0.1,
               subsample=0.8, colsample_bytree=0.8) if fast_mode else
          dict(n_estimators=400, max_depth=5, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8))

    for _, (tr, te) in enumerate(tscv.split(X)):
        m = xgb.XGBClassifier(**hp,
                               eval_metric="mlogloss", random_state=42, verbosity=0)
        m.fit(X[tr], y[tr])
        oof_preds[te] = m.predict_proba(X[te])

    final = xgb.XGBClassifier(**hp,
                               eval_metric="mlogloss", random_state=42, verbosity=0)
    final.fit(X, y)
    vm = oof_preds.sum(axis=1) > 0
    return final, oof_preds, log_loss(y[vm], oof_preds[vm])


# =============================================================================
# SECTION 5 — PROBABILITY CALIBRATION
# =============================================================================

class ProbabilityCalibrator:
    def __init__(self): self.cals = []
    def fit(self, y_true, y_pred):
        self.cals = []
        for i in range(3):
            c = IsotonicRegression(out_of_bounds="clip")
            c.fit(y_pred[:, i], (y_true == i).astype(int))
            self.cals.append(c)
    def transform(self, y_pred):
        out = np.zeros_like(y_pred)
        for i, c in enumerate(self.cals):
            out[:, i] = c.transform(y_pred[:, i])
        return out / np.clip(out.sum(axis=1, keepdims=True), 1e-8, None)

@st.cache_resource(show_spinner=False)
def fit_calibrator(_df, _oof_preds):
    cal = ProbabilityCalibrator()
    cal.fit(_df["outcome"].values, _oof_preds)
    return cal


# =============================================================================
# SECTION 6 — DIXON-COLES WITH TIME DECAY
# =============================================================================

class DixonColesTimeDecay:
    def __init__(self, xi=0.001, max_goals=8):
        self.xi=xi; self.max_goals=max_goals
        self.attack={}; self.defence={}
        self.home_adv=0.15; self.rho=0.0; self.teams=[]

    def fit(self, df):
        data = df.copy().sort_values("date")
        cutoff = data["date"].max() - pd.DateOffset(years=8)
        data   = data[data["date"] >= cutoff].reset_index(drop=True)

        self.teams = sorted(set(data["home_team"]) | set(data["away_team"]))
        n   = len(self.teams)
        t2i = {t: i for i, t in enumerate(self.teams)}

        hi = data["home_team"].map(t2i).values
        ai = data["away_team"].map(t2i).values
        hg = data["home_score"].values
        ag = data["away_score"].values
        w  = np.exp(-self.xi * data["days_ago"].values)

        def nll(params):
            att  = params[:n] - params[:n].mean()
            deff = params[n:2*n]
            home = params[2*n]; rho = params[2*n+1]
            lh = np.exp(home + att[hi] - deff[ai])
            la = np.exp(att[ai]         - deff[hi])
            p  = poisson.pmf(hg, lh) * poisson.pmf(ag, la)
            corr = np.ones(len(p))
            m00=(hg==0)&(ag==0); m01=(hg==0)&(ag==1)
            m10=(hg==1)&(ag==0); m11=(hg==1)&(ag==1)
            corr[m00]=np.maximum(1-lh[m00]*la[m00]*rho,1e-6)
            corr[m01]=np.maximum(1+lh[m01]*rho,1e-6)
            corr[m10]=np.maximum(1+la[m10]*rho,1e-6)
            corr[m11]=np.maximum(1-rho,1e-6)
            return -np.sum(w * np.log(np.maximum(p*corr, 1e-12)))

        x0     = np.concatenate([np.zeros(n), np.zeros(n), [0.15], [0.0]])
        bounds = [(-3,3)]*(2*n) + [(-1,1), (-0.15,0.15)]

        try:
            res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter":250,"ftol":1e-6})
            if not res.success:
                res = minimize(nll, res.x, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter":500,"ftol":1e-4})
            params = res.x
            att    = params[:n] - params[:n].mean()
            deff   = params[n:2*n]
            self.attack   = dict(zip(self.teams, att))
            self.defence  = dict(zip(self.teams, deff))
            self.home_adv = float(params[2*n])
            self.rho      = float(params[2*n+1])
        except Exception:
            self.attack  = {t: 0.0 for t in self.teams}
            self.defence = {t: 0.0 for t in self.teams}
        return self

    def predict(self, home, away, neutral=False):
        hb  = 0.0 if neutral else self.home_adv
        lh  = np.exp(hb + self.attack.get(home,0.) - self.defence.get(away,0.))
        la  = np.exp(     self.attack.get(away,0.) - self.defence.get(home,0.))
        mg  = self.max_goals
        S   = np.outer(poisson.pmf(range(mg+1),lh), poisson.pmf(range(mg+1),la))
        rho = self.rho
        S[0,0]=max(S[0,0]*(1-lh*la*rho),1e-10)
        S[0,1]=max(S[0,1]*(1+lh*rho),   1e-10)
        S[1,0]=max(S[1,0]*(1+la*rho),   1e-10)
        S[1,1]=max(S[1,1]*(1-rho),       1e-10)
        p_over = float(S[np.add.outer(range(mg+1),range(mg+1))>2].sum())
        return {
            "lambda_home": float(lh), "lambda_away": float(la),
            "prob_home":   float(np.tril(S,-1).sum()),
            "prob_draw":   float(np.trace(S)),
            "prob_away":   float(np.triu(S,1).sum()),
            "prob_over_25": p_over, "exp_goals": float(lh+la),
            "score_matrix": S,
        }

@st.cache_resource(show_spinner=False)
def fit_dixon_coles(_df):
    dc = DixonColesTimeDecay(xi=0.001)
    dc.fit(_df)
    return dc


# =============================================================================
# SECTION 7 — ENSEMBLE PREDICTOR
# =============================================================================

def get_team_features(home, away, df, feature_cols, elo_ratings,
                      neutral=False, tournament="FIFA World Cup"):
    def stats(rows, team):
        gs,gc,pts=[],[],[]
        for _,r in rows.iterrows():
            if r["home_team"]==team:
                gs.append(r["home_score"]); gc.append(r["away_score"])
                pts.append(3 if r["result"]=="H" else 1 if r["result"]=="D" else 0)
            else:
                gs.append(r["away_score"]); gc.append(r["home_score"])
                pts.append(3 if r["result"]=="A" else 1 if r["result"]=="D" else 0)
        return (np.mean(gs) if gs else 1.2,
                np.mean(gc) if gc else 1.0,
                np.mean(pts) if pts else 1.0)

    hr = df[(df["home_team"]==home)|(df["away_team"]==home)].tail(15)
    ar = df[(df["home_team"]==away)|(df["away_team"]==away)].tail(15)
    h_gs,h_gc,h_pts = stats(hr, home)
    a_gs,a_gc,a_pts = stats(ar, away)
    elo_h = elo_ratings.get(home, 1500)
    elo_a = elo_ratings.get(away, 1500)

    feats = {
        "elo_diff": elo_h-elo_a,
        "h_form_5":h_pts,"h_form_10":h_pts,"a_form_5":a_pts,"a_form_10":a_pts,
        "h_gs_5":h_gs,"h_gc_5":h_gc,"a_gs_5":a_gs,"a_gc_5":a_gc,
        "h_gs_10":h_gs,"h_gc_10":h_gc,"a_gs_10":a_gs,"a_gc_10":a_gc,
        "attack_diff_5":h_gs-a_gc,"defense_diff_5":h_gc-a_gs,
        "form_diff_5":h_pts-a_pts,"form_diff_10":h_pts-a_pts,
        "is_neutral":int(neutral),
        "is_wc":int("World Cup" in tournament),
        "is_friendly":int("Friendly" in tournament),
    }
    return np.array([feats[c] for c in list(feature_cols)])


def ensemble_predict(home, away, df, feature_cols, elo_ratings,
                     model, calibrator, dc_model,
                     neutral=False, tournament="FIFA World Cup", dc_w=0.55):
    dc      = dc_model.predict(home, away, neutral=neutral)
    fv      = get_team_features(home, away, df, feature_cols, elo_ratings,
                                neutral=neutral, tournament=tournament)
    raw_ml  = model.predict_proba(fv.reshape(1,-1))[0]
    cal_ml  = calibrator.transform(raw_ml.reshape(1,-1))[0]

    dc_p    = np.clip([dc["prob_home"],dc["prob_draw"],dc["prob_away"]], 1e-9,1-1e-9)
    ml_p    = np.clip(cal_ml, 1e-9,1-1e-9)
    blended = softmax(dc_w*np.log(dc_p) + (1-dc_w)*np.log(ml_p))

    return {
        "prob_home": float(blended[0]), "prob_draw": float(blended[1]),
        "prob_away": float(blended[2]),
        "lambda_home": dc["lambda_home"], "lambda_away": dc["lambda_away"],
        "prob_over_25": dc["prob_over_25"], "exp_goals": dc["exp_goals"],
        "confidence": float(blended.max()), "score_matrix": dc["score_matrix"],
    }


def most_likely_score(S):
    idx = np.unravel_index(np.argmax(S), S.shape)
    return idx[0], idx[1]


# =============================================================================
# SECTION 8 — HEAD-TO-HEAD
# =============================================================================

def head_to_head(df, a, b, n=10):
    mask = (((df["home_team"]==a)&(df["away_team"]==b))|
            ((df["home_team"]==b)&(df["away_team"]==a)))
    return df[mask].sort_values("date", ascending=False).head(n)


# =============================================================================
# SECTION 9 — TEAM STRENGTH TABLE
# =============================================================================

def build_strength_table(df, elo_ratings):
    rows = []
    for team in QUALIFIED_TEAMS:
        elo    = elo_ratings.get(team, 1500)
        recent = df[(df["home_team"]==team)|(df["away_team"]==team)].sort_values("date").tail(20)
        w=d=l=gf=ga=0
        for _,r in recent.iterrows():
            if r["home_team"]==team:
                gf+=r["home_score"]; ga+=r["away_score"]
                if r["result"]=="H": w+=1
                elif r["result"]=="D": d+=1
                else: l+=1
            else:
                gf+=r["away_score"]; ga+=r["home_score"]
                if r["result"]=="A": w+=1
                elif r["result"]=="D": d+=1
                else: l+=1
        p = w+d+l
        rows.append({"Team":team,"Elo":round(elo),"W":w,"D":d,"L":l,
                     "GF":gf,"GA":ga,"GD":gf-ga,
                     "Win%":round(w/p*100,1) if p else 0})
    return pd.DataFrame(rows).sort_values("Elo",ascending=False).reset_index(drop=True)


# =============================================================================
# SECTION 10 — TOURNAMENT SIMULATOR
# =============================================================================

WC2026_GROUPS = {
    "A": ["USA",        "Morocco",          "Poland",               "Panama"],
    "B": ["Mexico",     "Argentina",        "Cameroon",             "Jamaica"],
    "C": ["Canada",     "Brazil",           "Belgium",              "New Zealand"],
    "D": ["France",     "Uruguay",          "South Korea",          "Algeria"],
    "E": ["Spain",      "Germany",          "Japan",                "Costa Rica"],
    "F": ["Portugal",   "Colombia",         "Saudi Arabia",         "Tunisia"],
    "G": ["England",    "Netherlands",      "Iran",                 "Venezuela"],
    "H": ["Croatia",    "Ecuador",          "Senegal",              "Scotland"],
    "I": ["Switzerland","Paraguay",         "Egypt",                "Jordan"],
    "J": ["Austria",    "Denmark",          "Nigeria",              "Australia"],
    "K": ["Turkey",     "Czechia",          "DR Congo",             "Uzbekistan"],
    "L": ["Bosnia and Herzegovina","Côte d'Ivoire","Qatar",         "Ghana"],
}

def _ko_match(h, a, dc_model):
    pred = dc_model.predict(h, a, neutral=True)
    ph = pred["prob_home"] / max(pred["prob_home"]+pred["prob_away"], 1e-9)
    return h if random.random() < ph else a

@st.cache_data(show_spinner=False)
def run_monte_carlo(_dc_model, _elo_ratings, n_sims=5000):
    champ_ct   = defaultdict(int)
    top4_ct    = defaultdict(int)
    final_ct   = defaultdict(int)

    for _ in range(n_sims):
        # --- Group stage ---
        qualified = []
        for teams in WC2026_GROUPS.values():
            pts = defaultdict(int); gd = defaultdict(int)
            for i in range(len(teams)):
                for j in range(i+1, len(teams)):
                    h,a = teams[i], teams[j]
                    pred = _dc_model.predict(h, a, neutral=True)
                    r = random.random()
                    if r < pred["prob_home"]:
                        pts[h]+=3; gd[h]+=1; gd[a]-=1
                    elif r < pred["prob_home"]+pred["prob_draw"]:
                        pts[h]+=1; pts[a]+=1
                    else:
                        pts[a]+=3; gd[a]+=1; gd[h]-=1
            ranked = sorted(teams, key=lambda t:(pts[t],gd[t]), reverse=True)
            qualified.extend(ranked[:3])   # top 3 from each group

        # Pad/trim to 32 for clean bracket
        random.shuffle(qualified)
        while len(qualified) < 32: qualified.append(random.choice(qualified))
        qualified = qualified[:32]

        # --- R32 → R16 → QF → SF → Final ---
        bracket = qualified[:]
        for round_size in [32, 16, 8, 4]:
            next_r = []
            for i in range(0, round_size, 2):
                next_r.append(_ko_match(bracket[i], bracket[i+1], _dc_model))
            bracket = next_r

        # bracket now = [winner] after Final
        # Track SF losers = top4
        # Re-run to track properly:
        b = qualified[:]
        for rnd in ["R32","R16","QF"]:
            b = [_ko_match(b[i],b[i+1],_dc_model) for i in range(0,len(b),2)]
        # SF (4 teams)
        sf_w, sf_l = [], []
        for i in range(0,4,2):
            w = _ko_match(b[i],b[i+1],_dc_model)
            l = b[i] if w==b[i+1] else b[i+1]
            sf_w.append(w); sf_l.append(l)
        for t in sf_l: top4_ct[t]+=1
        for t in sf_w: final_ct[t]+=1; top4_ct[t]+=1
        champ = _ko_match(sf_w[0], sf_w[1], _dc_model)
        champ_ct[champ]+=1

    n = n_sims
    return ({t:v/n*100 for t,v in champ_ct.items()},
            {t:v/n*100 for t,v in top4_ct.items()},
            {t:v/n*100 for t,v in final_ct.items()})


# =============================================================================
# SECTION 11 — CHART HELPERS  (theme-aware)
# =============================================================================

def chart_layout(t):
    return dict(
        paper_bgcolor=t["chart_bg"],
        plot_bgcolor =t["chart_bg"],
        font=dict(color=t["chart_font"], family="Inter, sans-serif"),
        margin=dict(l=10,r=10,t=40,b=10),
    )

def prob_bar_chart(home, away, p_h, p_d, p_a, t):
    fig = go.Figure()
    for cat,val,col in zip(
        [f"🏠 {home}","Draw",f"✈️ {away}"],
        [p_h*100, p_d*100, p_a*100],
        [t["green"], t["orange"], t["red"]],
    ):
        fig.add_trace(go.Bar(x=[val],y=[cat],orientation="h",marker_color=col,
                             text=[f"{val:.1f}%"],textposition="outside",showlegend=False))
    fig.update_layout(**chart_layout(t), height=180,
        xaxis=dict(range=[0,100],showgrid=True,gridcolor=t["chart_grid"],ticksuffix="%"),
        yaxis=dict(showgrid=False), bargap=0.3,
        title=dict(text="Win Probability",font=dict(size=14,color=t["accent"])))
    return fig

def score_heatmap(S, home, away, t, max_g=6):
    M = S[:max_g+1,:max_g+1]*100
    cs = ([[0,t["chart_bg"]],[0.5,"#2b6cb0"],[1.0,t["accent"]]]
          if st.session_state["theme"]=="dark"
          else [[0,"#ebf8ff"],[0.5,"#3182ce"],[1.0,"#9a7a0a"]])
    fig = go.Figure(go.Heatmap(
        z=M, x=[str(i) for i in range(max_g+1)], y=[str(i) for i in range(max_g+1)],
        colorscale=cs,
        text=[[f"{M[i,j]:.1f}%" for j in range(max_g+1)] for i in range(max_g+1)],
        texttemplate="%{text}",
        hovertemplate=f"{home}: %{{y}} – {away}: %{{x}}<br>Prob: %{{text}}<extra></extra>",
        showscale=False,
    ))
    fig.update_layout(**chart_layout(t), height=300,
        xaxis=dict(title=f"{away} Goals"),
        yaxis=dict(title=f"{home} Goals"),
        title=dict(text="Score Probability Matrix (%)",font=dict(size=13,color=t["accent"])))
    return fig

def champion_bar(champ_pct, t, top_n=16):
    df_c = (pd.DataFrame(list(champ_pct.items()),columns=["Team","Pct"])
              .sort_values("Pct",ascending=True).tail(top_n))
    cols = [t["accent"] if i==len(df_c)-1 else t["accent2"] if i>=len(df_c)-4 else t["muted_text"]
            for i in range(len(df_c))]
    fig = go.Figure(go.Bar(
        x=df_c["Pct"], y=df_c["Team"], orientation="h",
        marker_color=cols, text=[f"{v:.1f}%" for v in df_c["Pct"]],
        textposition="outside", showlegend=False,
    ))
    fig.update_layout(**chart_layout(t), height=460,
        xaxis=dict(ticksuffix="%",showgrid=True,gridcolor=t["chart_grid"],
                   range=[0,df_c["Pct"].max()*1.28]),
        yaxis=dict(showgrid=False),
        title=dict(text=f"Top {top_n} — Championship Probability",
                   font=dict(size=14,color=t["accent"])))
    return fig

def elo_scatter(sdf, t):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sdf["Win%"], y=sdf["Elo"], mode="markers+text",
        text=sdf["Team"], textposition="top center", textfont=dict(size=9),
        marker=dict(
            size=sdf["Elo"].apply(lambda e: 6+(e-1400)/30).clip(6,20),
            color=sdf["Elo"],
            colorscale=([[0,t["muted_text"]],[0.5,t["accent2"]],[1.0,t["accent"]]]
                        if st.session_state["theme"]=="dark"
                        else [[0,"#e2e8f0"],[0.5,"#3182ce"],[1.0,"#9a7a0a"]]),
            showscale=False,
        ),
        hovertemplate="<b>%{text}</b><br>Elo: %{y}<br>Win%: %{x}%<extra></extra>",
    ))
    fig.update_layout(**chart_layout(t), height=500,
        xaxis=dict(title="Win % (last 20)",showgrid=True,gridcolor=t["chart_grid"]),
        yaxis=dict(title="Elo Rating",     showgrid=True,gridcolor=t["chart_grid"]),
        title=dict(text="Team Strength — Elo vs Recent Win %",
                   font=dict(size=14,color=t["accent"])))
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():

    # ---- Sidebar ----
    with st.sidebar:
        # Theme toggle at very top
        st.markdown("### 🎨 Display")
        new_theme = st.radio(
            "Color theme",
            ["🌙 Dark", "☀️ Light"],
            index=0 if st.session_state["theme"]=="dark" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state["theme"] = "dark" if new_theme=="🌙 Dark" else "light"

        t = T()   # active theme dict from here on

        st.markdown("---")
        st.markdown(f"""
        <div style="background:{t['sidebar_grad'] if 'gradient' not in t else ''};
                    background:{t['header_grad']};
                    padding:16px; border-radius:10px; margin-bottom:16px; text-align:center;">
            <div style="font-size:2rem;">🏆</div>
            <div style="color:#c8a415; font-weight:700; font-size:1.1rem;">WC 2026 Predictor</div>
            <div style="color:#90cdf4; font-size:0.78rem; margin-top:4px;">ML + Dixon-Coles Ensemble</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Settings")
        fast_mode = st.toggle(
            "⚡ Fast Mode",
            value=False,
            help="Lighter XGBoost (100 trees). ~3× faster startup. Recommended for first demo run.",
        )
        st.caption("⚡ Fast" if fast_mode else "🎯 Full accuracy mode")

        st.markdown("---")
        st.markdown(f"""
        <div style="color:{t['muted_text']}; font-size:0.78rem; line-height:1.7;">
        <b>Data:</b> martj42/international_results<br>
        <b>Model:</b> XGBoost + Dixon-Coles<br>
        <b>Calibration:</b> Isotonic regression<br>
        <b>Sim:</b> Monte Carlo (up to 10k runs)<br><br>
        <i>All models cached after first run.</i>
        </div>
        """, unsafe_allow_html=True)

    # Resolve active theme (sidebar toggle already updated session_state)
    t = T()

    # Inject CSS for current theme
    inject_css(t)

    # ---- Header ----
    st.markdown("""
    <div class="header-banner">
        <h1>🏆 2026 FIFA World Cup Predictor</h1>
        <p>ML + Dixon-Coles Ensemble &nbsp;·&nbsp; Monte Carlo Simulator &nbsp;·&nbsp; 48-Team Tournament</p>
    </div>
    """, unsafe_allow_html=True)

    # ================================================================
    # LOAD & TRAIN
    # ================================================================

    with st.spinner("📡 Downloading match data..."): df_raw = load_data()
    with st.spinner("📐 Computing Elo ratings..."): df_elo, elo_ratings = compute_elo(df_raw)
    with st.spinner("🔧 Engineering features..."): df_feat, feature_cols = engineer_features(df_elo)
    with st.spinner("🤖 Training XGBoost..."):
        final_model, oof_preds, oof_ll = train_model(df_feat, tuple(feature_cols), fast_mode=fast_mode)
    with st.spinner("📊 Calibrating probabilities..."): calibrator = fit_calibrator(df_feat, oof_preds)
    with st.spinner("⚙️ Fitting Dixon-Coles..."):       dc_model   = fit_dixon_coles(df_raw)

    # Stats bar
    c1,c2,c3,c4 = st.columns(4)
    for col,lbl,val,sub in zip(
        [c1,c2,c3,c4],
        ["Total Matches","Date Range","Teams","OOF Log-Loss"],
        [f"{len(df_raw):,}",
         f"{df_raw['date'].min().year}–{df_raw['date'].max().year}",
         str(df_raw['home_team'].nunique()), f"{oof_ll:.4f}"],
        ["Historical dataset","Coverage","Unique nations","Model accuracy"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{lbl}</div>
            <div class="value">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ================================================================
    # TABS
    # ================================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Match Predictor","🏆 Tournament Simulator","📊 Team Rankings","ℹ️ Methodology"
    ])

    # ---- TAB 1: MATCH PREDICTOR ----
    with tab1:
        st.markdown('<div class="section-header">⚽ Select Match Parameters</div>',
                    unsafe_allow_html=True)
        col_l, col_r = st.columns([1,2])

        with col_l:
            home_team = st.selectbox("🏠 Home / Team A", QUALIFIED_TEAMS,
                                     index=QUALIFIED_TEAMS.index("Argentina"))
            away_opts = [x for x in QUALIFIED_TEAMS if x != home_team]
            away_team = st.selectbox("✈️ Away / Team B", away_opts,
                                     index=away_opts.index("France") if "France" in away_opts else 0)
            venue = st.radio("📍 Venue", ["Neutral Ground","Home Advantage"], horizontal=True)
            stage = st.selectbox("🏟️ Stage",
                                 ["Group Stage","Round of 32","Quarter-Final","Semi-Final","Final"])
            predict_btn = st.button("⚡ Predict Match", use_container_width=True, type="primary")

        with col_r:
            # Auto-render on first load; re-render on button click
            if "pred_cache" not in st.session_state or predict_btn:
                neutral = (venue == "Neutral Ground")
                pred = ensemble_predict(
                    home_team, away_team, df_feat, feature_cols, elo_ratings,
                    final_model, calibrator, dc_model, neutral=neutral, tournament=stage)
                st.session_state["pred_cache"] = (home_team, away_team, pred)
            else:
                home_team, away_team, pred = st.session_state["pred_cache"]

            p_h=pred["prob_home"]; p_d=pred["prob_draw"]; p_a=pred["prob_away"]
            sh,sa = most_likely_score(pred["score_matrix"])

            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        background:{t['card_bg']};border:1px solid {t['card_border']};
                        border-radius:10px;padding:18px 24px;margin-bottom:14px;">
                <div style="text-align:center;">
                    <div style="font-size:2.5rem;">{flag(home_team)}</div>
                    <div style="color:{t['body_text']};font-weight:700;font-size:1.1rem;margin-top:4px;">{home_team}</div>
                    <div style="color:{t['green']};font-size:1.6rem;font-weight:800;">{p_h*100:.1f}%</div>
                </div>
                <div style="text-align:center;">
                    <div style="color:{t['accent']};font-size:1.2rem;font-weight:600;">VS</div>
                    <div style="color:{t['muted_text']};font-size:0.85rem;margin-top:6px;">
                        Draw: {p_d*100:.1f}%<br>
                        🎯 {sh} – {sa}<br>
                        xG: {pred['lambda_home']:.2f} – {pred['lambda_away']:.2f}
                    </div>
                    <div style="color:{t['accent2']};font-size:0.8rem;margin-top:4px;">
                        Confidence: {pred['confidence']*100:.0f}%
                    </div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:2.5rem;">{flag(away_team)}</div>
                    <div style="color:{t['body_text']};font-weight:700;font-size:1.1rem;margin-top:4px;">{away_team}</div>
                    <div style="color:{t['red']};font-size:1.6rem;font-weight:800;">{p_a*100:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            cp,ch = st.columns(2)
            with cp: st.plotly_chart(prob_bar_chart(home_team,away_team,p_h,p_d,p_a,t),
                                     use_container_width=True)
            with ch: st.plotly_chart(score_heatmap(pred["score_matrix"],home_team,away_team,t),
                                     use_container_width=True)

            st.markdown('<div class="section-header">📋 Market Breakdown</div>',
                        unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            for col_m,(lbl,val,ico) in zip([m1,m2,m3,m4],[
                ("Over 2.5 Goals",  f"{pred['prob_over_25']*100:.1f}%",       "🎯"),
                ("Under 2.5 Goals", f"{(1-pred['prob_over_25'])*100:.1f}%",   "🛡️"),
                ("Double Chance 1X",f"{(p_h+p_d)*100:.1f}%",                  "🔒"),
                ("Double Chance X2",f"{(p_d+p_a)*100:.1f}%",                  "🔓"),
            ]):
                col_m.markdown(f"""
                <div class="metric-card">
                    <div class="label">{ico} {lbl}</div>
                    <div class="value">{val}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header">🤝 Head-to-Head (Last 10)</div>',
                        unsafe_allow_html=True)
            h2h = head_to_head(df_raw, home_team, away_team)
            if h2h.empty:
                st.info(f"No H2H data found for {home_team} vs {away_team}.")
            else:
                for _,r in h2h.iterrows():
                    is_home = r["home_team"]==home_team
                    rc = (t["green"] if (is_home and r["result"]=="H") or (not is_home and r["result"]=="A")
                          else t["orange"] if r["result"]=="D" else t["red"])
                    st.markdown(f"""
                    <div class="ranking-row">
                        <span style="color:{t['muted_text']};width:90px;">{r['date'].strftime('%d %b %Y')}</span>
                        <span style="flex:1;">{flag(r['home_team'])} {r['home_team']}</span>
                        <span style="font-weight:700;color:{rc};width:50px;text-align:center;">
                            {int(r['home_score'])}–{int(r['away_score'])}
                        </span>
                        <span style="flex:1;text-align:right;">{r['away_team']} {flag(r['away_team'])}</span>
                        <span style="color:{t['muted_text']};font-size:0.78rem;margin-left:12px;width:130px;text-align:right;">
                            {str(r.get('tournament',''))[:28]}
                        </span>
                    </div>""", unsafe_allow_html=True)

    # ---- TAB 2: TOURNAMENT SIMULATOR ----
    with tab2:
        st.markdown('<div class="section-header">🌍 Monte Carlo World Cup Simulator</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
            Simulates the complete 2026 FIFA World Cup (48 teams, 12 groups × 4) using Dixon-Coles
            match probabilities. Each run: <b>Group Stage → R32 → R16 → QF → SF → Final</b>.
        </div>""", unsafe_allow_html=True)

        n_sims_opt = st.select_slider("Simulations", [1000,2500,5000,10000], value=5000)
        sim_btn = st.button("🚀 Run Tournament Simulation", type="primary", use_container_width=True)

        if sim_btn:
            with st.spinner(f"🎲 Running {n_sims_opt:,} simulations..."):
                champ_pct, top4_pct, final_pct = run_monte_carlo(dc_model, elo_ratings, n_sims=n_sims_opt)

            if not champ_pct:
                st.error("Simulation failed — check team name coverage.")
            else:
                top_team = max(champ_pct, key=champ_pct.get)
                st.markdown(f"""
                <div class="winner-card">
                    <h2>🏆 Predicted Champion</h2>
                    <h1>{flag(top_team)} {top_team}</h1>
                    <p>Championship probability: <b>{champ_pct[top_team]:.1f}%</b>
                       over {n_sims_opt:,} simulations</p>
                </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown('<div class="section-header">🏅 Top Contenders</div>',
                            unsafe_allow_html=True)
                top_sorted = sorted(champ_pct, key=champ_pct.get, reverse=True)[:16]
                tc1,tc2 = st.columns(2)
                for idx,team in enumerate(top_sorted[:8]):
                    tc1.markdown(f"""
                    <div class="ranking-row">
                        <span class="rank-num">#{idx+1}</span>
                        <span style="flex:1;">{flag(team)} {team}</span>
                        <span style="color:{t['accent']};font-weight:700;">{champ_pct[team]:.1f}%</span>
                        <span style="color:{t['accent2']};margin-left:12px;width:70px;text-align:right;">
                            Top4: {top4_pct.get(team,0):.0f}%</span>
                    </div>""", unsafe_allow_html=True)
                for idx,team in enumerate(top_sorted[8:],start=9):
                    tc2.markdown(f"""
                    <div class="ranking-row">
                        <span class="rank-num">#{idx}</span>
                        <span style="flex:1;">{flag(team)} {team}</span>
                        <span style="color:{t['accent']};font-weight:700;">{champ_pct[team]:.1f}%</span>
                        <span style="color:{t['accent2']};margin-left:12px;width:70px;text-align:right;">
                            Top4: {top4_pct.get(team,0):.0f}%</span>
                    </div>""", unsafe_allow_html=True)

                st.plotly_chart(champion_bar(champ_pct, t, top_n=16), use_container_width=True)

                st.markdown('<div class="section-header">📋 2026 Group Stage Draw</div>',
                            unsafe_allow_html=True)
                g_cols = st.columns(4)
                for gi,(gname,teams) in enumerate(WC2026_GROUPS.items()):
                    col = g_cols[gi % 4]
                    lines = "".join(
                        f"<div style='padding:3px 0;border-bottom:1px solid {t['row_border']};'>"
                        f"{flag(tm)} {tm}</div>" for tm in teams)
                    col.markdown(f"""
                    <div style="background:{t['card_bg']};border-radius:8px;padding:12px;
                                margin-bottom:12px;border:1px solid {t['card_border']};">
                        <div style="color:{t['accent']};font-weight:700;font-size:0.9rem;margin-bottom:6px;">
                            GROUP {gname}</div>
                        <div style="color:{t['body_text']};font-size:0.82rem;">{lines}</div>
                    </div>""", unsafe_allow_html=True)

    # ---- TAB 3: TEAM RANKINGS ----
    with tab3:
        st.markdown('<div class="section-header">📊 Team Strength & Rankings</div>',
                    unsafe_allow_html=True)
        with st.spinner("Building strength table..."):
            sdf = build_strength_table(df_raw, elo_ratings)
        st.plotly_chart(elo_scatter(sdf, t), use_container_width=True)

        st.markdown('<div class="section-header">📋 Full Rankings Table</div>',
                    unsafe_allow_html=True)
        search = st.text_input("🔍 Filter team", placeholder="Type team name...")
        ddf = sdf[sdf["Team"].str.contains(search, case=False)] if search else sdf.copy()
        ddf.insert(0,"Rank",range(1,len(ddf)+1))

        hcols = st.columns([1,4,2,1,1,1,1,1,1,2])
        for hc,hl in zip(hcols,["#","Team","Elo","W","D","L","GF","GA","GD","Win%"]):
            hc.markdown(f"<span style='color:{t['muted_text']};font-size:0.78rem;"
                        f"text-transform:uppercase;'>{hl}</span>", unsafe_allow_html=True)

        for _,row in ddf.head(48).iterrows():
            ec = t["accent"] if row["Elo"]>1700 else t["accent2"] if row["Elo"]>1600 else t["muted_text"]
            rc = st.columns([1,4,2,1,1,1,1,1,1,2])
            for ci,v in zip(rc,[
                f"<b style='color:{t['accent']}'>{int(row['Rank'])}</b>",
                f"{flag(row['Team'])} {row['Team']}",
                f"<span style='color:{ec}'>{int(row['Elo'])}</span>",
                str(int(row['W'])),str(int(row['D'])),str(int(row['L'])),
                str(int(row['GF'])),str(int(row['GA'])),
                f"{'+'if row['GD']>0 else ''}{int(row['GD'])}",
                f"<b>{row['Win%']}%</b>",
            ]):
                ci.markdown(
                    f"<div style='padding:6px 0;border-bottom:1px solid {t['row_border']};"
                    f"color:{t['body_text']};font-size:0.86rem;'>{v}</div>",
                    unsafe_allow_html=True)

    # ---- TAB 4: METHODOLOGY ----
    with tab4:
        st.markdown('<div class="section-header">📖 Model Architecture & Methodology</div>',
                    unsafe_allow_html=True)
        for heading,body in [
            ("🧠 Ensemble Design",
             """This predictor uses a <b>log-odds ensemble</b>:<br>
             • <b>Dixon-Coles Poisson (55%)</b> — time-decay weighting (ξ=0.001), rho low-score correction,
               per-team attack/defence parameters<br>
             • <b>XGBoost (45%)</b> — walk-forward TimeSeriesSplit(5), Elo + rolling form + matchup features<br>
             Combined via log-pooling: P ∝ P_dc^0.55 × P_ml^0.45, normalised with softmax."""),
            ("📐 Features",
             """• <b>Elo differential</b> — K=32 dynamic rating on full historical data<br>
             • <b>Rolling form (5 & 10)</b> — average PPG in last N matches<br>
             • <b>Rolling attack/defence</b> — goals scored/conceded averages<br>
             • <b>Matchup differentials</b> — home attack vs away defence<br>
             • <b>Venue & tournament flags</b> — neutral ground, World Cup, friendly"""),
            ("🎲 Tournament Simulation",
             """• 12 groups × 4 teams — top 3 from each group qualify (36 teams)<br>
             • Padded to 32 for R32 → R16 → QF → SF → Final<br>
             • No draws in knockout (head-to-head DC probability only)<br>
             • 5k–10k simulations → ±1–2% confidence intervals"""),
            ("📊 Data & Requirements",
             """<b>Data:</b> <code>martj42/international_results</code> — international matches since 1872<br>
             <b>requirements.txt</b> (place in same folder as app.py — Streamlit Cloud auto-detects it):<br>
             <code>streamlit pandas numpy scipy scikit-learn xgboost plotly</code>"""),
            ("🚀 Improvements",
             """• Live FIFA rankings API integration<br>
             • Player squad strength / injury flags<br>
             • Real xG data (StatsBomb / Opta)<br>
             • Bayesian tournament-pressure prior<br>
             • Grid-search optimal Dixon-Coles ξ"""),
        ]:
            st.markdown(f"""
            <div class="info-box">
                <h4>{heading}</h4>
                {body}
            </div>""", unsafe_allow_html=True)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
