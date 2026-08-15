"""
Football Predictor V2 — Streamlit App (single file)

Same leakage-controlled, walk-forward architecture as football_predictor_v2.ipynb,
now as one self-contained Streamlit script:
  - unified feature builder (training == live, no separate averaging logic)
  - venue-independent rest days via a team-match history table
  - Elo with goal-margin multiplier + attack/defence tracks
  - Dixon-Coles with walk-forward xi tuning
  - XGBoost trained walk-forward, calibrated on genuinely valid OOF rows only
  - ensemble weight chosen by walk-forward log-loss grid search
  - Asian Handicap settled from the score matrix (no logistic approximation)
  - no-vig market pricing, EV/edge/fair-odds, transparent Bet Score

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import poisson
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Football Predictor V2", page_icon="⚽", layout="wide")

# ============================================================================
# CORE MODEL LOGIC
# (team history, unified feature builder, Elo, Dixon-Coles, XGBoost walk-forward,
#  calibration, ensemble optimization, markets, Asian Handicap, Bet Score)
# ============================================================================

from scipy.stats import poisson
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

EPS = 1e-9
WINDOWS = [3, 5, 10]

LEAGUES = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Bundesliga": "D1",
    "Ligue 1": "F1",
}
LEAGUE_NAMES = list(LEAGUES.keys())

STAT_COLS = ["GF", "GA", "Shots", "SOT", "Corners", "Fouls", "Yellow", "Red",
             "Points", "Win", "Draw", "Loss", "CleanSheet", "BTTS"]

DEFAULTS = {
    "GF": 1.20, "GA": 1.20, "Shots": 11.0, "SOT": 4.0, "Corners": 5.0,
    "Fouls": 11.0, "Yellow": 1.8, "Red": 0.05, "Points": 1.2,
    "Win": 0.33, "Draw": 0.27, "Loss": 0.40, "CleanSheet": 0.28, "BTTS": 0.50,
}

DC_XI_CANDIDATES = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004]
ENSEMBLE_WEIGHT_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MIN_TRAIN_SEASONS = 2

BET_SCORE_WEIGHTS = {"probability": 25, "ev": 30, "edge": 20, "agreement": 15, "data_quality": 10}


# ---------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------
def get_last_n_seasons(n=5):
    current_year = datetime.now().year
    if datetime.now().month < 8:
        current_year -= 1
    seasons = [f"{str(current_year - i)[-2:]}{str(current_year - i + 1)[-2:]}" for i in range(n)]
    return seasons[::-1]


def load_raw_data(n_seasons=5, leagues=None, progress_callback=None):
    leagues = leagues or LEAGUES
    seasons = get_last_n_seasons(n_seasons)

    all_data = []
    total = len(leagues) * len(seasons)
    done = 0
    for league_name, league_code in leagues.items():
        for season in seasons:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
            try:
                d = pd.read_csv(url, encoding="latin1", on_bad_lines="skip")
                d["League"] = league_name
                d["Season"] = season
                all_data.append(d)
            except Exception:
                pass
            done += 1
            if progress_callback:
                progress_callback(done / total, f"{league_name} {season}")

    if not all_data:
        raise RuntimeError("No data could be loaded from football-data.co.uk. Check your internet connection.")

    df = pd.concat(all_data, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values(["League", "Date"]).reset_index(drop=True)

    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    df = df.dropna(subset=required)
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["Outcome"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})

    for c in ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]:
        if c not in df.columns:
            df[c] = np.nan

    return df


# ---------------------------------------------------------------------
# 2. TEAM-MATCH HISTORY TABLE
# ---------------------------------------------------------------------
def build_team_history(df):
    home = pd.DataFrame({
        "League": df["League"], "Season": df["Season"], "Date": df["Date"],
        "Team": df["HomeTeam"], "Opponent": df["AwayTeam"], "Venue": "H",
        "GF": df["FTHG"], "GA": df["FTAG"],
        "Shots": df["HS"], "SOT": df["HST"], "Corners": df["HC"],
        "Fouls": df["HF"], "Yellow": df["HY"], "Red": df["HR"],
        "Points": np.where(df["FTR"] == "H", 3, np.where(df["FTR"] == "D", 1, 0)),
        "Win": (df["FTR"] == "H").astype(int), "Draw": (df["FTR"] == "D").astype(int),
        "Loss": (df["FTR"] == "A").astype(int),
        "CleanSheet": (df["FTAG"] == 0).astype(int),
        "BTTS": ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int),
    })
    away = pd.DataFrame({
        "League": df["League"], "Season": df["Season"], "Date": df["Date"],
        "Team": df["AwayTeam"], "Opponent": df["HomeTeam"], "Venue": "A",
        "GF": df["FTAG"], "GA": df["FTHG"],
        "Shots": df["AS"], "SOT": df["AST"], "Corners": df["AC"],
        "Fouls": df["AF"], "Yellow": df["AY"], "Red": df["AR"],
        "Points": np.where(df["FTR"] == "A", 3, np.where(df["FTR"] == "D", 1, 0)),
        "Win": (df["FTR"] == "A").astype(int), "Draw": (df["FTR"] == "D").astype(int),
        "Loss": (df["FTR"] == "H").astype(int),
        "CleanSheet": (df["FTHG"] == 0).astype(int),
        "BTTS": ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int),
    })
    hist = pd.concat([home, away], ignore_index=True)
    return hist.sort_values(["Team", "Date"]).reset_index(drop=True)


def build_team_index(hist):
    idx = {}
    for team, g in hist.groupby("Team"):
        g = g.sort_values("Date")
        idx[team] = {
            "dates": g["Date"].values,
            "venue": g["Venue"].values,
            **{c: g[c].to_numpy(dtype=float) for c in STAT_COLS},
        }
    return idx


def _weighted_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    weights = np.linspace(1.0, 0.4, len(values))
    return float(np.average(values, weights=weights))


# ---------------------------------------------------------------------
# 3. UNIFIED MATCH-FEATURE BUILDER (training == live, one function)
# ---------------------------------------------------------------------
class FeatureBuilder:
    """Wraps the team index so build_match_features can be called both
    row-by-row over history and on demand for a live fixture."""

    def __init__(self, team_history):
        self.team_idx = build_team_index(team_history)

    def team_stats_asof(self, team, cutoff_date, venue=None, windows=WINDOWS):
        out = {"matches": 0}
        if team not in self.team_idx:
            for w in windows:
                for c in STAT_COLS:
                    out[f"{c}_L{w}"] = DEFAULTS[c]
            out["rest"] = 7.0
            return out

        d = self.team_idx[team]
        cutoff = np.datetime64(cutoff_date)
        overall_mask = d["dates"] < cutoff
        mask = overall_mask if venue is None else (overall_mask & (d["venue"] == venue))
        positions = np.nonzero(mask)[0]
        out["matches"] = int(len(positions))

        if len(positions) == 0:
            for w in windows:
                for c in STAT_COLS:
                    out[f"{c}_L{w}"] = DEFAULTS[c]
        else:
            for w in windows:
                sel = positions[-w:]
                for c in STAT_COLS:
                    out[f"{c}_L{w}"] = _weighted_mean(d[c][sel])

        overall_positions = np.nonzero(overall_mask)[0]
        if len(overall_positions) > 0:
            last_date = d["dates"][overall_positions[-1]]
            out["rest"] = float((cutoff - last_date) / np.timedelta64(1, "D"))
        else:
            out["rest"] = 7.0
        return out

    def build_match_features(self, league, home, away, match_date):
        home_all = self.team_stats_asof(home, match_date, venue=None)
        home_h = self.team_stats_asof(home, match_date, venue="H")
        away_all = self.team_stats_asof(away, match_date, venue=None)
        away_a = self.team_stats_asof(away, match_date, venue="A")

        f = {}
        for w in WINDOWS:
            f[f"home_gf_L{w}"] = home_all[f"GF_L{w}"]
            f[f"home_ga_L{w}"] = home_all[f"GA_L{w}"]
            f[f"away_gf_L{w}"] = away_all[f"GF_L{w}"]
            f[f"away_ga_L{w}"] = away_all[f"GA_L{w}"]
            f[f"home_home_gf_L{w}"] = home_h[f"GF_L{w}"]
            f[f"home_home_ga_L{w}"] = home_h[f"GA_L{w}"]
            f[f"away_away_gf_L{w}"] = away_a[f"GF_L{w}"]
            f[f"away_away_ga_L{w}"] = away_a[f"GA_L{w}"]
            f[f"home_points_L{w}"] = home_all[f"Points_L{w}"]
            f[f"away_points_L{w}"] = away_all[f"Points_L{w}"]
            f[f"home_shots_L{w}"] = home_all[f"Shots_L{w}"]
            f[f"away_shots_L{w}"] = away_all[f"Shots_L{w}"]
            f[f"home_sot_L{w}"] = home_all[f"SOT_L{w}"]
            f[f"away_sot_L{w}"] = away_all[f"SOT_L{w}"]
            f[f"home_corners_L{w}"] = home_all[f"Corners_L{w}"]
            f[f"away_corners_L{w}"] = away_all[f"Corners_L{w}"]
            f[f"home_yellow_L{w}"] = home_all[f"Yellow_L{w}"]
            f[f"away_yellow_L{w}"] = away_all[f"Yellow_L{w}"]

        f["attack_home_vs_away_defence"] = home_h["GF_L5"] - away_a["GA_L5"]
        f["attack_away_vs_home_defence"] = away_a["GF_L5"] - home_h["GA_L5"]
        f["form_diff"] = home_all["Points_L5"] - away_all["Points_L5"]
        f["shot_diff"] = home_all["Shots_L5"] - away_all["Shots_L5"]
        f["sot_diff"] = home_all["SOT_L5"] - away_all["SOT_L5"]
        f["corner_diff"] = home_all["Corners_L5"] - away_all["Corners_L5"]
        f["clean_sheet_diff"] = home_all["CleanSheet_L5"] - away_all["CleanSheet_L5"]
        f["btts_diff"] = home_all["BTTS_L5"] - away_all["BTTS_L5"]
        f["home_rest"] = home_all["rest"]
        f["away_rest"] = away_all["rest"]
        f["rest_diff"] = home_all["rest"] - away_all["rest"]
        f["home_sample"] = home_all["matches"]
        f["away_sample"] = away_all["matches"]

        for lg in LEAGUE_NAMES:
            f[f"league_{lg}"] = int(league == lg)

        return pd.Series(f)

    def build_all_features(self, df):
        df = df.sort_values(["League", "Date"]).reset_index(drop=True)
        rows = [self.build_match_features(r["League"], r["HomeTeam"], r["AwayTeam"], r["Date"])
                for _, r in df.iterrows()]
        feat_df = pd.DataFrame(rows)
        feat_df.index = df.index
        out = pd.concat([df, feat_df], axis=1)
        feature_cols = list(feat_df.columns)
        out = out[(out["home_sample"] >= 3) & (out["away_sample"] >= 3)].reset_index(drop=True)
        return out, feature_cols


# ---------------------------------------------------------------------
# 4. ELO (improved: goal-margin multiplier, attack/defence tracks)
# ---------------------------------------------------------------------
def build_elo(df, k=20, base=1500.0):
    data = df.sort_values("Date").copy()
    ratings, atk, dfc = {}, {}, {}
    h_elo, a_elo, h_atk, a_atk, h_def, a_def = [], [], [], [], [], []

    for _, row in data.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        for t in (h, a):
            ratings.setdefault(t, base)
            atk.setdefault(t, base)
            dfc.setdefault(t, base)

        rh, ra = ratings[h], ratings[a]
        h_elo.append(rh); a_elo.append(ra)
        h_atk.append(atk[h]); a_atk.append(atk[a])
        h_def.append(dfc[h]); a_def.append(dfc[a])

        expected_h = 1 / (1 + 10 ** ((ra - rh) / 400))
        actual_h = 1.0 if row["FTR"] == "H" else (0.5 if row["FTR"] == "D" else 0.0)
        margin = abs(row["FTHG"] - row["FTAG"])
        mult = 1.0 if margin <= 1 else (1.25 if margin == 2 else (1.5 if margin == 3 else 1.75))

        change = k * mult * (actual_h - expected_h)
        ratings[h] += change
        ratings[a] -= change

        atk_change_h = k * 0.5 * mult * np.tanh((row["FTHG"] - row["FTAG"]) / 3)
        atk[h] += atk_change_h
        dfc[a] -= atk_change_h
        atk_change_a = k * 0.5 * mult * np.tanh((row["FTAG"] - row["FTHG"]) / 3)
        atk[a] += atk_change_a
        dfc[h] -= atk_change_a

    data["ELO_home"] = h_elo
    data["ELO_away"] = a_elo
    data["ELO_diff"] = data["ELO_home"] - data["ELO_away"]
    data["ELO_atk_home"] = h_atk
    data["ELO_atk_away"] = a_atk
    data["ELO_def_home"] = h_def
    data["ELO_def_away"] = a_def
    return data, ratings, atk, dfc


# ---------------------------------------------------------------------
# 5. DIXON-COLES (proper walk-forward xi tuning)
# ---------------------------------------------------------------------
class DixonColes:
    def __init__(self, xi=0.002, max_goals=10):
        self.xi = xi
        self.max_goals = max_goals

    def fit(self, data):
        data = data.sort_values("Date").copy()
        self.teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
        n = len(self.teams)
        tidx = {t: i for i, t in enumerate(self.teams)}
        hi = data["HomeTeam"].map(tidx).values
        ai = data["AwayTeam"].map(tidx).values
        hg = data["FTHG"].values
        ag = data["FTAG"].values
        max_date = data["Date"].max()
        days = (max_date - data["Date"]).dt.days.values
        weights = np.exp(-self.xi * days)

        def nll(params):
            att = params[:n] - np.mean(params[:n])
            deff = params[n:2 * n]
            home_adv, rho = params[2 * n], params[2 * n + 1]
            lh = np.exp(home_adv + att[hi] - deff[ai])
            la = np.exp(att[ai] - deff[hi])
            p = poisson.pmf(hg, lh) * poisson.pmf(ag, la)
            corr = np.ones(len(data))
            m00 = (hg == 0) & (ag == 0); m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0); m11 = (hg == 1) & (ag == 1)
            corr[m00] = 1 - lh[m00] * la[m00] * rho
            corr[m01] = 1 + lh[m01] * rho
            corr[m10] = 1 + la[m10] * rho
            corr[m11] = 1 - rho
            p *= corr
            return -np.sum(weights * np.log(np.maximum(p, EPS)))

        x0 = np.r_[np.zeros(n), np.zeros(n), 0.2, 0.0]
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.1, 0.1)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
        p = res.x
        self.attack = dict(zip(self.teams, p[:n] - np.mean(p[:n])))
        self.defence = dict(zip(self.teams, p[n:2 * n]))
        self.home_adv, self.rho = p[2 * n], p[2 * n + 1]
        return self

    def _lambdas(self, home, away):
        lh = np.exp(self.home_adv + self.attack.get(home, 0) - self.defence.get(away, 0))
        la = np.exp(self.attack.get(away, 0) - self.defence.get(home, 0))
        return lh, la

    def score_matrix(self, home, away):
        lh, la = self._lambdas(home, away)
        g = np.arange(self.max_goals + 1)
        S = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        S[0, 0] *= (1 - lh * la * self.rho)
        S[0, 1] *= (1 + lh * self.rho)
        S[1, 0] *= (1 + la * self.rho)
        S[1, 1] *= (1 - self.rho)
        S = np.maximum(S, 0)
        S /= S.sum()
        return S, lh, la

    def predict(self, home, away):
        S, lh, la = self.score_matrix(home, away)
        g = np.arange(self.max_goals + 1)
        total = np.add.outer(g, g)
        return {
            "lambda_home": float(lh), "lambda_away": float(la),
            "prob_home": float(np.tril(S, -1).sum()),
            "prob_draw": float(np.trace(S)),
            "prob_away": float(np.triu(S, 1).sum()),
            "prob_over_05": float(S[total >= 1].sum()),
            "prob_over_15": float(S[total >= 2].sum()),
            "prob_over_25": float(S[total >= 3].sum()),
            "prob_over_35": float(S[total >= 4].sum()),
            "prob_under_25": float(S[total < 3].sum()),
            "btts_yes": float(S[(g[:, None] > 0) & (g[None, :] > 0)].sum()),
            "score_matrix": S,
            "exp_goals": float(lh + la),
        }

    def match_nll(self, data):
        total = 0.0
        for _, r in data.iterrows():
            lh, la = self._lambdas(r["HomeTeam"], r["AwayTeam"])
            p = poisson.pmf(r["FTHG"], lh) * poisson.pmf(r["FTAG"], la)
            hg, ag = r["FTHG"], r["FTAG"]
            corr = 1.0
            if hg == 0 and ag == 0: corr = 1 - lh * la * self.rho
            elif hg == 0 and ag == 1: corr = 1 + lh * self.rho
            elif hg == 1 and ag == 0: corr = 1 + la * self.rho
            elif hg == 1 and ag == 1: corr = 1 - self.rho
            p *= corr
            total += -np.log(max(p, EPS))
        return total


def tune_dixon_coles_xi(data, candidate_xis=DC_XI_CANDIDATES, min_train_seasons=MIN_TRAIN_SEASONS):
    seasons = sorted(data["Season"].unique())
    if len(seasons) <= min_train_seasons:
        return candidate_xis[len(candidate_xis) // 2], {}
    results = {}
    for xi in candidate_xis:
        total_nll, n_eval = 0.0, 0
        for i in range(min_train_seasons, len(seasons)):
            train = data[data["Season"].isin(seasons[:i])]
            test = data[data["Season"] == seasons[i]]
            if len(train) < 20 or len(test) == 0:
                continue
            m = DixonColes(xi=xi).fit(train)
            total_nll += m.match_nll(test)
            n_eval += len(test)
        results[xi] = total_nll / max(n_eval, 1)
    best_xi = min(results, key=results.get)
    return best_xi, results


# ---------------------------------------------------------------------
# 6. XGBOOST WALK-FORWARD TRAINING + OOF
# ---------------------------------------------------------------------
def walk_forward_train_xgb(df, feature_cols, min_train_seasons=MIN_TRAIN_SEASONS, progress_callback=None):
    df = df.sort_values("Date").reset_index(drop=True)
    seasons = sorted(df["Season"].unique())
    oof = np.full((len(df), 3), np.nan)
    oof_valid = np.zeros(len(df), dtype=bool)
    metrics = []
    last_model = None

    for i in range(min_train_seasons, len(seasons)):
        train_idx = df["Season"].isin(seasons[:i])
        test_idx = df["Season"] == seasons[i]
        train, test = df.loc[train_idx], df.loc[test_idx]
        if len(test) == 0 or len(train) < 20:
            continue

        Xtr, ytr = train[feature_cols].values, train["Outcome"].values
        Xte, yte = test[feature_cols].values, test["Outcome"].values

        model = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=300,
            learning_rate=0.04, max_depth=4, min_child_weight=6,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1,
            reg_lambda=2.0, gamma=0.05, random_state=42, eval_metric="mlogloss",
        )
        model.fit(Xtr, ytr)
        last_model = model
        preds = model.predict_proba(Xte)
        oof[test.index] = preds
        oof_valid[test.index] = True

        ll = log_loss(yte, preds, labels=[0, 1, 2])
        y_oh = np.zeros((len(yte), 3)); y_oh[np.arange(len(yte)), yte] = 1
        brier = np.mean(np.sum((preds - y_oh) ** 2, axis=1))
        acc = (preds.argmax(axis=1) == yte).mean()
        metrics.append({"season": seasons[i], "log_loss": ll, "brier": brier, "accuracy": acc, "n": len(test)})
        if progress_callback:
            progress_callback(i / len(seasons), f"Trained through season {seasons[i]}")

    return oof, oof_valid, metrics, last_model


def dc_walk_forward_oof(df, xi_by_league, min_train_seasons=MIN_TRAIN_SEASONS):
    df = df.sort_values("Date").reset_index(drop=True)
    oof = np.full((len(df), 3), np.nan)
    valid = np.zeros(len(df), dtype=bool)

    for league, g in df.groupby("League"):
        xi = xi_by_league.get(league, 0.002)
        seasons = sorted(g["Season"].unique())
        for i in range(min_train_seasons, len(seasons)):
            train = g[g["Season"].isin(seasons[:i])]
            test = g[g["Season"] == seasons[i]]
            if len(train) < 20 or len(test) == 0:
                continue
            m = DixonColes(xi=xi).fit(train)
            for idx, row in test.iterrows():
                pred = m.predict(row["HomeTeam"], row["AwayTeam"])
                oof[idx] = [pred["prob_home"], pred["prob_draw"], pred["prob_away"]]
                valid[idx] = True
    return oof, valid


def optimize_ensemble_weight(y_true, ml_probs, dc_probs, weight_grid=ENSEMBLE_WEIGHT_GRID):
    results = {}
    ml_c = np.clip(ml_probs, EPS, 1 - EPS)
    dc_c = np.clip(dc_probs, EPS, 1 - EPS)
    for w in weight_grid:
        combined = softmax(w * np.log(dc_c) + (1 - w) * np.log(ml_c), axis=1)
        results[w] = log_loss(y_true, combined, labels=[0, 1, 2])
    best_w = min(results, key=results.get)
    return best_w, results


# ---------------------------------------------------------------------
# 7. CALIBRATION (valid OOF rows only)
# ---------------------------------------------------------------------
class ProbabilityCalibrator:
    def __init__(self):
        self.calibrators = []

    def fit(self, y_true, y_pred_proba):
        self.calibrators = []
        for i in range(3):
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(y_pred_proba[:, i], (y_true == i).astype(int))
            self.calibrators.append(cal)

    def transform(self, y_pred_proba):
        out = np.zeros_like(y_pred_proba)
        for i, cal in enumerate(self.calibrators):
            out[:, i] = cal.transform(y_pred_proba[:, i])
        out /= np.clip(out.sum(axis=1, keepdims=True), 1e-8, None)
        return out


# ---------------------------------------------------------------------
# 8. MARKETS: no-vig pricing, EV/edge/fair-odds, Asian Handicap, Bet Score
# ---------------------------------------------------------------------
def no_vig_probs(odds_list):
    odds = np.array(odds_list, dtype=float)
    raw = 1 / odds
    overround = raw.sum() - 1
    fair = raw / raw.sum()
    return fair, overround


def value_metrics(model_prob, odds, market_fair_prob=None):
    if odds is None or pd.isna(odds) or odds <= 1:
        return None
    implied = 1 / odds
    ev = model_prob * odds - 1
    out = {
        "model_probability": model_prob, "odds": odds, "implied_probability": implied,
        "fair_odds": (1 / model_prob) if model_prob > 0 else np.inf,
        "EV": ev, "EV_pct": ev * 100,
    }
    if market_fair_prob is not None:
        out["market_fair_probability"] = market_fair_prob
        out["model_edge_vs_market"] = model_prob - market_fair_prob
    return out


def asian_handicap_settlement(score_matrix, line, side, max_goals=10):
    g = np.arange(max_goals + 1)
    diff = np.subtract.outer(g, g)

    def settle(single_line):
        adj = (diff + single_line) if side == "home" else (-diff + single_line)
        win = score_matrix[adj > 0].sum()
        push = score_matrix[adj == 0].sum()
        loss = score_matrix[adj < 0].sum()
        return win, push, loss

    is_quarter = np.isclose(abs(line * 4) % 1, 0.5)
    if is_quarter:
        l1, l2 = line - 0.25, line + 0.25
        w1, p1, lo1 = settle(l1)
        w2, p2, lo2 = settle(l2)
        win, push, loss = (w1 + w2) / 2, (p1 + p2) / 2, (lo1 + lo2) / 2
    else:
        win, push, loss = settle(line)
    return {"win": win, "push": push, "loss": loss}


def top_correct_scores(score_matrix, n=5, max_show_goals=5):
    S = score_matrix[:max_show_goals + 1, :max_show_goals + 1]
    flat = [((i, j), S[i, j]) for i in range(S.shape[0]) for j in range(S.shape[1])]
    flat.sort(key=lambda x: -x[1])
    return flat[:n]


def model_agreement(dc_prob, ml_prob):
    return float(max(0.0, 1.0 - abs(dc_prob - ml_prob) / 0.5))


def data_quality_score(home_sample, away_sample, min_matches=10):
    return float(min(1.0, min(home_sample, away_sample) / min_matches))


def calculate_bet_score(probability, ev, edge, agreement, data_quality):
    score = 0
    score += min(probability / 0.80, 1) * BET_SCORE_WEIGHTS["probability"]
    score += min(max(ev, 0) / 0.15, 1) * BET_SCORE_WEIGHTS["ev"]
    score += min(max(edge, 0) / 0.10, 1) * BET_SCORE_WEIGHTS["edge"]
    score += agreement * BET_SCORE_WEIGHTS["agreement"]
    score += data_quality * BET_SCORE_WEIGHTS["data_quality"]
    return round(min(score, 100), 1)


def classify_bet(probability, ev, edge, score):
    if probability >= 0.65 and ev >= 0.05 and edge >= 0.03 and score >= 70:
        return "BET"
    if probability >= 0.55 and ev >= 0.02 and edge >= 0.015 and score >= 55:
        return "LEAN"
    return "NO BET"


# ---------------------------------------------------------------------
# 9. ENSEMBLE PREDICTION (live == training feature path)
# ---------------------------------------------------------------------
def build_live_features(fb, elo_ratings, elo_attack, elo_defence, league, home, away, match_date, feature_cols):
    base = fb.build_match_features(league, home, away, match_date)
    elo_extra = pd.Series({
        "ELO_diff": elo_ratings.get(home, 1500.0) - elo_ratings.get(away, 1500.0),
        "ELO_atk_home": elo_attack.get(home, 1500.0),
        "ELO_atk_away": elo_attack.get(away, 1500.0),
        "ELO_def_home": elo_defence.get(home, 1500.0),
        "ELO_def_away": elo_defence.get(away, 1500.0),
    })
    full = pd.concat([base, elo_extra])
    return full.reindex(feature_cols).values.reshape(1, -1).astype(float)


def ensemble_predict(fb, elo_ratings, elo_attack, elo_defence, league, home, away,
                      match_date, dc_model, ml_model, feature_cols, dc_weight):
    dc_pred = dc_model.predict(home, away)
    dc_probs = np.clip([dc_pred["prob_home"], dc_pred["prob_draw"], dc_pred["prob_away"]], EPS, 1 - EPS)

    feats = build_live_features(fb, elo_ratings, elo_attack, elo_defence, league, home, away, match_date, feature_cols)
    ml_probs = np.clip(ml_model.predict_proba(feats)[0], EPS, 1 - EPS)

    combined_log = dc_weight * np.log(dc_probs) + (1 - dc_weight) * np.log(ml_probs)
    ens = softmax(combined_log)

    return {
        "prob_home": float(ens[0]), "prob_draw": float(ens[1]), "prob_away": float(ens[2]),
        "dc_prob_home": float(dc_probs[0]), "dc_prob_draw": float(dc_probs[1]), "dc_prob_away": float(dc_probs[2]),
        "ml_prob_home": float(ml_probs[0]), "ml_prob_draw": float(ml_probs[1]), "ml_prob_away": float(ml_probs[2]),
        "lambda_home": dc_pred["lambda_home"], "lambda_away": dc_pred["lambda_away"],
        "score_matrix": dc_pred["score_matrix"],
        "prob_over_05": dc_pred["prob_over_05"], "prob_over_15": dc_pred["prob_over_15"],
        "prob_over_25": dc_pred["prob_over_25"], "prob_over_35": dc_pred["prob_over_35"],
        "prob_under_25": dc_pred["prob_under_25"], "btts_yes": dc_pred["btts_yes"],
        "exp_goals": dc_pred["exp_goals"],
    }

# ============================================================================
# STREAMLIT UI
# ============================================================================


# ----------------------------------------------------------------------
# Minimal styling
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a8c 100%);
        padding: 1.6rem 2rem; border-radius: 12px; margin-bottom: 1.2rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.9rem; }
    .main-header p { color: #cbd8e8; margin: 0.3rem 0 0 0; font-size: 0.92rem; }
    .metric-card {
        background: #f8f9fb; border: 1px solid #e3e7ee; border-radius: 10px;
        padding: 1rem 1.2rem; text-align: center;
    }
    .decision-bet { background:#e6f6ea; border-left:5px solid #1e9e4a; padding:1rem 1.3rem; border-radius:8px; }
    .decision-lean { background:#fff8e1; border-left:5px solid #d4a017; padding:1rem 1.3rem; border-radius:8px; }
    .decision-nobet { background:#f1f2f4; border-left:5px solid #8a8f98; padding:1rem 1.3rem; border-radius:8px; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>⚽ Football Predictor V2</h1>
    <p>Leakage-controlled · walk-forward validated · Dixon-Coles + XGBoost ensemble · no-vig market comparison</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
for key in ["trained", "df", "feature_cols", "fb", "elo_ratings", "elo_attack", "elo_defence",
            "final_dc_models", "final_ml_model", "best_xi_by_league", "best_dc_weight",
            "calibrator", "wf_metrics", "oof_preds", "oof_valid", "dc_oof", "dc_valid",
            "raw_ll", "cal_ll"]:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.setdefault("trained", False)

# ----------------------------------------------------------------------
# Sidebar — data / training controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")
    n_seasons = st.slider("Seasons of history to load", 2, 6, 4)
    selected_leagues = st.multiselect(
        "Leagues", list(LEAGUES.keys()), default=list(LEAGUES.keys())
    )
    st.caption("Data source: football-data.co.uk (free historical results + odds)")

    train_clicked = st.button("🚀 Load Data & Train Models", type="primary", use_container_width=True)

    if st.session_state.trained:
        st.success(f"✅ Model ready — {len(st.session_state.df):,} matches")
        st.caption(f"Optimal DC ensemble weight: {st.session_state.best_dc_weight}")

# ----------------------------------------------------------------------
# Training pipeline
# ----------------------------------------------------------------------
if train_clicked:
    if not selected_leagues:
        st.sidebar.error("Select at least one league.")
    else:
        leagues_dict = {k: LEAGUES[k] for k in selected_leagues}
        progress = st.sidebar.progress(0.0, text="Loading data...")

        try:
            df_raw = load_raw_data(
                n_seasons=n_seasons, leagues=leagues_dict,
                progress_callback=lambda frac, msg: progress.progress(frac * 0.15, text=f"Loading: {msg}")
            )
        except Exception as e:
            st.sidebar.error(f"Data load failed: {e}")
            st.stop()

        progress.progress(0.18, text="Building team-match history...")
        team_history = build_team_history(df_raw)
        fb = FeatureBuilder(team_history)

        progress.progress(0.25, text="Building unified match features (train==live)...")
        df, feature_cols = fb.build_all_features(df_raw)

        progress.progress(0.40, text="Building chronological Elo ratings...")
        df, elo_ratings, elo_attack, elo_defence = build_elo(df)
        feature_cols = feature_cols + ["ELO_diff", "ELO_atk_home", "ELO_atk_away", "ELO_def_home", "ELO_def_away"]

        progress.progress(0.50, text="Tuning Dixon-Coles ξ (walk-forward NLL)...")
        best_xi_by_league = {}
        for league in df["League"].unique():
            ld = df[df["League"] == league]
            if len(ld) < 40:
                continue
            xi, _ = tune_dixon_coles_xi(ld)
            best_xi_by_league[league] = xi

        progress.progress(0.62, text="Walk-forward training XGBoost...")
        oof_preds, oof_valid, wf_metrics, _ = walk_forward_train_xgb(df, feature_cols)

        progress.progress(0.75, text="Walk-forward Dixon-Coles OOF...")
        dc_oof, dc_valid = dc_walk_forward_oof(df, best_xi_by_league)

        both_valid = oof_valid & dc_valid & ~np.isnan(oof_preds).any(axis=1) & ~np.isnan(dc_oof).any(axis=1)
        if both_valid.sum() > 10:
            best_dc_weight, _ = optimize_ensemble_weight(
                df.loc[both_valid, "Outcome"].values, oof_preds[both_valid], dc_oof[both_valid]
            )
        else:
            best_dc_weight = 0.5

        progress.progress(0.85, text="Calibrating probabilities (valid OOF rows only)...")
        valid_mask = oof_valid & ~np.isnan(oof_preds).any(axis=1)
        calibrator = ProbabilityCalibrator()
        calibrator.fit(df.loc[valid_mask, "Outcome"].values, oof_preds[valid_mask])
        raw_ll = log_loss(df.loc[valid_mask, "Outcome"], oof_preds[valid_mask], labels=[0, 1, 2])
        cal_ll = log_loss(df.loc[valid_mask, "Outcome"],
                              calibrator.transform(oof_preds[valid_mask]), labels=[0, 1, 2])

        progress.progress(0.93, text="Training final models on full history...")
        X_full, y_full = df[feature_cols].values, df["Outcome"].values
        final_ml_model = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=300,
            learning_rate=0.04, max_depth=4, min_child_weight=6,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1,
            reg_lambda=2.0, gamma=0.05, random_state=42, eval_metric="mlogloss",
        )
        final_ml_model.fit(X_full, y_full)

        final_dc_models = {}
        for league in df["League"].unique():
            xi = best_xi_by_league.get(league, 0.002)
            final_dc_models[league] = DixonColes(xi=xi).fit(df[df["League"] == league])

        progress.progress(1.0, text="Done!")

        st.session_state.update({
            "trained": True, "df": df, "feature_cols": feature_cols, "fb": fb,
            "elo_ratings": elo_ratings, "elo_attack": elo_attack, "elo_defence": elo_defence,
            "final_dc_models": final_dc_models, "final_ml_model": final_ml_model,
            "best_xi_by_league": best_xi_by_league, "best_dc_weight": best_dc_weight,
            "calibrator": calibrator, "wf_metrics": wf_metrics,
            "oof_preds": oof_preds, "oof_valid": oof_valid, "dc_oof": dc_oof, "dc_valid": dc_valid,
            "raw_ll": raw_ll, "cal_ll": cal_ll,
        })
        st.rerun()

# ----------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------
if not st.session_state.trained:
    st.info("👈 Set your options in the sidebar and click **Load Data & Train Models** to get started.")
    st.markdown("""
    ### What this app does
    - Downloads recent match results + odds for the leagues you pick from football-data.co.uk
    - Builds a **team-match history table** so rest days and form are computed correctly regardless of home/away role
    - Trains **XGBoost** with true walk-forward validation (never trains on the future)
    - Fits a **Dixon-Coles** Poisson model per league, with its time-decay ξ chosen by walk-forward held-out likelihood
    - Blends the two models at a weight chosen by walk-forward log-loss (not a fixed guess)
    - Calibrates probabilities on genuinely out-of-fold predictions only
    - Compares model probabilities to **no-vig** market prices to look for value, with a transparent Bet Score
    """)
    st.stop()

df = st.session_state.df
feature_cols = st.session_state.feature_cols
fb = st.session_state.fb

tab_predict, tab_perf, tab_backtest, tab_about = st.tabs(
    ["🔮 Predict a Match", "📊 Model Performance", "💰 Backtest", "ℹ️ About the Model"]
)

# ========================================================================
# TAB: PREDICT
# ========================================================================
with tab_predict:
    league_teams = {
        lg: sorted(set(df.loc[df["League"] == lg, "HomeTeam"]) | set(df.loc[df["League"] == lg, "AwayTeam"]))
        for lg in df["League"].unique()
    }

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        league = st.selectbox("League", sorted(league_teams.keys()))
    teams = league_teams[league]
    with c2:
        home = st.selectbox("Home team", teams, index=0)
    with c3:
        away_options = [t for t in teams if t != home]
        away = st.selectbox("Away team", away_options, index=0)

    st.markdown("##### Bookmaker odds (1X2) — required for value comparison")
    o1, o2, o3 = st.columns(3)
    with o1:
        odds_home = st.number_input("Home odds", min_value=1.01, value=2.00, step=0.01)
    with o2:
        odds_draw = st.number_input("Draw odds", min_value=1.01, value=3.40, step=0.01)
    with o3:
        odds_away = st.number_input("Away odds", min_value=1.01, value=4.00, step=0.01)

    with st.expander("Optional: Asian Handicap odds"):
        ah1, ah2, ah3 = st.columns(3)
        with ah1:
            ah_line = st.selectbox("Handicap line (applied to Home)",
                                    [-1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25], index=5)
        with ah2:
            ah_odds_home = st.number_input("Home AH odds", min_value=1.01, value=1.90, step=0.01)
        with ah3:
            ah_odds_away = st.number_input("Away AH odds", min_value=1.01, value=1.90, step=0.01)

    predict_clicked = st.button("🔮 Generate Prediction", type="primary")

    if predict_clicked:
        match_date = pd.Timestamp(datetime.now())
        dc_model = st.session_state.final_dc_models.get(league)
        if dc_model is None or home not in dc_model.teams or away not in dc_model.teams:
            st.error("One of the selected teams doesn't have enough history in this league yet.")
        else:
            pred = ensemble_predict(
                fb, st.session_state.elo_ratings, st.session_state.elo_attack, st.session_state.elo_defence,
                league, home, away, match_date, dc_model, st.session_state.final_ml_model,
                feature_cols, st.session_state.best_dc_weight,
            )
            fair, overround = no_vig_probs([odds_home, odds_draw, odds_away])

            agreement = np.mean([
                model_agreement(pred["dc_prob_home"], pred["ml_prob_home"]),
                model_agreement(pred["dc_prob_draw"], pred["ml_prob_draw"]),
                model_agreement(pred["dc_prob_away"], pred["ml_prob_away"]),
            ])
            home_stats = fb.team_stats_asof(home, match_date)
            away_stats = fb.team_stats_asof(away, match_date)
            dq = data_quality_score(home_stats["matches"], away_stats["matches"])

            st.markdown("---")
            st.subheader(f"{home} vs {away}  ·  {league}")

            # --- Probability comparison chart ---
            prob_df = pd.DataFrame({
                "Outcome": ["Home", "Draw", "Away"] * 3,
                "Model": ["Dixon-Coles"] * 3 + ["XGBoost"] * 3 + ["Ensemble"] * 3,
                "Probability": [
                    pred["dc_prob_home"], pred["dc_prob_draw"], pred["dc_prob_away"],
                    pred["ml_prob_home"], pred["ml_prob_draw"], pred["ml_prob_away"],
                    pred["prob_home"], pred["prob_draw"], pred["prob_away"],
                ],
            })
            fig = px.bar(prob_df, x="Outcome", y="Probability", color="Model", barmode="group",
                         color_discrete_map={"Dixon-Coles": "#4C78A8", "XGBoost": "#F58518", "Ensemble": "#54A24B"})
            fig.update_layout(yaxis_tickformat=".0%", height=340, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Model agreement", f"{agreement*100:.0f}/100")
            m2.metric("Data quality", f"{dq*100:.0f}/100")
            m3.metric("Expected goals (H-A)", f"{pred['lambda_home']:.2f} – {pred['lambda_away']:.2f}")
            m4.metric("Over 2.5 / BTTS", f"{pred['prob_over_25']*100:.0f}% / {pred['btts_yes']*100:.0f}%")

            # --- Correct scores ---
            st.markdown("##### Most likely scorelines")
            top_scores = top_correct_scores(pred["score_matrix"], n=5)
            score_cols = st.columns(5)
            for col, ((h, a), p) in zip(score_cols, top_scores):
                col.metric(f"{h}-{a}", f"{p*100:.1f}%")

            # --- Market comparison ---
            st.markdown("##### Market comparison (no-vig)")
            market_df = pd.DataFrame({
                "Selection": ["Home", "Draw", "Away"],
                "Model probability": [pred["prob_home"], pred["prob_draw"], pred["prob_away"]],
                "No-vig market probability": fair,
                "Bookmaker odds": [odds_home, odds_draw, odds_away],
            })
            market_df["Edge"] = market_df["Model probability"] - market_df["No-vig market probability"]
            market_df["Fair odds"] = 1 / market_df["Model probability"]
            st.dataframe(
                market_df.style.format({
                    "Model probability": "{:.1%}", "No-vig market probability": "{:.1%}",
                    "Edge": "{:+.1%}", "Bookmaker odds": "{:.2f}", "Fair odds": "{:.2f}",
                }),
                use_container_width=True, hide_index=True,
            )
            st.caption(f"Bookmaker overround: {overround*100:.1f}%")

            # --- Best value + decision ---
            best_bet, best_score = None, -1
            for sel, prob_key, fair_idx, odds_val in [
                ("Home", "prob_home", 0, odds_home), ("Draw", "prob_draw", 1, odds_draw), ("Away", "prob_away", 2, odds_away)
            ]:
                mprob = pred[prob_key]
                vm = value_metrics(mprob, odds_val, market_fair_prob=fair[fair_idx])
                if vm is None:
                    continue
                sc = calculate_bet_score(mprob, vm["EV"], vm["model_edge_vs_market"], agreement, dq)
                if sc > best_score:
                    best_score = sc
                    best_bet = (sel, mprob, odds_val, vm, sc)

            if best_bet:
                sel, mprob, odds_val, vm, sc = best_bet
                decision = classify_bet(mprob, vm["EV"], vm["model_edge_vs_market"], sc)
                css_class = {"BET": "decision-bet", "LEAN": "decision-lean", "NO BET": "decision-nobet"}[decision]
                emoji = {"BET": "🟢", "LEAN": "🟡", "NO BET": "⚪"}[decision]
                st.markdown(f"""
                <div class="{css_class}">
                    <h4 style="margin:0 0 0.4rem 0;">{emoji} {decision} — {sel}</h4>
                    Model probability <b>{mprob*100:.1f}%</b> vs no-vig market <b>{vm['market_fair_probability']*100:.1f}%</b>
                    (edge <b>{vm['model_edge_vs_market']*100:+.1f}%</b>)<br>
                    Fair odds <b>{vm['fair_odds']:.2f}</b> · Available odds <b>{odds_val:.2f}</b> ·
                    EV <b>{vm['EV_pct']:+.1f}%</b> · Bet Score <b>{sc}/100</b>
                </div>
                """, unsafe_allow_html=True)

            # --- Asian Handicap ---
            with st.expander("Asian Handicap settlement (score-matrix based, not a logistic approximation)"):
                ah_home = asian_handicap_settlement(pred["score_matrix"], ah_line, "home")
                ah_away = asian_handicap_settlement(pred["score_matrix"], -ah_line, "away")
                ah_c1, ah_c2 = st.columns(2)
                with ah_c1:
                    st.write(f"**Home {ah_line:+.2f}**")
                    st.write(f"Win {ah_home['win']*100:.1f}% · Push {ah_home['push']*100:.1f}% · Loss {ah_home['loss']*100:.1f}%")
                    ev_h = ah_home["win"] * (ah_odds_home - 1) - ah_home["loss"]
                    st.write(f"EV at odds {ah_odds_home:.2f}: **{ev_h*100:+.1f}%**")
                with ah_c2:
                    st.write(f"**Away {-ah_line:+.2f}**")
                    st.write(f"Win {ah_away['win']*100:.1f}% · Push {ah_away['push']*100:.1f}% · Loss {ah_away['loss']*100:.1f}%")
                    ev_a = ah_away["win"] * (ah_odds_away - 1) - ah_away["loss"]
                    st.write(f"EV at odds {ah_odds_away:.2f}: **{ev_a*100:+.1f}%**")

# ========================================================================
# TAB: MODEL PERFORMANCE
# ========================================================================
with tab_perf:
    st.subheader("Walk-forward validation (out-of-sample, season by season)")
    wf_df = pd.DataFrame(st.session_state.wf_metrics)
    if len(wf_df):
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wf_df["season"], y=wf_df["log_loss"], mode="lines+markers", name="Log-Loss"))
            fig.add_trace(go.Scatter(x=wf_df["season"], y=wf_df["brier"], mode="lines+markers", name="Brier"))
            fig.update_layout(height=320, margin=dict(t=20, b=20), yaxis_title="Score (lower = better)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("Mean walk-forward Log-Loss", f"{wf_df['log_loss'].mean():.4f}")
            st.metric("Mean walk-forward Brier", f"{wf_df['brier'].mean():.4f}")
            st.metric("Mean walk-forward Accuracy", f"{wf_df['accuracy'].mean()*100:.1f}%")
        st.dataframe(wf_df.style.format({"log_loss": "{:.4f}", "brier": "{:.4f}", "accuracy": "{:.1%}"}),
                     use_container_width=True, hide_index=True)

    st.subheader("Calibration (valid out-of-fold rows only)")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Raw OOF Log-Loss", f"{st.session_state.raw_ll:.4f}")
    cc2.metric("Calibrated Log-Loss", f"{st.session_state.cal_ll:.4f}")
    cc3.metric("Improvement", f"{st.session_state.raw_ll - st.session_state.cal_ll:.4f}")

    st.subheader("Dixon-Coles ξ (time-decay) chosen per league")
    xi_df = pd.DataFrame(list(st.session_state.best_xi_by_league.items()), columns=["League", "Best ξ"])
    st.dataframe(xi_df, use_container_width=True, hide_index=True)

    st.subheader("Ensemble weight")
    st.metric("Optimal Dixon-Coles weight (walk-forward selected)", st.session_state.best_dc_weight)
    st.caption("Remaining weight goes to XGBoost. Selected by grid search minimising walk-forward log-loss (Section 11 of the notebook).")

# ========================================================================
# TAB: BACKTEST
# ========================================================================
with tab_backtest:
    st.subheader("Walk-forward betting backtest")
    st.caption("Uses ONLY out-of-fold probabilities — no model here ever saw the outcome of a match it bet on.")

    b1, b2, b3 = st.columns(3)
    with b1:
        min_prob = st.slider("Minimum model probability", 0.50, 0.75, 0.55, 0.01)
    with b2:
        min_ev = st.slider("Minimum EV", 0.00, 0.10, 0.03, 0.01)
    with b3:
        bankroll = st.number_input("Starting bankroll", min_value=100, value=10000, step=100)

    run_backtest = st.button("▶️ Run Backtest")

    if run_backtest:
        class RiskManager:
            def __init__(self, bankroll=10000, kelly_frac=0.25, max_bet_pct=0.05,
                         max_daily_pct=0.20, max_bets_per_match=3, min_edge=0.015, drawdown_stop=0.25):
                self.bankroll = bankroll; self.peak_bankroll = bankroll
                self.kelly_frac = kelly_frac; self.max_bet_pct = max_bet_pct
                self.max_daily_pct = max_daily_pct; self.max_bets_per_match = max_bets_per_match
                self.min_edge = min_edge; self.drawdown_stop = drawdown_stop
                self.daily_exposure = 0; self.match_bets = {}; self.stopped = False

            def kelly_stake(self, prob, odds, edge, match_id=None):
                if self.stopped or edge < self.min_edge or prob <= 0.05 or prob >= 0.95 or odds <= 1.0:
                    return 0
                b = odds - 1
                kelly = (prob * b - (1 - prob)) / b
                if kelly <= 0:
                    return 0
                kelly = min(kelly * self.kelly_frac, self.max_bet_pct)
                stake = self.bankroll * kelly
                if self.daily_exposure + stake > self.bankroll * self.max_daily_pct:
                    return 0
                if match_id is not None:
                    if self.match_bets.get(match_id, 0) >= self.max_bets_per_match:
                        return 0
                    self.match_bets[match_id] = self.match_bets.get(match_id, 0) + 1
                stake = round(stake, 2)
                self.daily_exposure += stake
                return stake

            def record_bet(self, stake, won, odds):
                profit = stake * (odds - 1) if won else -stake
                self.bankroll += profit
                self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
                if (self.peak_bankroll - self.bankroll) / self.peak_bankroll >= self.drawdown_stop:
                    self.stopped = True
                return profit

            def reset_daily(self):
                self.daily_exposure = 0; self.match_bets = {}

        oof_ml, oof_dc = st.session_state.oof_preds, st.session_state.dc_oof
        valid_ml, valid_dc = st.session_state.oof_valid, st.session_state.dc_valid
        both_valid_local = valid_ml & valid_dc & ~np.isnan(oof_ml).any(axis=1) & ~np.isnan(oof_dc).any(axis=1)
        sub = df.loc[both_valid_local].copy().reset_index(drop=True)
        ml = np.clip(oof_ml[both_valid_local], EPS, 1 - EPS)
        dcp = np.clip(oof_dc[both_valid_local], EPS, 1 - EPS)
        order = np.argsort(sub["Date"].values)
        sub = sub.iloc[order].reset_index(drop=True)
        ml, dcp = ml[order], dcp[order]

        combined = softmax(st.session_state.best_dc_weight * np.log(dcp) + (1 - st.session_state.best_dc_weight) * np.log(ml), axis=1)
        combined = st.session_state.calibrator.transform(combined)

        risk = RiskManager(bankroll=bankroll)
        trades = []
        prev_date = None
        for i in range(len(sub)):
            row = sub.iloc[i]
            if prev_date is not None and row["Date"] != prev_date:
                risk.reset_daily()
            prev_date = row["Date"]
            match_id = f"{row['Date']}_{row['HomeTeam']}_{row['AwayTeam']}"
            probs = combined[i]
            markets = {"home": (probs[0], row.get("AvgH"), "H"), "draw": (probs[1], row.get("AvgD"), "D"),
                       "away": (probs[2], row.get("AvgA"), "A")}
            for market, (prob, odds, outcome) in markets.items():
                if odds is None or pd.isna(odds) or odds <= 1 or odds > 6.0 or prob < min_prob:
                    continue
                implied = 1 / odds
                ev = prob * odds - 1
                edge = prob - implied
                if ev < min_ev:
                    continue
                stake = risk.kelly_stake(prob, odds, edge, match_id)
                if stake <= 0:
                    continue
                won = row["FTR"] == outcome
                profit = risk.record_bet(stake, won, odds)
                closing_col = {"home": "MaxH", "draw": "MaxD", "away": "MaxA"}[market]
                closing_odds = row.get(closing_col)
                clv = (closing_odds - odds) / odds if closing_odds and not pd.isna(closing_odds) else None
                trades.append({"date": row["Date"], "market": market, "prob": prob, "odds": odds, "ev": ev,
                                "stake": stake, "won": won, "profit": profit, "bankroll": risk.bankroll, "clv": clv})

        trades_df = pd.DataFrame(trades)

        if len(trades_df) == 0:
            st.warning("No trades passed these filters on the out-of-fold data. Try lowering the thresholds.")
        else:
            winners = trades_df[trades_df["won"]]
            losers = trades_df[~trades_df["won"]]
            roi = (risk.bankroll - bankroll) / bankroll * 100
            yield_pct = trades_df["profit"].sum() / trades_df["stake"].sum() * 100
            max_dd = ((risk.peak_bankroll - trades_df["bankroll"].min()) / risk.peak_bankroll) * 100

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("ROI", f"{roi:+.1f}%")
            k2.metric("Yield", f"{yield_pct:+.1f}%")
            k3.metric("Trades", len(trades_df))
            k4.metric("Win rate", f"{(trades_df['won'].mean())*100:.1f}%")
            k5.metric("Max drawdown", f"{max_dd:.1f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trades_df["date"], y=trades_df["bankroll"], mode="lines", name="Bankroll"))
            fig.add_hline(y=bankroll, line_dash="dash", line_color="gray")
            fig.update_layout(height=350, margin=dict(t=20, b=20), yaxis_title="Bankroll")
            st.plotly_chart(fig, use_container_width=True)

            if trades_df["clv"].notna().any():
                cc1, cc2 = st.columns(2)
                cc1.metric("Average CLV", f"{trades_df['clv'].mean():.3f}")
                cc2.metric("Positive CLV %", f"{(trades_df['clv'] > 0).mean()*100:.1f}%")

            st.dataframe(
                trades_df[["date", "market", "prob", "odds", "ev", "stake", "won", "profit"]]
                .sort_values("date", ascending=False)
                .style.format({"prob": "{:.1%}", "odds": "{:.2f}", "ev": "{:+.1%}", "stake": "{:.2f}", "profit": "{:+.2f}"}),
                use_container_width=True, hide_index=True, height=300,
            )

            st.caption("⚠️ This backtest uses historical out-of-fold probabilities. Past out-of-sample performance "
                       "is informative but is not a guarantee of future results.")

# ========================================================================
# TAB: ABOUT
# ========================================================================
with tab_about:
    st.markdown("""
    ### Architecture

    | Layer | What it does |
    |---|---|
    | Team-match history | Every match → 2 rows (home/away perspective), so rest days and form are venue-independent |
    | Feature builder | **One function**, used for training AND live predictions — no separate averaging logic |
    | Elo | Chronological, goal-margin weighted, with separate attack/defence tracks |
    | Dixon-Coles | Time-decayed Poisson model per league; ξ chosen by walk-forward held-out likelihood |
    | XGBoost | Trained walk-forward (train on seasons before *i*, test on season *i*) |
    | Calibration | Isotonic regression fit **only** on genuinely valid out-of-fold rows |
    | Ensemble | Dixon-Coles / XGBoost blend weight chosen by walk-forward log-loss grid search |
    | Markets | No-vig pricing, EV/edge/fair-odds, Asian Handicap settled from the score matrix |
    | Decision | Transparent 0-100 Bet Score → BET / LEAN / NO BET |

    ### Limitations
    - Shot/corner/card stats are missing for some league-seasons on football-data.co.uk; those rows fall back to
      historical-average defaults rather than being fabricated.
    - No real Expected Goals (xG) data — the shot/SOT-based proxy is a reasonable stand-in, not a replacement.
    - Dixon-Coles is fit **per league**, so it can't currently model cross-league fixtures (e.g. cup ties).
    - This app is a decision-support tool, not a guarantee of betting profit. Always gamble responsibly.

    Companion notebook: `football_predictor_v2.ipynb` contains the full walk-forward validation report,
    including an OLD-vs-NEW comparison against the original feature engineering.
    """)
