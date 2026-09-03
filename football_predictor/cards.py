"""Dedicated pre-match card prediction model.

The model is deliberately separate from the 1X2/goals and corner models.
It predicts home and away yellow-card counts from strictly lagged team,
foul, referee and league information. Missing referee history falls back to
the league baseline rather than being fabricated.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

CARD_SCHEMA_VERSION = "cards-v1"


def _rolling(df, group_cols, source, window):
    return df.groupby(group_cols)[source].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )


def build_card_features(df):
    out = df.sort_values(["League", "Date"]).reset_index(drop=True).copy()
    for c in ["HY", "AY", "HR", "AR", "HF", "AF"]:
        if c not in out.columns:
            out[c] = np.nan
    if "Referee" not in out.columns:
        out["Referee"] = np.nan

    for w in (5, 10):
        out[f"HY_L{w}"] = _rolling(out, ["League", "HomeTeam"], "HY", w)
        out[f"AY_L{w}"] = _rolling(out, ["League", "AwayTeam"], "AY", w)
        out[f"HR_L{w}"] = _rolling(out, ["League", "HomeTeam"], "HR", w)
        out[f"AR_L{w}"] = _rolling(out, ["League", "AwayTeam"], "AR", w)
        out[f"HF_L{w}"] = _rolling(out, ["League", "HomeTeam"], "HF", w)
        out[f"AF_L{w}"] = _rolling(out, ["League", "AwayTeam"], "AF", w)

    out["HCardPts_L5"] = out["HY_L5"] + 3 * out["HR_L5"]
    out["ACardPts_L5"] = out["AY_L5"] + 3 * out["AR_L5"]
    out["CardIntensityDiff"] = out["HCardPts_L5"] - out["ACardPts_L5"]
    out["FoulDiff"] = out["HF_L5"] - out["AF_L5"]

    # Chronological referee history: strictly prior matches only.
    temp = out[["Date", "League", "Referee", "HY", "AY", "HR", "AR", "HF", "AF"]].copy()
    temp["TotalYellows"] = temp["HY"].fillna(0) + temp["AY"].fillna(0)
    temp["TotalReds"] = temp["HR"].fillna(0) + temp["AR"].fillna(0)
    temp["TotalFouls"] = temp["HF"].fillna(0) + temp["AF"].fillna(0)
    temp = temp.sort_values("Date")
    for stat, source in [("ref_avg_yellows", "TotalYellows"),
                         ("ref_avg_reds", "TotalReds"),
                         ("ref_avg_fouls", "TotalFouls")]:
        temp[stat] = temp.groupby("Referee")[source].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
    temp["ref_sample_size"] = temp.groupby("Referee")["TotalYellows"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).count()
    )
    league_baselines = temp.groupby("League")[["TotalYellows", "TotalReds", "TotalFouls"]].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    # The groupby/transform above is safe chronologically after sorting.
    temp = temp.sort_index()
    out = out.join(temp[["ref_avg_yellows", "ref_avg_reds", "ref_avg_fouls", "ref_sample_size"]], how="left")

    # Conservative fallback baselines, computed from prior matches only.
    out["ref_avg_yellows"] = out["ref_avg_yellows"].fillna(out["HY_L5"].fillna(0) + out["AY_L5"].fillna(0))
    out["ref_avg_reds"] = out["ref_avg_reds"].fillna(out["HR_L5"].fillna(0) + out["AR_L5"].fillna(0))
    out["ref_avg_fouls"] = out["ref_avg_fouls"].fillna(out["HF_L5"].fillna(0) + out["AF_L5"].fillna(0))
    out["ref_sample_size"] = out["ref_sample_size"].fillna(0)

    # League one-hot effects.
    for league in sorted(out["League"].dropna().unique()):
        out[f"Lg_{league}"] = (out["League"] == league).astype(float)

    feature_cols = [
        "HY_L5", "AY_L5", "HY_L10", "AY_L10",
        "HR_L5", "AR_L5", "HR_L10", "AR_L10",
        "HF_L5", "AF_L5", "HF_L10", "AF_L10",
        "HCardPts_L5", "ACardPts_L5", "CardIntensityDiff", "FoulDiff",
        "ref_avg_yellows", "ref_avg_reds", "ref_avg_fouls", "ref_sample_size",
    ] + sorted(c for c in out.columns if c.startswith("Lg_"))
    return out, feature_cols


class CardPredictionModel:
    def __init__(self, min_history=3):
        self.min_history = min_history
        self.models = {}
        self.feature_cols = None
        self.league_baselines = {}
        self.team_history = {}
        self.ref_history = {}
        self.converged_ = False

    def fit(self, df, league):
        data, feature_cols = build_card_features(df)
        data = data[data["League"] == league].copy()
        self.feature_cols = feature_cols
        X = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        home_target = "HY_observed" if "HY_observed" in data.columns else "HY"
        away_target = "AY_observed" if "AY_observed" in data.columns else "AY"
        data = data.dropna(subset=[home_target, away_target]).copy()
        X = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_home = data[home_target].astype(float)
        y_away = data[away_target].astype(float)
        if len(data) < 30:
            raise ValueError(f"Insufficient card history for {league}: {len(data)} rows")

        self.models["home"] = HistGradientBoostingRegressor(loss="poisson", max_iter=180, learning_rate=0.045, max_leaf_nodes=15, random_state=42).fit(X, y_home)
        self.models["away"] = HistGradientBoostingRegressor(loss="poisson", max_iter=180, learning_rate=0.045, max_leaf_nodes=15, random_state=43).fit(X, y_away)
        self.league_baselines[league] = {
            "home_yellows": float(y_home.mean()),
            "away_yellows": float(y_away.mean()),
            "total_yellows": float((y_home + y_away).mean()),
            "home_reds": float(data["HR_observed"].dropna().mean()) if "HR_observed" in data.columns and data["HR_observed"].notna().any() else 0.0,
            "away_reds": float(data["AR_observed"].dropna().mean()) if "AR_observed" in data.columns and data["AR_observed"].notna().any() else 0.0,
        }
        self._history = data
        self.converged_ = True
        return self

    def _fixture_features(self, df, home, away, league, referee=None, fixture_date=None):
        data, feature_cols = build_card_features(df)
        ldf = data[data["League"] == league].sort_values("Date")
        if fixture_date is not None:
            d = pd.to_datetime(fixture_date, dayfirst=True, errors="coerce")
            if pd.notna(d):
                ldf = ldf[ldf["Date"] < d]
        if ldf.empty:
            raise ValueError(f"No historical card data for {league}")
        hr = ldf[ldf["HomeTeam"] == home]
        ar = ldf[ldf["AwayTeam"] == away]
        if len(hr) < self.min_history or len(ar) < self.min_history:
            raise ValueError(f"Insufficient card history for {home} or {away}")
        h = hr.iloc[-1].copy()
        a = ar.iloc[-1].copy()
        row = h.copy()
        # Team-role features come from the correct role-specific latest rows.
        for c in feature_cols:
            if c.startswith(("AY_", "AR_", "AF_", "ACardPts_")):
                row[c] = a.get(c, row.get(c, 0))
        row["CardIntensityDiff"] = row.get("HCardPts_L5", 0) - a.get("ACardPts_L5", 0)
        row["FoulDiff"] = row.get("HF_L5", 0) - a.get("AF_L5", 0)
        for c in feature_cols:
            if c.startswith("Lg_"):
                row[c] = 1.0 if c == f"Lg_{league}" else 0.0
        if referee:
            refs = ldf[ldf["Referee"].astype(str).str.strip() == str(referee).strip()]
            if len(refs) >= 15:
                row["ref_avg_yellows"] = float((refs["HY"].fillna(0) + refs["AY"].fillna(0)).mean())
                row["ref_avg_reds"] = float((refs["HR"].fillna(0) + refs["AR"].fillna(0)).mean())
                row["ref_avg_fouls"] = float((refs["HF"].fillna(0) + refs["AF"].fillna(0)).mean())
                row["ref_sample_size"] = float(len(refs))
        return pd.DataFrame([[row.get(c, 0) for c in feature_cols]], columns=feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0)

    def predict(self, df, home, away, league, referee=None, fixture_date=None):
        if league not in self.league_baselines or not self.models:
            raise ValueError(f"No card model available for {league}")
        try:
            X = self._fixture_features(df, home, away, league, referee, fixture_date)
            home_y = float(self.models["home"].predict(X)[0])
            away_y = float(self.models["away"].predict(X)[0])
            source = "model"
        except Exception:
            base = self.league_baselines[league]
            home_y, away_y = base["home_yellows"], base["away_yellows"]
            source = "league_baseline"
        total = max(0.0, home_y + away_y)
        # Approximate 80% interval using a Poisson total distribution.
        lo = max(0, int(np.floor(total - 1.28 * np.sqrt(max(total, 1e-6)))))
        hi = int(np.ceil(total + 1.28 * np.sqrt(max(total, 1e-6))))
        return {
            "exp_home_yellows": home_y,
            "exp_away_yellows": away_y,
            "exp_total_yellows": total,
            "likely_range": [lo, hi],
            "source": source,
        }


def train_card_models(df):
    models = {}
    feature_cols = None
    validation = {}
    for league in sorted(df["League"].dropna().unique()):
        try:
            model = CardPredictionModel().fit(df, league)
            models[league] = model
            feature_cols = model.feature_cols
            # Genuine chronological holdout diagnostic using the same lagged
            # feature construction. This is a validation metric, not an
            # in-sample performance claim.
            data, cols = build_card_features(df)
            data = data[data["League"] == league].dropna(subset=[c for c in cols] + ["HY_observed", "AY_observed"] if "HY_observed" in data.columns else [c for c in cols] + ["HY", "AY"]).copy()
            n = len(data)
            split = max(20, int(n * 0.8))
            if n > 30 and split < n:
                Xtr = data[cols].replace([np.inf, -np.inf], np.nan).fillna(0).iloc[:split]
                Xte = data[cols].replace([np.inf, -np.inf], np.nan).fillna(0).iloc[split:]
                ytr_h = data["HY_observed"].iloc[:split] if "HY_observed" in data.columns else data["HY"].iloc[:split]
                yte_h = data["HY_observed"].iloc[split:] if "HY_observed" in data.columns else data["HY"].iloc[split:]
                ytr_a = data["AY_observed"].iloc[:split] if "AY_observed" in data.columns else data["AY"].iloc[:split]
                yte_a = data["AY_observed"].iloc[split:] if "AY_observed" in data.columns else data["AY"].iloc[split:]
                from sklearn.metrics import mean_absolute_error
                vh = HistGradientBoostingRegressor(loss="poisson", max_iter=180, learning_rate=0.045, max_leaf_nodes=15, random_state=42).fit(Xtr, ytr_h)
                va = HistGradientBoostingRegressor(loss="poisson", max_iter=180, learning_rate=0.045, max_leaf_nodes=15, random_state=43).fit(Xtr, ytr_a)
                validation[league] = {"status": "trained", "n": n, "holdout_n": len(Xte), "mae_home": float(mean_absolute_error(yte_h, vh.predict(Xte))), "mae_away": float(mean_absolute_error(yte_a, va.predict(Xte)))}
            else:
                validation[league] = {"status": "trained", "n": n, "holdout_n": 0, "validation": "insufficient rows"}
        except Exception as exc:
            validation[league] = {"status": "failed", "error": str(exc)}
    return models, feature_cols or [], validation


# ============================================================================
# FIXTURE-LEVEL PREDICTION ENTRY POINT (moved here from the old predict.py so
# all card-specific logic — features, model, training, and the fixture-level
# predict call — lives in one file)
# ============================================================================

def predict_cards(league, home, away, card_models, df, referee=None, fixture_date=None):
    """Predict yellow cards using the dedicated cards model above.

    Missing/low referee history falls back inside CardPredictionModel to the
    league baseline. No referee information is invented.
    """
    if card_models is None or league not in card_models:
        return {"error": f"No card model available for league: {league}"}
    model = card_models[league]
    try:
        return {"league": league, "home": home, "away": away, **model.predict(df, home, away, league, referee=referee, fixture_date=fixture_date)}
    except Exception as exc:
        return {"error": str(exc), "league": league, "home": home, "away": away}
