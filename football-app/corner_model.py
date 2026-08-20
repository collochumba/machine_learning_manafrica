"""
CORNER PREDICTION MODULE

A dedicated model for expected corners (HC/AC), separate from the 1X2/goals
pipeline in models.py + train.py. Corners are their own target variable —
this module does NOT derive corners from goals or reuse the Dixon-Coles
goal model in any way.

Contents
--------
1. Feature engineering  (build_corner_features)          — all time-safe
2. Model A: CornerStrengthModel (Poisson / Negative-Binomial team-strength,
   fit by MLE — structurally similar in spirit to DixonColesTimeDecay in
   models.py, but for corner counts, with its own likelihood and no
   low-score correlation correction, since that correction is specific to
   Dixon-Coles' documented small-goal-score bias and has no corner analogue)
3. Model B: CornerGBRModel (HistGradientBoostingRegressor, Poisson loss)
4. Walk-forward comparison of A vs B (MAE / RMSE / Poisson deviance)
5. Market probability construction (Over/Under totals, team totals, corner
   handicaps) via numerical convolution of the fitted count distributions —
   no closed-form assumption is made about the sum of two NB variables.

Every rolling feature below uses shift(1) before rolling, exactly like the
existing goal/shot/corner rolling features in train.py, so this module
introduces no leakage on its own. Train/serve consistency is handled by
predict_corners() in predict.py, which reconstructs the same feature set
for a hypothetical fixture using only historical data.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

CORNER_SCHEMA_VERSION = "corners-v1"

MAX_CORNERS = 20  # matrix truncation point for market-probability derivation


# ============================================================================
# 1. FEATURE ENGINEERING
# ============================================================================

def build_corner_features(df):
    """
    Build the corner-specific feature set. Returns (df_with_features,
    corner_feature_cols). Does not mutate the input df.

    All windows use shift(1) before .rolling(...) — this is mandatory and
    matches the leakage-prevention pattern already used for goals/shots in
    train.py's create_features_with_cache.
    """

    df = df.sort_values(['League', 'Date']).reset_index(drop=True).copy()

    windows = [5, 10]

    for w in windows:
        # Corners WON (attack)
        df[f'HCW_L{w}'] = df.groupby(['League', 'HomeTeam'])['HC'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'ACW_L{w}'] = df.groupby(['League', 'AwayTeam'])['AC'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        # Corners CONCEDED (defense) — home team concedes the away team's
        # corner count in that match, and vice versa.
        df[f'HCD_L{w}'] = df.groupby(['League', 'HomeTeam'])['AC'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'ACD_L{w}'] = df.groupby(['League', 'AwayTeam'])['HC'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )

    # Corner volatility (10-match window; min_periods=3 so it isn't NaN for
    # almost the whole early season for a given team)
    df['HCW_std10'] = df.groupby(['League', 'HomeTeam'])['HC'].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).std()
    )
    df['HCW_min10'] = df.groupby(['League', 'HomeTeam'])['HC'].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).min()
    )
    df['HCW_max10'] = df.groupby(['League', 'HomeTeam'])['HC'].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).max()
    )
    df['ACW_std10'] = df.groupby(['League', 'AwayTeam'])['AC'].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).std()
    )
    df['ACW_min10'] = df.groupby(['League', 'AwayTeam'])['AC'].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).min()
    )
    df['ACW_max10'] = df.groupby(['League', 'AwayTeam'])['AC'].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).max()
    )

    # Matchup features (naive average of "my attack" and "your defense" —
    # a simple, transparent baseline feature for the GBR model; the
    # strength model below does NOT use these, it fits its own attack/
    # defense parameters directly)
    df['ExpHomeCorners_naive'] = (df['HCW_L5'] + df['ACD_L5']) / 2
    df['ExpAwayCorners_naive'] = (df['ACW_L5'] + df['HCD_L5']) / 2
    df['ExpTotalCorners_naive'] = df['ExpHomeCorners_naive'] + df['ExpAwayCorners_naive']
    df['CornerAttackDiff'] = df['HCW_L5'] - df['ACW_L5']
    df['CornerDefenseDiff'] = df['HCD_L5'] - df['ACD_L5']

    corner_feature_cols = []
    for w in windows:
        corner_feature_cols += [f'HCW_L{w}', f'ACW_L{w}', f'HCD_L{w}', f'ACD_L{w}']
    corner_feature_cols += [
        'HCW_std10', 'HCW_min10', 'HCW_max10',
        'ACW_std10', 'ACW_min10', 'ACW_max10',
    ]
    corner_feature_cols += [
        'ExpHomeCorners_naive', 'ExpAwayCorners_naive', 'ExpTotalCorners_naive',
        'CornerAttackDiff', 'CornerDefenseDiff',
    ]
    if 'ELO_diff' in df.columns:
        corner_feature_cols.append('ELO_diff')

    return df, corner_feature_cols


# ============================================================================
# 2. MODEL A — POISSON / NEGATIVE-BINOMIAL TEAM-STRENGTH MODEL
# ============================================================================

class CornerStrengthModel:
    """
    Team-strength count model for corners: log(expected corners) = home_adv*
    is_home + attack[team] - defense[opponent], fit by MLE — the same
    modeling family as Dixon-Coles, applied to corner counts instead of
    goals. No low-score correlation correction is applied (that correction
    addresses a specific, well-documented small-goal-score bias in football
    scorelines; there's no equivalent published bias for corner counts, so
    adding one here would be an unjustified, unvalidated assumption).

    distribution: 'poisson' or 'negbinom'. For 'negbinom' a single
    dispersion parameter alpha is fit jointly with the strength parameters
    (variance = mean + alpha * mean**2).
    """

    def __init__(self, distribution='negbinom', xi=0.002):
        assert distribution in ('poisson', 'negbinom')
        self.distribution = distribution
        self.xi = xi
        self.teams = None
        self.attack = None
        self.defence = None
        self.home_adv = None
        self.alpha = None  # only used for negbinom

    def fit(self, df, league):
        data = df[df['League'] == league].copy()
        data = data.sort_values('Date')

        self.teams = sorted(set(data['HomeTeam']) | set(data['AwayTeam']))
        n = len(self.teams)
        team_to_idx = {t: i for i, t in enumerate(self.teams)}

        home_idx = data['HomeTeam'].map(team_to_idx).values
        away_idx = data['AwayTeam'].map(team_to_idx).values
        hc = data['HC'].values
        ac = data['AC'].values
        days = data['DaysSinceMatch'].values
        weights = np.exp(-self.xi * days)

        use_nb = self.distribution == 'negbinom'
        n_extra_params = 2 if use_nb else 1  # home_adv (+ alpha)

        def unpack(params):
            att = params[:n] - np.mean(params[:n])
            deff = params[n:2 * n]
            home = params[2 * n]
            alpha = np.exp(params[2 * n + 1]) if use_nb else None  # softplus-ish, kept positive via exp
            return att, deff, home, alpha

        def nll(params):
            att, deff, home, alpha = unpack(params)

            lam_h = np.exp(home + att[home_idx] - deff[away_idx])
            lam_a = np.exp(att[away_idx] - deff[home_idx])

            lam_h = np.clip(lam_h, 1e-6, 60)
            lam_a = np.clip(lam_a, 1e-6, 60)

            if use_nb:
                r = 1.0 / alpha
                p_h = r / (r + lam_h)
                p_a = r / (r + lam_a)
                logp_h = nbinom.logpmf(hc, r, p_h)
                logp_a = nbinom.logpmf(ac, r, p_a)
            else:
                logp_h = poisson.logpmf(hc, lam_h)
                logp_a = poisson.logpmf(ac, lam_a)

            ll = np.sum(weights * (logp_h + logp_a))
            return -ll

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.1]])
        if use_nb:
            x0 = np.concatenate([x0, [np.log(0.3)]])  # start alpha ~ 0.3 (mild overdispersion)

        bounds = [(-3, 3)] * (2 * n) + [(-1.0, 1.0)]
        if use_nb:
            bounds += [(np.log(1e-3), np.log(5.0))]

        res = minimize(nll, x0, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 300})

        att, deff, home, alpha = unpack(res.x)
        self.attack = dict(zip(self.teams, att))
        self.defence = dict(zip(self.teams, deff))
        self.home_adv = float(home)
        self.alpha = float(alpha) if use_nb else 0.0
        self.converged_ = bool(res.success)

        return self

    def predict(self, home, away):
        """Returns (exp_home_corners, exp_away_corners). Raises ValueError
        if either team is unknown to this model (caller must catch this and
        fall back — same contract as DixonColesTimeDecay.predict)."""
        if home not in self.attack or away not in self.attack:
            raise ValueError(f"Team not found in corner model: {home} or {away}")

        lam_h = np.exp(self.home_adv + self.attack[home] - self.defence[away])
        lam_a = np.exp(self.attack[away] - self.defence[home])
        return float(np.clip(lam_h, 1e-6, 60)), float(np.clip(lam_a, 1e-6, 60))


# ============================================================================
# 3. MODEL B — GRADIENT-BOOSTING REGRESSION
# ============================================================================

class CornerGBRModel:
    """
    Two independent HistGradientBoostingRegressor models (home corners, away
    corners), Poisson loss (count-appropriate, matches the target's
    distribution better than squared error). Uses HistGradientBoostingRegressor
    rather than XGBoost — it's already available via scikit-learn (already a
    project dependency), so no new dependency is introduced, per the
    instruction to avoid unnecessary dependencies.
    """

    def __init__(self, max_iter=200, max_depth=4, learning_rate=0.05, random_state=42):
        self.params = dict(loss='poisson', max_iter=max_iter, max_depth=max_depth,
                            learning_rate=learning_rate, random_state=random_state)
        self.home_model = None
        self.away_model = None
        self.feature_cols = None

    def fit(self, X, y_home, y_away, feature_cols):
        self.feature_cols = list(feature_cols)
        self.home_model = HistGradientBoostingRegressor(**self.params).fit(X, y_home)
        self.away_model = HistGradientBoostingRegressor(**self.params).fit(X, y_away)
        return self

    def predict(self, X):
        exp_home = self.home_model.predict(X)
        exp_away = self.away_model.predict(X)
        return np.clip(exp_home, 1e-6, 60), np.clip(exp_away, 1e-6, 60)


# ============================================================================
# 4. WALK-FORWARD VALIDATION / MODEL SELECTION
# ============================================================================

def _poisson_deviance(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_term = np.where(y_true > 0, y_true * np.log(np.where(y_true > 0, y_true, 1.0) / y_pred), 0.0)
    return float(2 * np.mean(ratio_term - (y_true - y_pred)))


def walk_forward_validate(df, corner_feature_cols, league, n_splits=4, min_train=200):
    """
    Walk-forward (chronological, TimeSeriesSplit) comparison of:
      - Poisson strength model
      - Negative-Binomial strength model
      - GBR (Poisson-loss HistGradientBoostingRegressor)

    Returns a dict of results keyed by model name, each with MAE, RMSE and
    Poisson deviance for home+away corners pooled, plus the recommended
    model name (lowest walk-forward MAE, ties broken by deviance).

    Only evaluated on rows with complete corner_feature_cols (i.e. rows with
    enough team history) — this is standard walk-forward practice and
    matches what train_ml_model does for the 1X2 model (dropna before
    scoring/fitting).
    """

    league_df = df[df['League'] == league].sort_values('Date').reset_index(drop=True)
    league_df = league_df.dropna(subset=corner_feature_cols + ['HC', 'AC'])

    if len(league_df) < min_train + 50:
        return {'error': f'Not enough rows ({len(league_df)}) for walk-forward validation in {league}'}

    X = league_df[corner_feature_cols].values
    y_home = league_df['HC'].values
    y_away = league_df['AC'].values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    preds = {
        'poisson_strength': {'home': [], 'away': [], 'idx': []},
        'negbinom_strength': {'home': [], 'away': [], 'idx': []},
        'gbr_poisson': {'home': [], 'away': [], 'idx': []},
    }

    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) < min_train:
            continue
        train_df = league_df.iloc[train_idx]
        test_df = league_df.iloc[test_idx]

        # Strength models need the raw match rows (HomeTeam/AwayTeam/HC/AC/
        # DaysSinceMatch/League), not the feature matrix.
        for dist, key in [('poisson', 'poisson_strength'), ('negbinom', 'negbinom_strength')]:
            model = CornerStrengthModel(distribution=dist).fit(train_df, league)
            for _, row in test_df.iterrows():
                try:
                    eh, ea = model.predict(row['HomeTeam'], row['AwayTeam'])
                except ValueError:
                    continue  # unseen team in this fold — skip, not a fallback test
                preds[key]['home'].append(eh)
                preds[key]['away'].append(ea)
                preds[key]['idx'].append(row.name)

        # GBR
        gbr = CornerGBRModel().fit(X[train_idx], y_home[train_idx], y_away[train_idx], corner_feature_cols)
        eh, ea = gbr.predict(X[test_idx])
        preds['gbr_poisson']['home'].extend(eh.tolist())
        preds['gbr_poisson']['away'].extend(ea.tolist())
        preds['gbr_poisson']['idx'].extend(test_df.index.tolist())

    results = {}
    for name, p in preds.items():
        if len(p['home']) < 20:
            results[name] = {'error': 'too few evaluable predictions'}
            continue
        idx = p['idx']
        yt_h = league_df.loc[idx, 'HC'].values
        yt_a = league_df.loc[idx, 'AC'].values
        ph = np.array(p['home'])
        pa = np.array(p['away'])

        mae = float((np.abs(yt_h - ph).mean() + np.abs(yt_a - pa).mean()) / 2)
        rmse = float((np.sqrt(((yt_h - ph) ** 2).mean()) + np.sqrt(((yt_a - pa) ** 2).mean())) / 2)
        dev = float((_poisson_deviance(yt_h, ph) + _poisson_deviance(yt_a, pa)) / 2)

        results[name] = {'mae': mae, 'rmse': rmse, 'poisson_deviance': dev, 'n': len(idx)}

    valid = {k: v for k, v in results.items() if 'error' not in v}
    if valid:
        best = min(valid, key=lambda k: valid[k]['mae'])
    else:
        best = None

    results['_recommended'] = best
    return results


# ============================================================================
# 5. MARKET PROBABILITIES (numeric convolution — no closed-form assumption)
# ============================================================================

def _count_pmf(exp_value, alpha, max_n=MAX_CORNERS):
    """PMF array [0..max_n] for either Poisson (alpha=0) or Negative
    Binomial (alpha>0) with the given mean. Using the actual PMF (not a
    normal approximation) keeps market probabilities honest for realistic
    corner counts (typically 0-18)."""
    exp_value = max(float(exp_value), 1e-6)
    ks = np.arange(max_n + 1)
    if alpha and alpha > 1e-6:
        r = 1.0 / alpha
        p = r / (r + exp_value)
        pmf = nbinom.pmf(ks, r, p)
    else:
        pmf = poisson.pmf(ks, exp_value)
    pmf = pmf / pmf.sum()  # renormalize after truncation at max_n
    return pmf


def corner_market_probabilities(exp_home, exp_away, alpha_home=0.0, alpha_away=0.0):
    """
    Build corner market probabilities from expected home/away corners,
    assuming independence between the two teams' corner counts (standard
    simplifying assumption for this market — no dataset in this project
    currently supports estimating their correlation safely). The joint
    distribution is built by outer-producting the two marginal PMFs (like
    the score_matrix in models.py for goals), so every derived probability
    is read directly off that matrix rather than approximated.

    Returns a dict: exp_home, exp_away, exp_total, and market probabilities
    for total-corner Over/Under lines, team-corner Over lines, and corner
    handicaps.
    """

    pmf_h = _count_pmf(exp_home, alpha_home)
    pmf_a = _count_pmf(exp_away, alpha_away)
    joint = np.outer(pmf_h, pmf_a)  # joint[i, j] = P(home=i, away=j)
    joint = joint / joint.sum()

    max_n = joint.shape[0] - 1
    home_counts = np.arange(max_n + 1).reshape(-1, 1)
    away_counts = np.arange(max_n + 1).reshape(1, -1)
    total = home_counts + away_counts

    markets = {
        'exp_home_corners': float(exp_home),
        'exp_away_corners': float(exp_away),
        'exp_total_corners': float(exp_home + exp_away),
    }

    # Total-corner Over/Under
    for line in [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
        over = float(joint[total > line].sum())
        markets[f'Corners Over {line}'] = over
        markets[f'Corners Under {line}'] = 1 - over

    # Team-corner Over (only where the marginal genuinely supports it —
    # i.e. always, since it's a direct marginal-CDF read, but we cap the
    # lines offered to ones with non-trivial spread)
    for line in [2.5, 3.5, 4.5, 5.5]:
        home_over = float(pmf_h[int(np.ceil(line)):].sum())
        away_over = float(pmf_a[int(np.ceil(line)):].sum())
        markets[f'Home Corners Over {line}'] = home_over
        markets[f'Away Corners Over {line}'] = away_over

    # Corner handicap (same construction as Asian Handicap in models.py)
    diff = home_counts - away_counts
    for h in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
        adjusted = diff + h
        home_covers = joint[adjusted > 0].sum()
        away_covers = joint[adjusted < 0].sum()
        label = f"{h:+.1f}"
        markets[f'Corner Hcp Home {label}'] = float(home_covers)
        markets[f'Corner Hcp Away {label}'] = float(away_covers)

    return markets
