"""
MODELS MODULE

Core football prediction mathematics AND the centralized fixture-level
prediction engine built on top of it:

  1. Dixon-Coles time-decay goal model + ML ensemble  (original models.py)
  2. Market-probability construction from a score matrix
  3. Team-name pass-through + fixture feature assembly, and the
     predict_with_fallback() / predict_multiple_fixtures() engine
     (moved here from the old predict.py, since this is where the DC+ML
     ensemble those functions call already lives)

Betting economics (no-vig pricing, EV, Kelly, value-bet filtering, CLV) live
in betting.py, imported below where needed — this module is only ever the
math that turns a fixture into probabilities, not the economics layered on
top of those probabilities.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from scipy.special import softmax

from betting import calculate_value, find_value_bets, calculate_kelly_stake

class DixonColesTimeDecay:
    """
    Dixon-Coles model with time decay weighting.

    Full implementation with:
    - Poisson-based goal modeling
    - Home advantage parameter
    - Low-score correction (rho)
    - Time decay weighting
    - Maximum likelihood estimation
    """

    def __init__(self, xi=0.002, max_goals=10):
        """
        Args:
            xi: Time decay parameter (higher = more recent weight)
            max_goals: Maximum goals to consider in calculations
        """
        self.xi = xi
        self.max_goals = max_goals
        self.teams = None
        self.attack = None
        self.defence = None
        self.home_adv = None
        self.rho = None
        self.converged_ = False
        self.optimizer_message_ = ''
        self.optimizer_status_ = -1
        self.optimizer_nit_ = -1

    def fit(self, df, league=None):
        """
        Fit Dixon-Coles model using MLE.

        Args:
            df: DataFrame with match history
            league: League to fit (None = all data)

        Returns:
            self
        """

        if league:
            data = df[df['League'] == league].copy()
        else:
            data = df.copy()

        data = data.sort_values('Date')

        self.teams = sorted(set(data['HomeTeam']) | set(data['AwayTeam']))
        n = len(self.teams)

        team_to_idx = {team: i for i, team in enumerate(self.teams)}

        home_idx = data['HomeTeam'].map(team_to_idx).values
        away_idx = data['AwayTeam'].map(team_to_idx).values
        hg = data['FTHG'].values
        ag = data['FTAG'].values
        days = data['DaysSinceMatch'].values

        weights = np.exp(-self.xi * days)

        def nll(params):
            """Negative log-likelihood with Dixon-Coles correction."""
            att = params[:n]
            deff = params[n:2*n]
            home = params[2*n]
            rho = params[2*n+1]

            att = att - np.mean(att)

            lh = np.exp(home + att[home_idx] - deff[away_idx])
            la = np.exp(att[away_idx] - deff[home_idx])

            p = poisson.pmf(hg, lh) * poisson.pmf(ag, la)

            corr = np.ones_like(p)
            mask00 = (hg == 0) & (ag == 0)
            mask01 = (hg == 0) & (ag == 1)
            mask10 = (hg == 1) & (ag == 0)
            mask11 = (hg == 1) & (ag == 1)

            corr[mask00] = 1 - lh[mask00] * la[mask00] * rho
            corr[mask01] = 1 + lh[mask01] * rho
            corr[mask10] = 1 + la[mask10] * rho
            corr[mask11] = 1 - rho

            p *= corr
            ll = np.sum(weights * np.log(np.maximum(p, 1e-12)))

            return -ll

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [0]])
        # FIX: bounds were (0, 0.5) for home advantage, which structurally
        # prevents the model from ever expressing a neutral or negative
        # home advantage (a real phenomenon — e.g. some leagues/periods,
        # or specific team pairings, have shown near-zero or slightly
        # negative home advantage). Similarly widened rho slightly beyond
        # the very tight (-0.1, 0.1) to give the low-score correction a
        # bit more room while still keeping it in a sane, well-studied range.
        bounds = [(-3, 3)] * (2 * n) + [(-0.3, 0.8), (-0.15, 0.15)]

        attempts = [
            (x0, {'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-6}),
            (
                np.asarray(x0, dtype=float)
                + np.random.default_rng(42).normal(0, 0.01, size=len(x0)),
                {'maxiter': 800, 'ftol': 1e-8, 'gtol': 1e-7},
            ),
        ]

        res = None
        for x0_try, options in attempts:
            candidate = minimize(
                nll, x0_try, method='L-BFGS-B',
                bounds=bounds, options=options
            )
            if res is None or candidate.fun < res.fun:
                res = candidate
            if candidate.success:
                break

        params = res.x
        att = params[:n] - np.mean(params[:n])
        deff = params[n:2*n]

        self.attack = dict(zip(self.teams, att))
        self.defence = dict(zip(self.teams, deff))
        self.home_adv = params[2*n]
        self.rho = params[2*n+1]
        self.converged_ = bool(res.success)
        self.optimizer_message_ = str(getattr(res, 'message', ''))
        self.optimizer_status_ = int(getattr(res, 'status', -1))
        self.optimizer_nit_ = int(getattr(res, 'nit', -1))

        return self

    def predict(self, home, away):
        """
        Generate full match prediction.

        Returns:
            Dictionary with probabilities, expected goals, score matrix
        """

        if home not in self.attack or away not in self.attack:
            raise ValueError(f"Team not found: {home} or {away}")

        lh = np.exp(self.home_adv + self.attack[home] - self.defence[away])
        la = np.exp(self.attack[away] - self.defence[home])

        max_g = self.max_goals

        home_probs = poisson.pmf(range(max_g+1), lh)
        away_probs = poisson.pmf(range(max_g+1), la)

        score_matrix = np.outer(home_probs, away_probs)

        score_matrix[0,0] *= (1 - lh*la*self.rho)
        score_matrix[0,1] *= (1 + lh*self.rho)
        score_matrix[1,0] *= (1 + la*self.rho)
        score_matrix[1,1] *= (1 - self.rho)

        # Safety: with the widened rho bound, the low-score correction terms
        # (1 - lh*la*rho) and (1 - rho) can in principle go negative for
        # extreme lambda/rho combinations. Clip before renormalizing so no
        # market probability derived from this matrix can be negative.
        score_matrix = np.clip(score_matrix, 0, None)

        # Re-normalize: the Dixon-Coles low-score correction perturbs four
        # cells of a matrix that was already truncated at max_goals, so the
        # matrix is no longer guaranteed to sum to 1. Renormalize before it
        # is used to derive any market probability.
        score_matrix = score_matrix / score_matrix.sum()

        prob_home = np.tril(score_matrix, -1).sum()
        prob_draw = np.trace(score_matrix)
        prob_away = np.triu(score_matrix, 1).sum()

        total_goals = np.add.outer(range(max_g+1), range(max_g+1))
        prob_over25 = score_matrix[total_goals > 2.5].sum()
        prob_under25 = 1 - prob_over25

        return {
            'lambda_home': float(lh),
            'lambda_away': float(la),
            'prob_home': float(prob_home),
            'prob_draw': float(prob_draw),
            'prob_away': float(prob_away),
            'prob_over_25': float(prob_over25),
            'prob_under_25': float(prob_under25),
            'exp_goals': float(lh + la),
            'score_matrix': score_matrix
        }


def ensemble_prediction(final_model, dc_models, league, home, away, features, dc_weight=0.6):
    """
    Ensemble prediction via log-odds pooling.

    Combines:
    - Dixon-Coles (Poisson-based)
    - XGBoost (ML-based)

    Args:
        final_model: Trained ML model
        dc_models: Dict of Dixon-Coles models by league
        league: League name
        home: Home team
        away: Away team
        features: Feature vector for ML
        dc_weight: Weight for Dixon-Coles (default 0.6)

    Returns:
        probs: [home, draw, away] probabilities
        dc_pred: Full Dixon-Coles prediction dict
    """

    # Dixon-Coles prediction.  A non-converged optimizer result must never be
    # silently treated as a production model.  Training records convergence
    # on each fitted model; if an older artifact contains a non-converged
    # model, raise ValueError so predict_with_fallback() can use its safe
    # league-average fallback instead of deploying an unreliable fit.
    dc_model = dc_models.get(league)
    if dc_model is None:
        raise ValueError(f"League not supported: {league}")
    if hasattr(dc_model, 'converged_') and not bool(dc_model.converged_):
        raise ValueError(
            f"Dixon-Coles model for {league} did not converge; "
            "safe fallback required."
        )

    dc_pred = dc_model.predict(home, away)

    dc_probs = np.array([
        dc_pred['prob_home'],
        dc_pred['prob_draw'],
        dc_pred['prob_away']
    ])

    # ML prediction
    ml_probs = final_model.predict_proba(features.reshape(1, -1))[0]

    # Safety clipping and normalization.  Some estimators/artifacts can
    # return probabilities with tiny floating-point drift; normalize before
    # pooling so the ensemble always starts from valid probability vectors.
    dc_probs = np.clip(dc_probs, 1e-9, 1 - 1e-9)
    ml_probs = np.clip(ml_probs, 1e-9, 1 - 1e-9)
    dc_probs = dc_probs / dc_probs.sum()
    ml_probs = ml_probs / ml_probs.sum()

    # Log-odds pooling
    dc_log = np.log(dc_probs)
    ml_log = np.log(ml_probs)

    combined_log = dc_weight * dc_log + (1 - dc_weight) * ml_log

    # Normalize
    probs = softmax(combined_log)

    return probs, dc_pred


def _goal_market_probs(score_matrix, max_g):
    """Derive Over/Under and BTTS probabilities directly from the score matrix."""

    markets = {}
    total_goals = np.add.outer(range(max_g + 1), range(max_g + 1))

    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        over = float(score_matrix[total_goals > line].sum())
        markets[f'Over {line}'] = over
        markets[f'Under {line}'] = 1 - over

    # BTTS: both teams score means home goals >= 1 AND away goals >= 1
    btts_yes = float(score_matrix[1:, 1:].sum())
    markets['BTTS Yes'] = btts_yes
    markets['BTTS No'] = 1 - btts_yes

    return markets


def _asian_handicap_probs(score_matrix, max_g):
    """
    Derive Asian Handicap probabilities directly from the score matrix,
    rather than approximating with a sigmoid on the lambda difference.

    For a given home handicap h (e.g. -0.5, -1.0, +0.5), the home side
    covers when (home_goals + h) > away_goals. Quarter lines (.25/.75)
    split the stake between the two adjacent half/whole lines, but we
    only expose the common half/whole lines here.
    """

    home_goals = np.arange(max_g + 1).reshape(-1, 1)
    away_goals = np.arange(max_g + 1).reshape(1, -1)
    diff = home_goals - away_goals  # broadcast to (max_g+1, max_g+1)

    markets = {}
    for h in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
        adjusted = diff + h
        home_covers = score_matrix[adjusted > 0].sum()
        away_covers = score_matrix[adjusted < 0].sum()
        push = score_matrix[adjusted == 0].sum()

        # For whole-number lines a push refunds the stake; report probability
        # of covering conditional on not pushing, which is the standard way
        # AH probabilities are quoted.
        denom = home_covers + away_covers
        if denom > 0:
            home_prob = home_covers / denom
            away_prob = away_covers / denom
        else:
            home_prob = away_prob = 0.5

        label = f"{h:+.1f}".rstrip('0').rstrip('.') if h != 0 else "0"
        markets[f'AH Home {label}'] = float(home_prob)
        markets[f'AH Away {label}'] = float(away_prob)
        markets[f'AH Push {label}'] = float(push)

    return markets


def build_market_probabilities(probs, dc_pred):
    """
    Build probabilities for all betting markets.

    Markets:
    - 1X2 (Home/Draw/Away)
    - Over/Under (0.5 through 4.5)
    - BTTS
    - Double Chance (1X, X2, 12)
    - Draw No Bet (DNB Home, DNB Away)
    - Asian Handicap (derived from the score matrix, not a sigmoid approximation)

    Args:
        probs: [home, draw, away] probabilities from ensemble
        dc_pred: Dixon-Coles prediction dict

    Returns:
        Dictionary of all market probabilities
    """

    markets = {}

    # 1X2
    markets['Home'] = float(probs[0])
    markets['Draw'] = float(probs[1])
    markets['Away'] = float(probs[2])

    # Double Chance
    markets['1X'] = probs[0] + probs[1]  # Home or Draw
    markets['X2'] = probs[1] + probs[2]  # Draw or Away
    markets['12'] = probs[0] + probs[2]  # Home or Away (no draw)

    # Draw No Bet
    home_dnb = probs[0] / (probs[0] + probs[2])
    away_dnb = probs[2] / (probs[0] + probs[2])
    markets['DNB Home'] = float(home_dnb)
    markets['DNB Away'] = float(away_dnb)

    score_matrix = dc_pred.get('score_matrix')

    if score_matrix is not None:
        max_g = score_matrix.shape[0] - 1

        # Goal markets + BTTS derived straight from the score matrix
        markets.update(_goal_market_probs(score_matrix, max_g))

        # Asian Handicap derived straight from the score matrix
        markets.update(_asian_handicap_probs(score_matrix, max_g))
    else:
        # Fallback path (e.g. unknown-team league-average fallback in predict.py,
        # where no score matrix is available). Only the 2.5 line is provided
        # by the caller in that case.
        markets['Over 2.5'] = dc_pred.get('prob_over_25', 0.5)
        markets['Under 2.5'] = dc_pred.get('prob_under_25', 0.5)

    return markets


def calculate_confidence_score(probs):
    """
    Calculate prediction confidence.

    Confidence is scaled relative to a uniform distribution (1/3 each).
    A score of 0 = maximally uncertain (all outcomes equal),
    a score of 1 = maximally confident (one outcome probability = 1).

    Args:
        probs: Array of [home, draw, away] probabilities

    Returns:
        Confidence score (0 = uncertain, 1 = very confident)
    """

    sorted_probs = sorted(probs, reverse=True)
    # Scale: 0 when max_prob = 1/3 (uniform), 1 when max_prob = 1
    confidence = (sorted_probs[0] - 1/3) * 1.5
    confidence = max(0.0, min(1.0, float(confidence)))

    return confidence


# ============================================================================
# PREDICTION ENGINE (moved here from the old predict.py)
# ============================================================================

def normalize_team_name(team_name, team_mapping, all_teams, league, df):
    """
    Pass through an ALREADY-CANONICAL team name for prediction.

    FIX #1: this used to be an independent, silently-substituting resolver
    — it ran its own difflib fuzzy match and used the top hit (e.g.
    'Le Mans' -> 'Lens') as the team to actually predict with, completely
    bypassing team_normalization.py's strict no-auto-substitution rule.
    team_normalization.py is the one authoritative resolver (exact /
    case-insensitive / normalized / alias only, otherwise reject) — every
    caller (the "Load Latest Fixtures" flow AND the manual-paste flow in
    app.py) now resolves through it BEFORE a fixture ever reaches this
    prediction module, so this function's job is no longer to re-validate
    the name — that would just be a second, conflicting source of truth.

    FIX #2: an earlier version of this fix rejected any name absent from
    `all_teams` (the historical training universe). That's wrong for a
    newly-promoted club with NO prior top-flight history in the 10-season
    training window at all (e.g. Coventry, Hull, Le Mans this season) —
    team_normalization.py correctly confirms these ARE valid current
    top-flight teams, they just have no fitted Dixon-Coles/Elo parameters
    yet. Rejecting them here would skip right past the league-average
    fallback that predict_with_fallback() already implements below for
    exactly this situation. So: pass the name straight through, and let
    the existing DC/Elo-level fallback (a ValueError from
    ensemble_prediction, caught further down) handle "no historical fit"
    gracefully, same as it always has.
    """
    if not team_name or not isinstance(team_name, str):
        return None, 0.0, {'original': team_name, 'error': 'Empty/invalid team name.'}
    return team_name, 1.0, None


# Feature-column classification used by get_fixture_features below.
# 'home' columns describe the HOME team's own recent home-match record and
# should be sourced from that team's most recent match AS HOME TEAM.
# 'away' columns describe the AWAY team's own recent away-match record and
# should be sourced from that team's most recent match AS AWAY TEAM.
# 'diff' columns are recomputed from the resolved home/away values rather
# than copied from either source row, since they compare the two teams
# directly and a copied value would reflect the wrong pairing.
_HOME_PREFIXES = ('HGS_', 'HGC_', 'HS_', 'HST_', 'HC_', 'HFormPPG', 'HSC_', 'HSTC_', 'HSTPct_', 'HF_', 'HY_', 'HR_', 'HCardPts_')
_AWAY_PREFIXES = ('AGS_', 'AGC_', 'AS_', 'AST_', 'AC_', 'AFormPPG', 'ASC_', 'ASTC_', 'ASTPct_', 'AF_', 'AY_', 'AR_', 'ACardPts_')
_DIFF_COLS = {
    'AttackDiff': ('HGS_L5', 'AGC_L5'),
    'DefenseDiff': ('HGC_L5', 'AGS_L5'),
    'ShotDiff': ('HS_L5', 'AS_L5'),
    'ShotTargetDiff': ('HST_L5', 'AST_L5'),
    'CornerDiff': ('HC_L5', 'AC_L5'),
    'ShotConcededDiff': ('HSC_L5', 'ASC_L5'),
    'FoulDiff': ('HF_L5', 'AF_L5'),
    'CardPtsDiff': ('HCardPts_L5', 'ACardPts_L5'),
}
# Match-level (not team-role-specific) columns. The referee of a FUTURE
# fixture is never known at prediction time (football-data.co.uk's
# fixtures.csv doesn't carry it), so these always fall back to the
# league-wide historical baseline rather than being copied from either
# team's most recent row — that's the "clearly defined league-level
# fallback" the referee-feature spec requires, applied unconditionally
# at prediction time since the alternative (a specific referee) is simply
# unavailable pre-match.
_LEAGUE_BASELINE_ONLY_COLS = ('RefFouls_hist', 'RefCards_hist')




# Feature-column classification used by get_fixture_features below.
# 'home' columns describe the HOME team's own recent home-match record and
# should be sourced from that team's most recent match AS HOME TEAM.
# 'away' columns describe the AWAY team's own recent away-match record and
# should be sourced from that team's most recent match AS AWAY TEAM.
# 'diff' columns are recomputed from the resolved home/away values rather
# than copied from either source row, since they compare the two teams
# directly and a copied value would reflect the wrong pairing.
_HOME_PREFIXES = ('HGS_', 'HGC_', 'HS_', 'HST_', 'HC_', 'HFormPPG', 'HSC_', 'HSTC_', 'HSTPct_', 'HF_', 'HY_', 'HR_', 'HCardPts_')
_AWAY_PREFIXES = ('AGS_', 'AGC_', 'AS_', 'AST_', 'AC_', 'AFormPPG', 'ASC_', 'ASTC_', 'ASTPct_', 'AF_', 'AY_', 'AR_', 'ACardPts_')
_DIFF_COLS = {
    'AttackDiff': ('HGS_L5', 'AGC_L5'),
    'DefenseDiff': ('HGC_L5', 'AGS_L5'),
    'ShotDiff': ('HS_L5', 'AS_L5'),
    'ShotTargetDiff': ('HST_L5', 'AST_L5'),
    'CornerDiff': ('HC_L5', 'AC_L5'),
    'ShotConcededDiff': ('HSC_L5', 'ASC_L5'),
    'FoulDiff': ('HF_L5', 'AF_L5'),
    'CardPtsDiff': ('HCardPts_L5', 'ACardPts_L5'),
}
# Match-level (not team-role-specific) columns. The referee of a FUTURE
# fixture is never known at prediction time (football-data.co.uk's
# fixtures.csv doesn't carry it), so these always fall back to the
# league-wide historical baseline rather than being copied from either
# team's most recent row — that's the "clearly defined league-level
# fallback" the referee-feature spec requires, applied unconditionally
# at prediction time since the alternative (a specific referee) is simply
# unavailable pre-match.
_LEAGUE_BASELINE_ONLY_COLS = ('RefFouls_hist', 'RefCards_hist')



def _find_cross_league_role_row(df, team, role, exclude_league, before_date):
    """
    Promotion/relegation bridge helper: look for `team`'s most recent
    role-specific row ('HomeTeam' or 'AwayTeam') in ANY league OTHER than
    `exclude_league`, strictly before `before_date`. Used only when a team
    has zero role-specific history in its *current* (target) league — the
    classic case being a side freshly promoted into, or relegated out of,
    the league it's about to play in.

    This never reassigns which league a team is predicted to compete in —
    it only borrows the team's own historical performance figures from
    wherever they actually played, subject to the same no-future-leakage
    date cutoff used everywhere else in this function. If nothing is found
    (a genuinely new team, or no data before this date), it returns None
    and the caller falls through to the existing league-baseline behavior.
    """
    if pd.isna(before_date):
        return None
    role_col = 'HomeTeam' if role == 'home' else 'AwayTeam'
    other = df[
        (df['League'] != exclude_league)
        & (df[role_col] == team)
        & (df['Date'] < before_date)
    ]
    if len(other) == 0:
        return None
    return other.sort_values('Date').iloc[-1]


def get_fixture_features(df, feature_cols, league, home, away, fixture_date=None):
    """
    Build a feature vector for a SPECIFIC upcoming fixture (home vs away),
    rather than grabbing the single most recent row involving either team.

    CRITICAL FIX (replaces the old get_latest_features):
    The old approach pulled the last row where either team appeared in
    ANY role (home or away, against ANY opponent) and used that row's
    features wholesale. Because feature columns are role-specific
    (e.g. 'HGS_L5' = the row's home team's scoring form), that row's
    values often did not correspond to `home` playing at home and `away`
    playing away at all — e.g. it could be a row for "Away vs Someone
    Else" where "Away" was actually the home team of that historical row.

    The fix: pull `home`'s own most recent HOME-role features, and
    `away`'s own most recent AWAY-role features, independently, then
    recompute the matchup-differential columns and set the league
    dummies for the target league. This mirrors exactly what train.py
    computes per-row, just assembled for a hypothetical fixture rather
    than read off an existing one.

    PROMOTION/RELEGATION BRIDGE:
    A team with zero role-specific rows in the TARGET league (e.g. just
    promoted from the Championship into the Premier League) previously
    fell straight through to the league-wide baseline for every one of
    its feature columns, discarding real, informative history the team
    does have — just filed under a different league. Before falling back
    to the baseline, we now look for the team's most recent role-specific
    row in any OTHER league (strictly before the fixture date — no future
    leakage), and if one exists, blend it 50/50 with the target league's
    own baseline for each column. The blend is a deliberately simple,
    conservative league-strength adjustment: a team's second-tier scoring
    rate is informative but shouldn't be assumed to transfer 1:1 into a
    tougher division, so it's damped toward the new league's typical
    level rather than trusted outright. Exact same-league history (when
    it exists) is always used in preference to this bridge; the bridge
    only ever applies when same-league role history is completely empty.
    """

    league_df = df[df['League'] == league].sort_values('Date')

    # For an upcoming fixture, never use a match dated on/after the fixture
    # when selecting historical rows or calculating rest. This makes the
    # prediction-time feature vector strictly pre-match.
    parsed_fixture_date = pd.to_datetime(fixture_date, dayfirst=True, errors='coerce') if fixture_date is not None else pd.NaT
    if pd.notna(parsed_fixture_date):
        league_df = league_df[league_df['Date'] < parsed_fixture_date].copy()

    if len(league_df) == 0:
        raise ValueError(f"No historical data at all for league: {league}")

    home_rows = league_df[league_df['HomeTeam'] == home]
    away_rows = league_df[league_df['AwayTeam'] == away]

    missing_data_flag = False
    used_promotion_bridge = False
    home_row = None
    away_row = None
    home_bridge_row = None
    away_bridge_row = None

    if len(home_rows) > 0:
        home_valid = home_rows.dropna(subset=[c for c in feature_cols if c.startswith(_HOME_PREFIXES) or c == 'ELO_home'])
        home_row = home_valid.iloc[-1] if len(home_valid) > 0 else home_rows.iloc[-1]
    else:
        # No home-role history for this team in this league at all. Try the
        # promotion/relegation bridge before giving up and flagging it as
        # missing data. We deliberately do NOT fall back to an away-role
        # row for this team here — doing so would put away-context stats
        # (how this team performs on the road) into home-context feature
        # slots, which is exactly the row-mismatch bug this function
        # exists to prevent.
        home_bridge_row = _find_cross_league_role_row(
            df, home, 'home', league, parsed_fixture_date
        )
        if home_bridge_row is not None:
            used_promotion_bridge = True
        else:
            missing_data_flag = True

    if len(away_rows) > 0:
        away_valid = away_rows.dropna(subset=[c for c in feature_cols if c.startswith(_AWAY_PREFIXES) or c == 'ELO_away'])
        away_row = away_valid.iloc[-1] if len(away_valid) > 0 else away_rows.iloc[-1]
    else:
        away_bridge_row = _find_cross_league_role_row(
            df, away, 'away', league, parsed_fixture_date
        )
        if away_bridge_row is not None:
            used_promotion_bridge = True
        else:
            missing_data_flag = True

    def _league_baseline(col):
        """League-wide average for `col`, used only when a team has no
        role-specific history at all. Returns NaN (not 0) if the league
        itself has no data for this column, so downstream NaN handling
        applies uniformly."""
        if col not in league_df.columns:
            return np.nan
        val = league_df[col].mean()
        return val if val == val else np.nan

    def _bridged_value(bridge_row, col):
        """50/50 blend of a cross-league borrowed value with this
        league's own baseline for that column — see PROMOTION/RELEGATION
        BRIDGE note above. Falls back to a pure league baseline if the
        borrowed row doesn't have this column or the league baseline
        itself is unavailable."""
        baseline = _league_baseline(col)
        borrowed = bridge_row.get(col, np.nan) if bridge_row is not None else np.nan
        if borrowed != borrowed and baseline != baseline:
            return np.nan
        if borrowed != borrowed:
            return baseline
        if baseline != baseline:
            return borrowed
        return 0.5 * borrowed + 0.5 * baseline

    values = {}

    for col in feature_cols:
        if col == 'ELO_home':
            if home_row is not None:
                values[col] = home_row.get('ELO_home', np.nan)
            elif home_bridge_row is not None:
                values[col] = _bridged_value(home_bridge_row, 'ELO_home')
            else:
                values[col] = _league_baseline('ELO_home')
        elif col == 'ELO_away':
            if away_row is not None:
                values[col] = away_row.get('ELO_away', np.nan)
            elif away_bridge_row is not None:
                values[col] = _bridged_value(away_bridge_row, 'ELO_away')
            else:
                values[col] = _league_baseline('ELO_away')
        elif col == 'ELO_diff':
            continue  # recomputed below
        elif col in _DIFF_COLS:
            continue  # recomputed below
        elif col.startswith('Lg_'):
            values[col] = 1.0 if col == f'Lg_{league}' else 0.0
        elif col in _LEAGUE_BASELINE_ONLY_COLS:
            values[col] = _league_baseline(col)
        elif col.startswith(_HOME_PREFIXES):
            if home_row is not None:
                values[col] = home_row.get(col, np.nan)
            elif home_bridge_row is not None:
                values[col] = _bridged_value(home_bridge_row, col)
            else:
                values[col] = _league_baseline(col)
        elif col.startswith(_AWAY_PREFIXES):
            if away_row is not None:
                values[col] = away_row.get(col, np.nan)
            elif away_bridge_row is not None:
                values[col] = _bridged_value(away_bridge_row, col)
            else:
                values[col] = _league_baseline(col)
        else:
            # Unrecognized column pattern: best-effort, prefer home row,
            # then away row, then league baseline.
            if home_row is not None and col in home_row and home_row.get(col, np.nan) == home_row.get(col, np.nan):
                values[col] = home_row.get(col)
            elif away_row is not None and col in away_row and away_row.get(col, np.nan) == away_row.get(col, np.nan):
                values[col] = away_row.get(col)
            else:
                values[col] = _league_baseline(col)

    # Rest features must be calculated from the most recent COMPLETED match
    # for each team before the target fixture, regardless of home/away role.
    if pd.notna(parsed_fixture_date):
        home_hist = league_df[(league_df['HomeTeam'] == home) | (league_df['AwayTeam'] == home)]
        away_hist = league_df[(league_df['HomeTeam'] == away) | (league_df['AwayTeam'] == away)]
        home_last = home_hist['Date'].max() if len(home_hist) else pd.NaT
        away_last = away_hist['Date'].max() if len(away_hist) else pd.NaT
        home_rest = (parsed_fixture_date - home_last).days if pd.notna(home_last) else np.nan
        away_rest = (parsed_fixture_date - away_last).days if pd.notna(away_last) else np.nan
        if 'HomeRest' in feature_cols:
            values['HomeRest'] = float(max(home_rest, 0)) if pd.notna(home_rest) else _league_baseline('HomeRest')
        if 'AwayRest' in feature_cols:
            values['AwayRest'] = float(max(away_rest, 0)) if pd.notna(away_rest) else _league_baseline('AwayRest')
        if 'RestDiff' in feature_cols:
            hr = values.get('HomeRest', np.nan)
            ar = values.get('AwayRest', np.nan)
            values['RestDiff'] = (hr if pd.notna(hr) else 7.0) - (ar if pd.notna(ar) else 7.0)

    # Recompute differential columns from the resolved values so they
    # actually compare `home` against `away`, not two mismatched rows.
    for diff_col, (a, b) in _DIFF_COLS.items():
        if diff_col in feature_cols:
            va = values.get(a, np.nan)
            vb = values.get(b, np.nan)
            values[diff_col] = (va if va == va else 0) - (vb if vb == vb else 0)

    if 'ELO_diff' in feature_cols:
        eh = values.get('ELO_home', np.nan)
        ea = values.get('ELO_away', np.nan)
        values['ELO_diff'] = (eh if eh == eh else 1500) - (ea if ea == ea else 1500)

    features = np.array([
        values.get(c, 0.0) if values.get(c, np.nan) == values.get(c, np.nan) else 0.0
        for c in feature_cols
    ], dtype=float)

    assert len(features) == len(feature_cols), \
        f"Feature count mismatch! Expected {len(feature_cols)}, got {len(features)}"

    return features, missing_data_flag, used_promotion_bridge




def predict_with_fallback(
    fixture,
    final_model,
    dc_models,
    feature_cols,
    df,
    team_mapping,
    all_teams,
    use_fallback=True,
    fixture_date=None
):
    """
    Predict with fallback for missing teams.

    CRITICAL FIX:
    - If team not found in DC → use league average parameters
    - No crash on unknown teams
    """

    league = fixture['league']
    home = fixture['home']
    away = fixture['away']

    # Normalize teams
    home_norm, home_conf, home_info = normalize_team_name(home, team_mapping, all_teams, league, df)
    away_norm, away_conf, away_info = normalize_team_name(away, team_mapping, all_teams, league, df)

    # Collect warnings/errors
    warnings = []

    if home_info and 'warning' in home_info:
        warnings.append(home_info['warning'])
    if away_info and 'warning' in away_info:
        warnings.append(away_info['warning'])

    if not home_norm or not away_norm:
        error_msg = ""
        if not home_norm and home_info:
            error_msg += home_info.get('error', f"Team not found: {home}")
        if not away_norm and away_info:
            error_msg += " | " + away_info.get('error', f"Team not found: {away}")

        raise ValueError(error_msg)

    # Check league
    if league not in dc_models:
        raise ValueError(f"League not supported: {league}")

    # Get fixture-specific features (FIXED: no longer a mismatched last row)
    features, low_data, used_bridge = get_fixture_features(
        df, feature_cols, league, home_norm, away_norm, fixture_date=fixture_date
    )
    if low_data:
        warnings.append(
            f"Limited home/away-specific history for {home_norm} or {away_norm}; "
            f"features may be less reliable."
        )
    elif used_bridge:
        # Distinct from `low_data`: the team has zero history in THIS
        # league specifically, but real history from a previous league
        # (promotion/relegation) was found and blended in — worth telling
        # the user, but this is meaningfully more reliable than a bare
        # league-average fallback, so it gets its own message rather than
        # being lumped in with "limited history."
        warnings.append(
            f"{home_norm} or {away_norm} has no {league} history yet this "
            f"season; using a blended estimate from their previous league."
        )

    # Try prediction with fallback
    try:
        # Standard prediction
        probs, dc_pred = ensemble_prediction(
            final_model, dc_models, league, home_norm, away_norm, features
        )
        used_fallback = False

    except ValueError as e:
        # Team not in Dixon-Coles model, OR the league's DC model exists
        # but did not converge (ensemble_prediction() raises ValueError in
        # both cases). Distinguish them in the warning so the fallback
        # reason shown to the user is accurate, not a generic guess.
        if not use_fallback:
            raise

        if "did not converge" in str(e):
            warnings.append(
                f"Dixon-Coles model for {league} did not converge; "
                f"using league average instead of an unreliable fit."
            )
        else:
            warnings.append(f"Using league average for unknown team")

        league_data = df[df['League'] == league].tail(100)

        avg_home_prob = (league_data['FTR'] == 'H').mean()
        avg_draw_prob = (league_data['FTR'] == 'D').mean()
        avg_away_prob = (league_data['FTR'] == 'A').mean()

        probs = np.array([avg_home_prob, avg_draw_prob, avg_away_prob])

        # Estimate over/under 2.5 from the league's actual recent scoring
        # rate instead of assuming a flat 50/50 split.
        total_goals = league_data['FTHG'] + league_data['FTAG']
        prob_over25 = float((total_goals > 2.5).mean()) if len(league_data) > 0 else 0.5
        prob_over25 = prob_over25 if prob_over25 == prob_over25 else 0.5  # guard NaN

        dc_pred = {
            'lambda_home': league_data['FTHG'].mean(),
            'lambda_away': league_data['FTAG'].mean(),
            'exp_goals': league_data['FTHG'].mean() + league_data['FTAG'].mean(),
            'prob_over_25': prob_over25,
            'prob_under_25': 1 - prob_over25,
            'score_matrix': None
        }

        used_fallback = True

    # Build market probabilities
    market_probs = build_market_probabilities(probs, dc_pred)

    # Calculate confidence
    confidence = calculate_confidence_score(probs)

    # Prepare result
    result = {
        'league': league,
        'home': home_norm,
        'away': away_norm,
        'prob_home': float(probs[0]),
        'prob_draw': float(probs[1]),
        'prob_away': float(probs[2]),
        'lambda_home': dc_pred['lambda_home'],
        'lambda_away': dc_pred['lambda_away'],
        'exp_goals': dc_pred['exp_goals'],
        'confidence': confidence,
        'market_probs': market_probs,
        'value_bets': [],
        'warnings': warnings,
        'used_fallback': used_fallback
    }

    # Calculate value if odds provided
    if 'odds' in fixture:
        all_values = calculate_value(market_probs, fixture['odds'])
        result['all_bets'] = all_values

    return result




def predict_multiple_fixtures(
    fixtures,
    final_model,
    dc_models,
    feature_cols,
    df,
    team_mapping,
    all_teams,
    min_prob=0.45,
    min_ev=0.03
):
    """
    Predict multiple fixtures with proper error handling.

    Returns:
        results, errors, warnings_collected
        (NOTE: this order is errors-then-warnings; callers must unpack in
        this order. See app.py FIX.)
    """

    results = []
    errors = []
    warnings_collected = []

    for i, fixture in enumerate(fixtures, 1):
        try:
            result = predict_with_fallback(
                fixture,
                final_model,
                dc_models,
                feature_cols,
                df,
                team_mapping,
                all_teams,
                use_fallback=True,
                fixture_date=fixture.get('date')
            )

            # Collect warnings
            if result['warnings']:
                warnings_collected.extend(result['warnings'])

            # Find value bets
            if 'all_bets' in result:
                value_bets = find_value_bets(
                    result['all_bets'],
                    min_prob=min_prob,
                    min_ev=min_ev
                )
                result['value_bets'] = value_bets

                # Add Kelly stakes
                for bet in value_bets:
                    bet['kelly_stake'] = calculate_kelly_stake(bet['prob'], bet['odds'])

            results.append(result)

        except Exception as e:
            errors.append({
                'fixture': f"{fixture['home']} vs {fixture['away']}",
                'error': str(e)
            })

    return results, errors, warnings_collected




def generate_summary_stats(results):
    """Generate summary statistics."""

    total_matches = len(results)

    matches_with_value = sum(1 for r in results if len(r['value_bets']) > 0)

    total_value_bets = sum(len(r['value_bets']) for r in results)

    if total_value_bets > 0:
        avg_ev = np.mean([bet['ev'] for r in results for bet in r['value_bets']])
        avg_prob = np.mean([bet['prob'] for r in results for bet in r['value_bets']])
        avg_odds = np.mean([bet['odds'] for r in results for bet in r['value_bets']])
    else:
        avg_ev = 0
        avg_prob = 0
        avg_odds = 0

    confidences = [r['confidence'] for r in results]
    avg_confidence = np.mean(confidences)

    exp_goals = [r['exp_goals'] for r in results]
    avg_exp_goals = np.mean(exp_goals)

    # Count fallback usage
    fallback_count = sum(1 for r in results if r.get('used_fallback', False))

    summary = {
        'total_matches': total_matches,
        'matches_with_value': matches_with_value,
        'total_value_bets': total_value_bets,
        'avg_ev': avg_ev,
        'avg_prob': avg_prob,
        'avg_odds': avg_odds,
        'avg_confidence': avg_confidence,
        'avg_exp_goals': avg_exp_goals,
        'hit_rate': matches_with_value / total_matches if total_matches > 0 else 0,
        'fallback_used': fallback_count
    }

    return summary


