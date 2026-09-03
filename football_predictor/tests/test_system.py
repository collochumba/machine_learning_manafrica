"""
CONSOLIDATED TEST SUITE

Tests for the current 15-league football predictor architecture.

Current architecture:
  - fixtures.py
      * team-name resolution
      * fixture loading
      * fixture cache refresh
      * artifact health
      * odds extraction
  - betting.py
      * betting economics
  - models.py
      * model utilities
  - config.py
      * authoritative 15-league configuration

IMPORTANT:
MLS is intentionally excluded from this project.

The supported leagues are:

  1. Premier League
  2. Championship
  3. La Liga
  4. Segunda Division
  5. Serie A
  6. Serie B
  7. Bundesliga
  8. 2. Bundesliga
  9. Ligue 1
 10. Ligue 2
 11. Eredivisie
 12. Belgian Pro League
 13. Primeira Liga
 14. Super Lig
 15. Greek Super League
"""

import pandas as pd
import numpy as np
import joblib
import pytest

from fixtures import (
    resolve_fixture,
    artifact_health,
    refresh_fixture_cache,
)

from config import (
    LEAGUE_CONFIG,
    SUPPORTED_LEAGUES,
    DIV_TO_LEAGUE,
)

from betting import calculate_value
from models import _asian_handicap_probs


# ============================================================================
# Team-name resolution
# ============================================================================

CURRENT = {
    "Test League": ["Alpha FC", "Beta FC"],
    "Other League": ["Gamma FC"],
}

ALL = [
    "Alpha FC",
    "Beta FC",
    "Gamma FC",
]


def test_backward_compatible_without_candidates():
    """
    resolve_fixture() must continue to work without fixture_candidates.
    """

    r = resolve_fixture(
        "Test League",
        "Alpha FC",
        "Beta FC",
        CURRENT,
        ALL,
    )

    assert r["status"] == "valid"


def test_fixture_candidates_resolve_live_name():
    """
    A valid fixture candidate should be accepted when it belongs to
    the requested league.
    """

    r = resolve_fixture(
        "Test League",
        "New Club",
        "Beta FC",
        CURRENT,
        ALL,
        fixture_candidates={
            "Test League": [
                "New Club",
                "Beta FC",
            ]
        },
    )

    assert r["status"] == "valid"
    assert r["home"]["method"] == "fixture_candidate"


def test_wrong_league_candidate_is_not_used():
    """
    A team known to belong to another league must never be substituted
    into the requested league.
    """

    r = resolve_fixture(
        "Test League",
        "Gamma FC",
        "Beta FC",
        CURRENT,
        ALL,
        fixture_candidates={
            "Test League": [
                "Gamma FC",
                "Beta FC",
            ]
        },
    )

    assert r["home"]["resolved"] is None


def test_no_cross_league_team_substitution():
    """
    A team from another league must not be resolved merely because
    its name exists in the global team list.
    """

    r = resolve_fixture(
        "Test League",
        "Gamma FC",
        "Beta FC",
        CURRENT,
        ALL,
    )

    assert r["home"]["resolved"] is None


def test_no_inter_miami_to_inter():
    """
    The resolver must not shorten or incorrectly substitute a team name.

    This test is intentionally generic and does not require MLS.
    """

    current = {
        "Test League": [
            "Inter Miami",
            "Inter",
        ]
    }

    all_teams = [
        "Inter",
        "Inter Miami",
    ]

    r = resolve_fixture(
        "Test League",
        "Inter Miami",
        "Inter",
        current,
        all_teams,
    )

    assert r["home"]["resolved"] == "Inter Miami"
    assert r["away"]["resolved"] == "Inter"


def test_unknown_team_is_not_substituted():
    """
    An unknown team must remain unresolved rather than being replaced
    with a similar team.
    """

    r = resolve_fixture(
        "Ligue 1",
        "Paris FC",
        "Paris SG",
        {
            "Ligue 1": [
                "Paris SG",
            ]
        },
        [
            "Paris SG",
            "Paris FC",
        ],
    )

    assert r["home"]["resolved"] is None


def test_alias_resolves_to_canonical_current_team():
    """
    A known alias (e.g. a team's full/alternate name) must resolve to the
    canonical current-season name, using the module's real ALIASES table.
    """

    from fixtures import ALIASES

    league = "Premier League"
    canonical = ALIASES[league]["manchester city"]  # "Man City"

    current = {league: [canonical]}

    r = resolve_fixture(
        league,
        "Manchester City",
        canonical,
        current,
        [canonical],
    )

    assert r["home"]["resolved"] == canonical
    assert r["home"]["method"] == "alias"
    assert r["status"] == "valid"


def test_ambiguous_team_returns_suggestions_without_substitution():
    """
    A name that plausibly matches more than one current team must remain
    unresolved and surface multiple suggestions — never auto-applied.
    """

    current = {
        "Test League": [
            "Rangers FC",
            "Rangers County",
        ],
    }
    all_teams = [
        "Rangers FC",
        "Rangers County",
    ]

    r = resolve_fixture(
        "Test League",
        "Rangers",
        "Rangers FC",
        current,
        all_teams,
    )

    assert r["home"]["resolved"] is None
    assert r["home"]["category"] == "unknown"
    assert len(r["home"]["suggestions"]) >= 2


# ============================================================================
# Betting economics
# ============================================================================

def test_dnb_push_aware_ev():
    """
    Draw-No-Bet EV must correctly account for the push probability.
    """

    probs = {
        "DNB Home": 0.6,
        "Draw": 0.2,
    }

    vals = calculate_value(
        probs,
        {
            "DNB Home": 2.0,
        },
    )

    assert vals[0]["push_prob"] == 0.2
    assert abs(vals[0]["ev"] - 0.16) < 1e-9


def test_ah_push_key():
    """
    Asian-handicap probability output should contain an AH Push result
    when applicable.
    """

    m = _asian_handicap_probs(
        np.eye(3),
        2,
    )

    assert any(
        key.startswith("AH Push")
        for key in m
    )


# ============================================================================
# Artifact health
# ============================================================================

def test_artifact_health_has_expected_sections(tmp_path):
    """
    artifact_health() should expose the expected health sections.
    """

    health = artifact_health(tmp_path)

    assert "files" in health
    assert "leagues" in health


# ============================================================================
# Fixture cache refresh
# ============================================================================

def test_cached_fixture_fallback_on_network_failure(
    tmp_path,
    monkeypatch,
):
    """
    If a fixture refresh fails, the last known-good cache must remain
    available and must not be destroyed.
    """

    cache = tmp_path / "fixtures.pkl"

    joblib.dump(
        {
            "raw_df": pd.DataFrame(
                {
                    "Div": ["E0"],
                    "HomeTeam": ["A"],
                    "AwayTeam": ["B"],
                    "Date": ["25/08/26"],
                }
            ),
            "source": "test",
            "fetched_at": (
                pd.Timestamp.now()
                .to_pydatetime()
            ),
        },
        cache,
    )

    monkeypatch.setattr(
        "fixtures.fetch_fixtures_bytes",
        lambda *a, **k: (
            None,
            "network failure",
        ),
    )

    cached, status = refresh_fixture_cache(
        cache,
        max_age_hours=0,
        force=True,
    )

    assert cached is not None
    assert status["using_last_good"] is True


# ============================================================================
# League configuration
# ============================================================================

def test_exactly_15_leagues_configured():
    """
    The current project intentionally supports exactly 15 leagues.

    MLS is excluded.
    """

    assert len(LEAGUE_CONFIG) == 15


def test_required_league_codes_are_present():
    """
    Verify the authoritative 15-league configuration.
    """

    expected = {
        "Premier League": "E0",
        "Championship": "E1",

        "La Liga": "SP1",
        "Segunda Division": "SP2",

        "Serie A": "I1",
        "Serie B": "I2",

        "Bundesliga": "D1",
        "2. Bundesliga": "D2",

        "Ligue 1": "F1",
        "Ligue 2": "F2",

        "Eredivisie": "N1",

        "Belgian Pro League": "B1",

        "Primeira Liga": "P1",

        "Super Lig": "T1",

        "Greek Super League": "G1",
    }

    actual = {
        k: v["code"]
        for k, v in LEAGUE_CONFIG.items()
    }

    assert actual == expected


def test_mls_is_excluded():
    """
    MLS must remain completely excluded from the authoritative
    configuration.

    This prevents a future edit from accidentally bringing MLS back.
    """

    assert "MLS" not in LEAGUE_CONFIG
    assert "MLS" not in SUPPORTED_LEAGUES
    assert "USA" not in DIV_TO_LEAGUE


def test_all_configured_leagues_are_main_source():
    """
    All 15 currently supported leagues use the main Football-Data
    source. There should be no extra-league source in the configuration.
    """

    assert len(LEAGUE_CONFIG) == 15

    for league, metadata in LEAGUE_CONFIG.items():
        assert metadata["source"] == "main"


def test_supported_leagues_match_configuration():
    """
    SUPPORTED_LEAGUES must contain exactly the same leagues as
    LEAGUE_CONFIG.
    """

    assert set(SUPPORTED_LEAGUES) == set(
        LEAGUE_CONFIG.keys()
    )


def test_division_mapping_matches_configuration():
    """
    Every configured main-league division code must resolve back to
    the correct league.
    """

    for league, metadata in LEAGUE_CONFIG.items():
        code = metadata["code"]

        assert DIV_TO_LEAGUE[code] == league


# ============================================================================
# Odds extraction
# ============================================================================

def test_optional_fixture_odds_never_invented():
    """
    1X2 odds should only be returned when those columns actually exist.
    """

    from fixtures import _extract_odds

    row = pd.Series(
        {
            "B365H": 2.0,
            "B365D": 3.4,
            "B365A": 3.8,
        }
    )

    odds = _extract_odds(row)

    assert odds == {
        "Home": 2.0,
        "Draw": 3.4,
        "Away": 3.8,
    }


def test_optional_over_under_and_asian_handicap_only_when_present():
    """
    Over/Under and Asian Handicap markets must only be extracted when
    the source row actually contains those markets.
    """

    from fixtures import _extract_odds

    row = pd.Series(
        {
            "B365H": 2.0,
            "B365D": 3.4,
            "B365A": 3.8,

            "B365O25": 1.9,
            "B365U25": 1.9,

            "B365AHH": 1.8,
            "B365AHA": 2.0,
        }
    )

    odds = _extract_odds(row)

    assert "OverUnder" in odds

    assert odds["OverUnder"]["Over2.5"] == 1.9
    assert odds["OverUnder"]["Under2.5"] == 1.9

    assert "AsianHandicap" in odds

    assert odds["AsianHandicap"]["Home"] == 1.8
    assert odds["AsianHandicap"]["Away"] == 2.0


def test_missing_optional_markets_are_not_invented():
    """
    The system must not fabricate Over/Under or Asian Handicap odds
    when those fields are absent from the source data.
    """

    from fixtures import _extract_odds

    row = pd.Series(
        {
            "B365H": 2.0,
            "B365D": 3.4,
            "B365A": 3.8,
        }
    )

    odds = _extract_odds(row)

    assert odds == {
        "Home": 2.0,
        "Draw": 3.4,
        "Away": 3.8,
    }

    assert "OverUnder" not in odds
    assert "AsianHandicap" not in odds


# ============================================================================
# Current-season team fallback (train.py)
# ============================================================================

def test_current_season_teams_fallback_to_last_usable_season():
    """
    If the newest requested season has no usable rows for a league (e.g.
    Bundesliga before its season has started), get_current_season_teams()
    must fall back to the newest season that actually has matches, rather
    than reporting an empty team list.
    """

    from train import get_current_season_teams

    df = pd.DataFrame(
        {
            "League": ["Bundesliga"] * 4,
            "Season": ["2526", "2526", "2526", "2526"],
            "HomeTeam": ["Bayern", "Dortmund", "Leipzig", "Leverkusen"],
            "AwayTeam": ["Dortmund", "Leipzig", "Leverkusen", "Bayern"],
        }
    )

    # "2627" is a source-status entry (the download succeeded / was
    # attempted) but produced zero usable rows for this league — this
    # mirrors the real football-data.co.uk behavior for a season that
    # hasn't started yet for a given league.
    load_report = {
        "loaded": {"Bundesliga": ["2526", "2627"]},
    }

    current_teams_by_league, latest_season_by_league = get_current_season_teams(
        df, load_report
    )

    assert latest_season_by_league["Bundesliga"] == "2526"
    assert current_teams_by_league["Bundesliga"] != []
    assert set(current_teams_by_league["Bundesliga"]) == {
        "Bayern",
        "Dortmund",
        "Leipzig",
        "Leverkusen",
    }


# ============================================================================
# Dixon-Coles convergence handling
# ============================================================================

def test_non_converged_dc_model_raises_for_safe_fallback():
    """
    A Dixon-Coles model that did not converge must never be silently used
    for a production prediction — ensemble_prediction() must raise so the
    caller (predict_with_fallback) can use its safe league-average
    fallback instead.
    """

    from models import ensemble_prediction

    class FakeNonConvergedDC:
        converged_ = False

    with pytest.raises(ValueError, match="did not converge"):
        ensemble_prediction(
            final_model=None,
            dc_models={"Test League": FakeNonConvergedDC()},
            league="Test League",
            home="Alpha FC",
            away="Beta FC",
            features=np.zeros(1),
        )


# ============================================================================
# Corner / card model bundle league-count handling
# ============================================================================

def test_corner_bundle_league_count_uses_models_key(tmp_path):
    """
    Corner model league coverage must be derived from bundle['models'],
    never from len(bundle) (which counts bundle metadata fields, not
    leagues).
    """

    bundle = {
        "models": {
            "League A": object(),
            "League B": object(),
            "League C": object(),
            "League D": object(),
            "League E": object(),
        },
        "schema_version": 3,
        "validation": {},
    }
    joblib.dump(bundle, tmp_path / "corner_model.pkl")

    health = artifact_health(tmp_path)

    assert health["leagues"]["corners"] == sorted(bundle["models"].keys())
    assert len(health["leagues"]["corners"]) == 5
    # len(bundle) (3 top-level keys) would be the wrong, buggy count.
    assert len(health["leagues"]["corners"]) != len(bundle)


def test_card_bundle_league_count_uses_models_key(tmp_path):
    """
    Card model league coverage must be derived from bundle['models'],
    never from len(bundle).
    """

    bundle = {
        "models": {
            "League A": object(),
            "League B": object(),
            "League C": object(),
            "League D": object(),
        },
        "schema_version": 2,
        "validation": {},
    }
    joblib.dump(bundle, tmp_path / "card_model.pkl")

    health = artifact_health(tmp_path)

    assert health["leagues"]["cards"] == sorted(bundle["models"].keys())
    assert len(health["leagues"]["cards"]) == 4
    assert len(health["leagues"]["cards"]) != len(bundle)


# ============================================================================
# Prediction failure isolation
# ============================================================================

def test_prediction_failure_isolation_continues_after_one_bad_fixture(monkeypatch):
    """
    One fixture failing during prediction must not prevent the remaining
    fixtures from being predicted and returned.
    """

    import models as models_module

    def fake_predict_with_fallback(fixture, *args, **kwargs):
        if fixture["home"] == "Bad Team":
            raise RuntimeError("simulated prediction failure")
        return {
            "league": fixture["league"],
            "home": fixture["home"],
            "away": fixture["away"],
            "prob_home": 0.4,
            "prob_draw": 0.3,
            "prob_away": 0.3,
            "lambda_home": 1.2,
            "lambda_away": 1.0,
            "exp_goals": 2.2,
            "confidence": 0.5,
            "market_probs": {},
            "value_bets": [],
            "warnings": [],
            "used_fallback": False,
        }

    monkeypatch.setattr(
        models_module, "predict_with_fallback", fake_predict_with_fallback
    )

    fixtures_list = [
        {"league": "Test League", "home": "Good Team", "away": "Other Team"},
        {"league": "Test League", "home": "Bad Team", "away": "Other Team"},
    ]

    results, errors, warnings_collected = models_module.predict_multiple_fixtures(
        fixtures_list,
        final_model=None,
        dc_models={},
        feature_cols=[],
        df=None,
        team_mapping={},
        all_teams=[],
    )

    assert len(results) == 1
    assert results[0]["home"] == "Good Team"
    assert len(errors) == 1
    assert "Bad Team" in errors[0]["fixture"]


# ============================================================================
# Corner model distribution fallback (train.py)
# ============================================================================

def test_corner_model_fallback_tries_other_distribution(monkeypatch):
    """
    Regression test: when the walk-forward-recommended distribution fails
    to converge, the fallback must try the OTHER distribution family
    (poisson<->negbinom), not silently retry the same distribution again
    — retrying an identical, deterministic optimization against the same
    data just reproduces the same failure and never actually fixes
    anything.
    """

    import train as train_module

    fit_calls = []

    class FakeCornerModel:
        def __init__(self, distribution="negbinom", xi=0.002):
            self.distribution = distribution
            self.home_adv = 0.1
            self.alpha = 0.05
            self.teams = ["A", "B"]

        def fit(self, df, league):
            fit_calls.append(self.distribution)
            # 'poisson' never converges in this test; 'negbinom' always does,
            # so a correct fallback must recover by switching families.
            self.converged_ = self.distribution == "negbinom"
            return self

    fake_df_c = pd.DataFrame(
        {
            "League": ["Test League"] * 2,
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )

    monkeypatch.setattr(
        train_module, "build_corner_features", lambda df: (fake_df_c, ["f1"])
    )
    monkeypatch.setattr(
        train_module,
        "walk_forward_validate",
        lambda df_c, cols, league, **kw: {"_recommended": "poisson_strength"},
    )
    monkeypatch.setattr(train_module, "CornerStrengthModel", FakeCornerModel)

    corner_models, corner_feature_cols, corner_validation = (
        train_module.train_corner_models(pd.DataFrame())
    )

    assert fit_calls == ["poisson", "negbinom"]
    assert "Test League" in corner_models
    assert corner_models["Test League"].distribution == "negbinom"


# ============================================================================
# Promotion/relegation history bridge (models.py get_fixture_features)
# ============================================================================

_BRIDGE_FEATURE_COLS = [
    "HGS_L5", "AGS_L5", "ELO_home", "ELO_away", "ELO_diff", "GS_diff",
    "Lg_Premier League", "Lg_Championship",
]


def test_promoted_team_uses_previous_league_history():
    """
    A team with zero home-role history in the target league, but real
    history in another league before the fixture date, must be blended in
    (missing_data_flag False, used_promotion_bridge True) rather than
    discarded in favor of a bare league average.
    """
    from models import get_fixture_features

    df = pd.DataFrame([
        {"League": "Championship", "Date": pd.Timestamp("2026-01-01"),
         "HomeTeam": "Promoted FC", "AwayTeam": "Other CH",
         "HGS_L5": 2.0, "AGS_L5": 1.0, "ELO_home": 1500, "ELO_away": 1400},
        {"League": "Premier League", "Date": pd.Timestamp("2026-06-01"),
         "HomeTeam": "Big Club", "AwayTeam": "Small Club",
         "HGS_L5": 1.8, "AGS_L5": 1.2, "ELO_home": 1600, "ELO_away": 1550},
        {"League": "Premier League", "Date": pd.Timestamp("2026-06-08"),
         "HomeTeam": "Small Club", "AwayTeam": "Big Club",
         "HGS_L5": 1.2, "AGS_L5": 1.9, "ELO_home": 1550, "ELO_away": 1610},
    ])

    features, missing, bridge = get_fixture_features(
        df, _BRIDGE_FEATURE_COLS, "Premier League", "Promoted FC", "Big Club",
        fixture_date="15/06/2026",
    )

    assert missing is False
    assert bridge is True
    # HGS_L5 must be a 50/50 blend of the borrowed Championship value (2.0)
    # and the Premier League baseline (mean of 1.8 and 1.2 = 1.5) = 1.75,
    # not the raw borrowed value and not a pure league average.
    hgs = dict(zip(_BRIDGE_FEATURE_COLS, features))["HGS_L5"]
    assert abs(hgs - 1.75) < 1e-9


def test_relegated_team_uses_previous_league_history():
    """
    Same bridge, exercised on the AWAY role (e.g. a side relegated out of
    a higher division playing away in its new league).
    """
    from models import get_fixture_features

    df = pd.DataFrame([
        {"League": "Premier League", "Date": pd.Timestamp("2026-01-01"),
         "HomeTeam": "Other PL", "AwayTeam": "Relegated FC",
         "HGS_L5": 1.0, "AGS_L5": 2.4, "ELO_home": 1400, "ELO_away": 1600},
        {"League": "Championship", "Date": pd.Timestamp("2026-06-01"),
         "HomeTeam": "Home Side", "AwayTeam": "Other CH",
         "HGS_L5": 1.4, "AGS_L5": 1.0, "ELO_home": 1450, "ELO_away": 1400},
        {"League": "Championship", "Date": pd.Timestamp("2026-06-08"),
         "HomeTeam": "Other CH 2", "AwayTeam": "Away Side",
         "HGS_L5": 1.3, "AGS_L5": 1.1, "ELO_home": 1440, "ELO_away": 1390},
    ])

    features, missing, bridge = get_fixture_features(
        df, _BRIDGE_FEATURE_COLS, "Championship", "Home Side", "Relegated FC",
        fixture_date="15/06/2026",
    )

    assert missing is False
    assert bridge is True


def test_genuinely_new_team_has_no_bridge():
    """
    A team with zero history anywhere — in the target league or any other
    — must fall back to the plain league-average path (missing_data_flag
    True, used_promotion_bridge False), not spuriously "find" a bridge.
    """
    from models import get_fixture_features

    df = pd.DataFrame([
        {"League": "Premier League", "Date": pd.Timestamp("2026-06-01"),
         "HomeTeam": "Big Club", "AwayTeam": "Small Club",
         "HGS_L5": 1.8, "AGS_L5": 1.2, "ELO_home": 1600, "ELO_away": 1550},
    ])

    features, missing, bridge = get_fixture_features(
        df, _BRIDGE_FEATURE_COLS, "Premier League", "Brand New FC", "Big Club",
        fixture_date="15/06/2026",
    )

    assert missing is True
    assert bridge is False


def test_promotion_bridge_does_not_leak_future_data():
    """
    A team's cross-league row that falls ON OR AFTER the fixture date must
    never be borrowed — the bridge must respect the exact same
    no-future-leakage cutoff as same-league history. With no *usable*
    pre-fixture data anywhere, this must behave exactly like a genuinely
    new team (missing_data_flag True, bridge False), not silently use the
    future row.
    """
    from models import get_fixture_features

    df = pd.DataFrame([
        # This Championship row is AFTER the fixture date being predicted —
        # must not be borrowed.
        {"League": "Championship", "Date": pd.Timestamp("2026-07-01"),
         "HomeTeam": "Future Data FC", "AwayTeam": "Other CH",
         "HGS_L5": 2.0, "AGS_L5": 1.0, "ELO_home": 1500, "ELO_away": 1400},
        {"League": "Premier League", "Date": pd.Timestamp("2026-06-01"),
         "HomeTeam": "Big Club", "AwayTeam": "Small Club",
         "HGS_L5": 1.8, "AGS_L5": 1.2, "ELO_home": 1600, "ELO_away": 1550},
    ])

    features, missing, bridge = get_fixture_features(
        df, _BRIDGE_FEATURE_COLS, "Premier League", "Future Data FC", "Big Club",
        fixture_date="15/06/2026",
    )

    assert missing is True
    assert bridge is False


def test_same_league_history_takes_priority_over_bridge():
    """
    If a team DOES have same-league role history, the bridge must never
    override it — exact same-league history always wins.
    """
    from models import get_fixture_features

    df = pd.DataFrame([
        {"League": "Premier League", "Date": pd.Timestamp("2026-05-01"),
         "HomeTeam": "Established FC", "AwayTeam": "Other PL",
         "HGS_L5": 3.0, "AGS_L5": 1.0, "ELO_home": 1700, "ELO_away": 1400},
        {"League": "Championship", "Date": pd.Timestamp("2026-01-01"),
         "HomeTeam": "Established FC", "AwayTeam": "Other CH",
         "HGS_L5": 1.0, "AGS_L5": 1.0, "ELO_home": 1400, "ELO_away": 1400},
        {"League": "Premier League", "Date": pd.Timestamp("2026-06-01"),
         "HomeTeam": "Other PL 2", "AwayTeam": "Big Club",
         "HGS_L5": 1.8, "AGS_L5": 1.2, "ELO_home": 1600, "ELO_away": 1550},
    ])

    features, missing, bridge = get_fixture_features(
        df, _BRIDGE_FEATURE_COLS, "Premier League", "Established FC", "Big Club",
        fixture_date="15/06/2026",
    )

    assert missing is False
    assert bridge is False  # same-league history was used, bridge never needed
    hgs = dict(zip(_BRIDGE_FEATURE_COLS, features))["HGS_L5"]
    assert abs(hgs - 3.0) < 1e-9  # the team's own Premier League row, unblended