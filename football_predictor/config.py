"""Authoritative league configuration for the football predictor.

The system intentionally supports exactly 15 Football-Data leagues.

MLS/USA has been permanently excluded from the project and must not be
included in configuration, training, fixture resolution, or model coverage.

Main Football-Data leagues use:

    https://www.football-data.co.uk/mmz4281/{season}/{code}.csv
"""

# ============================================================================
# AUTHORITATIVE 15-LEAGUE CONFIGURATION
# ============================================================================

LEAGUE_CONFIG = {
    # ------------------------------------------------------------------------
    # England
    # ------------------------------------------------------------------------
    "Premier League": {
        "country": "England",
        "code": "E0",
        "source": "main",
        "canonical": "Premier League",
        "display": "Premier League",
    },

    "Championship": {
        "country": "England",
        "code": "E1",
        "source": "main",
        "canonical": "Championship",
        "display": "Championship",
    },

    # ------------------------------------------------------------------------
    # Spain
    # ------------------------------------------------------------------------
    "La Liga": {
        "country": "Spain",
        "code": "SP1",
        "source": "main",
        "canonical": "La Liga",
        "display": "La Liga",
    },

    "Segunda Division": {
        "country": "Spain",
        "code": "SP2",
        "source": "main",
        "canonical": "Segunda Division",
        "display": "La Liga 2",
    },

    # ------------------------------------------------------------------------
    # Italy
    # ------------------------------------------------------------------------
    "Serie A": {
        "country": "Italy",
        "code": "I1",
        "source": "main",
        "canonical": "Serie A",
        "display": "Serie A",
    },

    "Serie B": {
        "country": "Italy",
        "code": "I2",
        "source": "main",
        "canonical": "Serie B",
        "display": "Serie B",
    },

    # ------------------------------------------------------------------------
    # Germany
    # ------------------------------------------------------------------------
    "Bundesliga": {
        "country": "Germany",
        "code": "D1",
        "source": "main",
        "canonical": "Bundesliga",
        "display": "Bundesliga",
    },

    "2. Bundesliga": {
        "country": "Germany",
        "code": "D2",
        "source": "main",
        "canonical": "2. Bundesliga",
        "display": "2. Bundesliga",
    },

    # ------------------------------------------------------------------------
    # France
    # ------------------------------------------------------------------------
    "Ligue 1": {
        "country": "France",
        "code": "F1",
        "source": "main",
        "canonical": "Ligue 1",
        "display": "Ligue 1",
    },

    "Ligue 2": {
        "country": "France",
        "code": "F2",
        "source": "main",
        "canonical": "Ligue 2",
        "display": "Ligue 2",
    },

    # ------------------------------------------------------------------------
    # Netherlands
    # ------------------------------------------------------------------------
    "Eredivisie": {
        "country": "Netherlands",
        "code": "N1",
        "source": "main",
        "canonical": "Eredivisie",
        "display": "Eredivisie",
    },

    # ------------------------------------------------------------------------
    # Belgium
    # ------------------------------------------------------------------------
    "Belgian Pro League": {
        "country": "Belgium",
        "code": "B1",
        "source": "main",
        "canonical": "Belgian Pro League",
        "display": "Belgian Pro League",
    },

    # ------------------------------------------------------------------------
    # Portugal
    # ------------------------------------------------------------------------
    "Primeira Liga": {
        "country": "Portugal",
        "code": "P1",
        "source": "main",
        "canonical": "Primeira Liga",
        "display": "Primeira Liga",
    },

    # ------------------------------------------------------------------------
    # Turkey
    # ------------------------------------------------------------------------
    "Super Lig": {
        "country": "Turkey",
        "code": "T1",
        "source": "main",
        "canonical": "Super Lig",
        "display": "Süper Lig",
    },

    # ------------------------------------------------------------------------
    # Greece
    # ------------------------------------------------------------------------
    "Greek Super League": {
        "country": "Greece",
        "code": "G1",
        "source": "main",
        "canonical": "Greek Super League",
        "display": "Greek Super League",
    },
}


# ============================================================================
# DERIVED LEAGUE MAPPINGS
# ============================================================================

# League name -> Football-Data division code
LEAGUES = {
    name: meta["code"]
    for name, meta in LEAGUE_CONFIG.items()
}


# Main Football-Data leagues only.
# Since MLS has been removed, this should contain all 15 leagues.
MAIN_LEAGUES = {
    name: meta["code"]
    for name, meta in LEAGUE_CONFIG.items()
    if meta["source"] == "main"
}


# No extra leagues are currently supported.
#
# This is deliberately generated rather than hard-coded so that the rest
# of the application can safely use EXTRA_LEAGUES without special cases.
EXTRA_LEAGUES = {
    name: meta["code"]
    for name, meta in LEAGUE_CONFIG.items()
    if meta["source"] == "extra"
}


# Tuple preserving the authoritative configuration order.
SUPPORTED_LEAGUES = tuple(LEAGUE_CONFIG.keys())


# Football-Data division code -> internal league name
DIV_TO_LEAGUE = {
    meta["code"]: name
    for name, meta in LEAGUE_CONFIG.items()
    if meta["source"] == "main"
}


# ============================================================================
# ENGLAND SAFETY CONFIGURATION
# ============================================================================

# Only these English divisions are intentionally supported.
ENGLAND_SUPPORTED_DIVISIONS = {
    "E0",
    "E1",
}


# ============================================================================
# OPERATIONAL SETTINGS
# ============================================================================

# Fixture cache is considered fresh for this many hours.
#
# The application should not download fixtures repeatedly on every
# Streamlit rerun while the cache is still fresh.
FIXTURE_REFRESH_HOURS = 6


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

# These are the exact 15 leagues that the application is expected to have.
EXPECTED_LEAGUES = {
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


def validate_league_config():
    """Validate the authoritative 15-league configuration.

    Raises:
        ValueError: if the configuration contains a wrong number of leagues,
                   unexpected leagues, wrong division codes, duplicate codes,
                   or an accidentally reintroduced MLS/USA entry.
    """

    # ------------------------------------------------------------------------
    # Exactly 15 leagues
    # ------------------------------------------------------------------------
    if len(LEAGUE_CONFIG) != 15:
        raise ValueError(
            f"Expected exactly 15 configured leagues, "
            f"found {len(LEAGUE_CONFIG)}."
        )

    # ------------------------------------------------------------------------
    # Exact league/code mapping
    # ------------------------------------------------------------------------
    actual = {
        name: meta["code"]
        for name, meta in LEAGUE_CONFIG.items()
    }

    if actual != EXPECTED_LEAGUES:
        missing = set(EXPECTED_LEAGUES) - set(actual)
        unexpected = set(actual) - set(EXPECTED_LEAGUES)

        wrong_codes = {
            name: (EXPECTED_LEAGUES.get(name), actual.get(name))
            for name in set(EXPECTED_LEAGUES) & set(actual)
            if EXPECTED_LEAGUES[name] != actual[name]
        }

        raise ValueError(
            "League configuration mismatch. "
            f"Missing={sorted(missing)}, "
            f"Unexpected={sorted(unexpected)}, "
            f"WrongCodes={wrong_codes}"
        )

    # ------------------------------------------------------------------------
    # MLS must remain excluded
    # ------------------------------------------------------------------------
    if "MLS" in LEAGUE_CONFIG:
        raise ValueError(
            "MLS must not be present in LEAGUE_CONFIG."
        )

    if "USA" in DIV_TO_LEAGUE:
        raise ValueError(
            "USA must not be present in DIV_TO_LEAGUE."
        )

    # ------------------------------------------------------------------------
    # All configured leagues must use the main Football-Data source.
    # ------------------------------------------------------------------------
    extra_leagues = [
        name
        for name, meta in LEAGUE_CONFIG.items()
        if meta.get("source") != "main"
    ]

    if extra_leagues:
        raise ValueError(
            "Unexpected extra-source leagues found: "
            f"{extra_leagues}"
        )

    # ------------------------------------------------------------------------
    # Division codes must be unique
    # ------------------------------------------------------------------------
    codes = [meta["code"] for meta in LEAGUE_CONFIG.values()]

    if len(codes) != len(set(codes)):
        raise ValueError(
            "Duplicate Football-Data division codes detected."
        )

    return True


# Validate immediately when config.py is imported.
validate_league_config()