"""
TEAM NORMALIZATION MODULE

Strict, auditable team-name resolution for incoming fixtures (from the
football-data.co.uk fixture file or manual paste) against the CURRENT
season's canonical team list (current_teams.pkl), NOT the historical
all_teams universe.

Design rule (non-negotiable): an unrecognised team name must NEVER be
silently substituted with an unrelated team. The matching hierarchy is:

    1. Exact canonical match
    2. Case-insensitive canonical match
    3. Punctuation/whitespace-normalized match
    4. Explicit alias match (hand-maintained table below)
    5. UNRESOLVED — fuzzy matching may only produce a *suggestion* to show
       the user; it never becomes the resolved team on its own.

This directly fixes the previously observed bad behaviour where fuzzy
matching silently turned "Nottingham" into "Tottenham", "Le Mans" into
"Lens", "Paris" into "Paris SG", etc.
"""

import re
from difflib import get_close_matches


# ---------------------------------------------------------------------------
# Known aliases: football-data.co.uk / common media names -> canonical name
# (canonical names are whatever current_teams.pkl / all_teams.pkl use).
# Keys are matched case-insensitively after normalization, so casing here is
# just for readability.
# ---------------------------------------------------------------------------

ALIASES = {
    "Premier League": {
        "aston villa fc": "Aston Villa",
        "brighton & hove albion": "Brighton",
        "brighton and hove albion": "Brighton",
        "leeds united": "Leeds",
        "manchester city": "Man City",
        "manchester united": "Man United",
        "man utd": "Man United",
        "newcastle united": "Newcastle",
        "nottingham forest": "Nott'm Forest",
        "nottm forest": "Nott'm Forest",
        "nott'm forest": "Nott'm Forest",
        "tottenham hotspur": "Tottenham",
        "spurs": "Tottenham",
        "west ham united": "West Ham",
        "wolverhampton": "Wolves",
        "wolverhampton wanderers": "Wolves",
    },
    "La Liga": {
        "atletico madrid": "Ath Madrid",
        "atlético madrid": "Ath Madrid",
        "atletico de madrid": "Ath Madrid",
        "atl. madrid": "Ath Madrid",
        "atl madrid": "Ath Madrid",
        "athletic bilbao": "Ath Bilbao",
        "athletic club": "Ath Bilbao",
        "ath. bilbao": "Ath Bilbao",
        "real betis": "Betis",
        "celta vigo": "Celta",
        "espanyol": "Espanol",
        "rcd espanyol": "Espanol",
        "real betis balompie": "Betis",
        "getafe cf": "Getafe",
        "girona fc": "Girona",
        "ca osasuna": "Osasuna",
        "real sociedad": "Sociedad",
        "valencia cf": "Valencia",
        "rayo vallecano": "Vallecano",
        "villarreal cf": "Villarreal",
        "sevilla fc": "Sevilla",
    },
    "Serie A": {
        "atalanta bc": "Atalanta",
        "bologna fc": "Bologna",
        "cagliari calcio": "Cagliari",
        "como 1907": "Como",
        "us cremonese": "Cremonese",
        "acf fiorentina": "Fiorentina",
        "genoa cfc": "Genoa",
        "fc internazionale": "Inter",
        "inter milan": "Inter",
        "internazionale": "Inter",
        "juventus fc": "Juventus",
        "ss lazio": "Lazio",
        "us lecce": "Lecce",
        "ac milan": "Milan",
        "ssc napoli": "Napoli",
        "as roma": "Roma",
        "us sassuolo": "Sassuolo",
        "udinese calcio": "Udinese",
        "hellas verona": "Verona",
    },
    "Bundesliga": {
        "bayern munich": "Bayern Munich",
        "borussia dortmund": "Dortmund",
        "bvb": "Dortmund",
        "eintracht frankfurt": "Ein Frankfurt",
        "cologne": "FC Koln",
        "koln": "FC Koln",
        "1. fc koln": "FC Koln",
        "sc freiburg": "Freiburg",
        "hamburger sv": "Hamburg",
        "tsg 1899 hoffenheim": "Hoffenheim",
        "tsg hoffenheim": "Hoffenheim",
        "bayer 04 leverkusen": "Leverkusen",
        "bayer leverkusen": "Leverkusen",
        "borussia monchengladbach": "M'gladbach",
        "borussia mönchengladbach": "M'gladbach",
        "monchengladbach": "M'gladbach",
        "m'gladbach": "M'gladbach",
        "rb leipzig": "RB Leipzig",
        "fc st. pauli": "St Pauli",
        "st. pauli": "St Pauli",
        "1. fc union berlin": "Union Berlin",
        "fc union berlin": "Union Berlin",
        "sv werder bremen": "Werder Bremen",
        "vfl wolfsburg": "Wolfsburg",
    },
    "Ligue 1": {
        "angers sco": "Angers",
        "aj auxerre": "Auxerre",
        "stade brestois": "Brest",
        "havre ac": "Le Havre",
        "rc lens": "Lens",
        "losc lille": "Lille",
        "fc lorient": "Lorient",
        "olympique lyonnais": "Lyon",
        "ol": "Lyon",
        "olympique marseille": "Marseille",
        "om": "Marseille",
        "fc metz": "Metz",
        "as monaco": "Monaco",
        "fc nantes": "Nantes",
        "ogc nice": "Nice",
        "paris saint germain": "Paris SG",
        "paris saint-germain": "Paris SG",
        "psg": "Paris SG",
        "stade rennais": "Rennes",
        "rc strasbourg": "Strasbourg",
        "toulouse fc": "Toulouse",
        # Deliberately NOT included: "paris" alone (ambiguous between Paris
        # SG and Paris FC — must not be auto-resolved) and "le mans" (not a
        # current top-flight team in any supported league).
    },
}


def _normalize_key(name):
    """Lowercase, strip, collapse whitespace, drop punctuation for comparison."""
    s = name.strip().lower()
    s = re.sub(r"[^\w\s]", "", s)  # drop punctuation (periods, apostrophes, &, etc.)
    s = re.sub(r"\s+", " ", s)
    return s


def resolve_team(raw_name, league, current_teams_by_league, all_teams=None):
    """
    Resolve a raw team name to a canonical current-season team name for the
    given league, using ONLY the strict hierarchy described in the module
    docstring. Never auto-substitutes an unrelated team.

    Args:
        raw_name: team name as it appeared in the fixture source
        league: canonical league name (e.g. 'Premier League')
        current_teams_by_league: dict league -> list of canonical current
            teams (from current_teams.pkl['current_teams_by_league'])
        all_teams: optional full historical team list, used only to widen
            fuzzy SUGGESTIONS (never to resolve automatically)

    Returns:
        dict with:
            'input': raw_name
            'league': league
            'resolved': canonical name, or None if unresolved
            'method': one of 'exact', 'case_insensitive', 'normalized',
                      'alias', 'unresolved'
            'suggestions': list of plausible canonical names (only
                populated when unresolved) — for DISPLAY ONLY, never
                auto-applied
    """

    canonical_list = current_teams_by_league.get(league, [])
    raw_stripped = raw_name.strip()

    # 1. Exact match
    if raw_stripped in canonical_list:
        return {'input': raw_name, 'league': league, 'resolved': raw_stripped,
                'method': 'exact', 'suggestions': []}

    # 2. Case-insensitive match
    lower_map = {c.lower(): c for c in canonical_list}
    if raw_stripped.lower() in lower_map:
        return {'input': raw_name, 'league': league, 'resolved': lower_map[raw_stripped.lower()],
                'method': 'case_insensitive', 'suggestions': []}

    # 3. Punctuation/whitespace-normalized match
    norm_key = _normalize_key(raw_stripped)
    norm_map = {_normalize_key(c): c for c in canonical_list}
    if norm_key in norm_map:
        return {'input': raw_name, 'league': league, 'resolved': norm_map[norm_key],
                'method': 'normalized', 'suggestions': []}

    # 4. Explicit alias match (checked against the normalized key, so
    # "Nottingham Forest", "nottingham   forest", "Nottingham-Forest" etc.
    # all hit the same alias entry)
    league_aliases = ALIASES.get(league, {})
    alias_norm_map = {_normalize_key(k): v for k, v in league_aliases.items()}
    if norm_key in alias_norm_map:
        canonical = alias_norm_map[norm_key]
        # Only trust the alias if it actually IS a current team for this
        # league (guards against a stale alias pointing to a relegated team).
        if canonical in canonical_list:
            return {'input': raw_name, 'league': league, 'resolved': canonical,
                    'method': 'alias', 'suggestions': []}

    # 5. Unresolved — offer suggestions for the human to confirm, but do
    # NOT resolve automatically.
    pool = list(canonical_list)
    if all_teams:
        pool = sorted(set(pool) | set(all_teams))
    suggestions = get_close_matches(raw_stripped, pool, n=3, cutoff=0.6)

    return {'input': raw_name, 'league': league, 'resolved': None,
            'method': 'unresolved', 'suggestions': suggestions}


def resolve_fixture(league, home_raw, away_raw, current_teams_by_league, all_teams=None):
    """
    Resolve both teams of a fixture. Returns a dict describing the fixture's
    validation status:

        'status': 'valid' | 'needs_review'
        'league', 'home_raw', 'away_raw'
        'home': resolve_team(...) result
        'away': resolve_team(...) result
    """

    home_res = resolve_team(home_raw, league, current_teams_by_league, all_teams)
    away_res = resolve_team(away_raw, league, current_teams_by_league, all_teams)

    status = 'valid' if (home_res['resolved'] and away_res['resolved']) else 'needs_review'

    return {
        'status': status,
        'league': league,
        'home_raw': home_raw,
        'away_raw': away_raw,
        'home': home_res,
        'away': away_res,
    }
