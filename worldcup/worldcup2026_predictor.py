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
import os
import random
import difflib


# =============================================================================
# TEAM NAME NORMALIZER
# Maps every app-facing display name → exact dataset name.
# Add new aliases here whenever a mismatch is discovered.
# =============================================================================

# ---------------------------------------------------------------------------
# Full alias table: APP/USER name  →  DATASET canonical name
# Lookup is done case-insensitively + stripped so capitalization,
# extra spaces, and minor accent differences are handled automatically.
# ---------------------------------------------------------------------------
# =============================================================================
# TEAM REGISTRY — single source of truth for all 48 WC 2026 teams
#
# display_name  : FIFA-official / user-facing label shown in dropdowns & UI
# dataset_name  : exact string used in martj42/international_results CSV
# confederation : for the statistics table
# flag          : emoji flag
# =============================================================================

# =============================================================================
# CANONICAL TEAM REGISTRY  —  single source of truth for all 48 WC 2026 teams
#
# Key  = canonical name (used in QUALIFIED_TEAMS, dropdowns, WC2026_GROUPS)
# Each entry:
#   display_name  : shown in UI labels and dropdowns
#   dataset_name  : exact string in martj42/international_results CSV
#   confederation : AFC | CAF | CONCACAF | CONMEBOL | OFC | UEFA
#   flag          : emoji flag
#
# Name-convention mappings:
#   Canonical         →  dataset_name
#   United States     →  United States   (hosts; stored correctly)
#   Czechia           →  Czech Republic  (UEFA branding vs FIFA dataset)
#   Côte d'Ivoire     →  Ivory Coast     (CAF official vs historical CSV)
#   Cabo Verde        →  Cape Verde      (CAF official vs historical CSV)
#   Curaçao           →  Curaçao         (stored with accent in CSV)
#   Türkiye           →  Turkey          (UEFA branding vs historical CSV)
#   DR Congo          →  DR Congo        (stored as DR Congo in CSV)
#   IR Iran           →  Iran            (AFC code vs historical CSV)
# =============================================================================

TEAM_REGISTRY: dict[str, dict] = {
    # ── Hosts (CONCACAF) ───────────────────────────────────────────────────
    "Canada":        {"display_name": "Canada",        "dataset_name": "Canada",        "confederation": "CONCACAF", "flag": "🇨🇦"},
    "Mexico":        {"display_name": "Mexico",        "dataset_name": "Mexico",        "confederation": "CONCACAF", "flag": "🇲🇽"},
    "United States": {"display_name": "United States", "dataset_name": "United States", "confederation": "CONCACAF", "flag": "🇺🇸"},

    # ── CAF — Africa (10 teams) ────────────────────────────────────────────
    "Algeria":       {"display_name": "Algeria",       "dataset_name": "Algeria",       "confederation": "CAF",      "flag": "🇩🇿"},
    "Cabo Verde":    {"display_name": "Cabo Verde",    "dataset_name": "Cape Verde",    "confederation": "CAF",      "flag": "🇨🇻"},
    "Côte d'Ivoire": {"display_name": "Côte d'Ivoire", "dataset_name": "Ivory Coast",   "confederation": "CAF",      "flag": "🇨🇮"},
    "DR Congo":      {"display_name": "DR Congo",      "dataset_name": "DR Congo",      "confederation": "CAF",      "flag": "🇨🇩"},
    "Egypt":         {"display_name": "Egypt",         "dataset_name": "Egypt",         "confederation": "CAF",      "flag": "🇪🇬"},
    "Ghana":         {"display_name": "Ghana",         "dataset_name": "Ghana",         "confederation": "CAF",      "flag": "🇬🇭"},
    "Morocco":       {"display_name": "Morocco",       "dataset_name": "Morocco",       "confederation": "CAF",      "flag": "🇲🇦"},
    "Senegal":       {"display_name": "Senegal",       "dataset_name": "Senegal",       "confederation": "CAF",      "flag": "🇸🇳"},
    "South Africa":  {"display_name": "South Africa",  "dataset_name": "South Africa",  "confederation": "CAF",      "flag": "🇿🇦"},
    "Tunisia":       {"display_name": "Tunisia",       "dataset_name": "Tunisia",       "confederation": "CAF",      "flag": "🇹🇳"},

    # ── AFC — Asia (9 teams) ───────────────────────────────────────────────
    "Australia":     {"display_name": "Australia",     "dataset_name": "Australia",     "confederation": "AFC",      "flag": "🇦🇺"},
    "IR Iran":       {"display_name": "IR Iran",       "dataset_name": "Iran",          "confederation": "AFC",      "flag": "🇮🇷"},
    "Iraq":          {"display_name": "Iraq",          "dataset_name": "Iraq",          "confederation": "AFC",      "flag": "🇮🇶"},
    "Japan":         {"display_name": "Japan",         "dataset_name": "Japan",         "confederation": "AFC",      "flag": "🇯🇵"},
    "Jordan":        {"display_name": "Jordan",        "dataset_name": "Jordan",        "confederation": "AFC",      "flag": "🇯🇴"},
    "Qatar":         {"display_name": "Qatar",         "dataset_name": "Qatar",         "confederation": "AFC",      "flag": "🇶🇦"},
    "Saudi Arabia":  {"display_name": "Saudi Arabia",  "dataset_name": "Saudi Arabia",  "confederation": "AFC",      "flag": "🇸🇦"},
    "South Korea":   {"display_name": "South Korea",   "dataset_name": "South Korea",   "confederation": "AFC",      "flag": "🇰🇷"},
    "Uzbekistan":    {"display_name": "Uzbekistan",    "dataset_name": "Uzbekistan",    "confederation": "AFC",      "flag": "🇺🇿"},

    # ── CONMEBOL — South America (6 teams) ────────────────────────────────
    "Argentina":     {"display_name": "Argentina",     "dataset_name": "Argentina",     "confederation": "CONMEBOL", "flag": "🇦🇷"},
    "Brazil":        {"display_name": "Brazil",        "dataset_name": "Brazil",        "confederation": "CONMEBOL", "flag": "🇧🇷"},
    "Colombia":      {"display_name": "Colombia",      "dataset_name": "Colombia",      "confederation": "CONMEBOL", "flag": "🇨🇴"},
    "Ecuador":       {"display_name": "Ecuador",       "dataset_name": "Ecuador",       "confederation": "CONMEBOL", "flag": "🇪🇨"},
    "Paraguay":      {"display_name": "Paraguay",      "dataset_name": "Paraguay",      "confederation": "CONMEBOL", "flag": "🇵🇾"},
    "Uruguay":       {"display_name": "Uruguay",       "dataset_name": "Uruguay",       "confederation": "CONMEBOL", "flag": "🇺🇾"},

    # ── CONCACAF non-hosts (3 teams) ──────────────────────────────────────
    "Curaçao":       {"display_name": "Curaçao",       "dataset_name": "Curaçao",       "confederation": "CONCACAF", "flag": "🇨🇼"},
    "Haiti":         {"display_name": "Haiti",         "dataset_name": "Haiti",         "confederation": "CONCACAF", "flag": "🇭🇹"},
    "Panama":        {"display_name": "Panama",        "dataset_name": "Panama",        "confederation": "CONCACAF", "flag": "🇵🇦"},

    # ── OFC — Oceania (1 team) ─────────────────────────────────────────────
    "New Zealand":   {"display_name": "New Zealand",   "dataset_name": "New Zealand",   "confederation": "OFC",      "flag": "🇳🇿"},

    # ── UEFA — Europe (16 teams) ───────────────────────────────────────────
    "Austria":       {"display_name": "Austria",       "dataset_name": "Austria",       "confederation": "UEFA",     "flag": "🇦🇹"},
    "Belgium":       {"display_name": "Belgium",       "dataset_name": "Belgium",       "confederation": "UEFA",     "flag": "🇧🇪"},
    "Bosnia and Herzegovina": {"display_name": "Bosnia and Herzegovina", "dataset_name": "Bosnia and Herzegovina", "confederation": "UEFA", "flag": "🇧🇦"},
    "Croatia":       {"display_name": "Croatia",       "dataset_name": "Croatia",       "confederation": "UEFA",     "flag": "🇭🇷"},
    "Czechia":       {"display_name": "Czechia",       "dataset_name": "Czech Republic","confederation": "UEFA",     "flag": "🇨🇿"},
    "England":       {"display_name": "England",       "dataset_name": "England",       "confederation": "UEFA",     "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    "France":        {"display_name": "France",        "dataset_name": "France",        "confederation": "UEFA",     "flag": "🇫🇷"},
    "Germany":       {"display_name": "Germany",       "dataset_name": "Germany",       "confederation": "UEFA",     "flag": "🇩🇪"},
    "Netherlands":   {"display_name": "Netherlands",   "dataset_name": "Netherlands",   "confederation": "UEFA",     "flag": "🇳🇱"},
    "Norway":        {"display_name": "Norway",        "dataset_name": "Norway",        "confederation": "UEFA",     "flag": "🇳🇴"},
    "Portugal":      {"display_name": "Portugal",      "dataset_name": "Portugal",      "confederation": "UEFA",     "flag": "🇵🇹"},
    "Scotland":      {"display_name": "Scotland",      "dataset_name": "Scotland",      "confederation": "UEFA",     "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿"},
    "Spain":         {"display_name": "Spain",         "dataset_name": "Spain",         "confederation": "UEFA",     "flag": "🇪🇸"},
    "Sweden":        {"display_name": "Sweden",        "dataset_name": "Sweden",        "confederation": "UEFA",     "flag": "🇸🇪"},
    "Switzerland":   {"display_name": "Switzerland",   "dataset_name": "Switzerland",   "confederation": "UEFA",     "flag": "🇨🇭"},
    "Türkiye":       {"display_name": "Türkiye",       "dataset_name": "Turkey",        "confederation": "UEFA",     "flag": "🇹🇷"},
}

# Silent startup assertion — catches registry errors immediately, never shown in UI
assert len(TEAM_REGISTRY) == 48,  f"Expected 48 teams, got {len(TEAM_REGISTRY)}"
assert all("display_name"  in v for v in TEAM_REGISTRY.values()), "Missing display_name"
assert all("dataset_name"  in v for v in TEAM_REGISTRY.values()), "Missing dataset_name"
assert all("confederation" in v for v in TEAM_REGISTRY.values()), "Missing confederation"
assert all("flag"          in v for v in TEAM_REGISTRY.values()), "Missing flag"

# ---------------------------------------------------------------------------
# Derived helpers — always computed from TEAM_REGISTRY (single source of truth)
# ---------------------------------------------------------------------------

# Primary sorted list: confederation order then alphabetical within each
QUALIFIED_TEAMS: list[str] = sorted(
    TEAM_REGISTRY.keys(),
    key=lambda t: (TEAM_REGISTRY[t]["confederation"], t),
)

# TEAM_FLAGS — keyed by canonical, display_name, and dataset_name for full coverage
TEAM_FLAGS: dict[str, str] = {}
for _cn, _rec in TEAM_REGISTRY.items():
    TEAM_FLAGS[_cn]                  = _rec["flag"]
    TEAM_FLAGS[_rec["display_name"]] = _rec["flag"]
    TEAM_FLAGS[_rec["dataset_name"]] = _rec["flag"]

# TEAM_NAME_MAP: any spelling variant (lowercase) → dataset_name
TEAM_NAME_MAP: dict[str, str] = {}
for _cn, _rec in TEAM_REGISTRY.items():
    _ds = _rec["dataset_name"]
    _dn = _rec["display_name"]
    TEAM_NAME_MAP[_cn.strip().lower()] = _ds
    TEAM_NAME_MAP[_dn.strip().lower()] = _ds
    TEAM_NAME_MAP[_ds.strip().lower()] = _ds

# Extra well-known aliases and abbreviations
TEAM_NAME_MAP.update({
    "usa":                    "United States",
    "united states":          "United States",
    "czechia":                "Czech Republic",
    "czech republic":         "Czech Republic",
    "ivory coast":            "Ivory Coast",
    "cape verde":             "Cape Verde",
    "cabo verde":             "Cape Verde",
    "ir iran":                "Iran",
    "iran":                   "Iran",
    "south korea":            "South Korea",
    "korea republic":         "South Korea",
    "dr congo":               "DR Congo",
    "congo dr":               "DR Congo",
    "bosnia-herzegovina":     "Bosnia and Herzegovina",
    "türkiye":                "Turkey",
    "turkey":                 "Turkey",
    "curaçao":                "Curaçao",
    "curacao":                "Curaçao",
})

# Fast case-insensitive lookup dict
_NORM_LOOKUP: dict[str, str] = {k: v for k, v in TEAM_NAME_MAP.items()}

# Reverse map: dataset_name → canonical key (for display_name() function)
DISPLAY_NAME_MAP: dict[str, str] = {
    _rec["dataset_name"]: _cn for _cn, _rec in TEAM_REGISTRY.items()
}

def normalize(team: str) -> str:
    """Return the dataset-canonical name for any team spelling variant."""
    if not team:
        return team
    return _NORM_LOOKUP.get(team.strip().lower(), team)


def display_name(dataset_team: str) -> str:
    """Return the user-facing display name for a dataset team name."""
    return DISPLAY_NAME_MAP.get(dataset_team, dataset_team)


def ds(team: str) -> str:
    """Shorthand: display name → dataset name (used everywhere in calculations)."""
    return normalize(team)


def validate_teams(df: "pd.DataFrame", teams: list) -> dict:
    """Internal validation — not shown in UI."""
    all_ds = set(df["home_team"]) | set(df["away_team"])
    validated, aliased, missing = [], [], []
    for app_name in teams:
        ds_name = normalize(app_name)
        if ds_name in all_ds:
            validated.append((app_name, ds_name))
            if app_name != ds_name:
                aliased.append((app_name, ds_name))
        else:
            close = difflib.get_close_matches(app_name, all_ds, n=3, cutoff=0.5)
            missing.append((app_name, close))
    return {"validated": validated, "aliased": aliased, "missing": missing}




def run_team_audit(df: "pd.DataFrame", dc_model=None, elo_ratings: dict = None) -> list[dict]:
    """
    Audit every team in QUALIFIED_TEAMS against the dataset.
    Returns a list of dicts with full diagnostic info per team.
    """
    vresult    = validate_teams(df, QUALIFIED_TEAMS)
    missing_map = {app: close for app, close in vresult["missing"]}
    aliased_set = {app for app, _ in vresult["aliased"]}
    all_ds      = set(df["home_team"]) | set(df["away_team"])
    rows = []
    for app_name in QUALIFIED_TEAMS:
        ds_name  = normalize(app_name)
        in_ds    = ds_name in all_ds
        matches  = int(len(df[(df["home_team"]==ds_name)|(df["away_team"]==ds_name)])) if in_ds else 0
        elo_ok   = (elo_ratings is not None) and (ds_name in elo_ratings)
        att_ok   = (dc_model    is not None) and (ds_name in getattr(dc_model, "attack",  {}))
        def_ok   = (dc_model    is not None) and (ds_name in getattr(dc_model, "defence", {}))
        form_ok  = matches >= 5
        h2h_ok   = in_ds
        closest  = ", ".join(missing_map.get(app_name, [])) or "—"
        rows.append({
            "App Name":      app_name,
            "Dataset Name":  ds_name,
            "In Dataset":    "✅" if in_ds  else "❌",
            "Closest Match": closest if not in_ds else "—",
            "Matches":       matches,
            "Elo":           "✅" if elo_ok  else "❌",
            "Attack":        "✅" if att_ok  else "❌",
            "Defence":       "✅" if def_ok  else "❌",
            "Form":          "✅" if form_ok else "❌",
            "H2H":           "✅" if h2h_ok  else "❌",
            "Alias Used":    "⚠️ yes" if app_name in aliased_set else "—",
        })
    return rows


def run_developer_coverage_report(df: "pd.DataFrame") -> None:
    """
    Internal startup integrity check — NEVER displayed in the Streamlit UI.
    Runs silently at app startup and prints a developer coverage report to
    stdout (visible in terminal / Streamlit Cloud logs only).

    Performs the checks required by the consistency audit:
      - len(QUALIFIED_TEAMS) == 48
      - len(TEAM_REGISTRY) == 48
      - every team resolves through normalize()
      - every team's dataset_name exists in the historical dataset
    Fails fast (raises) if any qualified team cannot be resolved at all,
    since that would silently corrupt every downstream calculation.
    """
    assert len(QUALIFIED_TEAMS) == 48, f"QUALIFIED_TEAMS has {len(QUALIFIED_TEAMS)}, expected 48"
    assert len(TEAM_REGISTRY)   == 48, f"TEAM_REGISTRY has {len(TEAM_REGISTRY)}, expected 48"

    all_ds_teams = set(df["home_team"]) | set(df["away_team"])

    missing_teams   = []
    alias_fixes     = []
    in_dataset_count = 0

    for canonical in QUALIFIED_TEAMS:
        rec     = TEAM_REGISTRY[canonical]
        ds_name = rec["dataset_name"]

        # normalize() must resolve the canonical name correctly
        resolved = normalize(canonical)
        if resolved != ds_name:
            # normalize() disagrees with registry — developer bug, fail fast
            raise RuntimeError(
                f"normalize({canonical!r}) returned {resolved!r}, "
                f"but TEAM_REGISTRY says dataset_name={ds_name!r}"
            )

        if ds_name in all_ds_teams:
            in_dataset_count += 1
            if canonical != ds_name:
                alias_fixes.append(f"{canonical} -> {ds_name}")
        else:
            missing_teams.append(canonical)

    report_lines = [
        "",
        "=" * 70,
        "DEVELOPER COVERAGE REPORT (internal — not shown in Streamlit UI)",
        "=" * 70,
        f"Total Teams:          48",
        f"Teams In Registry:    {len(TEAM_REGISTRY)}",
        f"Teams In Dataset:     {in_dataset_count}",
        f"Teams In Dropdown:    {len(QUALIFIED_TEAMS)}",
        "",
        f"Missing Teams: {missing_teams if missing_teams else '[]'}",
        "",
        "Alias Fixes Applied:",
    ]
    if alias_fixes:
        report_lines.extend(f"  {a}" for a in alias_fixes)
    else:
        report_lines.append("  (none)")
    report_lines.append("=" * 70)

    print("\n".join(report_lines))

    # Fail fast — a missing team means downstream Elo/Dixon-Coles/predictions
    # for that team would silently default to 1500/zero, corrupting results.
    if missing_teams:
        raise RuntimeError(
            f"DEVELOPER ERROR: {len(missing_teams)} qualified team(s) not found "
            f"in dataset after normalization: {missing_teams}. "
            f"Add the correct alias to TEAM_NAME_MAP / TEAM_REGISTRY."
        )


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

# QUALIFIED_TEAMS and TEAM_FLAGS are now derived from TEAM_REGISTRY above.

def flag(team: str) -> str:
    """Return emoji flag; accepts canonical, display, or dataset name."""
    return TEAM_FLAGS.get(team,
           TEAM_FLAGS.get(normalize(team),
           TEAM_FLAGS.get(display_name(team), "🌍")))


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

# Primary URL — GitHub raw CSV
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Mirror URLs tried in order if primary fails
DATA_MIRRORS = [
    DATA_URL,
    "https://github.com/martj42/international_results/raw/master/results.csv",
]

# Local cache path — next to the script so it persists across runs
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
LOCAL_CACHE   = os.path.join(_SCRIPT_DIR, "results_cache.csv")


def _process_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and feature-engineer a freshly loaded DataFrame."""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date","home_team","away_team","home_score","away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["result"]  = np.where(
        df["home_score"] > df["away_score"], "H",
        np.where(df["home_score"] < df["away_score"], "A", "D")
    )
    df["outcome"] = df["result"].map({"H": 0, "D": 1, "A": 2})
    df = df.sort_values("date").reset_index(drop=True)
    df["days_ago"]       = (df["date"].max() - df["date"]).dt.days
    df["home_team_norm"] = df["home_team"]
    df["away_team_norm"] = df["away_team"]
    return df


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Load international results with a three-tier fallback strategy:

    1. Try each mirror URL (primary GitHub raw URL first)
    2. If all URLs fail, load from local cache file (results_cache.csv)
       placed in the same folder as this script
    3. If no local cache exists either, raise a clear error with
       instructions to manually download the file

    The file is also saved locally after every successful download
    so future runs work offline automatically.
    """
    last_err = None

    # ── Tier 1: try network URLs ─────────────────────────────────────────────
    for url in DATA_MIRRORS:
        try:
            import urllib.request
            import io
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw_bytes = resp.read()
            df = pd.read_csv(io.BytesIO(raw_bytes))
            df = _process_raw(df)
            # Save a local copy for offline use
            try:
                df.to_csv(LOCAL_CACHE, index=False)
            except Exception:
                pass   # non-fatal — just means offline fallback won't update
            return df
        except Exception as e:
            last_err = e
            continue

    # ── Tier 2: local cache ──────────────────────────────────────────────────
    if os.path.exists(LOCAL_CACHE):
        try:
            df = pd.read_csv(LOCAL_CACHE)
            df = _process_raw(df)
            st.warning(
                "⚠️ **Offline mode** — could not reach GitHub. "
                f"Using cached data from: `{LOCAL_CACHE}`  "
                f"(last saved: {pd.Timestamp(os.path.getmtime(LOCAL_CACHE), unit='s').strftime('%Y-%m-%d %H:%M')})"
            )
            return df
        except Exception as e:
            last_err = e

    # ── Tier 3: clear error with manual download instructions ────────────────
    st.error(
        "❌ **Cannot load match data.**\n\n"
        "**Network error:** " + str(last_err) + "\n\n"
        "**To fix — choose one option:**\n\n"
        "**Option A — Fix your internet connection** then restart the app.\n\n"
        "**Option B — Download the file manually:**\n"
        "1. Open this URL in your browser:\n"
        "   https://raw.githubusercontent.com/martj42/international_results/master/results.csv\n"
        "2. Save the file as `results_cache.csv`\n"
        f"3. Place it here: `{_SCRIPT_DIR}\\results_cache.csv`\n"
        "4. Restart the app — it will load from the local file automatically.\n\n"
        "**Option C — Use a VPN** if GitHub is blocked in your region."
    )
    st.stop()


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
        home = normalize(home)
        away = normalize(away)
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
    # Resolve dataset-canonical names before every lookup
    home = normalize(home)
    away = normalize(away)

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

    hr = df[(df["home_team_norm"]==home)|(df["away_team_norm"]==home)].tail(15)
    ar = df[(df["home_team_norm"]==away)|(df["away_team_norm"]==away)].tail(15)
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
    home = normalize(home)
    away = normalize(away)
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
    """Lookup H2H using dataset-canonical names so aliases always resolve."""
    a = normalize(a); b = normalize(b)
    mask = (((df["home_team_norm"]==a)&(df["away_team_norm"]==b))|
            ((df["home_team_norm"]==b)&(df["away_team_norm"]==a)))
    return df[mask].sort_values("date", ascending=False).head(n)


# =============================================================================
# SECTION 9 — TEAM STRENGTH TABLE
# =============================================================================

def build_strength_table(df: "pd.DataFrame", elo_ratings: dict,
                         dc_model=None) -> "pd.DataFrame":
    """
    Build the full Team Statistics table for all 48 qualified teams.
    Columns: Team, Confederation, Matches, W, D, L, GF, GA, GD,
             Win%, Elo, Attack, Defence
    All dataset lookups use normalize() so aliases are handled automatically.
    """
    rows = []
    for canonical in QUALIFIED_TEAMS:
        rec     = TEAM_REGISTRY[canonical]
        ds_name = rec["dataset_name"]
        conf    = rec["confederation"]
        elo     = round(elo_ratings.get(ds_name, 1500))

        # All matches for last-date lookup (full history, not just tail 30)
        all_matches = df[
            (df["home_team_norm"] == ds_name) |
            (df["away_team_norm"] == ds_name)
        ].sort_values("date")

        recent = all_matches.tail(30)

        w = d = l = gf = ga = 0
        for _, r in recent.iterrows():
            if r["home_team_norm"] == ds_name:
                gf += r["home_score"]; ga += r["away_score"]
                if r["result"] == "H": w += 1
                elif r["result"] == "D": d += 1
                else: l += 1
            else:
                gf += r["away_score"]; ga += r["home_score"]
                if r["result"] == "A": w += 1
                elif r["result"] == "D": d += 1
                else: l += 1

        played     = w + d + l
        att        = round(getattr(dc_model, "attack",  {}).get(ds_name, 0.0), 4) if dc_model else 0.0
        deff       = round(getattr(dc_model, "defence", {}).get(ds_name, 0.0), 4) if dc_model else 0.0
        avg_gs     = round(gf / played, 2) if played else 0.0
        avg_gc     = round(ga / played, 2) if played else 0.0
        last_date  = all_matches["date"].max().strftime("%Y-%m-%d") if len(all_matches) else "N/A"
        total_mp   = len(all_matches)   # full career matches for "most experienced" card

        rows.append({
            "Flag":           rec["flag"],
            "Team":           canonical,
            "Confederation":  conf,
            "Matches Played": played,
            "W":              w,
            "D":              d,
            "L":              l,
            "Goals For":      gf,
            "Goals Against":  ga,
            "GD":             gf - ga,
            "Win %":          round(w / played * 100, 1) if played else 0.0,
            "Avg GS":         avg_gs,
            "Avg GC":         avg_gc,
            "Elo":            elo,
            "Attack":         att,
            "Defence":        deff,
            "Last Match":     last_date,
            "_total_mp":      total_mp,   # internal — filtered from display
        })

    return (pd.DataFrame(rows)
              .sort_values("Elo", ascending=False)
              .reset_index(drop=True))



# =============================================================================
# SECTION 10 — TOURNAMENT SIMULATOR
# =============================================================================

# Official 2026 FIFA World Cup group draw (BBC Sport, June 2026).
# Every key is a TEAM_REGISTRY canonical name — verified against TEAM_REGISTRY.
# Simulation uses normalize(team) internally so dataset_name is always resolved.
WC2026_GROUPS = {
    "A": ["Mexico",                   "South Korea",    "Czechia",          "South Africa"],
    "B": ["Switzerland",              "Canada",         "Qatar",            "Bosnia and Herzegovina"],
    "C": ["Scotland",                 "Morocco",        "Brazil",           "Haiti"],
    "D": ["United States",            "Australia",      "Türkiye",          "Paraguay"],
    "E": ["Germany",                  "Côte d'Ivoire",  "Ecuador",          "Curaçao"],
    "F": ["Sweden",                   "Japan",          "Netherlands",      "Tunisia"],
    "G": ["New Zealand",              "IR Iran",        "Belgium",          "Egypt"],
    "H": ["Uruguay",                  "Saudi Arabia",   "Spain",            "Cabo Verde"],
    "I": ["Norway",                   "France",         "Senegal",          "Iraq"],
    "J": ["Argentina",                "Austria",        "Jordan",           "Algeria"],
    "K": ["Portugal",                 "DR Congo",       "Uzbekistan",       "Colombia"],
    "L": ["England",                  "Croatia",        "Ghana",            "Panama"],
}

# Silent integrity check — catches any future typo immediately
_all_grp = [t for teams in WC2026_GROUPS.values() for t in teams]
assert len(_all_grp) == 48,                     f"Groups have {len(_all_grp)} teams, expected 48"
assert len(set(_all_grp)) == 48,                "Duplicate team in WC2026_GROUPS"
assert all(t in TEAM_REGISTRY for t in _all_grp),     "WC2026_GROUPS team not in TEAM_REGISTRY: " + str([t for t in _all_grp if t not in TEAM_REGISTRY])

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

    # Silent startup integrity check — never shown in UI, logs to console only
    run_developer_coverage_report(df_raw)
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Match Predictor","🏆 Tournament Simulator","📊 Team Rankings",
        "📊 Dataset Coverage","ℹ️ Methodology",
    ])

    # Build alphabetical, flag-prefixed dropdown list once (outside tab)
    _sorted_teams  = sorted(QUALIFIED_TEAMS, key=lambda x: normalize(x))
    _team_labels   = [f"{flag(t)} {t}" for t in _sorted_teams]
    _label_to_name = {f"{flag(t)} {t}": t for t in _sorted_teams}

    # ---- TAB 1: MATCH PREDICTOR ----
    with tab1:
        st.markdown('<div class="section-header">⚽ Select Match Parameters</div>',
                    unsafe_allow_html=True)

        col_l, col_r = st.columns([1,2])

        with col_l:
            _default_h = f"{flag('Argentina')} Argentina"
            home_label = st.selectbox(
                "🏠 Home / Team A",
                _team_labels,
                index=_team_labels.index(_default_h) if _default_h in _team_labels else 0,
                help="Type to search",
            )
            home_team = _label_to_name[home_label]

            _away_labels = [l for l in _team_labels if l != home_label]
            _default_a   = f"{flag('France')} France"
            away_label = st.selectbox(
                "✈️ Away / Team B",
                _away_labels,
                index=_away_labels.index(_default_a) if _default_a in _away_labels else 0,
                help="Type to search",
            )
            away_team = _label_to_name[away_label]
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
                    is_home = r["home_team"]==normalize(home_team)
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
            sdf = build_strength_table(df_raw, elo_ratings, dc_model=dc_model)

        # ── Elo vs Win% scatter ──────────────────────────────────────────────
        _scatter_df = sdf.rename(columns={
            "Win %": "Win%", "Matches Played": "Matches",
            "Goals For": "GF", "Goals Against": "GA"
        })
        st.plotly_chart(elo_scatter(_scatter_df, t), use_container_width=True)

        # ── Team Statistics table ────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Team Statistics</div>',
                    unsafe_allow_html=True)

        _sort_opts = {
            "Elo (desc)":          ("Elo",           False),
            "Matches Played":      ("Matches Played",False),
            "Win % (desc)":        ("Win %",         False),
            "Goals For (desc)":    ("Goals For",     False),
            "Goal Difference":     ("GD",            False),
            "Attack Rating":       ("Attack",        False),
            "Defence Rating":      ("Defence",       False),
            "Alphabetical":        ("Team",          True),
        }
        _r1, _r2 = st.columns([2, 2])
        with _r1:
            _confs  = ["All"] + sorted(sdf["Confederation"].unique().tolist())
            _cf_sel = st.selectbox("Filter by Confederation", _confs, index=0)
        with _r2:
            _sort_sel = st.selectbox("Sort by", list(_sort_opts.keys()), index=0)

        # Fuzzy + alias search: matches canonical name, display aliases, and
        # any TEAM_NAME_MAP key that maps to this team's dataset_name
        _search = st.text_input("🔍 Search (supports aliases: usa, ivory, czech, cap, tur…)",
                                placeholder="Type team name or alias…")

        def _fuzzy_match(team_canonical: str, query: str) -> bool:
            """Return True if query matches any name variant for this team."""
            q = query.strip().lower()
            if not q:
                return True
            rec = TEAM_REGISTRY[team_canonical]
            # Check canonical, display, dataset names
            candidates = [
                team_canonical.lower(),
                rec["display_name"].lower(),
                rec["dataset_name"].lower(),
            ]
            # Also check all TEAM_NAME_MAP keys that resolve to this dataset_name
            ds = rec["dataset_name"]
            for alias_key, alias_ds in TEAM_NAME_MAP.items():
                if alias_ds == ds:
                    candidates.append(alias_key.lower())
            return any(q in c for c in candidates)

        _tdf = sdf.copy()
        if _cf_sel != "All":
            _tdf = _tdf[_tdf["Confederation"] == _cf_sel]
        if _search:
            _tdf = _tdf[_tdf["Team"].apply(lambda t: _fuzzy_match(t, _search))]

        _sort_col, _sort_asc = _sort_opts[_sort_sel]
        _tdf = _tdf.sort_values(_sort_col, ascending=_sort_asc).reset_index(drop=True)
        _tdf.insert(0, "Rank", range(1, len(_tdf) + 1))

        _display_cols = [
            "Rank", "Flag", "Team", "Confederation",
            "Matches Played", "W", "D", "L",
            "Goals For", "Goals Against", "GD", "Win %",
            "Avg GS", "Avg GC", "Elo", "Attack", "Defence", "Last Match",
        ]

        st.dataframe(
            _tdf[_display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank":           st.column_config.NumberColumn("Rank",     width="small"),
                "Flag":           st.column_config.TextColumn("",           width="small"),
                "Team":           st.column_config.TextColumn("Team",       width="medium"),
                "Confederation":  st.column_config.TextColumn("Conf.",      width="small"),
                "Matches Played": st.column_config.NumberColumn("MP",       width="small"),
                "W":              st.column_config.NumberColumn("W",        width="small"),
                "D":              st.column_config.NumberColumn("D",        width="small"),
                "L":              st.column_config.NumberColumn("L",        width="small"),
                "Goals For":      st.column_config.NumberColumn("GF",       width="small"),
                "Goals Against":  st.column_config.NumberColumn("GA",       width="small"),
                "GD":             st.column_config.NumberColumn("GD",       width="small"),
                "Win %":          st.column_config.NumberColumn("Win %",    width="small", format="%.1f%%"),
                "Avg GS":         st.column_config.NumberColumn("Avg GS",   width="small", format="%.2f"),
                "Avg GC":         st.column_config.NumberColumn("Avg GC",   width="small", format="%.2f"),
                "Elo":            st.column_config.NumberColumn("Elo",      width="small"),
                "Attack":         st.column_config.NumberColumn("Attack",   width="small", format="%.4f"),
                "Defence":        st.column_config.NumberColumn("Defence",  width="small", format="%.4f"),
                "Last Match":     st.column_config.TextColumn("Last Match", width="small"),
            },
        )
        st.caption(f"Showing {len(_tdf)} of 48 teams · Last 30 matches per team · {_sort_sel}")

    # ---- TAB 4: DATASET COVERAGE ──────────────────────────────────────────
    # Internal audit runs silently — never displayed to users
    with tab4:
        st.markdown('<div class="section-header">📊 Dataset Coverage</div>',
                    unsafe_allow_html=True)

        # ── Build coverage table directly from historical dataset ────────────
        _all_ds_teams = set(df_raw["home_team"]) | set(df_raw["away_team"])
        _cov_rows = []
        for canonical in QUALIFIED_TEAMS:
            rec     = TEAM_REGISTRY[canonical]
            ds_name = rec["dataset_name"]
            in_ds   = ds_name in _all_ds_teams
            _team_df = df_raw[
                (df_raw["home_team"] == ds_name) | (df_raw["away_team"] == ds_name)
            ].sort_values("date")

            mp = len(_team_df)
            w2 = d2 = l2 = gf2 = ga2 = 0
            for _, _r in _team_df.iterrows():
                if _r["home_team"] == ds_name:
                    gf2 += _r["home_score"]; ga2 += _r["away_score"]
                    if _r["result"]=="H": w2+=1
                    elif _r["result"]=="D": d2+=1
                    else: l2+=1
                else:
                    gf2 += _r["away_score"]; ga2 += _r["home_score"]
                    if _r["result"]=="A": w2+=1
                    elif _r["result"]=="D": d2+=1
                    else: l2+=1

            last = _team_df["date"].max().strftime("%Y-%m-%d") if mp else "N/A"
            elo_val = round(elo_ratings.get(ds_name, 1500))

            _cov_rows.append({
                "Flag":           rec["flag"],
                "Team":           canonical,
                "Confederation":  rec["confederation"],
                "Dataset Name":   ds_name,
                "In Dataset":     "✅" if in_ds else "❌",
                "Matches Played": mp,
                "Wins":           w2,
                "Draws":          d2,
                "Losses":         l2,
                "Goals For":      gf2,
                "Goals Against":  ga2,
                "GD":             gf2 - ga2,
                "Win %":          round(w2/mp*100,1) if mp else 0.0,
                "Last Match":     last,
                "Elo":            elo_val,
            })

        _cov_df = pd.DataFrame(_cov_rows)
        _found  = _cov_df["In Dataset"].eq("✅").sum()

        # ── Summary cards ────────────────────────────────────────────────────
        _high_elo = _cov_df.loc[_cov_df["Elo"].idxmax(), "Team"]
        _low_elo  = _cov_df.loc[_cov_df["Elo"].idxmin(), "Team"]
        _most_exp = _cov_df.loc[_cov_df["Matches Played"].idxmax(), "Team"]
        _least_exp= _cov_df.loc[_cov_df["Matches Played"].idxmin(), "Team"]
        _avg_mp   = round(_cov_df["Matches Played"].mean(), 1)

        _sc = st.columns(4)
        for _col, _lbl, _val, _sub in zip(_sc, [
            "🌍 WC Teams", "✅ In Dataset", "📈 Avg Matches", "🏆 Highest Elo",
        ], [
            "48", str(_found), str(_avg_mp), _high_elo,
        ], [
            "2026 qualified", f"of 48 teams", "per team (all-time)", f"Elo: {_cov_df['Elo'].max()}",
        ]):
            _col.markdown(f"""<div class="metric-card">
                <div class="label">{_lbl}</div>
                <div class="value">{_val}</div>
                <div class="sub">{_sub}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        _sc2 = st.columns(4)
        for _col, _lbl, _val, _sub in zip(_sc2, [
            "📉 Lowest Elo", "🎖️ Most Experienced", "🆕 Least Experienced", "📅 Dataset Matches",
        ], [
            _low_elo, _most_exp, _least_exp,
            f"{len(df_raw):,}",
        ], [
            f"Elo: {_cov_df['Elo'].min()}",
            f"{_cov_df['Matches Played'].max()} matches",
            f"{_cov_df['Matches Played'].min()} matches",
            "total international",
        ]):
            _col.markdown(f"""<div class="metric-card">
                <div class="label">{_lbl}</div>
                <div class="value">{_val}</div>
                <div class="sub">{_sub}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Coverage filters ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Dataset Verification Table</div>',
                    unsafe_allow_html=True)

        _d1, _d2, _d3 = st.columns([2, 2, 2])
        with _d1:
            _dc_conf = st.selectbox("Confederation", ["All"]+sorted(_cov_df["Confederation"].unique().tolist()),
                                    key="dc_conf")
        with _d2:
            _dc_sort_opts = {
                "Elo (desc)":       ("Elo",            False),
                "Matches Played":   ("Matches Played", False),
                "Win % (desc)":     ("Win %",          False),
                "Alphabetical":     ("Team",           True),
            }
            _dc_sort = st.selectbox("Sort by", list(_dc_sort_opts.keys()), key="dc_sort")
        with _d3:
            _dc_search = st.text_input("🔍 Search team", placeholder="Type alias or name…", key="dc_search")

        _tdf2 = _cov_df.copy()
        if _dc_conf != "All":
            _tdf2 = _tdf2[_tdf2["Confederation"] == _dc_conf]
        if _dc_search:
            _tdf2 = _tdf2[_tdf2["Team"].apply(
                lambda t: any(_dc_search.strip().lower() in c for c in [
                    t.lower(),
                    TEAM_REGISTRY[t]["display_name"].lower(),
                    TEAM_REGISTRY[t]["dataset_name"].lower(),
                ] + [k for k,v in TEAM_NAME_MAP.items() if v == TEAM_REGISTRY[t]["dataset_name"]])
            )]

        _dc_sc, _dc_asc = _dc_sort_opts[_dc_sort]
        _tdf2 = _tdf2.sort_values(_dc_sc, ascending=_dc_asc).reset_index(drop=True)

        # Color rows by data quality: green ≥50, yellow 20–49, red <20
        def _cov_style(row):
            mp = row["Matches Played"]
            if mp >= 50:
                return [f"color:#48bb78"] * len(row)   # green
            if mp >= 20:
                return [f"color:#ed8936"] * len(row)   # amber
            return [f"color:#fc8181"] * len(row)        # red

        _cov_display = [
            "Flag","Team","Confederation","Dataset Name","In Dataset",
            "Matches Played","Wins","Draws","Losses",
            "Goals For","Goals Against","GD","Win %","Last Match","Elo",
        ]

        st.dataframe(
            _tdf2[_cov_display].style.apply(_cov_style, axis=1),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Flag":           st.column_config.TextColumn("",            width="small"),
                "Team":           st.column_config.TextColumn("Team",        width="medium"),
                "Confederation":  st.column_config.TextColumn("Conf.",       width="small"),
                "Dataset Name":   st.column_config.TextColumn("Dataset Name",width="medium"),
                "In Dataset":     st.column_config.TextColumn("In DB",       width="small"),
                "Matches Played": st.column_config.NumberColumn("MP",        width="small"),
                "Wins":           st.column_config.NumberColumn("W",         width="small"),
                "Draws":          st.column_config.NumberColumn("D",         width="small"),
                "Losses":         st.column_config.NumberColumn("L",         width="small"),
                "Goals For":      st.column_config.NumberColumn("GF",        width="small"),
                "Goals Against":  st.column_config.NumberColumn("GA",        width="small"),
                "GD":             st.column_config.NumberColumn("GD",        width="small"),
                "Win %":          st.column_config.NumberColumn("Win %",     width="small", format="%.1f%%"),
                "Last Match":     st.column_config.TextColumn("Last Match",  width="small"),
                "Elo":            st.column_config.NumberColumn("Elo",       width="small"),
            },
        )
        st.caption(
            "🟢 Green = ≥50 matches · 🟡 Amber = 20–49 matches · 🔴 Red = <20 matches  "
            f"| Showing {len(_tdf2)} of 48 teams | Source: martj42/international_results"
        )

    # ---- TAB 5: METHODOLOGY ──────────────────────────────────────────────
    with tab5:
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