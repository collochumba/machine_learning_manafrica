"""
PROFESSIONAL STREAMLIT BETTING APP
Complete production implementation with all features
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import json
import html
import textwrap

import fixtures
from config import LEAGUE_CONFIG, SUPPORTED_LEAGUES

# Always load files relative to this script's location,
# regardless of where Streamlit is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_file(name):
    """Return absolute path to a model file in the same folder as app.py."""
    return os.path.join(BASE_DIR, name)

# ============================================================================
# FIXTURE CACHE (deliberately SEPARATE from the training cache — this only
# ever holds the downloaded upcoming-fixtures file + a timestamp, and is
# never read by train.py or mixed with cache/raw_data.pkl / cache/features.pkl
# / processed_data.pkl. Loading fixtures never touches model artifacts.)
# ============================================================================

FIXTURE_CACHE_DIR = os.path.join(BASE_DIR, "cache")
FIXTURE_CACHE_FILE = os.path.join(FIXTURE_CACHE_DIR, "latest_fixtures.pkl")
FIXTURE_REFRESH_HOURS = 6


def save_fixture_cache(raw_df, source_label):
    os.makedirs(FIXTURE_CACHE_DIR, exist_ok=True)
    joblib.dump({
        'raw_df': raw_df,
        'source': source_label,
        'fetched_at': datetime.now(),
    }, FIXTURE_CACHE_FILE)


def load_fixture_cache():
    if os.path.exists(FIXTURE_CACHE_FILE):
        try:
            return joblib.load(FIXTURE_CACHE_FILE)
        except Exception:
            return None
    return None

from models import DixonColesTimeDecay, predict_multiple_fixtures, generate_summary_stats
from corners import predict_corners
from cards import predict_cards
from betting import rank_top_value_bets, simulate_bankroll

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="⚽ Pro Football Betting",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# THEME (Light / Dark)
# ============================================================================
# Stored in session_state so it survives reruns (every widget interaction
# reruns the whole script top-to-bottom). Rendered as the very first sidebar
# element below so it's always visible, before any other control.

if "app_theme" not in st.session_state:
    st.session_state["app_theme"] = "Light"

with st.sidebar:
    st.session_state["app_theme"] = st.radio(
        "🎨 Theme",
        ["Light", "Dark"],
        index=0 if st.session_state["app_theme"] == "Light" else 1,
        horizontal=True,
        key="theme_toggle",
    )

_THEME = st.session_state["app_theme"]

# Palette per theme. Kept as plain CSS custom properties on :root so every
# rule below just references var(--xxx) once and never needs an if/else.
if _THEME == "Dark":
    _PALETTE = {
        "bg": "#0e1117",
        "surface": "#161b22",
        "surface-alt": "#1c2128",
        "border": "#30363d",
        "text": "#e6edf3",
        "text-muted": "#8b949e",
        "banner-grad": "linear-gradient(135deg, #2c3e91 0%, #4a2f7a 100%)",
        "home-color": "#ff8a3d",
        "draw-color": "#4b5563",
        "away-color": "#2f9e44",
        "btts-yes": "#1f7a4d",
        "btts-no": "#5a3a1a",
        "value-bg": "linear-gradient(135deg, #123a24 0%, #0f2e1c 100%)",
        "value-border": "#2ea043",
        "conf-bg": "linear-gradient(135deg, #0d3a4a 0%, #0a2e3b 100%)",
        "conf-border": "#39b8d6",
        "warn-bg": "linear-gradient(135deg, #4a3a10 0%, #3a2d0c 100%)",
        "warn-border": "#d4a017",
        "metric-bg": "#1c2128",
    }
else:
    _PALETTE = {
        "bg": "#ffffff",
        "surface": "#ffffff",
        "surface-alt": "#f8f9fa",
        "border": "#dee2e6",
        "text": "#1f2937",
        "text-muted": "#6c757d",
        "banner-grad": "linear-gradient(135deg, #3949ab 0%, #6a3fa0 100%)",
        "home-color": "#f0862f",
        "draw-color": "#9aa1a8",
        "away-color": "#2f9e44",
        "btts-yes": "#28a745",
        "btts-no": "#e8ac1e",
        "value-bg": "linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)",
        "value-border": "#28a745",
        "conf-bg": "linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%)",
        "conf-border": "#17a2b8",
        "warn-bg": "linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%)",
        "warn-border": "#ffc107",
        "metric-bg": "#f8f9fa",
    }

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(f"""
<style>
:root {{
    --app-bg: {_PALETTE['bg']};
    --app-surface: {_PALETTE['surface']};
    --app-surface-alt: {_PALETTE['surface-alt']};
    --app-border: {_PALETTE['border']};
    --app-text: {_PALETTE['text']};
    --app-text-muted: {_PALETTE['text-muted']};
    --app-banner-grad: {_PALETTE['banner-grad']};
    --app-home: {_PALETTE['home-color']};
    --app-draw: {_PALETTE['draw-color']};
    --app-away: {_PALETTE['away-color']};
    --app-btts-yes: {_PALETTE['btts-yes']};
    --app-btts-no: {_PALETTE['btts-no']};
    --app-value-bg: {_PALETTE['value-bg']};
    --app-value-border: {_PALETTE['value-border']};
    --app-conf-bg: {_PALETTE['conf-bg']};
    --app-conf-border: {_PALETTE['conf-border']};
    --app-warn-bg: {_PALETTE['warn-bg']};
    --app-warn-border: {_PALETTE['warn-border']};
    --app-metric-bg: {_PALETTE['metric-bg']};
}}

/* App-wide background/text so Dark mode actually looks dark, not just the cards */
.stApp {{
    background-color: var(--app-bg);
    color: var(--app-text);
}}
section[data-testid="stSidebar"] {{
    background-color: var(--app-surface-alt);
}}

/* Native Streamlit controls: explicitly set foreground colors so the
   application theme cannot leave labels/captions invisible. These selectors
   are scoped to Streamlit's widget containers and do not affect custom
   prediction-card badges or banners. */
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label,
.stApp [data-testid="stWidgetLabel"] span,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricDelta"],
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] summary p,
.stApp [data-testid="stTab"] p,
.stApp [data-testid="stRadio"] label,
.stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stSelectbox"] label,
.stApp [data-testid="stMultiSelect"] label,
.stApp [data-testid="stSlider"] label,
.stApp [data-testid="stNumberInput"] label,
.stApp [data-testid="stTextInput"] label,
.stApp [data-testid="stTextArea"] label {{
    color: var(--app-text) !important;
}}

.stApp [data-testid="stCaptionContainer"] p {{
    color: var(--app-text-muted) !important;
}}

.stApp [data-testid="stRadio"] [role="radiogroup"] label,
.stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stSelectbox"] [role="combobox"],
.stApp [data-testid="stMultiSelect"] [role="combobox"],
.stApp input,
.stApp textarea {{
    color: var(--app-text) !important;
}}

.stApp [data-testid="stAlert"] {{
    color: var(--app-text) !important;
}}

/* Main styling */
.main {{
    padding: 0rem 1rem;
}}

/* Value bet highlighting */
.value-bet {{
    background: var(--app-value-bg);
    color: var(--app-text);
    padding: 15px;
    border-radius: 8px;
    border-left: 5px solid var(--app-value-border);
    margin: 10px 0;
}}

.high-confidence {{
    background: var(--app-conf-bg);
    color: var(--app-text);
    padding: 15px;
    border-radius: 8px;
    border-left: 5px solid var(--app-conf-border);
    margin: 10px 0;
}}

.low-confidence {{
    background: var(--app-warn-bg);
    color: var(--app-text);
    padding: 15px;
    border-radius: 8px;
    border-left: 5px solid var(--app-warn-border);
    margin: 10px 0;
}}

/* Metrics */
div[data-testid="metric-container"] {{
    background-color: var(--app-metric-bg);
    border: 1px solid var(--app-border);
    padding: 10px;
    border-radius: 5px;
}}

/* Headers */
h1 {{ color: var(--app-text); font-weight: 700; }}
h2 {{ color: var(--app-text); font-weight: 600; }}
h3 {{ color: var(--app-text-muted); font-weight: 500; }}

/* Buttons */
.stButton>button {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
}}

.stButton>button:hover {{
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}}

/* ============================================================
   MATCH CARD — "Probability Pro"-style layout: a league banner,
   the fixture name/kickoff, a 3-way Winning Prediction bar, and
   compact Goal-Goal (BTTS) badges, mirroring the reference UI.
   ============================================================ */

.league-banner {{
    background: var(--app-banner-grad);
    color: white;
    padding: 8px 16px;
    border-radius: 8px 8px 0 0;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}}

.league-banner {{
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.league-count {{
    font-size: 0.78rem;
    font-weight: 600;
    opacity: 0.9;
}}

.match-card {{
    background: var(--app-surface);
    border: 1px solid var(--app-border);
    border-radius: 0 0 10px 10px;
    padding: 14px 16px 16px 16px;
    margin-bottom: 4px;
}}

.match-card .fixture-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 10px;
}}

.match-card .fixture-teams {{
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--app-text);
}}

.match-card .fixture-meta {{
    color: var(--app-text-muted);
    font-size: 0.85rem;
}}

.pred-label {{
    color: var(--app-text-muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}}

.pred-bar {{
    display: flex;
    width: 100%;
    border-radius: 6px;
    overflow: hidden;
    height: 38px;
    margin-bottom: 10px;
}}

.pred-seg {{
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 0.85rem;
    white-space: nowrap;
}}

.pred-seg.home {{ background-color: var(--app-home); }}
.pred-seg.draw {{ background-color: var(--app-draw); }}
.pred-seg.away {{ background-color: var(--app-away); }}
.pred-seg.winner {{ font-weight: 900; box-shadow: inset 0 0 0 3px rgba(255,255,255,.95); }}
.pred-seg:not(.winner) {{ opacity: 0.82; }}
.winner-mark {{ margin-right: 4px; }}

.goalgoal-row {{
    display: flex;
    gap: 8px;
    align-items: center;
}}

.badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 14px;
    font-size: 0.78rem;
    font-weight: 700;
    color: white;
}}

.badge.yes {{ background-color: var(--app-btts-yes); }}
.badge.no {{ background-color: var(--app-btts-no); }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_all_models():
    """Load all trained models and data."""
    try:
        final_model = joblib.load(load_file("final_model.pkl"))
        dc_models = joblib.load(load_file("dc_models.pkl"))
        feature_cols = joblib.load(load_file("feature_cols.pkl"))
        df = joblib.load(load_file("processed_data.pkl"))
        team_mapping = joblib.load(load_file("team_mapping.pkl"))
        all_teams = joblib.load(load_file("all_teams.pkl"))

        # NEW (optional): current-season team metadata produced by train.py.
        # Older model artifacts won't have this file — degrade gracefully
        # rather than breaking the app.
        try:
            current_teams_meta = joblib.load(load_file("current_teams.pkl"))
        except Exception:
            current_teams_meta = None

        # NEW (optional): dedicated corner model (Task 2). Older model
        # artifacts won't have this either — the app still works fully for
        # 1X2/goals/etc, it just won't show the Corner Predictions section.
        try:
            corner_bundle = joblib.load(load_file("corner_model.pkl"))
            corner_models = corner_bundle.get('models')
        except Exception:
            corner_models = None

        try:
            card_bundle = joblib.load(load_file("card_model.pkl"))
            card_models = card_bundle.get('models')
        except Exception:
            card_models = None

        return final_model, dc_models, feature_cols, df, team_mapping, all_teams, current_teams_meta, corner_models, card_models, None

    except Exception as e:
        return None, None, None, None, None, None, None, None, None, str(e)

# Automatic fixture refresh. refresh_fixture_cache() already checks the
# cache's age internally and returns immediately (no network call) when the
# cache is still fresh — so it is cheap and safe to call on every Streamlit
# rerun, not just once per session. The previous 'startup_fixture_refresh_done'
# session-state gate ran this exactly once per browser session: if a tab was
# left open past the 6-hour freshness window, the app would keep serving an
# aging cache until the page was reloaded, since nothing re-checked the age
# after that first run. Calling it unconditionally on every rerun instead
# means: cache fresh -> instant no-op; cache stale -> attempt refresh, and
# fall back to the last known-good cache on any network/parse failure. It
# never triggers model training.
try:
    _startup_cache, _startup_status = fixtures.refresh_fixture_cache(
        FIXTURE_CACHE_FILE, max_age_hours=FIXTURE_REFRESH_HOURS, force=False
    )
    st.session_state['startup_fixture_refresh_status'] = _startup_status
except Exception as _exc:
    st.session_state['startup_fixture_refresh_status'] = {'error': str(_exc)}

# Load models
with st.spinner("🔄 Loading trained models..."):
    final_model, dc_models, feature_cols, df, team_mapping, all_teams, current_teams_meta, corner_models, card_models, error = load_all_models()

# Lightweight system health/status panel. This does not trigger training.
with st.sidebar.expander("📡 Data & Model Status", expanded=True):
    _startup_status = st.session_state.get('startup_fixture_refresh_status', {})
    if _startup_status.get('error'):
        st.warning(f"🟡 Fixture refresh issue: {_startup_status['error']}")
    fixture_cache = load_fixture_cache()
    if fixture_cache is None:
        st.write("Fixtures: 🔴 unavailable")
    else:
        age_h = (datetime.now() - fixture_cache['fetched_at']).total_seconds() / 3600.0
        status_icon = "🟢" if age_h < FIXTURE_REFRESH_HOURS else "🟡"
        st.write(f"Fixtures: {status_icon} {age_h:.1f}h old")
        st.caption(f"Updated: {fixture_cache['fetched_at'].strftime('%Y-%m-%d %H:%M')}")
        try:
            live_raw, _, _ = fixtures.extract_supported_fixtures(fixture_cache['raw_df'])
            st.write(f"Upcoming fixtures: {len(live_raw)}")
        except Exception:
            st.write("Upcoming fixtures: unavailable")

    configured_count = len(SUPPORTED_LEAGUES)
    trained_ml = int(df['League'].nunique()) if df is not None and 'League' in df.columns else 0
    # Count only DC models that actually converged as "trained" coverage.
    # A non-converged model is still deployed in dc_models (so
    # ensemble_prediction() can fall back safely per-fixture at predict
    # time), but it must not be counted as if it were a normal, reliable
    # trained model in this summary.
    trained_dc_total = len(dc_models) if isinstance(dc_models, dict) else 0
    dc_non_converged_leagues = sorted(
        lg for lg, m in (dc_models or {}).items() if not getattr(m, 'converged_', True)
    ) if isinstance(dc_models, dict) else []
    trained_dc = trained_dc_total - len(dc_non_converged_leagues)
    trained_corners = len(corner_models) if isinstance(corner_models, dict) else 0
    trained_cards = len(card_models) if isinstance(card_models, dict) else 0
    corners_missing_leagues = sorted(
        set(SUPPORTED_LEAGUES) - set(corner_models.keys())
    ) if isinstance(corner_models, dict) else []
    cards_missing_leagues = sorted(
        set(SUPPORTED_LEAGUES) - set(card_models.keys())
    ) if isinstance(card_models, dict) else []

    st.write(f"Configured leagues: {configured_count}")
    st.write(f"Historical data leagues: {trained_ml}/{configured_count}")
    st.write(f"ML models: {trained_ml}/{configured_count}")
    st.write(f"Dixon-Coles models: {trained_dc}/{configured_count} converged")
    st.write(f"Corner models: {trained_corners}/{configured_count}")
    st.write(f"Card models: {trained_cards}/{configured_count}")

    # Coverage numbers above are always derived from the actually-loaded
    # artifacts (dc_models/corner_models/card_models in memory right now),
    # not from training_manifest.json — an artifact reflects exactly what
    # got deployed this session, whereas a manifest file can silently go
    # stale relative to the artifacts if only one of the two ever gets
    # regenerated. The manifest is still useful as a timestamped record of
    # *when* and *from what dataset* those artifacts were produced, so it's
    # shown here for context only, never as the source of the counts above.
    try:
        with open(load_file('training_manifest.json')) as _mf:
            _manifest = json.load(_mf)
        _trained_at = _manifest.get('training_timestamp', '')[:19].replace('T', ' ')
        _match_count = _manifest.get('dataset', {}).get('match_count')
        if _trained_at:
            st.caption(
                f"Artifacts trained: {_trained_at}"
                + (f" · {_match_count:,} matches" if _match_count else "")
            )
    except Exception:
        pass

    if dc_non_converged_leagues:
        st.caption(
            f"⚠️ Dixon-Coles non-converged (safe fallback used): "
            f"{', '.join(dc_non_converged_leagues)}"
        )
    if corners_missing_leagues:
        st.caption(f"⚠️ No corner model: {', '.join(corners_missing_leagues)}")
    if cards_missing_leagues:
        st.caption(f"⚠️ No card model: {', '.join(cards_missing_leagues)}")

# Error handling
if error:
    st.error(f"""
    ## ❌ Error Loading Models
    
    **Error:** {error}
    
    **Solution:**
    1. Run `python train.py` to train models
    2. Ensure these files exist:
       - final_model.pkl
       - dc_models.pkl  
       - feature_cols.pkl
       - processed_data.pkl
       - team_mapping.pkl
       - all_teams.pkl
    3. Restart the app
    """)
    st.stop()

st.success("✅ Models loaded successfully!")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("🎯 Value Bet Filters")

    # FIX 2: Replace invalid format="%.0%%" with valid format strings
    min_prob = st.slider(
        "Minimum Probability",
        min_value=0.30,   # Raised floor — below 30% signal is unreliable
        max_value=0.70,
        value=0.45,
        step=0.05,
        format="%.2f",
        help="Only show bets with model probability above this threshold. Below 0.30 the signal is unreliable."
    )

    min_ev = st.slider(
        "Minimum Expected Value",
        min_value=0.02,   # Hard floor — never allow zero/negative EV bets through
        max_value=0.20,   # Raised ceiling for more range
        value=0.05,       # Conservative default
        step=0.01,
        format="%.2f",
        help="Only show bets with positive EV above this threshold. Never set below 0.02."
    )

    st.markdown("---")

    st.subheader("💰 Bankroll Settings")

    bankroll = st.number_input(
        "Initial Bankroll",
        min_value=100,
        max_value=100000,
        value=1000,
        step=100,
        help="Your starting bankroll for simulation"
    )

    st.markdown("---")

    st.subheader("📊 System Info")

    st.info(f"""
    **Matches:** {len(df):,}
    
    **Date range:** {df['Date'].min().date()} to {df['Date'].max().date()}
    
    **Features:** {len(feature_cols)}
    
    **Leagues:** {trained_ml} configured ({trained_dc} with a converged Dixon-Coles model)
    
    **Teams:** {df['HomeTeam'].nunique()}
    """)

    st.info(f"⚽ **Corner model:** {'✅ loaded (' + str(len(corner_models)) + '/' + str(configured_count) + ' leagues)' if corner_models else '⚠️ not available — run train.py to generate corner_model.pkl'}")
    st.info(f"🟨 **Card model:** {'✅ loaded (' + str(len(card_models)) + '/' + str(configured_count) + ' leagues)' if card_models else '⚠️ not available — run train.py to generate card_model.pkl'}")

    # NEW: training-history / current-season metadata, if available
    # (produced by the updated train.py — degrades gracefully if the model
    # artifacts predate this).
    if current_teams_meta:
        seasons_req = current_teams_meta.get('seasons_requested', [])
        loaded = current_teams_meta.get('seasons_loaded', {})
        current_by_league = current_teams_meta.get('current_teams_by_league', {})
        latest_by_league = current_teams_meta.get('latest_season_by_league', {})

        n_loaded_total = sum(len(v) for v in loaded.values())
        n_requested_total = len(seasons_req) * max(len(loaded), 1)

        current_teams_lines = "\n    ".join(
            f"- **{lg}** ({latest_by_league.get(lg, '?')}): {len(teams)} teams"
            for lg, teams in current_by_league.items()
        )

        st.info(f"""
        **Training history:** {len(seasons_req)} seasons requested ({seasons_req[0] if seasons_req else '?'}–{seasons_req[-1] if seasons_req else '?'})

        **Current teams loaded:**
        {current_teams_lines}
        """)

    st.markdown("---")

    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 12px;">
    <strong>⚽ Pro Football Betting</strong><br>
    XGBoost + Dixon-Coles<br>
    <small>For educational purposes only</small>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN INTERFACE
# ============================================================================

st.title("⚽ PROFESSIONAL FOOTBALL BETTING MODEL")
st.markdown("**AI-Powered Value Bet Finder** | Dixon-Coles + XGBoost Ensemble")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Predictions",
    "🏆 Top Bets",
    "💰 Bankroll Simulator",
    "📊 Statistics",
    "ℹ️ Guide"
])

# ============================================================================
# TAB 1: PREDICTIONS
# ============================================================================

with tab1:
    st.header("📋 Input Fixtures")

    input_mode = st.radio(
        "How do you want to provide fixtures?",
        ["🔄 Load Latest Fixtures", "📝 Paste Fixtures Manually"],
        horizontal=True,
    )

    # These get set by whichever mode runs below; the prediction block
    # further down triggers on (run_predictions and selected_fixtures),
    # regardless of which input method populated `selected_fixtures`.
    #
    # NOTE: deliberately named `selected_fixtures`, not `fixtures` — the
    # `fixtures` name at this scope is the fixtures.py MODULE imported at
    # the top of this file (fixtures.resolve_fixture, fixtures.refresh_
    # fixture_cache, etc. are called throughout this tab); reusing it for
    # this list would silently shadow the module for the rest of the script.
    selected_fixtures = []
    run_predictions = False

    # ========================================================================
    # MODE: LOAD LATEST FIXTURES (football-data.co.uk)
    # ========================================================================
    if input_mode == "🔄 Load Latest Fixtures":

        st.caption(
            "Source: football-data.co.uk latest fixtures. "
            "Fixture odds reflect the latest downloadable snapshot and may "
            "not be live bookmaker odds."
        )

        loader_col1, loader_col2 = st.columns(2)

        with loader_col1:
            load_clicked = st.button("🔄 Load Latest Fixtures", type="primary", width='stretch')
        with loader_col2:
            uploaded_fixture_file = st.file_uploader(
                "...or upload a fixtures.csv / fixtures.xlsx file",
                type=['csv', 'xlsx'],
                key='fixture_upload',
            )

        # Handle download button
        if load_clicked:
            with st.spinner("Downloading latest fixtures..."):
                content, err = fixtures.fetch_fixtures_bytes(fixtures.FIXTURES_CSV_URL)
                if err:
                    st.error(f"❌ {err}")
                    st.info("You can still use **📝 Paste Fixtures Manually** instead.")
                else:
                    raw_df, perr = fixtures.parse_fixture_bytes(content, "fixtures.csv")
                    if perr:
                        st.error(f"❌ {perr}")
                    else:
                        save_fixture_cache(raw_df, "football-data.co.uk (downloaded)")
                        st.success("✅ Latest fixtures downloaded")

        # Handle file upload
        if uploaded_fixture_file is not None:
            content = uploaded_fixture_file.read()
            raw_df, perr = fixtures.parse_fixture_bytes(content, uploaded_fixture_file.name)
            if perr:
                st.error(f"❌ {perr}")
            else:
                save_fixture_cache(raw_df, f"uploaded file: {uploaded_fixture_file.name}")
                st.success("✅ Fixture file loaded")

        # Automatic refresh: do not delete the last-good cache. If the network
        # is unavailable, fixture_loader returns the previous cache and the UI
        # explicitly reports that it is being used.
        cached, refresh_status = fixtures.refresh_fixture_cache(
            FIXTURE_CACHE_FILE, max_age_hours=FIXTURE_REFRESH_HOURS, force=load_clicked
        )

        if refresh_status.get('error'):
            st.warning(
                f"🟡 Fixture refresh failed. Using last known good cache. "
                f"{refresh_status['error']}"
                if cached is not None else
                f"🔴 Fixture refresh failed and no cached fixtures are available. "
                f"{refresh_status['error']}"
            )

        if cached is None:
            st.info("No fixtures loaded yet. Click **🔄 Refresh Fixtures** or upload a file above.")
        else:
            age = refresh_status.get('age_hours')
            age_text = f"{age:.1f}h old" if isinstance(age, (int, float)) else "age unknown"
            refresh_col1, refresh_col2 = st.columns([3, 1])
            with refresh_col1:
                st.caption(
                    f"Loaded from: {cached['source']} · "
                    f"{cached['fetched_at'].strftime('%Y-%m-%d %H:%M:%S')} · {age_text}"
                )
            with refresh_col2:
                if st.button("🔄 Refresh Fixtures", width='stretch'):
                    refreshed, status = fixtures.refresh_fixture_cache(
                        FIXTURE_CACHE_FILE, max_age_hours=FIXTURE_REFRESH_HOURS, force=True
                    )
                    if status.get('error'):
                        st.warning(f"Refresh failed; retaining last-good cache: {status['error']}")
                    else:
                        st.success("✅ Fixture cache refreshed")
                    st.rerun()

            try:
                supported_raw, excluded_counts, total_rows = fixtures.extract_supported_fixtures(cached['raw_df'])
            except ValueError as e:
                st.error(f"❌ {e}")
                supported_raw, excluded_counts, total_rows = [], {}, 0

            # Resolve teams against CURRENT-season teams (not historical all_teams),
            # corrected for the season-rollover staleness gap via the verified
            # SEASON_ROLLOVER_OVERRIDES table (see fixtures.py).
            current_teams_by_league = (current_teams_meta or {}).get('current_teams_by_league', {})
            current_teams_by_league, season_override_changes = fixtures.apply_season_overrides(
                current_teams_by_league
            )

            fixture_candidates = {}
            for f in supported_raw:
                fixture_candidates.setdefault(f['league'], set()).update([f['home_raw'], f['away_raw']])
            fixture_candidates = {k: sorted(v) for k, v in fixture_candidates.items()}

            resolved_fixtures = []
            for f in supported_raw:
                res = fixtures.resolve_fixture(
                    f['league'], f['home_raw'], f['away_raw'], current_teams_by_league, all_teams,
                    fixture_candidates=fixture_candidates
                )
                res['date'] = f['date']
                res['time'] = f['time']
                res['odds'] = f['odds']
                resolved_fixtures.append(res)

            n_valid = sum(1 for r in resolved_fixtures if r['status'] == 'valid')
            n_review = len(resolved_fixtures) - n_valid
            excluded_named = fixtures.summarize_excluded(excluded_counts)

            st.markdown("#### ✅ Latest fixtures loaded")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total in file", total_rows)
            m2.metric("Supported fixtures", len(resolved_fixtures))
            m3.metric("Resolved", n_valid)
            m4.metric("Needs review", n_review)

            if excluded_named:
                with st.expander(f"ℹ️ Excluded {sum(excluded_named.values())} fixtures from unsupported leagues"):
                    for league_name, count in sorted(excluded_named.items(), key=lambda x: -x[1]):
                        st.write(f"- {league_name}: {count}")

            if season_override_changes:
                with st.expander("ℹ️ Season-rollover corrections applied to current team lists"):
                    st.caption(
                        "current_teams.pkl lags one season behind right after promotion/"
                        "relegation (see fixtures.py docstring). These verified "
                        "corrections are applied on top of it before resolving fixtures:"
                    )
                    for c in season_override_changes:
                        st.write(f"- {c}")

            if n_review > 0:
                with st.expander(f"⚠️ {n_review} fixture(s) need review", expanded=True):
                    for r in resolved_fixtures:
                        if r['status'] != 'needs_review':
                            continue
                        bad_side = 'home' if not r['home']['resolved'] else 'away'
                        bad = r[bad_side]
                        st.warning(
                            f"**{r['league']}**: {r['home_raw']} vs {r['away_raw']}\n\n"
                            f"{bad['message']}"
                        )
                    st.caption(
                        "Unrecognised teams are never auto-substituted. Fix the name in "
                        "**📝 Paste Fixtures Manually** if you want to predict this match."
                    )

            valid_fixtures = [r for r in resolved_fixtures if r['status'] == 'valid']

            if valid_fixtures:
                st.markdown("#### Select fixtures to predict")

                by_league = {}
                for r in valid_fixtures:
                    by_league.setdefault(r['league'], []).append(r)

                select_all = st.checkbox("☑ Select All", value=True, key='select_all_fixtures')

                selected_keys = set()
                for league_name in SUPPORTED_LEAGUES:
                    league_fixtures = by_league.get(league_name)
                    if not league_fixtures:
                        continue

                    league_select = st.checkbox(
                        f"Select {league_name} ({len(league_fixtures)})",
                        value=select_all,
                        key=f'select_league_{league_name}',
                    )
                    st.markdown(f"**{league_name}**")
                    for i, r in enumerate(league_fixtures):
                        odds_label = "odds available" if r['odds'] else "odds unavailable"
                        label = f"{r['home']['resolved']} vs {r['away']['resolved']}  ·  {odds_label}"
                        checked = st.checkbox(label, value=league_select, key=f'fx_{league_name}_{i}')
                        if checked:
                            selected_keys.add((league_name, i))

                    for i, r in enumerate(league_fixtures):
                        if (league_name, i) in selected_keys:
                            fx = {'league': league_name, 'home': r['home']['resolved'], 'away': r['away']['resolved']}
                            if r.get('date'):
                                fx['date'] = r['date']
                            if r.get('time'):
                                fx['time'] = r['time']
                            if r['odds']:
                                fx['odds'] = r['odds']
                            selected_fixtures.append(fx)

                predict_clicked = st.button("⚽ Predict Selected Fixtures", type="primary", width='stretch')
                run_predictions = predict_clicked and bool(selected_fixtures)
                if predict_clicked and not selected_fixtures:
                    st.warning("No fixtures selected.")

    # ========================================================================
    # MODE: PASTE FIXTURES MANUALLY (unchanged backup workflow)
    # ========================================================================
    else:
        st.markdown("""
        **Format:** `League, Home Team, Away Team, Home Odds, Draw Odds, Away Odds`
        
        **Supported Leagues:**
        - Premier League
        - Championship
        - La Liga
        - Segunda Division
        - Serie A
        - Serie B
        - Bundesliga
        - 2. Bundesliga
        - Ligue 1
        - Ligue 2
        - Eredivisie
        - Belgian Pro League
        - Primeira Liga
        - Super Lig
        - Greek Super League
        
        **Example:**
        ```
        Premier League, Arsenal, Chelsea, 2.10, 3.40, 3.50
        La Liga, Real Madrid, Barcelona, 1.95, 3.60, 3.80
        Serie A, Juventus, Inter, 2.30, 3.20, 3.30
        ```
        """)

        input_text = st.text_area(
            "Paste fixtures here (one per line):",
            height=250,
            placeholder="Premier League, Arsenal, Chelsea, 2.10, 3.40, 3.50\nLa Liga, Real Madrid, Barcelona, 1.95, 3.60, 3.80"
        )

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            predict_button = st.button("🔮 PREDICT ALL", type="primary", width='stretch')

        with col2:
            clear_button = st.button("🗑️ Clear", width='stretch')
            if clear_button:
                st.rerun()

        with col3:
            lines = [l.strip() for l in input_text.split('\n') if l.strip()] if input_text else []
            st.info(f"📊 {len(lines)} fixtures ready")

    # ============================================================================
    # MANUAL-PASTE PARSING (only runs in paste mode; appends into the shared
    # `selected_fixtures` list so the prediction block below is common to
    # both modes)
    # ============================================================================

    if input_mode == "📝 Paste Fixtures Manually" and predict_button and input_text:

        parse_errors = []

        for i, line in enumerate(input_text.split('\n'), 1):
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            try:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) != 6:
                    parse_errors.append(f"Line {i}: Expected 6 values, got {len(parts)}")
                    continue

                league, home, away = parts[0], parts[1], parts[2]

                # FIX 4: Safe float parsing with clear error messages
                try:
                    odds_h = float(parts[3])
                    odds_d = float(parts[4])
                    odds_a = float(parts[5])
                except ValueError as ve:
                    parse_errors.append(f"Line {i}: Invalid odds value — {ve}")
                    continue

                # Basic sanity check on odds
                if any(o <= 1.0 for o in [odds_h, odds_d, odds_a]):
                    parse_errors.append(
                        f"Line {i}: Odds must be > 1.0 (got H={odds_h}, D={odds_d}, A={odds_a})"
                    )
                    continue

                # FIX: manual-paste fixtures used to skip team_normalization
                # entirely and flow straight into prediction, where
                # predict.py's own (now-removed) fuzzy fallback could
                # silently substitute an unrelated team. Route through the
                # SAME strict resolver used for the "Load Latest Fixtures"
                # flow — fixtures.py is the one authoritative
                # source of truth for every entry point into this app.
                current_teams_by_league_manual = (current_teams_meta or {}).get('current_teams_by_league', {})
                current_teams_by_league_manual, _ = fixtures.apply_season_overrides(
                    current_teams_by_league_manual
                )
                res = fixtures.resolve_fixture(
                    league, home, away, current_teams_by_league_manual, all_teams,
                    fixture_candidates={league: [home, away]}
                )
                if res['status'] != 'valid':
                    bad_side = 'home' if not res['home']['resolved'] else 'away'
                    parse_errors.append(f"Line {i}: {res[bad_side]['message']}")
                    continue

                selected_fixtures.append({
                    'league': league,
                    'home': res['home']['resolved'],
                    'away': res['away']['resolved'],
                    'odds': {
                        'Home': odds_h,
                        'Draw': odds_d,
                        'Away': odds_a
                    }
                })

            except Exception as e:
                parse_errors.append(f"Line {i}: {str(e)}")

        # Show parse errors
        if parse_errors:
            with st.expander("⚠️ Parse Errors", expanded=True):
                for err in parse_errors:
                    st.warning(err)

        if not selected_fixtures:
            st.error("❌ No valid fixtures to predict!")

        run_predictions = bool(selected_fixtures)

    # ============================================================================
    # PREDICTION LOGIC (common to both input modes)
    # ============================================================================

    if run_predictions and selected_fixtures:

        # Generate predictions
        with st.spinner(f"🔄 Analyzing {len(selected_fixtures)} fixtures..."):

            # FIX 1: Pass all_teams as positional argument per the required signature:
            # predict_multiple_fixtures(selected_fixtures, final_model, dc_models,
            #                           feature_cols, df, team_mapping, all_teams,
            #                           min_prob, min_ev)
            try:
                # FIX (bug #27): predict_multiple_fixtures returns
                # (results, errors, warnings_collected) — errors second,
                # warnings third. This used to be unpacked backwards,
                # which silently swapped the "Warnings" and "Errors"
                # expanders below.
                results, prediction_errors, prediction_warnings = predict_multiple_fixtures(
                    selected_fixtures,
                    final_model,
                    dc_models,
                    feature_cols,
                    df,
                    team_mapping,
                    all_teams,
                    min_prob=min_prob,
                    min_ev=min_ev
                )
            except Exception as e:
                st.error(f"❌ Prediction pipeline error: {e}")
                st.stop()

        # Aggregate warnings instead of repeating the same message once per
        # fixture (e.g. "Dixon-Coles model for Segunda Division did not
        # converge..." showing up 20+ times, once per Segunda Division
        # fixture). Group by the exact message text and show each unique
        # warning once with how many fixtures it applied to.
        if prediction_warnings:
            warning_counts = {}
            for warn in prediction_warnings:
                text = (
                    f"{warn.get('fixture', 'Unknown')}: {warn.get('warning', warn)}"
                    if isinstance(warn, dict) else str(warn)
                )
                warning_counts[text] = warning_counts.get(text, 0) + 1

            total_unique = len(warning_counts)
            with st.expander(
                f"⚠️ Model warnings ({total_unique} unique, "
                f"{len(prediction_warnings)} total)",
                expanded=False,
            ):
                for text, count in sorted(
                    warning_counts.items(), key=lambda kv: -kv[1]
                ):
                    suffix = f" — affected {count} fixtures" if count > 1 else ""
                    st.warning(f"{text}{suffix}")

        # FIX 4: Display prediction errors cleanly
        if prediction_errors:
            with st.expander("❌ Prediction Errors", expanded=False):
                for err in prediction_errors:
                    if isinstance(err, dict):
                        st.error(f"{err.get('fixture', 'Unknown')}: {err.get('error', err)}")
                    else:
                        st.error(str(err))

        if not results:
            st.error("❌ No successful predictions!")

        else:
            st.success(f"✅ Predicted {len(results)} fixtures successfully!")

            # Persist immediately. Rendering must not depend on optional corner/card
            # models or summary/ranking calculations succeeding.
            st.session_state['results'] = results

            # ====================================================================
            # MATCH PREDICTIONS — Probability-Pro style cards
            # ====================================================================
            st.markdown("---")
            st.header("🎯 Match Predictions")
            st.caption(f"Showing {len(results)} successful predictions")

            def _safe_probability(value, default=0.0):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    return default
                return value if np.isfinite(value) else default

            # Group results by league so each configured league gets ONE
            # heading, with all of that league's fixtures underneath it.
            # This preserves the fixture order inside each league and means
            # 15 active leagues = at most 15 league headings.
            grouped_results = {}
            for result in results:
                league_key = str(result.get('league', 'Unknown league'))
                grouped_results.setdefault(league_key, []).append(result)

            # Render configured leagues first, then any unexpected league names
            # at the end instead of silently dropping them.
            league_order = [lg for lg in SUPPORTED_LEAGUES if lg in grouped_results]
            league_order += [lg for lg in grouped_results if lg not in league_order]

            display_index = 0
            for league_name in league_order:
                league_results = grouped_results[league_name]
                safe_league = html.escape(league_name)

                # ONE league heading for the whole group.
                st.markdown(textwrap.dedent(f"""
                    <div class="league-banner">
                        <span>⚽ {safe_league}</span>
                        <span class="league-count">{len(league_results)} match{'es' if len(league_results) != 1 else ''}</span>
                    </div>
                """).strip(), unsafe_allow_html=True)

                for result in league_results:
                    display_index += 1
                    try:
                        home = str(result.get('home', 'Unknown home'))
                        away = str(result.get('away', 'Unknown away'))
                        league = league_name
                        value_bets = result.get('value_bets') or []
                        has_value = bool(value_bets)
                        confidence = _safe_probability(result.get('confidence'))
    
                        ph = max(0.0, _safe_probability(result.get('prob_home')))
                        pd_ = max(0.0, _safe_probability(result.get('prob_draw')))
                        pa = max(0.0, _safe_probability(result.get('prob_away')))
                        total = ph + pd_ + pa
                        if total <= 0:
                            ph = pd_ = pa = 1.0 / 3.0
                        else:
                            ph, pd_, pa = ph / total, pd_ / total, pa / total
    
                        w_h, w_d, w_a = (max(x, 0.06) for x in (ph, pd_, pa))
                        w_sum = w_h + w_d + w_a
                        w_h, w_d, w_a = (100 * x / w_sum for x in (w_h, w_d, w_a))
                        winner = int(np.argmax([ph, pd_, pa]))
    
                        btts = result.get('market_probs')
                        if not isinstance(btts, dict):
                            btts = {}
                        yes_raw = btts.get('BTTS Yes')
                        no_raw = btts.get('BTTS No')
                        yes = _safe_probability(yes_raw) if yes_raw is not None else None
                        no = _safe_probability(no_raw) if no_raw is not None else None
    
                        goalgoal_html = ""
                        if yes is not None or no is not None:
                            yes_v = yes if yes is not None else 0.0
                            no_v = no if no is not None else 0.0
                            gg_yes = yes_v >= no_v
                            gg_prob = max(yes_v, no_v)
                            goalgoal_html = textwrap.dedent(f"""
                                <div class="goalgoal-row">
                                    <span class="pred-label" style="margin-bottom:0;">GOAL GOAL:</span>
                                    <span class="badge {'yes' if gg_yes else 'no'}">
                                        {'YES' if gg_yes else 'NO'} {gg_prob:.0%}
                                    </span>
                                </div>
                            """).strip()
    
                        segment_specs = [
                            ('home', 'Home', ph, w_h, winner == 0),
                            ('draw', 'Draw', pd_, w_d, winner == 1),
                            ('away', 'Away', pa, w_a, winner == 2),
                        ]
                        segment_html = []
                        for css_name, label, prob, width, is_winner in segment_specs:
                            winner_class = ' winner' if is_winner else ''
                            star = '<span class="winner-mark">★</span>' if is_winner else ''
                            segment_html.append(
                                f'<div class="pred-seg {css_name}{winner_class}" '
                                f'style="width:{width:.1f}%;">{star}{label} {prob:.0%}</div>'
                            )
                        segment_html = ''.join(segment_html)
    
                        style_class = 'value-bet' if has_value else (
                            'high-confidence' if confidence > 0.3 else 'low-confidence'
                        )
                        icon = '✅' if has_value else ('🔵' if confidence > 0.3 else '⚠️')
    
                        st.markdown(textwrap.dedent(f"""
                            <div class="match-card">
                                <div class="fixture-row">
                                    <span class="fixture-teams">#{display_index} · {html.escape(home)} vs {html.escape(away)}</span>
                                    <span class="fixture-meta">{'✅ Value bet' if has_value else ''}</span>
                                </div>
                                <div class="pred-label">Winning Prediction</div>
                                <div class="pred-bar">{segment_html}</div>
                                {goalgoal_html}
                            </div>
                        """).strip(), unsafe_allow_html=True)
    
                        # Optional markets are calculated per fixture immediately before
                        # the details expander. A failure here is isolated to this fixture
                        # and can never suppress the primary 1X2 card or other fixtures.
                        if 'corners' not in result:
                            try:
                                result['corners'] = predict_corners(
                                    league, home, away, corner_models,
                                    df=df, fixture_date=result.get('date')
                                )
                                if not isinstance(result.get('corners'), dict):
                                    result['corners'] = {'error': 'Corner model returned an invalid result.'}
                            except Exception as exc:
                                result['corners'] = {'error': f'Corner prediction unavailable: {exc}'}
    
                        if 'cards' not in result:
                            try:
                                result['cards'] = predict_cards(
                                    league, home, away, card_models, df,
                                    referee=result.get('referee'), fixture_date=result.get('date')
                                )
                                if not isinstance(result.get('cards'), dict):
                                    result['cards'] = {'error': 'Card model returned an invalid result.'}
                            except Exception as exc:
                                result['cards'] = {'error': f'Card prediction unavailable: {exc}'}
    
                        with st.expander(
                            f"{icon} #{display_index}: Full breakdown — {home} vs {away}",
                            expanded=False
                        ):
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric('Home Win', f'{ph:.1%}')
                            c2.metric('Draw', f'{pd_:.1%}')
                            c3.metric('Away Win', f'{pa:.1%}')
                            c4.metric('xG', f'{_safe_probability(result.get("exp_goals")):.2f}')
    
                            st.markdown(f"""
                            **Expected Goals:**
                            - {home}: {_safe_probability(result.get('lambda_home')):.2f}
                            - {away}: {_safe_probability(result.get('lambda_away')):.2f}
                            """)
    
                            conf_color = '🟢' if confidence > 0.3 else ('🟡' if confidence > 0.15 else '🔴')
                            st.markdown(f'**Confidence:** {conf_color} {confidence:.1%}')
    
                            if value_bets:
                                st.markdown('### 💰 Value Bets')
                                for j, bet in enumerate(value_bets, 1):
                                    try:
                                        st.markdown(f"""
                                        <div class="{style_class}">
                                        <strong>#{j}: {bet.get('market', 'Unknown')}</strong><br>
                                        Probability: <strong>{_safe_probability(bet.get('prob')):.1%}</strong> |
                                        Odds: <strong>{_safe_probability(bet.get('odds')):.2f}</strong> |
                                        Edge: <strong>{_safe_probability(bet.get('edge')):+.2%}</strong> |
                                        EV: <strong>{_safe_probability(bet.get('ev')):+.1%}</strong><br>
                                        Kelly Stake: <strong>{_safe_probability(bet.get('kelly_stake'))*100:.1f}%</strong> of bankroll
                                        </div>
                                        """, unsafe_allow_html=True)
                                    except Exception as exc:
                                        st.warning(f'Value bet #{j} could not be rendered: {exc}')
                            else:
                                st.info('ℹ️ No value bets found with current filters')
    
                            corners = result.get('corners')
                            st.markdown('### ⚽ Corner Predictions')
                            if not isinstance(corners, dict) or 'error' in corners:
                                st.info(f"ℹ️ {(corners or {}).get('error', 'Corner model unavailable')}")
                            else:
                                try:
                                    cm = corners['market_probs']
                                    cc1, cc2, cc3 = st.columns(3)
                                    cc1.metric(f'{home} Corners', f"{_safe_probability(cm.get('exp_home_corners')):.2f}")
                                    cc2.metric(f'{away} Corners', f"{_safe_probability(cm.get('exp_away_corners')):.2f}")
                                    cc3.metric('Total Corners', f"{_safe_probability(cm.get('exp_total_corners')):.2f}")
                                    rows = []
                                    for line in [7.5, 8.5, 9.5, 10.5, 11.5]:
                                        ok, uk = f'Corners Over {line}', f'Corners Under {line}'
                                        if ok in cm and uk in cm:
                                            rows.append({'Line': line, 'Over %': f"{_safe_probability(cm[ok]):.1%}", 'Under %': f"{_safe_probability(cm[uk]):.1%}"})
                                    if rows:
                                        st.table(pd.DataFrame(rows).set_index('Line'))
                                except Exception as exc:
                                    st.info(f'ℹ️ Corner prediction unavailable: {exc}')
    
                            cards = result.get('cards')
                            st.markdown('### 🟨 Card Predictions')
                            if not isinstance(cards, dict) or 'error' in cards:
                                st.info(f"ℹ️ {(cards or {}).get('error', 'Card model unavailable')}")
                            else:
                                try:
                                    k1, k2, k3 = st.columns(3)
                                    k1.metric(f'{home} Yellow Cards', f"{_safe_probability(cards.get('exp_home_yellows')):.2f}")
                                    k2.metric(f'{away} Yellow Cards', f"{_safe_probability(cards.get('exp_away_yellows')):.2f}")
                                    k3.metric('Expected Total Cards', f"{_safe_probability(cards.get('exp_total_yellows')):.2f}")
                                    if cards.get('likely_range'):
                                        lo, hi = cards['likely_range']
                                        st.caption(f"Likely range: {lo}–{hi} yellow cards · Source: {cards.get('source', 'model')}")
                                except Exception as exc:
                                    st.info(f'ℹ️ Card prediction unavailable: {exc}')
    
                    except Exception as exc:
                        st.error(f'❌ Could not render fixture #{display_index}: {exc}')

            try:
                st.session_state['summary'] = generate_summary_stats(results)
            except Exception:
                st.session_state['summary'] = None
            try:
                st.session_state['top_bets'] = rank_top_value_bets(results, n=7)
            except Exception:
                st.session_state['top_bets'] = []


# ============================================================================
# TAB 2: TOP BETS
# ============================================================================

with tab2:
    st.header("🏆 Top 7 Value Bets")

    if 'top_bets' in st.session_state and st.session_state['top_bets']:

        top_bets = st.session_state['top_bets']

        st.markdown(f"**Showing top {len(top_bets)} bets ranked by Expected Value**")

        for i, bet in enumerate(top_bets, 1):

            with st.expander(f"#{i} - {bet['match']} | {bet['market']}", expanded=(i <= 3)):

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Probability", f"{bet['prob']:.1%}")
                    st.metric("Odds", f"{bet['odds']:.2f}")
                    st.metric("Edge", f"{bet['edge']:+.2%}")

                with col2:
                    st.metric("Expected Value", f"{bet['ev']:+.1%}", delta=None)
                    st.metric("Kelly Stake", f"{bet['kelly_stake']*100:.1f}%")
                    st.metric("Confidence", f"{bet['confidence']:.1%}")

                st.markdown(f"""
                **Match Details:**
                - League: {bet['league']}
                - Expected Goals: {bet['exp_goals']:.2f}
                
                **Recommended Action:**
                - Stake: **{bet['kelly_stake']*100:.1f}%** of bankroll
                - On ${bankroll:,.0f} bankroll = **${bankroll * bet['kelly_stake']:.2f}**
                """)

    else:
        st.info("ℹ️ No predictions yet. Go to Predictions tab first.")

# ============================================================================
# TAB 3: BANKROLL SIMULATOR
# ============================================================================

with tab3:
    st.header("💰 Bankroll Simulator")

    if 'top_bets' in st.session_state and st.session_state['top_bets']:

        top_bets = st.session_state['top_bets']

        st.markdown(f"**Simulating Kelly criterion staking on top {len(top_bets)} bets**")

        simulation = simulate_bankroll(top_bets, bankroll)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Initial Bankroll", f"${simulation['initial_bankroll']:,.2f}")
        with col2:
            st.metric("Total Staked", f"${simulation['total_staked']:.2f}")
        with col3:
            st.metric(
                "Expected Profit",
                f"${simulation['expected_profit']:.2f}",
                delta=f"{simulation['expected_roi']:.1f}%"
            )
        with col4:
            st.metric(
                "Expected Bankroll",
                f"${simulation['expected_bankroll']:.2f}"
            )

        # Bet breakdown
        st.markdown("### 📋 Bet Breakdown")

        bet_df = pd.DataFrame(simulation['bets'])
        bet_df['Expected Return'] = bet_df['stake'] * (1 + bet_df['ev'])
        bet_df['Expected Profit'] = bet_df['stake'] * bet_df['ev']

        # Format columns
        bet_df['Stake'] = bet_df['stake'].apply(lambda x: f"${x:.2f}")
        bet_df['Odds'] = bet_df['odds'].apply(lambda x: f"{x:.2f}")
        bet_df['Prob'] = bet_df['prob'].apply(lambda x: f"{x:.1%}")
        bet_df['EV'] = bet_df['ev'].apply(lambda x: f"{x:+.1%}")
        bet_df['Exp Profit'] = bet_df['Expected Profit'].apply(lambda x: f"${x:.2f}")

        display_df = bet_df[['match', 'market', 'Stake', 'Odds', 'Prob', 'EV', 'Exp Profit']]

        st.dataframe(display_df, width='stretch')

        # Visualization
        st.markdown("### 📊 Expected Returns")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Stake',
            x=bet_df['market'],
            y=bet_df['stake'],
            marker_color='lightblue'
        ))

        fig.add_trace(go.Bar(
            name='Expected Return',
            x=bet_df['market'],
            y=bet_df['Expected Return'],
            marker_color='lightgreen'
        ))

        fig.update_layout(
            title="Stake vs Expected Return by Market",
            xaxis_title="Market",
            yaxis_title="Amount ($)",
            barmode='group',
            height=400
        )

        st.plotly_chart(fig, width='stretch')

        # Warning
        st.warning("""
        ⚠️ **Important Notes:**
        - This is a **theoretical simulation** based on expected values
        - Actual results will vary due to variance
        - Past performance does not guarantee future results
        - Never bet more than you can afford to lose
        - Use Kelly criterion as a guideline, not a rule
        """)

    else:
        st.info("ℹ️ No predictions yet. Go to Predictions tab first.")

# ============================================================================
# TAB 4: STATISTICS
# ============================================================================

with tab4:
    st.header("📊 Prediction Statistics")

    if 'summary' in st.session_state:

        summary = st.session_state['summary']

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Matches", summary['total_matches'])
        with col2:
            st.metric("Matches with Value", summary['matches_with_value'])
        with col3:
            st.metric("Total Value Bets", summary['total_value_bets'])
        with col4:
            st.metric("Hit Rate", f"{summary['hit_rate']:.1%}")

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Avg Probability", f"{summary['avg_prob']:.1%}")
        with col2:
            st.metric("Avg Expected Value", f"{summary['avg_ev']:+.1%}")
        with col3:
            st.metric("Avg Odds", f"{summary['avg_odds']:.2f}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Avg Confidence", f"{summary['avg_confidence']:.1%}")
        with col2:
            st.metric("Avg Expected Goals", f"{summary['avg_exp_goals']:.2f}")

        # Visualizations
        if 'results' in st.session_state:

            results = st.session_state['results']

            st.markdown("### 📈 Distributions")

            col1, col2 = st.columns(2)

            with col1:
                # Confidence distribution
                confidences = [r['confidence'] for r in results]

                fig = go.Figure(data=[go.Histogram(
                    x=confidences,
                    nbinsx=20,
                    marker_color='lightblue'
                )])

                fig.update_layout(
                    title="Confidence Score Distribution",
                    xaxis_title="Confidence",
                    yaxis_title="Count",
                    height=300
                )

                st.plotly_chart(fig, width='stretch')

            with col2:
                # Expected goals distribution
                exp_goals = [r['exp_goals'] for r in results]

                fig = go.Figure(data=[go.Histogram(
                    x=exp_goals,
                    nbinsx=20,
                    marker_color='lightgreen'
                )])

                fig.update_layout(
                    title="Expected Goals Distribution",
                    xaxis_title="xG",
                    yaxis_title="Count",
                    height=300
                )

                st.plotly_chart(fig, width='stretch')

    else:
        st.info("ℹ️ No statistics yet. Make predictions first.")

# ============================================================================
# TAB 5: GUIDE
# ============================================================================

with tab5:
    st.header("ℹ️ User Guide")

    st.markdown(f"""
    ## 📖 How to Use This App
    
    ### 1️⃣ Input Fixtures
    
    Go to the **Predictions** tab and paste your fixtures in this format:
    ```
    League, Home Team, Away Team, Home Odds, Draw Odds, Away Odds
    ```
    
    **Supported Leagues:**
    - Premier League
    - Championship
    - La Liga
    - Segunda Division
    - Serie A
    - Serie B
    - Bundesliga
    - 2. Bundesliga
    - Ligue 1
    - Ligue 2
    - Eredivisie
    - Belgian Pro League
    - Primeira Liga
    - Super Lig
    - Greek Super League
    
    ### 2️⃣ Adjust Filters
    
    Use the **sidebar** to set:
    - **Min Probability:** Model confidence threshold (recommend 0.45)
    - **Min EV:** Expected value threshold (recommend 0.03)
    
    ### 3️⃣ Generate Predictions
    
    Click **PREDICT ALL** to:
    - Calculate probabilities
    - Identify value bets
    - Rank opportunities
    
    ### 4️⃣ Review Results
    
    - **Predictions Tab:** See all matches with value bets highlighted
    - **Top Bets Tab:** View top 7 opportunities ranked by EV
    - **Bankroll Simulator:** Calculate Kelly stakes and simulate returns
    - **Statistics Tab:** Analyze prediction quality
    
    ---
    
    ## 🎯 Understanding the Model
    
    ### What is Dixon-Coles?
    
    A Poisson-based model that predicts goal counts by:
    - Modeling team attack and defense strengths
    - Adjusting for home advantage
    - Correcting for low-score scenarios
    
    ### What is XGBoost?
    
    A machine learning model trained on {len(feature_cols)} features:
    - Rolling goal averages
    - Shot statistics
    - Corner statistics
    - Form (points)
    - ELO ratings
    - Matchup differentials
    
    ### How is the Ensemble Created?
    
    - 60% Dixon-Coles + 40% XGBoost
    - Combined via log-odds pooling
    - Normalized with softmax
    
    ---
    
    ## 💰 Understanding Value Betting
    
    ### What is Expected Value (EV)?
    
    ```
    EV = (Probability × Odds) - 1
    ```
    
    **Positive EV** = Good bet (model thinks odds are generous)
    
    **Example:**
    - Model prob: 55%
    - Bookmaker odds: 2.10
    - EV = (0.55 × 2.10) - 1 = +15.5%
    
    ### What is Kelly Criterion?
    
    Optimal stake sizing formula:
    ```
    Kelly = (Prob × Odds - 1) / (Odds - 1)
    ```
    
    **We use Quarter Kelly (25%)** for safety.
    
    ---
    
    ## ⚠️ Important Warnings
    
    ### This Model Is NOT Perfect
    
    - Expected accuracy: 55-65%
    - Variance is high in sports betting
    - Value bets can still lose
    - Never bet more than you can afford to lose
    
    ### Responsible Betting
    
    - Set a strict bankroll limit
    - Never chase losses
    - Track your results
    - Take breaks if needed
    - Betting should be entertainment, not income
    
    ---
    
    ## 🔧 Technical Details
    
    ### Model Architecture
    
    - **ML Model:** XGBoost (300 trees, depth 5)
    - **Calibration:** Isotonic regression
    - **Validation:** Walk-forward time series CV
    - **Features:** {len(feature_cols)} engineered features
    - **Dixon-Coles:** Time-decay weighted MLE
    
    ### Training Data
    
    - **Matches:** {len(df):,}
    - **Leagues:** {trained_ml} configured ({trained_dc} Dixon-Coles converged, {trained_corners} corner models, {trained_cards} card models)
    - **Seasons:** {len(current_teams_meta.get('seasons_requested', [])) if current_teams_meta else 10}
    - **Date range:** {df['Date'].min().date()} to {df['Date'].max().date()}
    
    ---
    
    ## 📞 Support
    
    **For issues:**
    - Check team names match exactly
    - Ensure odds are in decimal format
    - Verify league names are correct
    
    **Common Issues:**
    - "Team not found" → Check spelling and use exact names
    - "League not supported" → Check the configured league list and retrain if its artifact is not yet available
    - "No value bets" → Lower filters or try different matches
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 20px;">
    <p><strong>⚽ Professional Football Betting Model</strong></p>
    <p>Dixon-Coles + XGBoost Ensemble | Built with Streamlit</p>
    <p><small>For educational and research purposes only. Bet responsibly.</small></p>
</div>
""", unsafe_allow_html=True)
