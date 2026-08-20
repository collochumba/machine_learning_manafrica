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

import fixture_loader
import team_normalization

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

from models import DixonColesTimeDecay
from predict import (
    predict_multiple_fixtures,
    predict_corners,
    generate_summary_stats,
    rank_top_value_bets,
    simulate_bankroll
)

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
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
/* Main styling */
.main {
    padding: 0rem 1rem;
}

/* Value bet highlighting */
.value-bet {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    padding: 15px;
    border-radius: 8px;
    border-left: 5px solid #28a745;
    margin: 10px 0;
}

.high-confidence {
    background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
    padding: 15px;
    border-radius: 8px;
    border-left: 5px solid #17a2b8;
    margin: 10px 0;
}

.low-confidence {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    padding: 15px;
    border-radius: 8px;
    border-left: 5px solid #ffc107;
    margin: 10px 0;
}

/* Metrics */
div[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    padding: 10px;
    border-radius: 5px;
}

/* Headers */
h1 {
    color: #2c3e50;
    font-weight: 700;
}

h2 {
    color: #34495e;
    font-weight: 600;
}

h3 {
    color: #7f8c8d;
    font-weight: 500;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}
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

        return final_model, dc_models, feature_cols, df, team_mapping, all_teams, current_teams_meta, corner_models, None

    except Exception as e:
        return None, None, None, None, None, None, None, None, str(e)

# Load models
with st.spinner("🔄 Loading trained models..."):
    final_model, dc_models, feature_cols, df, team_mapping, all_teams, current_teams_meta, corner_models, error = load_all_models()

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
    
    **Leagues:** {len(dc_models)}
    
    **Teams:** {df['HomeTeam'].nunique()}
    """)

    st.info(f"⚽ **Corner model:** {'✅ loaded (' + str(len(corner_models)) + ' leagues)' if corner_models else '⚠️ not available — run train.py to generate corner_model.pkl'}")

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
    # further down triggers on (run_predictions and fixtures), regardless
    # of which input method populated `fixtures`.
    fixtures = []
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
            load_clicked = st.button("🔄 Load Latest Fixtures", type="primary", use_container_width=True)
        with loader_col2:
            uploaded_fixture_file = st.file_uploader(
                "...or upload a fixtures.csv / fixtures.xlsx file",
                type=['csv', 'xlsx'],
                key='fixture_upload',
            )

        # Handle download button
        if load_clicked:
            with st.spinner("Downloading latest fixtures..."):
                content, err = fixture_loader.fetch_fixtures_bytes(fixture_loader.FIXTURES_CSV_URL)
                if err:
                    st.error(f"❌ {err}")
                    st.info("You can still use **📝 Paste Fixtures Manually** instead.")
                else:
                    raw_df, perr = fixture_loader.parse_fixture_bytes(content, "fixtures.csv")
                    if perr:
                        st.error(f"❌ {perr}")
                    else:
                        save_fixture_cache(raw_df, "football-data.co.uk (downloaded)")
                        st.success("✅ Latest fixtures downloaded")

        # Handle file upload
        if uploaded_fixture_file is not None:
            content = uploaded_fixture_file.read()
            raw_df, perr = fixture_loader.parse_fixture_bytes(content, uploaded_fixture_file.name)
            if perr:
                st.error(f"❌ {perr}")
            else:
                save_fixture_cache(raw_df, f"uploaded file: {uploaded_fixture_file.name}")
                st.success("✅ Fixture file loaded")

        cached = load_fixture_cache()

        if cached is None:
            st.info("No fixtures loaded yet. Click **🔄 Load Latest Fixtures** or upload a file above.")
        else:
            refresh_col1, refresh_col2 = st.columns([3, 1])
            with refresh_col1:
                st.caption(f"Loaded from: {cached['source']} · {cached['fetched_at'].strftime('%Y-%m-%d %H:%M:%S')}")
            with refresh_col2:
                if st.button("🔄 Refresh Fixtures", use_container_width=True):
                    if os.path.exists(FIXTURE_CACHE_FILE):
                        os.remove(FIXTURE_CACHE_FILE)
                    st.rerun()

            try:
                supported_raw, excluded_counts, total_rows = fixture_loader.extract_supported_fixtures(cached['raw_df'])
            except ValueError as e:
                st.error(f"❌ {e}")
                supported_raw, excluded_counts, total_rows = [], {}, 0

            # Resolve teams against CURRENT-season teams (not historical all_teams)
            current_teams_by_league = (current_teams_meta or {}).get('current_teams_by_league', {})

            resolved_fixtures = []
            for f in supported_raw:
                res = team_normalization.resolve_fixture(
                    f['league'], f['home_raw'], f['away_raw'], current_teams_by_league, all_teams
                )
                res['date'] = f['date']
                res['time'] = f['time']
                res['odds'] = f['odds']
                resolved_fixtures.append(res)

            n_valid = sum(1 for r in resolved_fixtures if r['status'] == 'valid')
            n_review = len(resolved_fixtures) - n_valid
            excluded_named = fixture_loader.summarize_excluded(excluded_counts)

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

            if n_review > 0:
                with st.expander(f"⚠️ {n_review} fixture(s) need review", expanded=True):
                    for r in resolved_fixtures:
                        if r['status'] != 'needs_review':
                            continue
                        bad_side = 'home' if not r['home']['resolved'] else 'away'
                        bad = r[bad_side]
                        st.warning(
                            f"**{r['league']}**: {r['home_raw']} vs {r['away_raw']}\n\n"
                            f"❌ Unrecognised team: **{bad['input']}**"
                            + (f"\n\nSuggested: **{', '.join(bad['suggestions'])}**" if bad['suggestions'] else "\n\nNo close match found.")
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
                for league_name in ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]:
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
                            if r['odds']:
                                fx['odds'] = r['odds']
                            fixtures.append(fx)

                predict_clicked = st.button("⚽ Predict Selected Fixtures", type="primary", use_container_width=True)
                run_predictions = predict_clicked and bool(fixtures)
                if predict_clicked and not fixtures:
                    st.warning("No fixtures selected.")

    # ========================================================================
    # MODE: PASTE FIXTURES MANUALLY (unchanged backup workflow)
    # ========================================================================
    else:
        st.markdown("""
        **Format:** `League, Home Team, Away Team, Home Odds, Draw Odds, Away Odds`
        
        **Supported Leagues:**
        - Premier League
        - La Liga
        - Serie A
        - Bundesliga
        - Ligue 1
        
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
            predict_button = st.button("🔮 PREDICT ALL", type="primary", use_container_width=True)

        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
            if clear_button:
                st.rerun()

        with col3:
            lines = [l.strip() for l in input_text.split('\n') if l.strip()] if input_text else []
            st.info(f"📊 {len(lines)} fixtures ready")

    # ============================================================================
    # MANUAL-PASTE PARSING (only runs in paste mode; appends into the shared
    # `fixtures` list so the prediction block below is common to both modes)
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

                fixtures.append({
                    'league': league,
                    'home': home,
                    'away': away,
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

        if not fixtures:
            st.error("❌ No valid fixtures to predict!")

        run_predictions = bool(fixtures)

    # ============================================================================
    # PREDICTION LOGIC (common to both input modes)
    # ============================================================================

    if run_predictions and fixtures:

        # Generate predictions
        with st.spinner(f"🔄 Analyzing {len(fixtures)} fixtures..."):

            # FIX 1: Pass all_teams as positional argument per the required signature:
            # predict_multiple_fixtures(fixtures, final_model, dc_models, feature_cols,
            #                           df, team_mapping, all_teams, min_prob, min_ev)
            try:
                # FIX (bug #27): predict_multiple_fixtures returns
                # (results, errors, warnings_collected) — errors second,
                # warnings third. This used to be unpacked backwards,
                # which silently swapped the "Warnings" and "Errors"
                # expanders below.
                results, prediction_errors, prediction_warnings = predict_multiple_fixtures(
                    fixtures,
                    final_model,
                    dc_models,
                    feature_cols,
                    df,
                    team_mapping,
                    all_teams,
                    min_prob=min_prob,
                    min_ev=min_ev
                )
            except TypeError:
                # Fallback: some versions may return only (results, errors)
                raw = predict_multiple_fixtures(
                    fixtures,
                    final_model,
                    dc_models,
                    feature_cols,
                    df,
                    team_mapping,
                    all_teams,
                    min_prob=min_prob,
                    min_ev=min_ev
                )
                if len(raw) == 3:
                    results, prediction_errors, prediction_warnings = raw
                else:
                    results, prediction_errors = raw
                    prediction_warnings = []
            except Exception as e:
                st.error(f"❌ Prediction pipeline error: {e}")
                st.stop()

        # FIX 4: Display warnings returned from prediction
        if prediction_warnings:
            with st.expander("⚠️ Prediction Warnings", expanded=False):
                for warn in prediction_warnings:
                    if isinstance(warn, dict):
                        st.warning(f"{warn.get('fixture', 'Unknown')}: {warn.get('warning', warn)}")
                    else:
                        st.warning(str(warn))

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

            # Corner predictions (Task 2) — computed separately from the
            # 1X2/goals pipeline above, using each result's already-
            # NORMALIZED home/away names so both models agree on which
            # fixture is which. Missing corner_model.pkl (older artifact
            # set) or an unrecognized team degrades gracefully — each
            # result just gets result['corners'] = {'error': ...} and the
            # UI shows an info message instead of a crash.
            for result in results:
                result['corners'] = predict_corners(
                    result['league'], result['home'], result['away'], corner_models
                )

            # Store results in session state
            st.session_state['results'] = results
            st.session_state['summary'] = generate_summary_stats(results)
            st.session_state['top_bets'] = rank_top_value_bets(results, n=7)

            # Display predictions
            st.markdown("---")
            st.header("🎯 Match Predictions")

            for i, result in enumerate(results, 1):

                has_value = len(result['value_bets']) > 0
                confidence = result['confidence']

                # Determine styling
                if has_value:
                    style_class = "value-bet"
                    icon = "✅"
                elif confidence > 0.3:
                    style_class = "high-confidence"
                    icon = "🔵"
                else:
                    style_class = "low-confidence"
                    icon = "⚠️"

                with st.expander(
                    f"{icon} #{i}: {result['home']} vs {result['away']} ({result['league']})",
                    expanded=has_value
                ):

                    # Probabilities
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Home Win", f"{result['prob_home']:.1%}")
                    with col2:
                        st.metric("Draw", f"{result['prob_draw']:.1%}")
                    with col3:
                        st.metric("Away Win", f"{result['prob_away']:.1%}")
                    with col4:
                        st.metric("xG", f"{result['exp_goals']:.2f}")

                    # Expected goals breakdown
                    st.markdown(f"""
                    **Expected Goals:**
                    - {result['home']}: {result['lambda_home']:.2f}
                    - {result['away']}: {result['lambda_away']:.2f}
                    """)

                    # Confidence score
                    conf_color = "🟢" if confidence > 0.3 else ("🟡" if confidence > 0.15 else "🔴")
                    st.markdown(f"**Confidence:** {conf_color} {confidence:.1%}")

                    # Value bets
                    if result['value_bets']:
                        st.markdown("### 💰 Value Bets")

                        for j, bet in enumerate(result['value_bets'], 1):
                            st.markdown(f"""
                            <div class="{style_class}">
                            <strong>#{j}: {bet['market']}</strong><br>
                            Probability: <strong>{bet['prob']:.1%}</strong> | 
                            Odds: <strong>{bet['odds']:.2f}</strong> | 
                            Edge: <strong>{bet['edge']:+.2%}</strong> | 
                            EV: <strong style="color: #28a745;">{bet['ev']:+.1%}</strong><br>
                            Kelly Stake: <strong>{bet['kelly_stake']*100:.1f}%</strong> of bankroll
                            </div>
                            """, unsafe_allow_html=True)

                            # High-EV sanity warning
                            if bet['ev'] > 0.25:
                                st.warning(
                                    f"⚠️ **Unusually high EV ({bet['ev']:.1%})** on {bet['market']}. "
                                    "Verify these odds are current and from a reputable source before acting. "
                                    "EV above 25% often indicates stale odds or model overconfidence."
                                )

                    else:
                        st.info("ℹ️ No value bets found with current filters")

                    # ============================================================
                    # ⚽ CORNER PREDICTIONS (Task 2 — dedicated corner model)
                    # ============================================================
                    corners = result.get('corners')
                    if corners is None:
                        pass  # corner_model.pkl not loaded at all — say nothing per-fixture, sidebar already notes this
                    elif 'error' in corners:
                        st.markdown("### ⚽ Corner Predictions")
                        st.info(f"ℹ️ {corners['error']}")
                    else:
                        st.markdown("### ⚽ Corner Predictions")

                        cm_probs = corners['market_probs']
                        ccol1, ccol2, ccol3 = st.columns(3)
                        with ccol1:
                            st.metric(f"{result['home']} Corners", f"{cm_probs['exp_home_corners']:.2f}")
                        with ccol2:
                            st.metric(f"{result['away']} Corners", f"{cm_probs['exp_away_corners']:.2f}")
                        with ccol3:
                            st.metric("Total Corners", f"{cm_probs['exp_total_corners']:.2f}")

                        ou_lines = [7.5, 8.5, 9.5, 10.5, 11.5]
                        ou_rows = []
                        for line in ou_lines:
                            over_key = f'Corners Over {line}'
                            under_key = f'Corners Under {line}'
                            if over_key in cm_probs:
                                ou_rows.append({
                                    'Line': line,
                                    'Over %': f"{cm_probs[over_key]:.1%}",
                                    'Under %': f"{cm_probs[under_key]:.1%}",
                                })
                        if ou_rows:
                            st.table(pd.DataFrame(ou_rows).set_index('Line'))

                        # Value bets on corner markets, ONLY if corner odds
                        # were actually supplied by the fixture source —
                        # never inferred from a high model probability alone.
                        if corners['value_bets']:
                            st.markdown("#### 💰 Corner Value Bets")
                            for j, bet in enumerate(corners['value_bets'], 1):
                                st.markdown(f"""
                                <div class="{style_class}">
                                <strong>#{j}: {bet['market']}</strong><br>
                                Probability: <strong>{bet['prob']:.1%}</strong> |
                                Odds: <strong>{bet['odds']:.2f}</strong> |
                                Edge: <strong>{bet['edge']:+.2%}</strong> |
                                EV: <strong style="color: #28a745;">{bet['ev']:+.1%}</strong>
                                </div>
                                """, unsafe_allow_html=True)



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

        st.dataframe(display_df, use_container_width=True)

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

        st.plotly_chart(fig, use_container_width=True)

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

                st.plotly_chart(fig, use_container_width=True)

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

                st.plotly_chart(fig, use_container_width=True)

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
    - La Liga
    - Serie A
    - Bundesliga
    - Ligue 1
    
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
    - **Leagues:** {len(dc_models)} (top European leagues)
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
    - "League not supported" → Only 5 leagues available
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
