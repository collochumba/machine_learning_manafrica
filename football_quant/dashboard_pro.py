"""
Professional Streamlit Dashboard
Hedge fund quality interface for football predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from inference import InferenceEngine
from backtester import InstitutionalBacktester
from betting_optimizer import PortfolioOptimizer
from asian_handicap import AsianHandicapPredictor

# Page config
st.set_page_config(
    page_title="Football Quant Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .value-bet {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .strong-value {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'inference_engine' not in st.session_state:
    st.session_state.inference_engine = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def load_inference_engine():
    """Load or initialize inference engine."""
    if st.session_state.inference_engine is None:
        with st.spinner("Loading models..."):
            try:
                engine = InferenceEngine()
                engine.load_models()
                st.session_state.inference_engine = engine
                st.success("✅ Models loaded successfully!")
            except Exception as e:
                st.error(f"Error loading models: {e}")
                return None
    return st.session_state.inference_engine

def tab_todays_matches():
    """TAB 1: Today's Matches."""
    st.markdown('<h2 class="main-header">📅 Today\'s Matches</h2>', unsafe_allow_html=True)
    
    engine = load_inference_engine()
    if engine is None:
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        leagues = st.multiselect(
            "Select Leagues",
            ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'],
            default=['Premier League']
        )
    
    with col2:
        min_edge = st.slider("Min Edge %", 0.0, 15.0, 3.0, 0.5) / 100
    
    if st.button("🔮 Generate Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            try:
                predictions = engine.predict_today(leagues=leagues)
                st.session_state.predictions = predictions
            except Exception as e:
                st.error(f"Error generating predictions: {e}")
                return
    
    if st.session_state.predictions is None or st.session_state.predictions.empty:
        st.info("No predictions available. Click 'Generate Predictions' to start.")
        return
    
    df = st.session_state.predictions
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(df))
    with col2:
        value_bets = (df['best_edge'] >= min_edge).sum()
        st.metric("Value Bets", value_bets)
    with col3:
        avg_edge = df[df['best_edge'] >= min_edge]['best_edge'].mean() * 100
        st.metric("Avg Edge", f"{avg_edge:.1f}%" if not np.isnan(avg_edge) else "N/A")
    with col4:
        total_stake = df[df['best_edge'] >= min_edge]['kelly_stake'].sum()
        st.metric("Total Stake", f"${total_stake:.0f}")
    
    st.markdown("---")
    
    # Match predictions table
    for idx, row in df.iterrows():
        has_value = row['best_edge'] >= min_edge
        
        if has_value:
            st.markdown(f'<div class="value-bet">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            st.markdown(f"**{row['league']}**")
            st.markdown(f"### {row['home_team']} vs {row['away_team']}")
        
        with col2:
            # 1X2 Probabilities
            st.markdown("**1X2 Probabilities**")
            prob_cols = st.columns(3)
            with prob_cols[0]:
                st.metric("Home", f"{row['prob_home']*100:.1f}%")
            with prob_cols[1]:
                st.metric("Draw", f"{row['prob_draw']*100:.1f}%")
            with prob_cols[2]:
                st.metric("Away", f"{row['prob_away']*100:.1f}%")
            
            # Expected Goals
            st.markdown(f"**Expected Goals:** {row['exp_goals_home']:.2f} - {row['exp_goals_away']:.2f}")
            
            # Asian Handicap
            if 'ah_line' in row:
                st.markdown(f"**Asian Handicap Fair Line:** {row['ah_line']:.2f}")
        
        with col3:
            if has_value:
                st.markdown("### 💰 VALUE BET")
                st.metric("Best Market", row['best_market'].upper())
                st.metric("Edge", f"{row['best_edge']*100:.1f}%")
                st.metric("Kelly Stake", f"${row['kelly_stake']:.2f}")
                st.metric("Model Prob", f"{row['best_prob']*100:.1f}%")
                st.metric("Bookmaker Odds", f"{row['best_odds']:.2f}")
        
        if has_value:
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")

def tab_week_predictions():
    """TAB 2: This Week Predictions."""
    st.markdown('<h2 class="main-header">📆 This Week Predictions</h2>', unsafe_allow_html=True)
    
    engine = load_inference_engine()
    if engine is None:
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        leagues = st.multiselect(
            "Filter by League",
            ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'],
            default=['Premier League', 'La Liga']
        )
    
    with col2:
        min_edge = st.slider("Min Edge % (Week)", 0.0, 15.0, 2.0, 0.5) / 100
    
    with col3:
        sort_by = st.selectbox("Sort by", ['Edge', 'Probability', 'Date'])
    
    if st.button("🔮 Predict Week", type="primary"):
        with st.spinner("Loading week's fixtures..."):
            try:
                predictions = engine.predict_week(leagues=leagues, days_ahead=7)
                st.session_state.week_predictions = predictions
            except Exception as e:
                st.error(f"Error: {e}")
                return
    
    if 'week_predictions' not in st.session_state:
        st.info("Click 'Predict Week' to load fixtures.")
        return
    
    df = st.session_state.week_predictions
    
    # Filter by edge
    df_filtered = df[df['best_edge'] >= min_edge].copy()
    
    # Sort
    if sort_by == 'Edge':
        df_filtered = df_filtered.sort_values('best_edge', ascending=False)
    elif sort_by == 'Probability':
        df_filtered = df_filtered.sort_values('best_prob', ascending=False)
    else:
        df_filtered = df_filtered.sort_values('date')
    
    st.markdown(f"**Showing {len(df_filtered)} value bets from {len(df)} total matches**")
    
    # Display as table
    display_df = df_filtered[[
        'date', 'league', 'home_team', 'away_team',
        'best_market', 'best_prob', 'best_odds', 'best_edge', 'kelly_stake'
    ]].copy()
    
    display_df['best_prob'] = (display_df['best_prob'] * 100).round(1).astype(str) + '%'
    display_df['best_edge'] = (display_df['best_edge'] * 100).round(1).astype(str) + '%'
    display_df['best_odds'] = display_df['best_odds'].round(2)
    display_df['kelly_stake'] = '$' + display_df['kelly_stake'].round(0).astype(str)
    
    display_df.columns = ['Date', 'League', 'Home', 'Away', 'Market', 'Prob', 'Odds', 'Edge', 'Stake']
    
    st.dataframe(display_df, use_container_width=True, height=600)

def tab_high_probability():
    """TAB 3: High Probability Outcomes."""
    st.markdown('<h2 class="main-header">🎯 High Probability Outcomes</h2>', unsafe_allow_html=True)
    
    engine = load_inference_engine()
    if engine is None:
        return
    
    prob_threshold = st.slider("Minimum Probability %", 50, 80, 60, 5) / 100
    
    if st.button("🔍 Find High Probability Bets"):
        with st.spinner("Analyzing..."):
            try:
                predictions = engine.predict_week(days_ahead=7)
                st.session_state.high_prob = predictions
            except Exception as e:
                st.error(f"Error: {e}")
                return
    
    if 'high_prob' not in st.session_state:
        st.info("Click 'Find High Probability Bets' to analyze.")
        return
    
    df = st.session_state.high_prob
    
    # Filter high probability
    df_high = df[df['best_prob'] >= prob_threshold].copy()
    df_high = df_high.sort_values('best_prob', ascending=False)
    
    st.markdown(f"**Found {len(df_high)} high-confidence outcomes (≥{prob_threshold*100:.0f}%)**")
    
    for idx, row in df_high.head(20).iterrows():
        st.markdown(f'<div class="strong-value">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 2, 2])
        
        with col1:
            st.markdown(f"**{row['league']}**")
            st.markdown(f"### {row['home_team']} vs {row['away_team']}")
            st.markdown(f"**Market:** {row['best_market'].upper()}")
        
        with col2:
            st.metric("Model Probability", f"{row['best_prob']*100:.1f}%")
            st.metric("Fair Odds", f"{1/row['best_prob']:.2f}")
            st.metric("Bookmaker Odds", f"{row['best_odds']:.2f}")
        
        with col3:
            st.metric("Edge", f"{row['best_edge']*100:.1f}%")
            st.metric("Kelly Stake", f"${row['kelly_stake']:.2f}")
            implied = 1 / row['best_odds']
            st.metric("Implied Prob", f"{implied*100:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)

def tab_portfolio_optimizer():
    """TAB 4: Portfolio Optimizer."""
    st.markdown('<h2 class="main-header">💼 Portfolio Optimizer</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Portfolio Settings")
        
        bankroll = st.number_input("Bankroll ($)", 1000, 100000, 10000, 1000)
        risk_tolerance = st.slider("Risk Tolerance", 0.5, 2.0, 1.0, 0.1)
        max_exposure = st.slider("Max Exposure %", 5, 25, 10, 5) / 100
        correlation_adj = st.checkbox("Correlation Adjustment", value=True)
        
        if st.button("🎲 Optimize Portfolio"):
            if 'predictions' not in st.session_state or st.session_state.predictions is None:
                st.warning("Generate predictions first in Tab 1")
                return
            
            with st.spinner("Optimizing portfolio..."):
                try:
                    optimizer = PortfolioOptimizer(
                        bankroll=bankroll,
                        risk_tolerance=risk_tolerance,
                        max_total_exposure=max_exposure
                    )
                    
                    predictions = st.session_state.predictions
                    portfolio = optimizer.optimize_portfolio(
                        predictions,
                        correlation_aware=correlation_adj
                    )
                    
                    st.session_state.portfolio = portfolio
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
    
    with col2:
        if 'portfolio' not in st.session_state:
            st.info("Configure settings and click 'Optimize Portfolio'")
            return
        
        portfolio = st.session_state.portfolio
        
        st.markdown("### Recommended Bets")
        
        # Summary metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Total Bets", len(portfolio))
        with metric_cols[1]:
            total_stake = portfolio['stake'].sum()
            st.metric("Total Stake", f"${total_stake:.0f}")
        with metric_cols[2]:
            exposure_pct = (total_stake / bankroll) * 100
            st.metric("Exposure", f"{exposure_pct:.1f}%")
        with metric_cols[3]:
            expected_roi = portfolio['expected_value'].sum() / total_stake * 100
            st.metric("Expected ROI", f"{expected_roi:.1f}%")
        
        st.markdown("---")
        
        # Portfolio table
        display_portfolio = portfolio[[
            'match', 'market', 'probability', 'odds', 'edge', 'stake', 'expected_value'
        ]].copy()
        
        display_portfolio['probability'] = (display_portfolio['probability'] * 100).round(1).astype(str) + '%'
        display_portfolio['edge'] = (display_portfolio['edge'] * 100).round(1).astype(str) + '%'
        display_portfolio['stake'] = '$' + display_portfolio['stake'].round(0).astype(str)
        display_portfolio['expected_value'] = '$' + display_portfolio['expected_value'].round(2).astype(str)
        
        display_portfolio.columns = ['Match', 'Market', 'Prob', 'Odds', 'Edge', 'Stake', 'EV']
        
        st.dataframe(display_portfolio, use_container_width=True)

def tab_backtest():
    """TAB 5: Backtest Analytics."""
    st.markdown('<h2 class="main-header">📈 Backtest Analytics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Backtest Settings")
        
        initial_bankroll = st.number_input("Initial Bankroll", 1000, 100000, 10000)
        kelly_frac = st.slider("Kelly Fraction", 0.1, 0.5, 0.25, 0.05)
        min_edge = st.slider("Min Edge (BT)", 0.0, 10.0, 3.0, 0.5) / 100
        
        if st.button("▶️ Run Backtest"):
            with st.spinner("Running walk-forward backtest..."):
                try:
                    # Load historical data
                    engine = load_inference_engine()
                    historical = engine.load_historical_for_backtest()
                    
                    # Run backtest
                    backtester = InstitutionalBacktester(
                        initial_bankroll=initial_bankroll,
                        kelly_fraction=kelly_frac,
                        min_edge=min_edge
                    )
                    
                    stats = backtester.run_backtest(
                        predictions_df=historical['predictions'],
                        results_df=historical['results']
                    )
                    
                    st.session_state.backtest_stats = stats
                    st.session_state.backtester = backtester
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
    
    with col2:
        if 'backtest_stats' not in st.session_state:
            st.info("Configure and run backtest")
            return
        
        stats = st.session_state.backtest_stats
        backtester = st.session_state.backtester
        
        # Overview metrics
        st.markdown("### Performance Overview")
        metric_cols = st.columns(5)
        
        overview = stats['overview']
        with metric_cols[0]:
            st.metric("Final Bankroll", f"${overview['final_bankroll']:.0f}")
        with metric_cols[1]:
            st.metric("Total Profit", f"${overview['total_profit']:.0f}")
        with metric_cols[2]:
            st.metric("ROI", f"{overview['roi_pct']:.1f}%")
        with metric_cols[3]:
            st.metric("CAGR", f"{overview['cagr_pct']:.1f}%")
        with metric_cols[4]:
            st.metric("Win Rate", f"{overview['win_rate_pct']:.1f}%")
        
        st.markdown("---")
        
        # Equity curve
        equity = backtester.get_equity_curve()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity['date'],
            y=equity['bankroll'],
            mode='lines',
            name='Bankroll',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_hline(y=initial_bankroll, line_dash="dash", line_color="gray", annotation_text="Initial")
        
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Bankroll ($)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown curve
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=equity['date'],
            y=equity['drawdown_pct'],
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#d62728', width=2)
        ))
        
        fig_dd.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            height=300
        )
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Additional stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Risk Metrics")
            risk = stats['risk']
            st.metric("Max Drawdown", f"{risk['max_drawdown_pct']:.1f}%")
            st.metric("Sharpe Ratio", f"{risk['sharpe_ratio']:.2f}")
            st.metric("Avg Stake", f"{risk['avg_stake_pct']:.2f}%")
        
        with col2:
            st.markdown("### Edge Quality")
            edge = stats['edge_quality']
            st.metric("Avg Edge", f"{edge['avg_edge_pct']:.1f}%")
            st.metric("Avg CLV", f"{edge['avg_clv_pct']:.1f}%")
            st.metric("Positive CLV", f"{edge['positive_clv_pct']:.1f}%")

# Main app
def main():
    st.sidebar.title("⚽ Football Quant Pro")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Today's Matches", "This Week", "High Probability", "Portfolio Optimizer", "Backtest Analytics"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    if st.session_state.inference_engine is not None:
        st.sidebar.success("🟢 Models Loaded")
    else:
        st.sidebar.warning("🟡 Models Not Loaded")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Version:** 2.0 Pro")
    st.sidebar.markdown("**Last Updated:** " + datetime.now().strftime("%Y-%m-%d"))
    
    # Route to page
    if page == "Today's Matches":
        tab_todays_matches()
    elif page == "This Week":
        tab_week_predictions()
    elif page == "High Probability":
        tab_high_probability()
    elif page == "Portfolio Optimizer":
        tab_portfolio_optimizer()
    else:
        tab_backtest()

if __name__ == "__main__":
    main()
