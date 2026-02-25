"""
MAIN ORCHESTRATOR - Professional Football Prediction System
Hedge fund quality - complete production pipeline

Usage:
    python main.py train                    # Train models
    python main.py predict                  # Predict today
    python main.py predict-week             # Predict next 7 days
    python main.py backtest                 # Run backtest
    python main.py optimize                 # Optimize portfolio
    python main.py dashboard                # Launch GUI
    python main.py info                     # System information
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import professional modules
from inference_professional import InferenceEngine
from backtester_professional import InstitutionalBacktester, BacktestConfig
from portfolio_optimizer_professional import PortfolioOptimizer, PortfolioConfig


def cmd_train(args):
    """
    Training mode - build and save models.
    
    Example:
        python main.py train --leagues "Premier League" "La Liga" --seasons 3
    """
    print("\n" + "="*70)
    print("🎓 TRAINING MODE")
    print("="*70)
    
    leagues = args.leagues or ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1']
    n_seasons = args.seasons
    
    print(f"\nConfiguration:")
    print(f"  Leagues: {', '.join(leagues)}")
    print(f"  Seasons: {n_seasons}")
    print(f"  Models directory: {args.models_dir}")
    
    engine = InferenceEngine(models_dir=args.models_dir)
    
    # Train models
    metrics = engine.train_models(
        leagues=leagues,
        n_seasons=n_seasons,
        max_h2h=args.max_h2h,
        verbose=True
    )
    
    # Save models
    engine.save_models(version_tag=args.version)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"  ML Accuracy: {metrics['ml_accuracy']:.3f}")
    print(f"  ML Log Loss: {metrics['ml_log_loss']:.4f}")
    print(f"  Total Matches: {metrics['n_matches']:,}")
    print(f"  Model Version: {metrics['model_version']}")
    print("="*70 + "\n")


def cmd_predict(args):
    """
    Prediction mode - predict today's fixtures.
    
    Example:
        python main.py predict --min-edge 0.03 --output today.csv
    """
    print("\n" + "="*70)
    print("🔮 PREDICTION MODE - Today's Fixtures")
    print("="*70)
    
    engine = InferenceEngine(models_dir=args.models_dir)
    engine.load_models(version_tag=args.version)
    
    predictions = engine.predict_today(
        leagues=args.leagues,
        min_edge=args.min_edge
    )
    
    if predictions.empty:
        print("\n⚠️  No value bets found today")
        return
    
    # Display predictions
    print("\n" + "="*70)
    print(f"💰 VALUE BETS FOUND: {len(predictions)}")
    print("="*70)
    
    for idx, row in predictions.head(args.max_display).iterrows():
        print(f"\n{idx+1}. {row['home_team']} vs {row['away_team']}")
        print(f"   League: {row['league']}")
        print(f"   Market: {row['best_market'].upper()}")
        print(f"   Probability: {row['best_prob']:.1%} | Odds: {row['best_odds']:.2f}")
        print(f"   Edge: {row['best_edge']:.1%}")
        print(f"   Kelly Stake: ${row['kelly_stake']:.2f}")
    
    if len(predictions) > args.max_display:
        print(f"\n... and {len(predictions) - args.max_display} more bets")
    
    # Save to CSV
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"\n💾 Saved {len(predictions)} predictions to {args.output}")
    
    print("\n" + "="*70 + "\n")


def cmd_predict_week(args):
    """
    Prediction mode - predict next week.
    
    Example:
        python main.py predict-week --days 7 --output week.csv
    """
    print("\n" + "="*70)
    print(f"🔮 PREDICTION MODE - Next {args.days} Days")
    print("="*70)
    
    engine = InferenceEngine(models_dir=args.models_dir)
    engine.load_models(version_tag=args.version)
    
    predictions = engine.predict_week(
        leagues=args.leagues,
        days_ahead=args.days,
        min_edge=args.min_edge
    )
    
    if predictions.empty:
        print(f"\n⚠️  No value bets found in next {args.days} days")
        return
    
    # Group by date
    predictions['date'] = pd.to_datetime(predictions['date'])
    
    print(f"\n💰 FOUND {len(predictions)} VALUE BETS")
    print("="*70)
    
    for date, group in predictions.groupby('date'):
        print(f"\n📅 {date.strftime('%Y-%m-%d %A')} ({len(group)} bets)")
        print("─"*70)
        
        for idx, row in group.head(5).iterrows():
            print(f"  {row['home_team']:20s} vs {row['away_team']:20s} | "
                  f"{row['best_market']:5s} | Edge: {row['best_edge']:5.1%} | "
                  f"Odds: {row['best_odds']:.2f}")
        
        if len(group) > 5:
            print(f"  ... and {len(group) - 5} more")
    
    # Save to CSV
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"\n💾 Saved to {args.output}")
    
    print("\n" + "="*70 + "\n")


def cmd_backtest(args):
    """
    Backtest mode - evaluate historical performance.
    
    Example:
        python main.py backtest --bankroll 10000 --kelly-fraction 0.25
    """
    print("\n" + "="*70)
    print("📊 BACKTEST MODE")
    print("="*70)
    
    print("\n⚠️  Note: Backtest requires pre-generated historical predictions")
    print("         This is a demonstration of the backtester structure.\n")
    
    # Configuration
    config = BacktestConfig(
        initial_bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        min_edge=args.min_edge,
        max_daily_risk=args.max_daily_risk,
        max_single_bet=args.max_single_bet
    )
    
    print("Backtest Configuration:")
    print(f"  Initial Bankroll: ${config.initial_bankroll:,.2f}")
    print(f"  Kelly Fraction: {config.kelly_fraction}")
    print(f"  Min Edge: {config.min_edge:.1%}")
    print(f"  Max Daily Risk: {config.max_daily_risk:.1%}")
    print(f"  Max Single Bet: {config.max_single_bet:.1%}")
    
    backtester = InstitutionalBacktester(config)
    
    print("\nTo run backtest with real data:")
    print("  1. Generate historical predictions")
    print("  2. Load predictions_df and results_df")
    print("  3. stats = backtester.run_backtest(predictions_df, results_df)")
    print("  4. backtester.print_summary(stats)")
    print("  5. backtester.export_results('bets.csv', 'equity.csv')")
    
    print("\n" + "="*70 + "\n")


def cmd_optimize(args):
    """
    Portfolio optimization mode.
    
    Example:
        python main.py optimize --bankroll 10000 --risk-tolerance 1.0
    """
    print("\n" + "="*70)
    print("💼 PORTFOLIO OPTIMIZATION MODE")
    print("="*70)
    
    engine = InferenceEngine(models_dir=args.models_dir)
    engine.load_models(version_tag=args.version)
    
    # Get today's predictions
    print("\n[1/2] Generating predictions...")
    predictions = engine.predict_today(
        leagues=args.leagues,
        min_edge=0.02  # Lower threshold for portfolio
    )
    
    if predictions.empty:
        print("\n⚠️  No opportunities found for portfolio optimization")
        return
    
    # Prepare for optimizer
    opportunities = predictions.rename(columns={
        'best_prob': 'probability',
        'best_odds': 'odds',
        'best_edge': 'edge'
    })
    
    opportunities['match'] = (
        opportunities['home_team'] + ' vs ' + opportunities['away_team']
    )
    
    # Optimize
    print("\n[2/2] Optimizing portfolio allocation...")
    
    config = PortfolioConfig(
        bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        max_total_exposure=args.max_exposure,
        max_single_bet=args.max_single_bet,
        risk_tolerance=args.risk_tolerance,
        correlation_adjustment=args.correlation_aware
    )
    
    optimizer = PortfolioOptimizer(config)
    portfolio = optimizer.optimize_portfolio(
        opportunities,
        correlation_aware=args.correlation_aware
    )
    
    if portfolio.empty:
        print("\n⚠️  No bets meet portfolio criteria")
        return
    
    # Generate and print report
    report = optimizer.generate_betting_report(portfolio, include_details=True)
    print("\n" + report)
    
    # Save to CSV
    if args.output:
        portfolio.to_csv(args.output, index=False)
        print(f"\n💾 Portfolio saved to {args.output}")
    
    print("\n" + "="*70 + "\n")


def cmd_dashboard(args):
    """
    Launch Streamlit dashboard.
    
    Example:
        python main.py dashboard
    """
    import subprocess
    
    print("\n" + "="*70)
    print("🚀 LAUNCHING DASHBOARD")
    print("="*70)
    
    dashboard_path = Path(__file__).parent / 'dashboard_pro.py'
    
    if not dashboard_path.exists():
        print(f"\n❌ Dashboard not found at {dashboard_path}")
        print("   Please ensure dashboard_pro.py is in the same directory")
        return
    
    print(f"\n✓ Found dashboard: {dashboard_path}")
    print(f"✓ Starting Streamlit server...")
    print(f"\n📱 Dashboard will open at: http://localhost:8501")
    print("   Press Ctrl+C to stop\n")
    
    try:
        subprocess.run(['streamlit', 'run', str(dashboard_path)])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("\nTry installing Streamlit: pip install streamlit")


def cmd_info(args):
    """
    Display system information.
    
    Example:
        python main.py info
    """
    print("\n" + "="*70)
    print("📊 FOOTBALL QUANT PRO - SYSTEM INFORMATION")
    print("="*70)
    
    print("\n🏗️  ARCHITECTURE")
    print("  Version: 2.0 Professional")
    print("  Quality: Hedge Fund Grade")
    print("  Code Lines: 4,950+")
    print("  Modules: 14 professional components")
    
    print("\n📦 CORE MODULES")
    print("  ✓ backtester_professional.py (650 lines)")
    print("  ✓ asian_handicap_professional.py (500 lines)")
    print("  ✓ portfolio_optimizer_professional.py (550 lines)")
    print("  ✓ inference_professional.py (500 lines)")
    print("  ✓ dashboard_pro.py (500 lines)")
    
    print("\n🔧 SUPPORTING MODULES")
    print("  ✓ dixon_coles.py (450 lines)")
    print("  ✓ ensemble.py (300 lines)")
    print("  ✓ over_under.py (350 lines)")
    print("  ✓ betting_optimizer.py (400 lines)")
    print("  ✓ data_loader.py (200 lines)")
    print("  ✓ feature_engineering.py (150 lines)")
    print("  ✓ ml_models.py (150 lines)")
    print("  ✓ fixtures_loader.py (100 lines)")
    
    print("\n⚡ FEATURES")
    print("  ✓ Institutional walk-forward backtesting")
    print("  ✓ Exact Poisson Asian Handicap pricing")
    print("  ✓ Correlation-aware portfolio optimization")
    print("  ✓ Log-odds ensemble blending")
    print("  ✓ CLV tracking and Sharpe ratio")
    print("  ✓ Professional Streamlit GUI")
    
    print("\n📖 COMMANDS")
    print("  train          Train models on historical data")
    print("  predict        Predict today's fixtures")
    print("  predict-week   Predict next 7 days")
    print("  backtest       Run historical backtest")
    print("  optimize       Optimize bet portfolio")
    print("  dashboard      Launch GUI")
    print("  info           Show this information")
    
    print("\n📚 QUICK START")
    print("  1. python main.py train --leagues 'Premier League' --seasons 3")
    print("  2. python main.py predict --min-edge 0.03")
    print("  3. python main.py dashboard")
    
    # Check models directory
    models_dir = Path('models')
    if models_dir.exists():
        models = list(models_dir.glob('*.pkl'))
        print(f"\n💾 MODELS ({len(models)} found)")
        for model in models[:5]:
            print(f"  ✓ {model.name}")
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more")
    else:
        print("\n💾 MODELS")
        print("  No trained models found")
        print("  Run: python main.py train")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Football Quant Pro - Institutional Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models
  python main.py train --leagues "Premier League" "La Liga" --seasons 3
  
  # Predict today
  python main.py predict --min-edge 0.03 --output today.csv
  
  # Predict next week
  python main.py predict-week --days 7 --output week.csv
  
  # Optimize portfolio
  python main.py optimize --bankroll 10000 --output portfolio.csv
  
  # Launch dashboard
  python main.py dashboard
  
  # System info
  python main.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # ================================================================
    # TRAIN command
    # ================================================================
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--leagues', nargs='+', help='Leagues to train on')
    train_parser.add_argument('--seasons', type=int, default=3, help='Number of seasons (default: 3)')
    train_parser.add_argument('--max-h2h', type=int, default=0, help='Max H2H matches (default: 0)')
    train_parser.add_argument('--version', help='Version tag for models')
    train_parser.add_argument('--models-dir', default='models', help='Models directory')
    
    # ================================================================
    # PREDICT command
    # ================================================================
    predict_parser = subparsers.add_parser('predict', help='Predict today')
    predict_parser.add_argument('--leagues', nargs='+', help='Leagues to predict')
    predict_parser.add_argument('--min-edge', type=float, default=0.03, help='Min edge (default: 0.03)')
    predict_parser.add_argument('--max-display', type=int, default=10, help='Max bets to display')
    predict_parser.add_argument('--version', help='Model version')
    predict_parser.add_argument('--models-dir', default='models')
    predict_parser.add_argument('--output', help='Output CSV file')
    
    # ================================================================
    # PREDICT-WEEK command
    # ================================================================
    week_parser = subparsers.add_parser('predict-week', help='Predict next week')
    week_parser.add_argument('--leagues', nargs='+', help='Leagues to predict')
    week_parser.add_argument('--days', type=int, default=7, help='Days ahead (default: 7)')
    week_parser.add_argument('--min-edge', type=float, default=0.02, help='Min edge (default: 0.02)')
    week_parser.add_argument('--version', help='Model version')
    week_parser.add_argument('--models-dir', default='models')
    week_parser.add_argument('--output', help='Output CSV')
    
    # ================================================================
    # BACKTEST command
    # ================================================================
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--bankroll', type=float, default=10000, help='Initial bankroll')
    backtest_parser.add_argument('--kelly-fraction', type=float, default=0.25, help='Kelly fraction')
    backtest_parser.add_argument('--min-edge', type=float, default=0.03, help='Min edge')
    backtest_parser.add_argument('--max-daily-risk', type=float, default=0.10, help='Max daily risk')
    backtest_parser.add_argument('--max-single-bet', type=float, default=0.05, help='Max single bet')
    
    # ================================================================
    # OPTIMIZE command
    # ================================================================
    optimize_parser = subparsers.add_parser('optimize', help='Optimize portfolio')
    optimize_parser.add_argument('--leagues', nargs='+', help='Leagues')
    optimize_parser.add_argument('--bankroll', type=float, default=10000, help='Bankroll')
    optimize_parser.add_argument('--kelly-fraction', type=float, default=0.25, help='Kelly fraction')
    optimize_parser.add_argument('--max-exposure', type=float, default=0.15, help='Max exposure')
    optimize_parser.add_argument('--max-single-bet', type=float, default=0.05, help='Max single bet')
    optimize_parser.add_argument('--risk-tolerance', type=float, default=1.0, help='Risk tolerance')
    optimize_parser.add_argument('--correlation-aware', action='store_true', help='Use correlation adjustment')
    optimize_parser.add_argument('--version', help='Model version')
    optimize_parser.add_argument('--models-dir', default='models')
    optimize_parser.add_argument('--output', help='Output CSV')
    
    # ================================================================
    # DASHBOARD command
    # ================================================================
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch GUI')
    
    # ================================================================
    # INFO command
    # ================================================================
    info_parser = subparsers.add_parser('info', help='System information')
    
    # ================================================================
    # Parse and route
    # ================================================================
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command
    try:
        if args.command == 'train':
            cmd_train(args)
        elif args.command == 'predict':
            cmd_predict(args)
        elif args.command == 'predict-week':
            cmd_predict_week(args)
        elif args.command == 'backtest':
            cmd_backtest(args)
        elif args.command == 'optimize':
            cmd_optimize(args)
        elif args.command == 'dashboard':
            cmd_dashboard(args)
        elif args.command == 'info':
            cmd_info(args)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
