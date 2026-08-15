# Football Predictor V2

## Files
- `football_predictor_v2.ipynb` — full upgraded notebook (data loading, team-match
  history, unified feature engineering, Elo, Dixon-Coles with walk-forward xi
  tuning, XGBoost walk-forward OOF, calibration, ensemble weight optimization,
  Asian Handicap settlement, no-vig market pricing, EV/Bet Score, walk-forward
  backtest, CLV, GUI, and an OLD-vs-NEW performance report).
- `app.py` — **single self-contained Streamlit app**: all the model logic (team
  history, feature builder, Elo, Dixon-Coles, XGBoost, calibration, ensemble,
  markets, Asian Handicap, Bet Score) plus the UI, in one file. No other project
  files needed.
- `requirements.txt` — Python dependencies.

## Running the Streamlit app
```
pip install -r requirements.txt
streamlit run app.py
```
In the sidebar: pick how many seasons/leagues to load, click
**"Load Data & Train Models"** (downloads from football-data.co.uk and trains
all models — takes a minute or two), then use the tabs:
- **Predict a Match** — pick a fixture, enter odds, get ensemble probabilities,
  model agreement, expected goals, top scorelines, no-vig market comparison,
  a BET/LEAN/NO BET decision, and optional Asian Handicap settlement.
- **Model Performance** — walk-forward log-loss/Brier by season, calibration
  before/after, per-league Dixon-Coles ξ, optimal ensemble weight.
- **Backtest** — adjustable probability/EV thresholds, bankroll curve,
  ROI/yield/drawdown/CLV, trade log.

## Notes
- The app uses the exact same feature-building function for training and live
  prediction — no separate "averaging" logic for manual entries.
- The notebook's Section 22 report re-runs your original V1 feature-engineering
  function through the same walk-forward harness as V2 for an honest,
  non-cherry-picked comparison.
- Both files were validated end-to-end against a synthetic dataset in this
  sandbox (network access to football-data.co.uk is blocked here) — run them
  in your own environment for real data.
