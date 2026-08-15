# NY Liquidity Sweep EA V1 — Setup & Usage

## What this is
An MT5 Expert Advisor that treats your hypothesis — *"during the Asian
session, gold sweeps the previous New York session's high/low, then
reverses"* — as something to **measure**, not assume. It has two modes:

- **Research Mode** (default): never places trades. Every time a sweep
  happens, it watches what price does next for ~2 hours and logs the
  result — did it reverse, how far, max favorable/adverse excursion — to
  a CSV file.
- **Trading Mode**: uses the exact same sweep/structure detection to place
  real (or demo) trades with the entry/SL/TP/risk rules you configure.

**Use Research Mode first, across years of history, before ever switching
to Trading Mode.** A few weeks of visual chart-watching is not evidence;
hundreds of logged sweeps with a measured win rate is.

## Installation
1. Open MetaEditor (from MT5: Tools → MetaQuotes Language Editor, or F4).
2. File → Open → select `NY_Liquidity_Sweep_EA_V1.mq5` (or drop it into
   `MQL5/Experts/` first, then open it from there).
3. Press F7 to compile. Fix any compiler warnings your build's MT5
   version surfaces — MQL5 syntax has changed slightly across builds, so
   treat this file as a strong, well-commented starting point rather than
   a guaranteed zero-warning compile on every terminal version.
4. Attach it to an XAUUSD chart (any timeframe — internally it pulls M1
   data for precision regardless of the chart period).

## Recommended first run: pure research
1. Set `InpMode = MODE_RESEARCH`.
2. Open Strategy Tester (Ctrl+R), select this EA, symbol XAUUSD.
3. Set the date range as far back as your broker's tick data goes.
4. Model: "Every tick based on real ticks" if available (most accurate),
   otherwise "Every tick".
5. Run. Since Research Mode places no orders, this is effectively a free,
   fast simulation.
6. Find the CSV in `MQL5/Files` (or the "Common" files folder — the code
   opens it with `FILE_COMMON` so it survives across terminal profiles).
   Open it in Excel or a notebook and look at:
   - Reversal rate (did price reverse after the sweep?)
   - Average/median MFE vs MAE
   - Does it differ meaningfully between high-sweeps and low-sweeps?
   - Does it hold up across different years / volatility regimes?

If the win rate and MFE/MAE profile don't show a real edge after costs,
the hypothesis doesn't hold as stated — that's a valid and useful result,
not a failure of the EA.

## If the data supports the hypothesis: Trading Mode
1. Set `InpMode = MODE_TRADING`.
2. Configure entry type, stop-loss method, take-profit method, and risk
   management inputs to match whatever showed the best expectancy in
   Research Mode.
3. Re-run in the Strategy Tester with realistic spread/slippage settings
   for your broker before ever going live.
4. Forward-test on a demo account for a meaningful sample size before
   risking real capital.

## Important honesty notes
- The **BOS/CHoCH detection** here is a straightforward fractal-swing
  implementation (configurable lookback). It's a reasonable, commented
  starting point, but ICT/SMC practitioners sometimes want more nuanced
  structure logic (e.g. internal vs external structure, multi-timeframe
  confirmation). Treat `CheckStructureConfirmation()` as the place to
  extend if your definition differs.
- The **Fair Value Gap filter** is a plain 3-candle imbalance check, not
  a full gap-fill/mitigation model.
- Backtest quality depends entirely on your broker's historical tick
  data quality for gold — always cross-check a few logged sweeps by hand
  against the actual chart before trusting the aggregate stats.
- Nothing here is financial advice; it's a tool for testing a trading
  idea rigorously instead of on gut feel.

## Extending later (Step 17 — ML-ready structure)
The code is organized so new filters slot into `PassesAllFilters()` and
new confirmation logic slots into `CheckStructureConfirmation()` without
touching the rest of the EA — session tracking, sweep detection, risk
management, and logging are all decoupled from the entry-signal logic.
