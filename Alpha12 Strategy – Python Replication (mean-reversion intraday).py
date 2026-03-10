#!/usr/bin/env python
# coding: utf-8

# In[3]:


# # Alpha12 Strategy — Multi-Stock Backtester
# 
# This notebook replicates the **Alpha12 Pine Script strategy** in Python and runs it across multiple stocks to identify where it performs best.
# 
# **Strategy Logic:**
# - Alpha = -(VWAP / Open) × (High/Low)^1.5
# - Normalize alpha over a rolling window (default 60 bars)
# - **Long** when normalized factor < -1 (oversold)
# - **Short** when normalized factor > +1 (overbought)
# 
# ---

# ## 1. Install & Import Dependencies

# Install required libraries (run once)
get_ipython().system('pip install yfinance pandas numpy matplotlib seaborn --quiet')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', '{:.4f}'.format)
plt.style.use('dark_background')
print('✅ Libraries loaded successfully')

# ## 2. Configuration — Stocks & Parameters

# ─────────────────────────────────────────────
#  EDIT THESE TO CUSTOMISE YOUR BACKTEST
# ─────────────────────────────────────────────

tickers = [
    # Large-cap Tech (expanded Magnificent + AI/semicon/cloud leaders)
    'AAPL', 'MSFT', 'NVDA', 'META', 'GOOGL', 'GOOG', 'AMZN', 'TSLA',
    'AVGO', 'AMD', 'QCOM', 'INTC', 'CRM', 'ORCL', 'ADBE', 'NOW', 'PANW',
    'SNOW', 'PLTR', 'SHOP', 'CRWD', 'DDOG', 'ZS', 'MDB', 'NET', 'APP',
    'SMCI', 'ARM', 'MRVL', 'MU', 'KLAC', 'LRCX', 'AMAT', 'CDNS', 'SNPS',

    # Finance (major banks, investment banks, payments, fintech)
    'JPM', 'BAC', 'GS', 'WFC', 'C', 'MS', 'SCHW', 'BLK', 'BX', 'USB',
    'PNC', 'TFC', 'COF', 'MET', 'AIG', 'PRU', 'TRV', 'CB', 'AFL', 'ALL',
    'PYPL', 'SQ', 'HOOD', 'SOFI', 'MA', 'V',

    # ETFs (broad market, sector, growth/value)
    'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'ARKK', 'XLK', 'XLF', 'XLE',
    'XLV', 'XLY', 'XLI', 'XLC', 'XLB', 'XLU', 'SMH', 'IBB', 'XBI',

    # Volatile / High-beta / Disruptors (crypto proxies, growth, meme-ish)
    'COIN', 'MSTR', 'MARATHON', 'RIOT', 'BITO', 'RBLX', 'UPST', 'PATH',
    'RIVN', 'LCID', 'NIO', 'F', 'GM', 'DKNG', 'ROKU', 'U', 'ZM', 'PINS',
    'SNAP', 'BILI', 'TTD', 'OKTA', 'ESTC',

    # Energy (traditional oil/gas + some midstream/renewables)
    'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO',
    'HAL', 'BKR', 'WMB', 'KMI', 'ENB', 'TRGP', 'DVN', 'FANG', 'CTRA',

    # Additional S&P 500 heavyweights / diversifiers (to reach ~200)
    'BRK.B', 'LLY', 'JNJ', 'UNH', 'MRK', 'ABBV', 'PFE', 'ABT', 'MDT', 'TMO',
    'DHR', 'ISRG', 'REGN', 'VRTX', 'BSX', 'SYK', 'BDX', 'EW', 'ZTS',
    'COST', 'WMT', 'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'KMB', 'GIS',
    'NKE', 'LULU', 'SBUX', 'MCD', 'YUM', 'DPZ', 'CMG', 'HD', 'LOW', 'CAT',
    'DE', 'GE', 'HON', 'RTX', 'LMT', 'BA', 'GD', 'NOC', 'UPS', 'FDX', 'UNP',
    'CSX', 'NSC', 'CPRT', 'CTAS', 'PAYX', 'GILD', 'AMGN', 'BIIB', 'ILMN',
    'VRTX', 'CI', 'ELV', 'HCA', 'MCK', 'CAH', 'COR', 'CVS', 'BMY', 'GILD',
    'DUK', 'SO', 'NEE', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'ED', 'XEL',
    'T', 'VZ', 'CMCSA', 'CHTR', 'DIS', 'NFLX', 'PARA', 'WBD',
]

print(f"Total tickers: {len(tickers)}")  # Should be around 190–205 depending on exact duplicates/removals
# Strategy parameters (match your Pine Script inputs)
NORM_LEN   = 60      # Normalisation look-back period
THRESHOLD  = 1.0     # Entry threshold (|factor| > threshold)

# Data range
START_DATE = '2022-01-01'
END_DATE   = '2025-12-31'
INTERVAL   = '1d'    # '1d' daily | '1h' hourly (max 730 days for hourly)

# Commission & slippage per trade (fraction of trade value)
COMMISSION = 0.001   # 0.1%

print(f'📋 {len(TICKERS)} tickers configured | {START_DATE} → {END_DATE} | Interval: {INTERVAL}')

# ## 3. Alpha12 Strategy Engine

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Approximate intraday VWAP using cumulative (TP * Volume) / cumulative Volume.
    For daily data this is equivalent to the typical-price VWAP used in Pine.
    """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cum_tpv = (tp * df['Volume']).cumsum()
    cum_vol  = df['Volume'].cumsum()
    return cum_tpv / cum_vol


def alpha12_signals(df: pd.DataFrame, norm_len: int = 60, threshold: float = 1.0) -> pd.DataFrame:
    """
    Replicates the Pine Script Alpha12 logic:
      alpha  = -(vwap / open) * (high/low)^1.5
      factor = (alpha - sma(alpha, N)) / stdev(alpha, N)
      long   when factor < -threshold
      short  when factor >  threshold
    """
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]

    vwap  = compute_vwap(d)
    alpha = -(vwap / d['Open']) * (d['High'] / d['Low']) ** 1.5

    mean   = alpha.rolling(norm_len).mean()
    std    = alpha.rolling(norm_len).std()
    factor = (alpha - mean) / std

    d['vwap']   = vwap
    d['alpha']  = alpha
    d['factor'] = factor
    d['long']   = factor < -threshold
    d['short']  = factor >  threshold

    # Position: +1 long, -1 short, carry forward until opposite signal
    position = pd.Series(0, index=d.index, dtype=float)
    pos = 0
    for i, (lg, sh) in enumerate(zip(d['long'], d['short'])):
        if lg:
            pos = 1
        elif sh:
            pos = -1
        position.iloc[i] = pos

    d['position'] = position
    return d


def backtest(df: pd.DataFrame, commission: float = 0.001) -> dict:
    """
    Simple vectorised backtest on the signal dataframe.
    Returns a dict of performance metrics.
    """
    d = df.dropna(subset=['factor']).copy()

    # Daily log returns of the underlying
    d['ret']    = np.log(d['Close'] / d['Close'].shift(1))

    # Strategy return = position(t-1) * ret(t)
    d['strat_ret'] = d['position'].shift(1) * d['ret']

    # Detect trades (position changes) and apply commission
    d['trade']     = d['position'].diff().abs()
    d['strat_ret'] -= (d['trade'] > 0) * commission

    d['cum_ret']   = d['strat_ret'].cumsum().apply(np.exp)  # equity curve
    d['bh_ret']    = d['ret'].cumsum().apply(np.exp)        # buy-and-hold

    # ── Metrics ──────────────────────────────────────────────
    total_ret  = d['cum_ret'].iloc[-1] - 1
    bh_total   = d['bh_ret'].iloc[-1] - 1
    n_days     = len(d)
    ann_factor = 252 / n_days
    ann_ret    = (1 + total_ret) ** ann_factor - 1
    ann_vol    = d['strat_ret'].std() * np.sqrt(252)
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    roll_max   = d['cum_ret'].cummax()
    drawdown   = (d['cum_ret'] - roll_max) / roll_max
    max_dd     = drawdown.min()

    # Win rate
    trades_mask = d['strat_ret'] != 0
    wins        = (d.loc[trades_mask, 'strat_ret'] > 0).sum()
    total_trades = trades_mask.sum()
    win_rate    = wins / total_trades if total_trades > 0 else 0

    # Calmar ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        'total_return':   round(total_ret * 100, 2),
        'bh_return':      round(bh_total  * 100, 2),
        'ann_return':     round(ann_ret   * 100, 2),
        'ann_volatility': round(ann_vol   * 100, 2),
        'sharpe':         round(sharpe,  3),
        'max_drawdown':   round(max_dd   * 100, 2),
        'calmar':         round(calmar,  3),
        'win_rate':       round(win_rate * 100, 2),
        'num_trades':     int(total_trades),
        'n_bars':         n_days,
        '_df':            d,   # keep for plotting
    }


print('✅ Strategy engine defined')

# ## 4. Download Data & Run Backtest on All Stocks

results = {}
dfs     = {}
failed  = []

print(f'⬇️  Downloading {len(TICKERS)} tickers...\n')

for ticker in TICKERS:
    try:
        raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                          interval=INTERVAL, progress=False, auto_adjust=True)
        if raw.empty or len(raw) < NORM_LEN + 20:
            print(f'  ⚠️  {ticker}: not enough data — skipped')
            failed.append(ticker)
            continue

        # Flatten multi-level columns that yfinance sometimes returns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        sig = alpha12_signals(raw, norm_len=NORM_LEN, threshold=THRESHOLD)
        met = backtest(sig, commission=COMMISSION)

        results[ticker] = met
        dfs[ticker]     = met.pop('_df')   # store equity curve df separately
        print(f'  ✅ {ticker:6s}  Sharpe={met["sharpe"]:+.2f}  '
              f'Total={met["total_return"]:+.1f}%  '
              f'MaxDD={met["max_drawdown"]:+.1f}%')

    except Exception as e:
        print(f'  ❌ {ticker}: {e}')
        failed.append(ticker)

print(f'\n✅ Completed: {len(results)} stocks | Failed: {len(failed)}')

# ## 5. Results Leaderboard

summary = pd.DataFrame(results).T
summary.index.name = 'Ticker'

# Sort by Sharpe ratio (best strategy performance)
summary_sorted = summary.sort_values('sharpe', ascending=False)

print('=' * 75)
print('  ALPHA12 STRATEGY — LEADERBOARD  (sorted by Sharpe Ratio)')
print('=' * 75)

display_cols = ['sharpe', 'total_return', 'bh_return', 'ann_return',
                'ann_volatility', 'max_drawdown', 'calmar', 'win_rate', 'num_trades']

print(summary_sorted[display_cols].to_string())
print('\n🏆 Top 5 by Sharpe Ratio:')
print(summary_sorted.head(5)[['sharpe', 'total_return', 'max_drawdown', 'win_rate']])

# ## 6. Visual Ranking Charts

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Alpha12 Strategy — Cross-Stock Comparison', fontsize=16, fontweight='bold', y=1.01)

palette = ['#00d4aa' if v > 0 else '#ff4d6d' for v in summary_sorted['sharpe']]

# 1. Sharpe Ratio
ax = axes[0, 0]
bars = ax.barh(summary_sorted.index, summary_sorted['sharpe'],
               color=['#00d4aa' if v > 0 else '#ff4d6d' for v in summary_sorted['sharpe']])
ax.axvline(0, color='white', linewidth=0.8, linestyle='--')
ax.set_title('Sharpe Ratio', fontweight='bold')
ax.set_xlabel('Sharpe')

# 2. Total Return vs Buy-and-Hold
ax = axes[0, 1]
x = np.arange(len(summary_sorted))
w = 0.35
ax.bar(x - w/2, summary_sorted['total_return'], w, label='Strategy', color='#00d4aa', alpha=0.85)
ax.bar(x + w/2, summary_sorted['bh_return'],    w, label='Buy & Hold', color='#7b8cff', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(summary_sorted.index, rotation=45, ha='right', fontsize=8)
ax.axhline(0, color='white', linewidth=0.6)
ax.set_title('Total Return % vs Buy & Hold', fontweight='bold')
ax.set_ylabel('%')
ax.legend()

# 3. Max Drawdown
ax = axes[1, 0]
ax.barh(summary_sorted.index, summary_sorted['max_drawdown'].abs(),
        color='#ff4d6d', alpha=0.85)
ax.set_title('Max Drawdown % (lower is better)', fontweight='bold')
ax.set_xlabel('Drawdown %')

# 4. Win Rate
ax = axes[1, 1]
ax.barh(summary_sorted.index, summary_sorted['win_rate'],
        color=['#00d4aa' if v >= 50 else '#ff9a3c' for v in summary_sorted['win_rate']])
ax.axvline(50, color='white', linewidth=0.8, linestyle='--', label='50% line')
ax.set_title('Win Rate %', fontweight='bold')
ax.set_xlabel('Win Rate %')
ax.legend()

plt.tight_layout()
plt.savefig('alpha12_ranking.png', dpi=150, bbox_inches='tight')
plt.show()
print('📊 Chart saved as alpha12_ranking.png')

# ## 7. Equity Curves — Top 6 Performers

top6 = summary_sorted.head(6).index.tolist()

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Alpha12 — Equity Curves: Top 6 Stocks', fontsize=15, fontweight='bold')
axes = axes.flatten()

for i, ticker in enumerate(top6):
    ax  = axes[i]
    d   = dfs[ticker]
    met = results[ticker]

    ax.plot(d.index, d['cum_ret'],  color='#00d4aa', linewidth=1.5, label='Alpha12')
    ax.plot(d.index, d['bh_ret'],   color='#7b8cff', linewidth=1.2, linestyle='--', label='Buy & Hold')
    ax.fill_between(d.index, d['cum_ret'], 1,
                    where=d['cum_ret'] >= 1, alpha=0.15, color='#00d4aa')
    ax.fill_between(d.index, d['cum_ret'], 1,
                    where=d['cum_ret'] <  1, alpha=0.15, color='#ff4d6d')

    title = (f"{ticker}  |  Sharpe: {met['sharpe']:+.2f}  "
             f"Ret: {met['total_return']:+.1f}%  DD: {met['max_drawdown']:.1f}%")
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_ylabel('Equity (log=1 baseline)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('alpha12_equity_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print('📊 Chart saved as alpha12_equity_curves.png')

# ## 8. Factor Signal Plot — Best Stock Deep Dive

best_ticker = summary_sorted.index[0]
d = dfs[best_ticker]

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1.5, 1.5], hspace=0.08)

# ── Price + signals ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(d.index, d['Close'], color='white', linewidth=1, label='Close')
ax1.scatter(d.index[d['long']],  d.loc[d['long'],  'Close'],
            marker='^', color='#00d4aa', s=50, zorder=5, label='Long Entry')
ax1.scatter(d.index[d['short']], d.loc[d['short'], 'Close'],
            marker='v', color='#ff4d6d',  s=50, zorder=5, label='Short Entry')
ax1.set_title(f'{best_ticker} — Alpha12 Signal Overlay', fontweight='bold', fontsize=13)
ax1.legend(fontsize=9)
ax1.set_ylabel('Price ($)')
ax1.grid(alpha=0.2)
ax1.set_xticklabels([])

# ── Factor ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(d.index, d['factor'], color='#f9a825', linewidth=1)
ax2.axhline( THRESHOLD, color='#ff4d6d', linestyle='--', linewidth=0.9, label=f'+{THRESHOLD}')
ax2.axhline(-THRESHOLD, color='#00d4aa', linestyle='--', linewidth=0.9, label=f'-{THRESHOLD}')
ax2.axhline(0, color='grey', linewidth=0.5)
ax2.fill_between(d.index, d['factor'],  THRESHOLD,
                 where=d['factor'] >  THRESHOLD, alpha=0.25, color='#ff4d6d')
ax2.fill_between(d.index, d['factor'], -THRESHOLD,
                 where=d['factor'] < -THRESHOLD, alpha=0.25, color='#00d4aa')
ax2.set_ylabel('Factor')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.2)
ax2.set_xticklabels([])

# ── Equity curve ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.plot(d.index, d['cum_ret'], color='#00d4aa', linewidth=1.5, label='Alpha12')
ax3.plot(d.index, d['bh_ret'],  color='#7b8cff', linewidth=1,   linestyle='--', label='B&H')
ax3.axhline(1, color='grey', linewidth=0.5)
ax3.set_ylabel('Equity')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.2)

plt.savefig(f'alpha12_{best_ticker}_deep_dive.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'📊 Deep-dive chart saved for {best_ticker}')

# ## 9. Heatmap — All Metrics at a Glance

heat_cols = ['sharpe', 'total_return', 'ann_return', 'max_drawdown', 'calmar', 'win_rate']
heat_df   = summary_sorted[heat_cols].astype(float)

# Normalise each column to [0, 1] for colouring (flip max_drawdown so green = better)
normed = heat_df.copy()
for col in heat_cols:
    mn, mx = normed[col].min(), normed[col].max()
    if mx != mn:
        normed[col] = (normed[col] - mn) / (mx - mn)
        if col == 'max_drawdown':  # lower drawdown is better
            normed[col] = 1 - normed[col]

fig, ax = plt.subplots(figsize=(12, max(6, len(heat_df) * 0.45)))
sns.heatmap(normed, annot=heat_df.round(2), fmt='g',
            cmap='RdYlGn', linewidths=0.4, linecolor='#111',
            ax=ax, cbar_kws={'label': 'Relative Score (green = better)'})
ax.set_title('Alpha12 — Metrics Heatmap', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('')
ax.set_xticklabels(heat_cols, rotation=25, ha='right')

plt.tight_layout()
plt.savefig('alpha12_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('📊 Heatmap saved as alpha12_heatmap.png')

# ## 10. Parameter Sensitivity — Threshold Sweep on Best Stock

# Re-run the best stock across a range of threshold values
best_raw = yf.download(best_ticker, start=START_DATE, end=END_DATE,
                       interval=INTERVAL, progress=False, auto_adjust=True)
if isinstance(best_raw.columns, pd.MultiIndex):
    best_raw.columns = best_raw.columns.get_level_values(0)

thresholds   = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
sweep_results = []

for t in thresholds:
    sig = alpha12_signals(best_raw, norm_len=NORM_LEN, threshold=t)
    met = backtest(sig, commission=COMMISSION)
    met.pop('_df', None)
    met['threshold'] = t
    sweep_results.append(met)

sweep_df = pd.DataFrame(sweep_results).set_index('threshold')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f'{best_ticker} — Threshold Sensitivity', fontsize=13, fontweight='bold')

for ax, col, color in zip(axes,
                          ['sharpe', 'total_return', 'max_drawdown'],
                          ['#00d4aa', '#7b8cff', '#ff4d6d']):
    ax.plot(sweep_df.index, sweep_df[col], 'o-', color=color, linewidth=2, markersize=7)
    ax.axvline(THRESHOLD, color='white', linewidth=0.8, linestyle='--', label=f'Default ({THRESHOLD})')
    ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
    ax.set_xlabel('Threshold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'alpha12_{best_ticker}_sensitivity.png', dpi=150, bbox_inches='tight')
plt.show()

print('\n📊 Threshold sweep results:')
print(sweep_df[['sharpe', 'total_return', 'max_drawdown', 'num_trades']])

# ## 11. 📋 Final Recommendation

top3 = summary_sorted.head(3)

print('=' * 60)
print('  🏆  ALPHA12 STRATEGY — FINAL RECOMMENDATION')
print('=' * 60)

medals = ['🥇', '🥈', '🥉']
for rank, (ticker, row) in enumerate(top3.iterrows()):
    print(f"""
{medals[rank]}  #{rank+1}: {ticker}
   Sharpe Ratio  : {row['sharpe']:+.3f}
   Total Return  : {row['total_return']:+.1f}%
   vs Buy & Hold : {row['bh_return']:+.1f}%
   Ann. Return   : {row['ann_return']:+.1f}%
   Max Drawdown  : {row['max_drawdown']:.1f}%
   Calmar Ratio  : {row['calmar']:.3f}
   Win Rate      : {row['win_rate']:.1f}%
   No. Trades    : {int(row['num_trades'])}
""")

print('─' * 60)
print(f'➡️  Apply the Alpha12 Pine Script to {summary_sorted.index[0]} in TradingView')
print(f'   Settings: norm_len={NORM_LEN}, threshold={THRESHOLD}')
print('─' * 60)

