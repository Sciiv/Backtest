import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # For Python 3.9+; use 'pytz' if needed

# Initialize MT5 connection
if not mt5.initialize():
    print("MT5 initialize() failed, error code:", mt5.last_error())
    quit()

# Set timezone manually (e.g., CDT for your location)
timezone = ZoneInfo("America/Chicago")  # CDT (UTC-5 during DST on Sep 24, 2025)
utc_from = datetime.now(tz=ZoneInfo("UTC")) - timedelta(days=7)  # Last 7 days

# Pull historical M1 data for EURUSD
symbol = "EURUSD"
rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, utc_from, 10080)  # 10080 M1 bars = 7 days

if rates is None:
    print("Failed to pull rates:", mt5.last_error())
else:
    df = pd.DataFrame(rates)
    # Convert time column to datetime and set as index
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    # Convert to local timezone (CDT)
    df.index = df.index.tz_convert(timezone)
    print(f"Pulled {len(df)} M1 bars for {symbol}")
    print(df.tail())  # Sample output

# Pull recent trade history (e.g., last 10 deals) and compute returns
deals = mt5.history_deals_get(utc_from, datetime.now(tz=ZoneInfo("UTC")))
if deals:
    trade_returns = []
    for deal in deals[-10:]:  # Last 10 deals
        if deal.symbol == symbol and deal.type in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL]:
            profit = deal.profit  # Raw profit in account currency
            volume = deal.volume
            return_pct = (profit / (volume * 100000)) * 100  # Approx % return (adjust for pip value)
            trade_returns.append(return_pct / 100)  # As decimal
    print(f"Recent trade returns: {trade_returns}")
else:
    # Fallback: Simulate returns from rates (e.g., based on EMA signals)
    df['ema_fast'] = df['close'].ewm(span=2).mean()  # Reduced to 2 for more signals
    df['ema_slow'] = df['close'].ewm(span=5).mean()  # Reduced to 5 for more signals
    # Convert signals to pandas Series with df index
    signals = pd.Series(np.where((df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) & (df['ema_fast'] > df['ema_slow']), 1,  # Buy
                                 np.where((df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)) & (df['ema_fast'] < df['ema_slow']), -1, 0)),  # Sell
                        index=df.index)
    # Improved simulation: Hold trade until reverse signal or end, compute cumulative return
    position = 0
    entry_price = 0.0
    trade_returns = []
    trade_entries = []  # Store entry points for plotting
    trade_exits = []   # Store exit points for plotting
    for i in range(1, len(df)):
        signal = signals.iloc[i]
        if signal == 1 and position == 0:  # Open buy
            position = 1
            entry_price = df['close'].iloc[i]
            trade_entries.append((df.index[i], entry_price))
        elif signal == -1 and position == 0:  # Open sell
            position = -1
            entry_price = df['close'].iloc[i]
            trade_entries.append((df.index[i], entry_price))
        elif (signal == -1 and position == 1) or (signal == 1 and position == -1):  # Reverse
            # Close current, open new
            close_price = df['close'].iloc[i]
            ret = position * (close_price - entry_price) / entry_price
            trade_returns.append(ret)
            trade_exits.append((df.index[i-1], close_price))  # Exit at previous bar's close
            position = signal
            entry_price = close_price
            trade_entries.append((df.index[i], entry_price))
    # Close last position if open
    if position != 0:
        close_price = df['close'].iloc[-1]
        ret = position * (close_price - entry_price) / entry_price
        trade_returns.append(ret)
        trade_exits.append((df.index[-1], close_price))
    trade_returns = trade_returns[-10:]  # Last 10 "trades"
    print(f"Simulated trade returns from data: {trade_returns}")

# Quick Monte Carlo (resample returns for 1000 paths)
np.random.seed(42)
n_sims = 1000
final_returns = [np.prod(1 + np.random.choice(trade_returns, len(trade_returns), replace=True)) - 1 for _ in range(n_sims)]
mc_mean = np.mean(final_returns)
mc_p99_dd = np.percentile(final_returns, 1)  # 99% VaR proxy
print(f"MC Mean Return: {mc_mean:.4f}, 99% Worst Drawdown Proxy: {mc_p99_dd:.4f}")

# Bayesian Posterior (simple normal update)
sample_mean = np.mean(trade_returns)
posterior_mean = 0.5 * 0  # Neutral prior mean
posterior_mean += 0.5 * sample_mean  # Shrinkage example
print(f"Bayesian Posterior Mean Return: {posterior_mean:.4f}")

# Calculate VWAP for vertical histogram
df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
df['volume_weight'] = df['typical_price'] * df['tick_volume']
vwap = df['volume_weight'].sum() / df['tick_volume'].sum()
print(f"VWAP: {vwap:.5f}")

# Visualization with error handling
try:
    plt.figure(figsize=(15, 6))

    # Plot 1: Price action with trades
    plt.subplot(1, 2, 1)
    plt.plot(df.index, df['close'], label='Close Price', alpha=0.5)
    plt.plot(df.index, df['ema_fast'], label='EMA Fast (2)', alpha=0.7)
    plt.plot(df.index, df['ema_slow'], label='EMA Slow (5)', alpha=0.7)
    for entry in trade_entries:
        plt.plot(entry[0], entry[1], 'go', markersize=8, label='Entry' if entry == trade_entries[0] else "")
    for exit_ in trade_exits:
        plt.plot(exit_[0], exit_[1], 'ro', markersize=8, label='Exit' if exit_ == trade_exits[0] else "")
    plt.axhline(y=vwap, color='k', linestyle='--', label=f'VWAP: {vwap:.5f}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Price Action with Trades (Last 7 Days)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    # Plot 2: Histogram of Monte Carlo final returns
    plt.subplot(1, 2, 2)
    plt.hist(final_returns * 100, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(mc_mean * 100, color='r', linestyle='--', label=f'Mean: {mc_mean*100:.2f}%')
    plt.axvline(mc_p99_dd * 100, color='g', linestyle='--', label=f'99% DD: {mc_p99_dd*100:.2f}%')
    plt.xlabel('Final Return (%)')
    plt.ylabel('Frequency')
    plt.title('Monte Carlo: Distribution of Final Returns')
    plt.legend()
    plt.grid(True)

    # Sidebar: Vertical histogram of VWAP deviation
    plt.figure(figsize=(3, 6))
    price_deviations = (df['close'] - vwap) / vwap * 100  # Percentage deviation from VWAP
    plt.hist(price_deviations, bins=50, orientation='horizontal', alpha=0.7, edgecolor='black')
    plt.axhline(y=0, color='k', linestyle='--', label='VWAP')
    plt.ylabel('Price Deviation from VWAP (%)')
    plt.title('VWAP Deviation Histogram')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('trades_plot.png')  # Save the price action plot as PNG
    plt.show()
except Exception as e:
    print(f"Error in visualization: {str(e)}")

# Export analysis for EA
analysis_data = [posterior_mean, mc_p99_dd]
with open('analysis.csv', 'w') as f:
    for val in analysis_data:
        f.write(f"{val}\n")
print("Exported analysis to analysis.csv for EA.")

# Logic to suggest revisions
print("\nStrategy Performance Evaluation and Suggestions:")
if mc_mean < 0 and abs(mc_mean) > 0.001:  # Significant negative mean return
    print("- The strategy shows a consistent loss (MC Mean Return: {:.4f}%). Consider:".format(mc_mean))
    print("  - Increasing EMA periods (e.g., from 2/5 to 5/10) to filter noise in a flat market.")
    print("  - Adding a volatility filter (e.g., using ATR) to trade only during trending conditions.")
    print("  - Extending the data range (e.g., 14 days) to capture more market cycles.")
elif mc_p99_dd < -0.002:  # High drawdown risk
    print("- The drawdown risk is significant (99% DD: {:.4f}%). Consider:".format(mc_p99_dd))
    print("  - Tightening StopLoss (e.g., from 20 pips to 15 pips) to limit losses.")
    print("  - Reducing position sizing (e.g., lowering RiskPercent base in EA).")
else:
    print("- The strategy is marginally viable (MC Mean: {:.4f}%, 99% DD: {:.4f}%).".format(mc_mean, mc_p99_dd))
    print("  - Monitor performance with more data or optimize EMA periods for better signal quality.")
    print("  - Consider adding TakeProfit adjustments based on volatility (e.g., ATR-based TP).")

# Shutdown MT5
mt5.shutdown()