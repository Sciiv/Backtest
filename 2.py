import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # For Python 3.9+; use 'pytz' if needed

# Initialize MT5 connection
if not mt5.initialize():
    print("MT5 initialize() failed, error code:", mt5.last_error())
    quit()

# Set timezone manually (e.g., CDT for your location)
timezone = ZoneInfo("America/Chicago")  # CDT (UTC-5 during DST on Sep 24, 2025)
utc_from = datetime.now(tz=ZoneInfo("UTC")) - timedelta(days=1)  # Last 24 hours

# Pull historical M1 data for EURUSD
symbol = "EURUSD"
rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, utc_from, 1440)  # 1440 M1 bars = 1 day

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
    df['ema_fast'] = df['close'].ewm(span=3).mean()
    df['ema_slow'] = df['close'].ewm(span=10).mean()
    # Convert signals to pandas Series with df index
    signals = pd.Series(np.where((df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) & (df['ema_fast'] > df['ema_slow']), 1,  # Buy
                                 np.where((df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)) & (df['ema_fast'] < df['ema_slow']), -1, 0)),  # Sell
                        index=df.index)
    returns = df['close'].pct_change() * signals.shift(-1)  # Simplified return calc
    trade_returns = returns.dropna().tail(10).tolist()  # Last 10 "trades"
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

# Export analysis for EA
analysis_data = [posterior_mean, mc_p99_dd]
with open('analysis.csv', 'w') as f:
    for val in analysis_data:
        f.write(f"{val}\n")
print("Exported analysis to analysis.csv for EA.")

# Shutdown MT5
mt5.shutdown()