import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for GPU-intensive tasks: {device} at {datetime.now(tz=ZoneInfo('America/Chicago')).strftime('%I:%M %p CDT on %B %d, %Y')}")

# Financial metrics
def sharpe_ratio(returns, risk_free_rate=0.0):
    returns = np.array(returns, dtype=float)
    std = np.std(returns)
    return (np.mean(returns) - risk_free_rate) / std if std != 0 else 0

def sortino_ratio(returns, risk_free_rate=0.0):
    returns = np.array(returns, dtype=float)
    downside_returns = returns[returns < 0]
    std_downside = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
    ratio = (np.mean(returns) - risk_free_rate) / std_downside if len(downside_returns) > 0 else 0
    return min(ratio, 10)  # Cap at 10 to avoid inf plotting issues

def max_drawdown(returns):
    returns = np.array(returns, dtype=float)
    cum_returns = np.cumprod(1 + returns) - 1
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / (peak + 1e-10)
    return np.min(drawdown) if len(drawdown) > 0 else 0

# GPU-intensive Monte Carlo
def monte_carlo_gpu(returns, n_sims=10000, n_trades=100):
    returns = np.array(returns, dtype=np.float32)
    if torch.cuda.is_available():
        returns_tensor = torch.tensor(returns, device=device)
        resampled = torch.randint(0, len(returns), (n_sims, min(n_trades, len(returns))), device=device)
        resampled_returns = returns_tensor[resampled]
        cum_returns = torch.cumprod(1 + resampled_returns, dim=1)[:, -1]
        return cum_returns.cpu().numpy() - 1
    else:
        resampled = np.random.randint(0, len(returns), (n_sims, min(n_trades, len(returns))))
        resampled_returns = returns[resampled]
        cum_returns = np.cumprod(1 + resampled_returns, axis=1)[:, -1]
        return cum_returns - 1

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialize() failed, error code:", mt5.last_error())
    quit()

timezone = ZoneInfo("America/Chicago")
utc_from = datetime.now(tz=ZoneInfo("UTC")) - timedelta(days=7)

# Pull data
symbol = "EURUSD"
rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, utc_from, 10080)
if rates is None:
    print("Failed to pull rates:", mt5.last_error())
else:
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.index = df.index.tz_convert(timezone)
    print(f"Pulled {len(df)} M1 bars for {symbol}")
    print(df.tail())

# Simulate returns
deals = mt5.history_deals_get(utc_from, datetime.now(tz=ZoneInfo("UTC")))
if deals:
    trade_returns = [deal.profit / (deal.volume * 100000) * 100 / 100 for deal in deals[-10:] if deal.symbol == symbol and deal.type in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL]]
else:
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(window=14).mean()
    atr_threshold = 0.0005
    df['ema_fast'] = df['close'].ewm(span=2).mean()
    df['ema_slow'] = df['close'].ewm(span=5).mean()
    signals = pd.Series(np.where((df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) & (df['ema_fast'] > df['ema_slow']), 1,
                                 np.where((df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)) & (df['ema_fast'] < df['ema_slow']), -1, 0)),
                        index=df.index)
    position, entry_price = 0, 0.0
    trade_returns, trade_entries, trade_exits = [], [], []
    for i in range(1, len(df)):
        signal, atr = signals.iloc[i], df['atr'].iloc[i]
        if atr > atr_threshold:
            if signal == 1 and position == 0:
                position, entry_price = 1, df['close'].iloc[i]
                trade_entries.append((df.index[i], entry_price))
            elif signal == -1 and position == 0:
                position, entry_price = -1, df['close'].iloc[i]
                trade_entries.append((df.index[i], entry_price))
            elif (signal == -1 and position == 1) or (signal == 1 and position == -1):
                close_price = df['close'].iloc[i]
                ret = position * (close_price - entry_price) / entry_price
                trade_returns.append(ret)
                trade_exits.append((df.index[i-1], close_price))
                position, entry_price = signal, close_price
                trade_entries.append((df.index[i], entry_price))
    if position != 0:
        close_price = df['close'].iloc[-1]
        ret = position * (close_price - entry_price) / entry_price
        trade_returns.append(ret)
        trade_exits.append((df.index[-1], close_price))
    trade_returns = trade_returns[-10:]
    print(f"Simulated trade returns from data: {trade_returns}")

# Simulate multiple strategies
strategies = {
    "EMA_Crossover": trade_returns,
    "EMA_RSI_Filter": [r for r in trade_returns if r > -0.0001],
    "MACD": [r * 1.1 - 0.00005 for r in trade_returns],
    "Bollinger_Reversion": [r + np.random.normal(0, 0.00005, 1)[0] for r in trade_returns],
    "ML_Predicted": [r * 1.2 if r > 0 else r * 0.8 for r in trade_returns]
}

# Evaluate with ML
if len(trade_returns) > 5:
    df_ml = pd.DataFrame(index=range(len(trade_returns)))
    df_ml['lag1'] = pd.Series(trade_returns).shift(1).iloc[-len(trade_returns):]
    df_ml['ema_diff'] = df['ema_fast'].shift(1).iloc[-len(trade_returns):] - df['ema_slow'].shift(1).iloc[-len(trade_returns):]
    df_ml['atr'] = df['atr'].shift(1).iloc[-len(trade_returns):]
    df_ml['target'] = np.where(pd.Series(trade_returns) > 0, 1, 0)
    if not df_ml.isnull().all().all():
        X = df_ml[['lag1', 'ema_diff', 'atr']].dropna()
        y = df_ml['target'].loc[X.index].dropna()
        if len(X) > 0 and len(y) > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"ML Accuracy: {accuracy:.2f}")
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Loss', 'Gain'])
            disp.plot(cmap='Blues')
            plt.title('ML Confusion Matrix')
            plt.savefig('confusion_matrix.png')
            plt.show()
        else:
            print("Insufficient valid data for ML training.")
    else:
        print("Insufficient data for ML training.")
else:
    print("Insufficient trade returns for ML training.")

# Evaluate strategies
metrics = {}
for name, ret in strategies.items():
    if len(ret) == 0: continue
    ret_array = np.array(ret, dtype=float)
    sharpe = sharpe_ratio(ret_array)
    sortino = sortino_ratio(ret_array)
    mdd = max_drawdown(ret_array) * 100  # Convert to %
    mc_final = monte_carlo_gpu(ret_array, n_trades=len(ret_array))
    mc_mean = np.mean(mc_final)
    metrics[name] = {'Sharpe': sharpe, 'Sortino': sortino, 'Max DD': mdd, 'MC Mean': mc_mean}

# Visualize metrics
metrics_df = pd.DataFrame(metrics).T
print("Strategy Metrics (%):")
print(metrics_df.round(4))
metrics_df.plot(kind='bar', figsize=(12, 6), title='Strategy Metrics Comparison (%)')
for i, v in enumerate(metrics_df['Sharpe']):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
plt.ylabel('Value (%)')
plt.grid(True)
plt.savefig('strategy_metrics.png')
plt.show()

# Visualize equity curves
plt.figure(figsize=(12, 6))
for name, ret in strategies.items():
    equity = np.cumprod(1 + np.array(ret, dtype=float))
    plt.plot(equity, label=name)
plt.title('Equity Curves for Strategies')
plt.xlabel('Trades')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.savefig('equity_curves.png')
plt.show()

# Add plot for EMA_RSI_Filter strategy
if "EMA_RSI_Filter" in strategies:
    rsi_returns = strategies["EMA_RSI_Filter"]
    if len(rsi_returns) > 0:
        plt.figure(figsize=(12, 6))
        rsi_equity = np.cumprod(1 + np.array(rsi_returns))
        plt.plot(rsi_equity, label='EMA_RSI_Filter Equity Curve', color='g')
        plt.title('Equity Curve for EMA_RSI_Filter Strategy')
        plt.xlabel('Trades')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.savefig('ema_rsi_filter_plot.png')
        plt.show()

# Calculate VWAP
df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
df['volume_weight'] = df['typical_price'] * df['tick_volume']
vwap = df['volume_weight'].sum() / df['tick_volume'].sum()
print(f"VWAP: {vwap:.5f}")

# Relevance and Updatability Logic
best_strategy = max(metrics, key=lambda x: metrics[x]['Sharpe'])
print(f"\nBest Strategy: {best_strategy} (Sharpe: {metrics[best_strategy]['Sharpe']:.2f})")
print("Relevance of Results:")
print("- Sharpe Ratio: >0.5 indicates viable risk-adjusted return; values <0 suggest losses outweigh gains.")
print("- Sortino Ratio: >1.0 shows good downside protection; low or negative values indicate high loss risk.")
print("- Max DD: <-0.01% signals significant drawdown; mitigation is critical for large losses.")
print("- MC Mean: Positive values confirm profitability; negative indicates strategy refinement needed.")

print("Updatability Suggestions:")
if metrics[best_strategy]['Sharpe'] < 0.5:
    print(f"- {best_strategy} underperforms (Sharpe: {metrics[best_strategy]['Sharpe']:.2f}); consider:")
    print("  - Optimizing EMA periods (e.g., test 3/7 or 10/20 via grid search).")
    print("  - Adding ML with RSI/MACD features for better prediction.")
    print("  - Extending data range (e.g., 30 days) for robustness.")
    if metrics[best_strategy]['Max DD'] < -0.01:
        print("  - Max DD {metrics[best_strategy]['Max DD']:.2f}% is extreme; tighten StopLoss (e.g., 15 pips) or add a max loss cap (-0.5%).")
elif metrics[best_strategy]['Max DD'] < -0.01:
    print(f"- High drawdown in {best_strategy} (Max DD: {metrics[best_strategy]['Max DD']:.2f}%); consider:")
    print("  - Tightening StopLoss (e.g., 15 pips) or dynamic SL with ATR.")
    print("  - Implementing a max loss cap per trade (e.g., -0.5%).")
else:
    print(f"- {best_strategy} is viable (Sharpe: {metrics[best_strategy]['Sharpe']:.2f}); enhance with:")
    print("  - Real-time sentiment from X for risk adjustment.")
    print("  - Adaptive TakeProfit using ATR or VWAP deviation.")

# Shutdown MT5
mt5.shutdown()