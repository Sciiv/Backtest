import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Replace with your actual MT5 trade returns (e.g., (exit_price - entry_price) / entry_price per trade)
trade_returns = np.array([0.002, -0.001, 0.0025, -0.0005, 0.0015, 0.003, -0.0015, 0.002, 0.001, -0.0008])

# Monte Carlo: Bootstrap resampling
np.random.seed(42)
n_simulations = 1000
n_trades = len(trade_returns)
final_equities = []

for _ in range(n_simulations):
    resampled_returns = np.random.choice(trade_returns, size=n_trades, replace=True)
    final_equity = np.prod(1 + resampled_returns) - 1  # Cumulative return
    final_equities.append(final_equity)

final_equities = np.array(final_equities)

# MC Stats
mean_return = np.mean(final_equities)
median_return = np.median(final_equities)
std_return = np.std(final_equities)
win_prob = np.mean(final_equities > 0)
max_dd = np.min(final_equities)

print(f"Monte Carlo Results:")
print(f"Mean Final Return: {mean_return:.4f}")
print(f"Median Final Return: {median_return:.4f}")
print(f"Std Dev of Returns: {std_return:.4f}")
print(f"Probability of Profit: {win_prob:.4f}")
print(f"Worst Case Return: {max_dd:.4f}")

# Bayesian: Conjugate normal update for mean return
prior_mean = 0.0
prior_std = 0.001
likelihood_std = np.std(trade_returns)
n = len(trade_returns)
sample_mean = np.mean(trade_returns)
posterior_precision = (1 / prior_std**2) + (n / likelihood_std**2)
posterior_mean = (prior_mean / prior_std**2 + n * sample_mean / likelihood_std**2) / posterior_precision
posterior_std = np.sqrt(1 / posterior_precision)

print(f"\nBayesian Analysis (Posterior for Mean Trade Return):")
print(f"Prior Mean: {prior_mean:.4f}, Prior Std: {prior_std:.4f}")
print(f"Sample Mean: {sample_mean:.4f}")
print(f"Posterior Mean: {posterior_mean:.4f}")
print(f"Posterior Std: {posterior_std:.4f}")
print(f"95% Credible Interval: [{posterior_mean - 1.96*posterior_std:.4f}, {posterior_mean + 1.96*posterior_std:.4f}]")

# Plots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# MC Histogram
axs[0].hist(final_equities * 100, bins=50, alpha=0.7, edgecolor='black')
axs[0].axvline(mean_return * 100, color='r', linestyle='--', label=f'Mean: {mean_return*100:.2f}%')
axs[0].set_xlabel('Final Return (%)')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Monte Carlo: Distribution of Final Returns')
axs[0].legend()

# Bayesian Posterior
x = np.linspace(posterior_mean - 3*posterior_std, posterior_mean + 3*posterior_std, 100)
axs[1].plot(x, stats.norm.pdf(x, posterior_mean, posterior_std), 'b-', label='Posterior')
axs[1].axvline(sample_mean, color='g', linestyle='--', label=f'Sample Mean: {sample_mean:.4f}')
axs[1].set_xlabel('Mean Trade Return')
axs[1].set_ylabel('Density')
axs[1].set_title('Bayesian: Posterior for Mean Return')
axs[1].legend()

plt.tight_layout()
plt.show()  # Or plt.savefig('simulation_results.png') to save