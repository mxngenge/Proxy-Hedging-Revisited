# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from OptionPricing import (bachelier_put_price, bachelier_put_delta, proxy_sensitivities,
                           generate_option_values, generate_option_deltas)
from utils import read_xlsx_file
import statsmodels.api as sm
import GeneralMetrics as gm  

# %%
# Data loading
file = "Combined_data.xlsx"
Fund = "QQQ"
ProxyCols = ["amazon", "AVGO"]  # Adjust based on Combined_data.xlsx
data = read_xlsx_file(file)
maturityDate = 252

# Extract columns
Dates = data["Date"]
X_full = data[ProxyCols].to_numpy().T  # p x N
Y_full = (data[Fund] * 100_000_000).to_numpy()  #rescale

#splitting into training and testing
train_days = 90
X_train = X_full[:, :train_days]  
Y_train = Y_full[:train_days]
X_test = X_full[:, train_days:]   
Y_test = Y_full[train_days:]

# sklearn wants data as N x p
X_train_sklearn = X_train.T  
X_test_sklearn = X_test.T    
X_full_sklearn = X_full.T   
ProxyNames = np.array(ProxyCols)

# proxy volatilities and corr mat (training data)
sigma_proxies = np.std(X_train, axis=1, ddof=1)
correlation_matrix = np.corrcoef(X_train)

# %%
# just to make sure everything is looking good
print(f"#### Training OLS Linear Model for {Fund} replication using first {train_days} days ####")
print(f"Training samples: {len(X_train_sklearn)}")
print(f"Testing samples: {len(X_test_sklearn)}")
print(f"Total samples: {len(X_full_sklearn)}")

# %%
# fit ols
model = LinearRegression(fit_intercept=True)
model.fit(X_train_sklearn, Y_train)

y_train_pred = model.predict(X_train_sklearn)
train_residuals = Y_train - y_train_pred

# extract training at test metrics
train_metrics = {
    'r_squared': r2_score(Y_train, y_train_pred),
    'rmse': np.sqrt(mean_squared_error(Y_train, y_train_pred)),
    'mae': mean_absolute_error(Y_train, y_train_pred),
    'residual_std': np.std(train_residuals)
}


y_test_pred = model.predict(X_test_sklearn)
test_residuals = Y_test - y_test_pred

test_metrics = {
    'r_squared': r2_score(Y_test, y_test_pred),
    'rmse': np.sqrt(mean_squared_error(Y_test, y_test_pred)),
    'mae': mean_absolute_error(Y_test, y_test_pred),
    'residual_std': np.std(test_residuals)
}


y_full_pred = model.predict(X_full_sklearn) # will also be used later for plotting
full_residuals = Y_full - y_full_pred

print(f"\nTraining Set Performance (first {train_days} days):")
print(f"  R-squared: {train_metrics['r_squared']:.4f}")
print(f"  RMSE: {train_metrics['rmse']:.2f}")
print(f"  MAE: {train_metrics['mae']:.2f}")
print(f"  Residual Std: {train_metrics['residual_std']:.2f}")

print(f"\nTest Set Performance (remaining {len(X_test_sklearn)} days):")
print(f"  R-squared: {test_metrics['r_squared']:.4f}")
print(f"  RMSE: {test_metrics['rmse']:.2f}")
print(f"  MAE: {test_metrics['mae']:.2f}")
print(f"  Residual Std: {test_metrics['residual_std']:.2f}")


# Residual histogram, QQ-plot, and ACF for training residuals
gm.plot_residual_histogram(train_residuals, title="Training Residual Histogram")
gm.plot_qq(train_residuals, title="Training Residual Q-Q Plot")
gm.plot_residual_acf(train_residuals, title="Training Residual ACF")

# Heteroscedasticity tests on training residuals
X_train_sm = sm.add_constant(X_train_sklearn)  # add intercept column for tests
het_results = gm.heteroscedasticity_tests(train_residuals, X_train_sm)
print("\nHeteroscedasticity Tests (Training Residuals):")
print(het_results)

# Out-of-sample metrics on test set
mape = gm.mean_absolute_percentage_error(Y_test, y_test_pred)
med_ae = gm.median_absolute_error(Y_test, y_test_pred)
print(f"\nTest Set MAPE: {mape:.4f}%")
print(f"Test Set Median Absolute Error: {med_ae:.4f}")


# %%
#
plt.figure(figsize=(15, 12))

# Plot 1: Residuals through time with training/test split
plt.subplot(2, 3, 1)
plt.plot(range(train_days), train_residuals, 'b-', label='Training', alpha=0.8)
plt.plot(range(train_days, len(full_residuals)), test_residuals, 'r-', label='Test', alpha=0.8)
plt.axvline(x=train_days, color='k', linestyle='--', alpha=0.5, label=f'Train/Test Split (day {train_days})')
plt.title("OLS Residuals Through Time")
plt.xlabel("Time")
plt.ylabel("Error")
plt.legend()

# Plot 2: Fund replication comparison
FundVals = Y_full / 100_000_000
weights = model.coef_ / 100_000_000
FundRep = y_full_pred / 100_000_000

plt.subplot(2, 3, 2)
plt.plot(FundVals, label=f"Actual {Fund}", alpha=0.8)
plt.plot(FundRep, label="OLS Replication", alpha=0.8)
plt.axvline(x=train_days, color='k', linestyle='--', alpha=0.5, label=f'Train/Test Split (day {train_days})')
plt.title(f"Fund Replication for {Fund}")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")

# Plot 3: Static weights
plt.subplot(2, 3, 3)
bars = plt.bar(range(len(weights)), weights)
plt.title(f"OLS Weights for {Fund} (trained on first {train_days} days)")
plt.xticks(range(len(ProxyNames)), ProxyNames, rotation=45)
plt.ylabel("Weight")
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{weights[i]:.3f}", ha='center', va='bottom')


# Option pricing analysis
r = 0.03        # annual risk-free rate
T = train_days + maturityDate  # end of training day is "present"
K = FundVals[train_days] * np.exp(r * (T - train_days) / 252)  # arbitrage-free pricing

# Calculate option values and sensitivities for full dataset
option_values = []
option_deltas = []
proxy_sens_series = []

for t in range(len(FundVals)):
    F_t = FundVals[t]

    # Option value and delta (volatility computed internally)
    V_t = bachelier_put_price(F_t, K, weights, sigma_proxies, correlation_matrix, T, t, r)
    delta_t = bachelier_put_delta(F_t, K, weights, sigma_proxies, correlation_matrix, T, t, r)

    option_values.append(V_t)
    option_deltas.append(delta_t)

    # Proxy sensitivities using updated proxy_sensitivities
    sens_t = proxy_sensitivities(F_t, K, weights, sigma_proxies, correlation_matrix, T, t, r)
    proxy_sens_series.append(sens_t)


# Convert lists to numpy arrays
option_values = np.array(option_values)
option_deltas = np.array(option_deltas)
proxy_sens_series = np.array(proxy_sens_series)

# Plot 4: Option values (full timeline, keep from 0 for context)
plt.subplot(2, 3, 4)
plt.plot(range(train_days, len(option_values)), option_values[train_days:], label="Option Value")
plt.axvline(x=train_days, color='k', linestyle='--', alpha=0.5, label=f'Train/Test Split (day {train_days})')
plt.axvline(x=T, color='red', linestyle='--', alpha=0.8, label=f'Maturity (T={T})')
plt.title("European Put Option Values")
plt.xlabel("Time")
plt.ylabel("Option Value")
plt.legend()

# Plot 5: Option deltas (start at train_days)
plt.subplot(2, 3, 5)
plt.plot(range(train_days, len(option_deltas)), option_deltas[train_days:], label="Option Delta")
plt.axvline(x=train_days, color='k', linestyle='--', alpha=0.5, label=f'Train/Test Split (day {train_days})')
plt.axvline(x=T, color='red', linestyle='--', alpha=0.8, label=f'Maturity (T={T})')
plt.title("Option Delta (∂P/∂F)")
plt.xlabel("Time")
plt.ylabel("Delta")
plt.legend()

# Plot 6: Proxy sensitivities (start at train_days)
plt.subplot(2, 3, 6)
for i, name in enumerate(ProxyNames):
    plt.plot(range(train_days, len(proxy_sens_series)), proxy_sens_series[train_days:, i], label=f"{name}")
plt.axvline(x=train_days, color='k', linestyle='--', alpha=0.5, label=f'Train/Test Split (day {train_days})')
plt.axvline(x=T, color='red', linestyle='--', alpha=0.8, label=f'Maturity (T={T})')
plt.title("Proxy Sensitivities (∂P/∂S_i)")
plt.xlabel("Time")
plt.ylabel("Sensitivity")
plt.legend()

plt.tight_layout()
plt.show()



# %%
# Additional plots, zooming in on the tracking
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.plot(FundVals, label=f"Actual {Fund}", alpha=0.8)
plt.plot(FundRep, label="OLS Replication", alpha=0.8)
plt.axvline(x=train_days, color='k', linestyle='--', alpha=0.5, label=f'Train/Test Split (day {train_days})')
plt.title(f"Full Period Replication for {Fund}")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(2, 2, 2)
replication_error = FundVals - FundRep
plt.plot(replication_error, alpha=0.8)
plt.axvline(x=train_days, color='k', linestyle='--', alpha=0.5, label=f'Train/Test Split (day {train_days})')
plt.title(f"Replication Error (full std={np.std(replication_error):.4f})")
plt.xlabel("Time")
plt.ylabel("Error")
plt.legend()

# Training period zoom
plt.subplot(2, 2, 3)
plt.plot(range(train_days), FundVals[:train_days], label=f"Actual {Fund}", alpha=0.8)
plt.plot(range(train_days), FundRep[:train_days], label="OLS Replication", alpha=0.8)
plt.title(f"Training Period (first {train_days} days)")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")

# Test period zoom
plt.subplot(2, 2, 4)
test_range = range(train_days, len(FundVals))
plt.plot(test_range, FundVals[train_days:], label=f"Actual {Fund}", alpha=0.8)
plt.plot(test_range, FundRep[train_days:], label="OLS Replication", alpha=0.8)
plt.title(f"Test Period (remaining {len(test_range)} days)")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")

plt.tight_layout()
plt.show()

# %%
# Summary statistics
print(f"\nOLS Summary for {Fund} (trained on first {train_days} days):")
print(f"Training period: {Dates.iloc[0]} to {Dates.iloc[train_days-1]}")
print(f"Test period: {Dates.iloc[train_days]} to {Dates.iloc[-1]}")
print(f"Intercept: {model.intercept_ / 100_000_000:.6f}")
print("Static weights (learned from training data):")
for name, weight in zip(ProxyNames, weights):
    print(f"  {name}: {weight:.6f}")

print(f"\nOption Analysis (Strike K = {K:.2f}):")
print(f"Current fund value: {FundVals[-1]:.2f}")
print(f"Current option value: {option_values[-1]:.4f}")
print(f"Current option delta: {option_deltas[-1]:.4f}")
print("Current proxy sensitivities:")
for i, name in enumerate(ProxyNames):
    print(f"  ∂P/∂{name}: {proxy_sens_series[-1, i]:.6f}")

# Hedging error and PnL calculations
hedge_pnl = np.sum(proxy_sens_series[:-1] * np.diff(X_full, axis=1).T, axis=1)
option_pnl = np.diff(option_values)
hedge_error = option_pnl - hedge_pnl


# Slice arrays accordingly for metrics
hedge_error_window = hedge_error[train_days: T]
proxy_sens_window = proxy_sens_series[train_days: T]
option_pnl_window = option_pnl[train_days: T]
hedge_pnl_window = hedge_pnl[train_days: T]
fund_values_window = Y_full[train_days: T] 

# Print hedging performance stats for that window
print(f"\nHedging Performance from end of training (day {train_days}) to maturity (day { T}):")
print(f"Hedge error std dev: {np.std(hedge_error_window):.6f}")
print(f"Hedge error mean: {np.mean(hedge_error_window):.6f}")

tracking_vol = gm.tracking_error_volatility(
    hedge_error_window,
    fund_values_window / 1e9  # scale fund values down
)
hedge_vol = gm.hedge_ratio_stability(proxy_sens_series, start_idx=train_days, end_idx= T)
hedge_eff_ratio = gm.hedge_effectiveness_ratio(option_pnl, hedge_pnl, start_idx=train_days, end_idx= T)
max_dd = gm.max_drawdown(hedge_error, start_idx=train_days, end_idx= T)
var_5 = gm.value_at_risk(hedge_error, alpha=0.05, start_idx=train_days, end_idx= T)
cvar_5 = gm.conditional_value_at_risk(hedge_error, alpha=0.05, start_idx=train_days, end_idx= T)
skew = gm.skewness(hedge_error, start_idx=train_days, end_idx= T)
kurt = gm.kurtosis_excess(hedge_error, start_idx=train_days, end_idx= T)

print("Hedge error sample:", hedge_error_window[:5])
print("Fund values sample:", fund_values_window[:5])


print(f"Strike K: {K:.4f}")
print(f"Fund value at maturity (scaled): {FundVals[-1]:.4f}")
print(f"Fund value just before maturity (scaled): {FundVals[T-1]:.4f}")

V_T_minus_1 = bachelier_put_price(FundVals[T-1], K, weights, sigma_proxies, correlation_matrix, T, T-1, r)

V_T = bachelier_put_price(FundVals[-1], K, weights, sigma_proxies, correlation_matrix, T, T, r)


print(f"Option value one day before maturity: {V_T_minus_1:.4f}")
print(f"Option value at maturity: {V_T:.4f}")


print(f"Tracking Error Volatility: {tracking_vol:.4f}%")
print(f"Hedge Ratio Volatility per Proxy: {hedge_vol}")
print(f"Hedge Effectiveness Ratio: {hedge_eff_ratio:.4f}")
print(f"Max Drawdown of Hedge Error: {max_dd:.6f}")
print(f"VaR(5%): {var_5:.6f}")
print(f"CVaR(5%): {cvar_5:.6f}")
print(f"Skewness of Hedge Error: {skew:.6f}")
print(f"Excess Kurtosis of Hedge Error: {kurt:.6f}")
