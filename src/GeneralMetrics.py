import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from scipy.stats import skew, kurtosis

def plot_residual_histogram(residuals, bins=30, title="Residual Histogram"):
    plt.hist(residuals, bins=bins, alpha=0.7)
    plt.title(title)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.show()

def plot_qq(residuals, title="Q-Q Plot of Residuals"):
    sm.qqplot(residuals, line='s')
    plt.title(title)
    plt.show()

def plot_residual_acf(residuals, lags=40, title="Residual Autocorrelation"):
    sm.graphics.tsa.plot_acf(residuals, lags=lags)
    plt.title(title)
    plt.show()

def heteroscedasticity_tests(residuals, exog):
    """
    Run Breusch-Pagan and White tests for heteroscedasticity
    """
    bp_test = het_breuschpagan(residuals, exog)
    white_test = het_white(residuals, exog)

    results = {
        "Breusch-Pagan": {
            "Lagrange multiplier statistic": bp_test[0],
            "p-value": bp_test[1],
            "f-value": bp_test[2],
            "f p-value": bp_test[3]
        },
        "White": {
            "Test statistic": white_test[0],
            "p-value": white_test[1],
            "f-value": white_test[2],
            "f p-value": white_test[3]
        }
    }
    return results

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def median_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs(y_true - y_pred))

def tracking_error_volatility(replication_errors, fund_values, start_idx=None, end_idx=None):
    if start_idx is not None and end_idx is not None:
        replication_errors = replication_errors[start_idx:end_idx]
        fund_values = fund_values[start_idx:end_idx]
    relative_errors = replication_errors / fund_values
    return np.std(relative_errors) * 100  # percent

def hedge_ratio_stability(proxy_sensitivities, start_idx=None, end_idx=None):
    """
    Returns volatility (std dev) per proxy time series
    """
    if start_idx is not None and end_idx is not None:
        proxy_sensitivities = proxy_sensitivities[start_idx:end_idx]
    return np.std(proxy_sensitivities, axis=0)

def hedge_effectiveness_ratio(option_pnl, hedge_pnl, start_idx=None, end_idx=None):
    if start_idx is not None and end_idx is not None:
        option_pnl = option_pnl[start_idx:end_idx]
        hedge_pnl = hedge_pnl[start_idx:end_idx]
    var_unhedged = np.var(option_pnl)
    var_hedged = np.var(option_pnl - hedge_pnl)
    if var_hedged == 0:
        return np.inf
    return var_unhedged / var_hedged

def max_drawdown(series, start_idx=None, end_idx=None):
    if start_idx is not None and end_idx is not None:
        series = series[start_idx:end_idx]
    cumulative = np.cumsum(series)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = peak - cumulative
    return np.max(drawdowns)

def value_at_risk(series, alpha=0.05, start_idx=None, end_idx=None):
    if start_idx is not None and end_idx is not None:
        series = series[start_idx:end_idx]
    return np.percentile(series, 100 * alpha)

def conditional_value_at_risk(series, alpha=0.05, start_idx=None, end_idx=None):
    if start_idx is not None and end_idx is not None:
        series = series[start_idx:end_idx]
    var = value_at_risk(series, alpha)
    tail_losses = series[series <= var]
    if len(tail_losses) == 0:
        return var
    return np.mean(tail_losses)

def skewness(series, start_idx=None, end_idx=None):
    if start_idx is not None and end_idx is not None:
        series = series[start_idx:end_idx]
    return skew(series)

def kurtosis_excess(series, start_idx=None, end_idx=None):
    if start_idx is not None and end_idx is not None:
        series = series[start_idx:end_idx]
    return kurtosis(series) - 3
