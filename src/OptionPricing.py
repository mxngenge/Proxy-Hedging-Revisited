import numpy as np
from scipy.stats import norm

def bachelier_put_price(F_t, K, weights, sigma_proxies, correlation_matrix, T, t, r=0.0):
    """
    Computes and returns the price of a European put option under Bachelier 
    Note that volatility is calculated internally from the given portfolio weights, proxy volatilities, and correlation matrix
    """
    tau = (T - t) / 252

    if tau == 0:
        return max(K - F_t, 0)
    if tau < 0:  
        return 0  # for plots

    # Compute variance from proxy assets
    variance = calculate_fund_variance(weights, sigma_proxies, correlation_matrix, tau, r)
    sigma_F = np.sqrt(variance)

    # Risk-neutral expected value at maturity
    mu_t = np.exp(r * tau) * F_t

    if sigma_F == 0:
        return np.exp(-r * tau) * max(K - mu_t, 0)

    # Standardized variable
    d = (K - mu_t) / sigma_F

    # Bachelier put price
    put_price = np.exp(-r * tau) * ((K - mu_t) * norm.cdf(d) + sigma_F * norm.pdf(d))

    return put_price


def bachelier_put_delta(F_t, K, weights, sigma_proxies, correlation_matrix, T, t, r=0.0):
    """
    Computes the delta (sensitivity to F_t) of a European put option under bachelier. Same params as bachelier_put_price
    """
    tau = (T - t) / 252

    if tau < 0:
        return 0.0  # no sensitivity after maturity
    if tau == 0:
        return -1.0 if F_t < K else 0.0  # delta at maturity is indicator function

    # Compute variance from proxy assets
    variance = calculate_fund_variance(weights, sigma_proxies, correlation_matrix, tau, r)
    sigma_F = np.sqrt(variance)

    mu_t = np.exp(r * tau) * F_t

    if sigma_F == 0:
        return -np.exp(-r * tau) if F_t < K else 0.0

    d = (K - mu_t) / sigma_F

    # Delta with respect to F_t
    delta = -np.exp(-r * tau) * norm.cdf(d) * np.exp(r * tau)

    return delta


def bachelier_put_gamma(F_t, K, weights, sigma_proxies, correlation_matrix, T, t, r=0.0):
    """
    Computes the gamma (second derivative w.r.t. F_t) of a European put option
    """
    tau = (T - t) / 252
    if tau <= 0:
        return 0.0

    # Compute variance from proxy assets
    variance = calculate_fund_variance(weights, sigma_proxies, correlation_matrix, tau, r)
    sigma_F = np.sqrt(variance)

    mu_t = np.exp(r * tau) * F_t

    if sigma_F == 0:
        return 0.0

    d = (K - mu_t) / sigma_F

    # Gamma
    gamma = np.exp(-r * tau) * norm.pdf(d) * np.exp(r * tau) / sigma_F

    return gamma


def bachelier_put_vega(F_t, K, weights, sigma_proxies, correlation_matrix, T, t, r=0.0):
    """
    Computes the vega (sensitivity to volatility) of a European put option
    """
    tau = (T - t) / 252
    if tau <= 0:
        return 0.0

    # Compute variance from proxy assets
    variance = calculate_fund_variance(weights, sigma_proxies, correlation_matrix, tau, r)
    sigma_F = np.sqrt(variance)

    mu_t = np.exp(r * tau) * F_t

    if sigma_F == 0:
        return 0.0

    d = (K - mu_t) / sigma_F

    # Vega
    vega = np.exp(-r * tau) * norm.pdf(d) * np.sqrt(tau)

    return vega


def proxy_sensitivities(F_t, K, weights, sigma_proxies, correlation_matrix, T, t, r=0.0):
    """
    Computes sensitivities of put option price to individual proxy assets
    """
    # Get the delta of the put option w.r.t. F_t (volatility computed internally)
    delta_F = bachelier_put_delta(F_t, K, weights, sigma_proxies, correlation_matrix, T, t, r)

    # Chain rule: ∂P_t/∂S_t^i = (∂P_t/∂F_t) * (∂F_t/∂S_t^i) = delta_F * α_i
    sensitivities = delta_F * np.array(weights).flatten()

    return sensitivities


def calculate_fund_variance(weights, sigma_proxies, correlation_matrix, tau, r=0.0):
    weights = np.array(weights).flatten()
    sigma_proxies = np.array(sigma_proxies)
    cov_matrix = np.outer(sigma_proxies, sigma_proxies) * correlation_matrix
    portfolio_var = weights.T @ cov_matrix @ weights

    if r != 0:
        variance = (np.exp(2 * r * tau) - 1) / (2 * r) * portfolio_var
    else:
        variance = tau * portfolio_var
    return variance

def generate_option_values(F_series, K, sigma, T, r=0.0):
    V_series = []
    for t in range(len(F_series)):
        V_t = bachelier_put_price(F_series[t], K, sigma, T, t, r)
        V_series.append(V_t)
    return np.array(V_series)

def generate_option_deltas(F_series, K, sigma, T, r=0.0):
    deltas = []
    for t in range(len(F_series)):
        delta_t = bachelier_put_delta(F_series[t], K, sigma, T, t, r)
        deltas.append(delta_t)
    return np.array(deltas)

def hedging_portfolio_value(proxy_prices, sensitivities):
    return np.sum(np.array(proxy_prices) * np.array(sensitivities))