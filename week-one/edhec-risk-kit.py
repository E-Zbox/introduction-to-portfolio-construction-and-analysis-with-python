import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import minimize


def create_drawdown(return_data: pd.Series, capital=1000):
    """
    Takes a time series of asset returns.
    returns a DataFrame with columns for
        - the wealth index
        - the previous peaks, and
        - the percentage drawdown
    """
    wealth_index = capital * (return_data + 1).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks

    return pd.DataFrame(index=return_data.index, data={
        "Drawdown": drawdown,
        "Peaks": previous_peaks,
        "Wealth Index": wealth_index
    })


def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by Marketcap
    """
    me_m = pd.read_csv("../data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    returns = me_m[['Lo 10', 'Hi 10']]
    returns.columns = ['SmallCap', 'LargeCap']

    returns = returns / 100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period("M")

    return returns


def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("../data/edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period("M")

    return hfi


def get_industry_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    industry_returns = pd.read_csv(
        "../data/ind30_m_vw_rets.csv", header=0, index_col=0, parse_dates=True, na_values=-99.99) / 100
    industry_returns.index = pd.to_datetime(
        industry_returns.index, format="%Y%m").to_period("M")

    industry_returns.columns = industry_returns.columns.str.strip()

    return industry_returns


def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def skewness(r: pd.Series):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()

    return exp / (sigma_r ** 3)


def kurtosis(r: pd.Series):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 4).mean()

    return exp / (sigma_r ** 4)


def is_normal(r: pd.Series, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    statistics, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")


def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = scipy.stats.norm.ppf(level / 100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (
            z +
            (z**2 - 1)*s/6 +
            (z**3 - 3*z)*(k-3)/24 -
            (2*z**3 - 5*z) * (s ** 2)/36
        )

    return -(r.mean() + z * r.std(ddof=0))


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def annualize_volatility(returns: pd.Series, months_per_year=12):
    """
    Annualizes the volatility of a set of returns
    """

    return returns.std() * np.sqrt(months_per_year)


def annualize_returns(returns: pd.Series, months_per_year: int = 12):
    """
    Annualizes a set of returns
    """
    n_months = returns.shape[0]
    return (returns + 1).prod() ** (months_per_year / n_months) - 1


def sharpe_ratio(returns: pd.Series, months_per_year: int, risk_free_rate=0.03):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    annualized_return = annualize_returns(returns, months_per_year)
    annualized_vol = annualize_volatility(returns)

    return (annualized_return - risk_free_rate) / annualized_vol


def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns


def portfolio_volatility(weights, covariance_matrix):
    """
    Weights -> Volatility
    """
    return (weights.T @ covariance_matrix @ weights) ** 0.5


def plot_efficient_frontier(n_points, returns, covariance_matrix, style=".-"):
    """
    Plots the 2-asset efficient frontier
    """
    if returns.shape[0] != 2:
        raise ValueError("Plot_ef2 can only plot 2-asset frontiers")

    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    portfolio_returns = [portfolio_return(w, returns) for w in weights]
    portfolio_volatilities = [portfolio_volatility(
        w, covariance_matrix) for w in weights]

    df_portfolio = pd.DataFrame({
        "Portfolio Returns": portfolio_returns,
        "Portfolio Volatilities": portfolio_volatilities
    })

    figure = plt.figure(figsize=(20, 20))

    return df_portfolio.plot(x="Portfolio Volatilities", y="Portfolio Returns", style=style)


def minimize_volatility(target_return, expected_return, cov):
    """
    target_return -> w (Weight vector)
    """
    n = expected_return.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n

    # constraints #1
    return_is_target = {
        'type': 'eq',
        'args': (expected_return,),
        'fun': lambda weights, expected_return: target_return - portfolio_return(weights, expected_return)
    }

    # constraints #2
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    results = minimize(
        portfolio_volatility, init_guess,
        args=(cov,), method="SLSQP", options={"disp": False},
        constraints=(return_is_target, weights_sum_to_1),
        bounds=bounds
    )

    return results.x


def optimal_weights(n_points, expected_return, cov):
    """
    Generates list of weights to run the optimizer on to minimize the volatility
    """
    target_returns = np.linspace(
        expected_return.min(), expected_return.max(), n_points)
    weights = [minimize_volatility(
        target_return=target_return, expected_return=expected_return, cov=cov) for target_return in target_returns]

    return weights


def plot_efficient_frontier(n_points, expected_return, covariance_matrix):
    """
    Plots the multi-asset efficient frontier
    """

    weights = optimal_weights(n_points, expected_return, covariance_matrix)
    portfolio_returns = [portfolio_return(
        w, expected_return) for w in weights]
    portfolio_volatilities = [portfolio_volatility(
        w, covariance_matrix) for w in weights]

    df_portfolio = pd.DataFrame({
        "Portfolio Returns": portfolio_returns,
        "Portfolio Volatilities": portfolio_volatilities
    })

    figure = plt.figure(figsize=(20, 20))

    return df_portfolio.plot(x="Portfolio Volatilities", y="Portfolio Returns", style=".-")
