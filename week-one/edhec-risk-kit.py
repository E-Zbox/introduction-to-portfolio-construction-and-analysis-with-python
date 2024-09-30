import pandas as pd


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
    me_m = pd.read_csv("./data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    returns = me_m[['Lo 10', 'Hi 10']]
    returns.columns = ['SmallCap', 'LargeCap']

    returns = returns / 100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period("M")

    return returns
