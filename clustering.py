import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def jump_condition(data, std_threshold):
    """
    Determines if the return jumps beyond a certain number of standard deviations from the moving average.

    Parameters:
    data (pd.DataFrame): DataFrame with columns 'returns', 'ma', and 'volatility'.
    std_threshold (float): The threshold in terms of number of standard deviations.

    Returns:
    pd.Series: Series of 1 and 0 where 1 indicates a jump in returns.
    """
    return ((data['returns'] > data['ma'] + std_threshold * data['volatility']) | 
            (data['returns'] < data['ma'] - std_threshold * data['volatility'])).astype(int)

def autocorr(data, window, autocorr_lag):
    """
    Calculates the autocorrelation of 'returns' over a rolling window.

    Parameters:
    data (pd.DataFrame): DataFrame with 'returns' data.
    window (int): The number of observations used for calculating the autocorrelation.
    autocorr_lag (int): The lag for the autocorrelation.

    Returns:
    pd.Series: The autocorrelation of 'returns' over a rolling window.
    """
    return data['returns'].rolling(window).apply(lambda x: x.autocorr(lag=autocorr_lag), raw=False)

def engineer_features(prices, window=20, std_threshold=2, autocorr_lag=2):
    """
    Generates features such as returns, volatility, moving average, skewness, kurtosis, autocorrelation, and jumps 
    from price data.

    Parameters:
    prices (pd.Series): The price data.
    window (int): The number of observations used for calculating the features. Default is 20.
    std_threshold (float): The threshold in terms of number of standard deviations for jump condition. Default is 2.
    autocorr_lag (int): The lag for the autocorrelation. Default is 2.

    Returns:
    pd.DataFrame: DataFrame with the newly created features.
    """
    data = pd.DataFrame()
    data['prices'] = prices
    data['returns'] = np.log(data.prices).diff()
    data['volatility'] = data['returns'].rolling(window).std()
    data['ma'] = data['returns'].rolling(window).mean()
    data['skew'] = data['returns'].rolling(window).skew()
    data['kurt'] = data['returns'].rolling(window).kurt()
    data['autocorr'] = autocorr(data, window, autocorr_lag)
    data['jump'] = jump_condition(data, std_threshold)
    return data[window:]

def extract_features(data):
    """
    Extracts all columns from the DataFrame that don't contain 'prices'.

    Parameters:
    data (pd.DataFrame): DataFrame from which to extract the columns.

    Returns:
    list: List of column names that don't contain 'prices'.
    """
    return [col for col in data if 'prices' not in col]

def kmeans_cluster(data, random_state, n_init='auto'):
    """
    Performs KMeans clustering on the DataFrame, excluding columns containing 'prices'.

    Parameters:
    data (pd.DataFrame): DataFrame on which to perform the clustering.
    random_state (int): The seed used by the random number generator.
    n_init (str, optional): Number of time the k-means algorithm will be run with different centroid seeds. Default is 'auto'.

    Returns:
    KMeans: Trained KMeans object.
    """
    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=n_init)
    features = data[[column for column in data if 'prices' not in column]]
    kmeans.fit(features)
    return kmeans

def scale(relevant_features):
    """
    Scales the features except the 'jump' variable using the StandardScaler. 

    Parameters:
    - relevant_features (pandas.DataFrame): DataFrame of relevant features including 'jump' column.

    Returns:
    - numpy.array: Scaled features array with re-appended 'jump' column.
    """
    scaler = StandardScaler()
    binary_ommitted_df = relevant_features.drop(columns='jump') # Omit the jump variable.
    scaled_data = scaler.fit_transform(np.array(binary_ommitted_df).T)
    jump_feature = np.array(relevant_features['jump']).reshape(-1, 1)
    return np.concatenate((scaled_data.T, jump_feature), axis=1) # Re-append the jump variable.

def compute_profit(i, trades, fees):
    """
    Computes profit for a given trade after accounting for exit fees.

    Parameters:
    i (int): Index for the specific trade.
    trades (pd.DataFrame): DataFrame containing details of each trade.
    fees (float): Fee associated with each trade.

    Returns:
    pd.DataFrame: Updated DataFrame with the profit calculated for the specific trade.
    """
    if trades.loc[i, 'position'] == 'short':
        trades.loc[i, 'exit_fee'] = trades.loc[i, 'shares'] * trades.loc[i, 'exit_price'] * fees
        trades.loc[i, 'profit'] = (trades.loc[i, 'shares'] * trades.loc[i, 'entry_price']) - \
                                  (trades.loc[i, 'shares'] * trades.loc[i, 'exit_price'] + trades.loc[i, 'exit_fee'])
    else:
        trades.loc[i, 'exit_fee'] = trades.loc[i, 'shares'] * trades.loc[i, 'exit_price'] * fees
        trades.loc[i, 'profit'] = (trades.loc[i, 'shares'] * trades.loc[i, 'exit_price'] - trades.loc[i, 'exit_fee']) - \
                                  (trades.loc[i, 'shares'] * trades.loc[i, 'entry_price'])
    return trades

def compute_stats(i, trades, unseen_data, prev_end_balance, fees):
    """
    Computes various statistics associated with a trade including start balance, entry fee, shares, profit, 
    end balance, and returns.

    Parameters:
    i (int): Index for the specific trade.
    trades (pd.DataFrame): DataFrame containing details of each trade.
    prev_end_balance (float): The ending balance of the previous trade.
    fees (float): Fee associated with each trade.

    Returns:
    pd.DataFrame: Updated DataFrame with statistics calculated for the specific trade.
    """
    trades.loc[i, 'start_balance'] = prev_end_balance
    trades.loc[i, 'entry_fee'] = trades.loc[i, 'start_balance'] * fees
    trades.loc[i, 'shares'] = (trades.loc[i, 'start_balance'] - trades.loc[i, 'entry_fee']) / trades.loc[i, 'entry_price']
    trades = compute_profit(i, trades, fees)
    trades.loc[i, 'fees'] = trades.loc[i, 'entry_fee'] + trades.loc[i, 'exit_fee']
    trades.loc[i, 'end_balance'] = trades.loc[i, 'start_balance'] + trades.loc[i, 'profit']
    trades.loc[i, 'returns'] = trades.loc[i, 'profit'] / trades.loc[i, 'start_balance']
    trades.loc[i, 'vol'] = unseen_data.loc[trades.loc[i, 'entry']:trades.loc[i, 'exit']]['returns'].std()
    return trades

def log_entry(current_trade, in_position, position, unseen_data, i):
    """
    Logs entry for a specific trade.

    Parameters:
    current_trade (dict): Dictionary holding the details of the current trade.
    in_position (bool): Indicates whether the algorithm is currently in a trade.
    position (str): Position type for the current trade ('short' or 'long').
    unseen_data (pd.DataFrame): DataFrame containing unseen data.
    i (int): Index for the specific trade.

    Returns:
    tuple: Updated trade dictionary and in_position status.
    """
    current_trade['position'] = position
    current_trade['entry_price'] = unseen_data.prices[i + 1]
    current_trade['entry'] = unseen_data.index[i + 1]
    in_position = True
    return current_trade, in_position

def log_exit(current_trade, in_position, unseen_data, i):
    """
    Logs exit for a specific trade.

    Parameters:
    current_trade (dict): Dictionary holding the details of the current trade.
    in_position (bool): Indicates whether the algorithm is currently in a trade.
    unseen_data (pd.DataFrame): DataFrame containing unseen data.
    i (int): Index for the specific trade.

    Returns:
    tuple: Updated trade dictionary and in_position status.
    """
    current_trade['exit_price'] = unseen_data.prices[i + 1]
    current_trade['exit'] = unseen_data.index[i + 1]
    in_position = False
    return current_trade, in_position

def set_trade_columns():
    """
    Defines the column names for a trade log.

    Returns:
    list: List of column names.
    """
    return [
        'position', 'entry', 'exit', 'entry_price', 'exit_price', 'shares', 
        'entry_fee', 'exit_fee', 'fees', 'taxes', 'profit', 'start_balance', 'end_balance', 'returns', 'vol'
    ]


def initialize_trade_log():
    """
    Initializes a DataFrame to log trades with specified column names.

    Returns:
    pd.DataFrame: Initialized DataFrame to log trades.
    """
    return pd.DataFrame(columns=set_trade_columns())

def set_regimes(data, unseen_data, kmeans, features):
    """
    Assigns a regime to each data point in data and unseen_data based on kmeans clustering.

    Parameters:
    data (pd.DataFrame): DataFrame with historical data.
    unseen_data (pd.DataFrame): DataFrame with unseen data.
    kmeans (KMeans): Trained KMeans clustering model.
    features (list): List of features used in the kmeans model.

    Returns:
    tuple: data and unseen_data DataFrames with the new 'regime' column.
    """
    data['regime'] = kmeans.predict((data[features]))
    unseen_data['regime'] = kmeans.predict(unseen_data[features])
    return data, unseen_data

def compute_total_return(trades):
    """
    Computes the total return from the logged trades.

    Parameters:
    trades (pd.DataFrame): DataFrame with logged trades.

    Returns:
    float: Total return from the trades.
    """
    return trades.end_balance.iloc[-1] / trades.start_balance.iloc[0] - 1

def compute_buy_hold_return(data):
    """
    Computes the buy-and-hold return from the data.

    Parameters:
    data (pd.DataFrame): DataFrame with price data.

    Returns:
    float: Buy-and-hold return.
    """
    return data.prices.iloc[-1] / data.prices.iloc[0] - 1

class TradeMetrics:
    """
    Compute various metrics and statistics for a set of trades.

    Attributes:
    trades (pd.DataFrame): DataFrame of trade logs.
    unseen_data (pd.DataFrame): DataFrame of unseen data.
    risk_free_rate (float): Assumed risk-free rate.
    """
    def __init__(self, trades, unseen_data, risk_free_rate):
        self.trades = trades
        self.unseen_data = unseen_data
        self.risk_free_rate = risk_free_rate
    
    def get(self):
        """
        Retrieve all the computed metrics in a DataFrame.

        Returns:
        pd.DataFrame: DataFrame of computed metrics.
        """
        stats = {}
        for method in dir(self):
            if method.startswith('_get'):
                stat_name = method[5:]
                stats[stat_name] = getattr(self, method)()
        stats = pd.DataFrame([stats])
        stats = self.__reorder_columns(stats).T
        stats.rename(columns={0: 'Trade Statistics'}, inplace=True)
        stats.index = [label.replace('_', ' ').title() for label in stats.index]
        return stats

    def _get_total_return(self):
        """Computes the total return from the trades."""
        return self.trades['end_balance'].iloc[-1] / self.trades['start_balance'].iloc[0] - 1

    def _get_total_trades(self):
        """Counts the total number of trades executed."""
        return len(self.trades)
    
    def _get_wins(self):
        """Counts the total number of profitable trades."""
        return len(self.trades[self.trades['profit'] > 0])
    
    def _get_losses(self):
        """Counts the total number of losing trades."""
        return len(self.trades[self.trades['profit'] < 0])

    def _get_win_rate(self):
        """Computes the win rate of the backtest."""
        return self._get_wins()/self._get_total_trades()
    
    def _get_average_trade_return(self):
        """Computes the average return per trade."""
        return self.trades['returns'].mean()
    
    def _get_max_drawdown(self):
        """Computes the max drawdown (%) of the backtest."""
        return ((self.trades['growth']) - self.trades['growth'].cummax()).min()
    
    def _get_max_drawdown_abs(self):
        """Computes the max drawdown ($) of the backtest."""
        return ((self.trades['profit']) - self.trades['profit'].cummax()).min()
    
    def _get_average_profit_per_winning_trade(self):
        """Computes the average of the winning trades."""
        return self.trades.loc[self.trades['profit'] > 0, 'profit'].mean()

    def _get_average_loss_per_losing_trade(self):
        """Computes the average of the losing trades."""
        return self.trades.loc[self.trades['profit'] < 0, 'profit'].mean()

    def _get_largest_winning_trade(self):
        """Computes the largest winning trade."""
        return self.trades['profit'].max()

    def _get_largest_losing_trade(self):
        """Computes the largest losing trade."""
        return self.trades['profit'].min()

    def _get_profit_factor(self):
        """Computes the profit factor."""
        return self.trades.loc[self.trades['profit'] > 0, 'profit'].sum() / -self.trades.loc[self.trades['profit'] < 0, 'profit'].sum()
    
    def _get_risk_reward_ratio(self):
        """Computes the risk-to-reward ratio."""
        return abs(self._get_average_loss_per_losing_trade() / self._get_average_profit_per_winning_trade())

    def _get_average_trade_duration(self):
        """Computes the average duration of trades."""
        return f"{(self.trades['exit'] - self.trades['entry']).mean().days}D"

    def _get_median_trade_duration(self):
        """Computes the median duration of trades."""
        return f"{(self.trades['exit'] - self.trades['entry']).median().days}D"

    def _get_profit_to_drawdown_ratio(self):
        """Computes the profit to maximum drawdown ratio."""
        return abs(self.trades['profit'].sum() / self._get_max_drawdown_abs())

    def _get_sharpe_ratio(self):
        """Computes the Sharpe ratio of the trades."""
        return ((self.trades['returns'].mean() * 252) - self.risk_free_rate) / (self.trades['returns'].std() * np.sqrt(252))

    def _get_sortino_ratio(self):
        """Computes the Sortino ratio of the trades."""
        return ((self.trades['returns'].mean() * 252) - self.risk_free_rate) / (self.trades.loc[self.trades['returns'] < 0, 'returns'].std() * np.sqrt(252))

    def _get_calmar_ratio(self):
        """Computes the Calmar ratio of the trades."""
        return self.trades['returns'].mean() / self._get_max_drawdown()

    def _get_regime_distribution(self):
        """Gets the distribution of the trading regime."""
        values = self.unseen_data['regime'].value_counts(normalize=True)
        index = values.index
        values = values.values
        pairs = list(zip(index, np.round(values, 4)))
        return pairs

    def _get_max_consecutive_losses(self):
        """Gets the maximum number of consecutive losing trades."""
        return self.trades['profit'].lt(0).astype(int).groupby(self.trades['profit'].lt(0).ne(self.trades['profit'].lt(0).shift()).cumsum()).cumsum().max()
   
    def _get_max_consecutive_wins(self):
        """Gets the maximum number of consecutive winning trades."""
        return self.trades['profit'].gt(0).astype(int).groupby(self.trades['profit'].gt(0).ne(self.trades['profit'].gt(0).shift()).cumsum()).cumsum().max()

    def _get_long_profits(self):
        """Computes the total profits from long trades."""
        return self.trades.loc[self.trades['position'] == 'long', 'profit'].sum()

    def _get_short_profits(self):
        """Computes the total profits from short trades."""
        return self.trades.loc[self.trades['position'] == 'short', 'profit'].sum()
    
    def _get_alpha(self):
        """Computes the alpha (excess return) of the strategy over a buy-and-hold strategy."""
        algo_ret = self.trades['end_balance'].iloc[-1] / self.trades['start_balance'].iloc[0] - 1
        bh_ret = self.trades['exit_price'].iloc[-1] / self.trades['entry_price'].iloc[0] - 1
        return algo_ret - bh_ret
    
    @staticmethod
    def __reorder_columns(stats):
        """Reorders the columns of the DataFrame according to the provided list."""
        return stats[
            [
               'total_return',
               'total_trades',
               'alpha',
               'wins', 
               'losses', 
               'win_rate',
               'average_trade_duration', 
               'median_trade_duration',
               'average_trade_return',
               'average_profit_per_winning_trade',
               'average_loss_per_losing_trade',
               'largest_winning_trade',
               'largest_losing_trade',
               'profit_factor',
               'risk_reward_ratio',
               'long_profits',
               'short_profits',
               'max_consecutive_wins',
               'max_consecutive_losses',
               'max_drawdown',
               'profit_to_drawdown_ratio',
               'sharpe_ratio',
               'sortino_ratio',
               'calmar_ratio',
               'regime_distribution'
            ]
        ]

def trade_by_regime(data, unseen_data, allocation, random_state, momentum_threshold=0.05, fees=0.002, regime_stability_period=1, rfr=.03):
    """
    Performs a trading strategy based on regime clustering and momentum signals.

    Parameters:
    - data (pandas.DataFrame): Historical data for model training.
    - unseen_data (pandas.DataFrame): Unseen data on which the strategy will be performed.
    - allocation (float): Initial amount of money allocated for trading.
    - random_state (int): Seed for the random number generator, for reproducibility.
    - momentum_threshold (float, optional): Threshold for considering momentum change. Default is 0.05.
    - fees (float, optional): Transaction fees. Default is 0.002.
    - regime_stability_period (int, optional): Period for checking regime stability. Default is 1.
    - rfr (float, optional): Risk-free rate. Default is 0.03.

    Returns:
    - log (dict): A dictionary containing two pandas.DataFrame: 'log' which is a DataFrame of trades, and 'stats' which is a DataFrame of computed metrics.
    """
    features = extract_features(data)
    kmeans = kmeans_cluster(data, 2, random_state)
    data, unseen_data = set_regimes(data, unseen_data, kmeans, features)
    unseen_data['momentum'] = unseen_data['prices'].pct_change(regime_stability_period)

    in_position, position, current_trade = False, None, {}
    trades = initialize_trade_log()

    for i in range(regime_stability_period - 1, len(unseen_data) - 1):

        current_regime = unseen_data.regime[i]
        past_regimes = unseen_data.regime[i-regime_stability_period:i]
        is_stable_regime = (past_regimes == current_regime).all()
        long_signal = is_stable_regime and current_regime == 1
        exit_long = position == 'long' and (current_regime == 0 or unseen_data.momentum[i] < -momentum_threshold)
        short_signal = is_stable_regime and current_regime == 0
        exit_short = position == 'short' and (current_regime == 1 or unseen_data.momentum[i] > momentum_threshold)

        if not in_position and long_signal:
            position = 'long'
            current_trade, in_position = log_entry(current_trade, in_position, position, unseen_data, i)
        
        if not in_position and short_signal:
            position = 'short'
            current_trade, in_position = log_entry(current_trade, in_position, position, unseen_data, i)

        if in_position and exit_long:
            current_trade, in_position = log_exit(current_trade, in_position, unseen_data, i)
            trades = pd.concat([trades, pd.DataFrame([current_trade])], ignore_index=True)
            position, current_trade = None, {}
            
        if in_position and exit_short:
            current_trade, in_position = log_exit(current_trade, in_position, unseen_data, i)
            trades = pd.concat([trades, pd.DataFrame([current_trade])], ignore_index=True)
            position, current_trade = None, {}

    prev_end_balance = allocation
    profit = 0
    for i in range(len(trades)):
        trades = compute_stats(i, trades, unseen_data, prev_end_balance, fees)
        prev_end_balance = trades.loc[i, 'end_balance']
        profit += trades.loc[i, 'profit']
            
    trades['growth'] = (1 + trades['returns']).cumprod() - 1
    trades.index.name = 'trade'
    trades.index += 1
    metrics = TradeMetrics(trades, unseen_data, rfr)
    log = {'log': trades, 'stats': metrics.get()}
    return log