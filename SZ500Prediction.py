from __future__ import print_function
import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from strategy import Strategy
from event import SignalEvent
from backtest import Backtest
from data import TushareDataHandler
from execution import SimulatedExecutionHandler
from portfolio import BenchmarkPortfolio

def create_lagged_series(lags=5):
    """
    This creates a Pandas DataFrame that stores the
    percentage returns of the adjusted closing value of
    a stock obtained from Yahoo Finance, along with a
    number of lagged returns from the prior trading days
    (lags defaults to 5 days). Trading volume, as well as
    the Direction from the previous day, are also included.
    :param df: ts dataframe indexed on datetime '2018-10-02'
    """
    # Obtain stock information from Yahoo Finance
    df = pd.read_csv('/Users/pzh/Documents/华兴资管/DINAH/TestData/000100.SZ.csv', index_col=0)
    ts = df.set_index('datetime').sort_index()
    start_date = ts.index[0]
    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)
    tslag["today"] = ts["pre_close"]
    tslag["vol"] = ts["vol"]
    # Create the shifted lag series of prior trading period close values
    for i in range(0, lags):
        tslag["lag%s" % str(i+1)] = ts["pre_close"].shift(i+1)
        # Create the returns DataFrame
        tsret = pd.DataFrame(index=tslag.index)
        tsret["vol"] = tslag["vol"]
        tsret["pct_chg"] = tslag["today"].pct_change()*100.0
    # If any of the values of percentage returns equal zero, set them to # a small number (stops issues with QDA model in Scikit-Learn)
    for i,x in enumerate(tsret["pct_chg"]):
        if (abs(x) < 0.0001):
            tsret["pct_chg"].loc[i] = 0.0001
            # Create the lagged percentage returns columns
            for i in range(0, lags):
                tsret["lag%s" % str(i+1)] = \
                tslag["lag%s" % str(i+1)].pct_change()*100.0
                # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["direction"] = np.sign(tsret["pct_chg"])
    tsret = tsret[tsret.index >= start_date]
    return tsret

class SZ500DailyForecastStrategy(Strategy):
    """
    SZ500 forecast strategy. It uses a Random Forest Classifier
    to predict the returns for a subsequent time period
    and then generated long/exit signals based on the prediction.
    """

    def __init__(self, bars, events):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.datetime_now = datetime.datetime.utcnow()
        #self.model_start_date = datetime.datetime(2001, 1, 10)
        #self.model_end_date = datetime.datetime(2005, 12, 31)
        self.model_start_test_date = str(datetime.datetime(2021, 1, 1))
        self.long_market = False
        self.short_market = False
        self.bar_index = 0
        self.model = self.create_symbol_forecast_model()

    def create_symbol_forecast_model(self):
        # Create a lagged series of the S&P500 US stock market index
        snpret = create_lagged_series(lags = 5).dropna()
        # Use the prior two days of returns as predictor # values, with direction as the response
        X = snpret[["lag1", "lag2"]]
        y = snpret["direction"]
        # Create training and test sets
        start_test = self.model_start_test_date
        X_train = X[X.index < start_test]
        # X_test = X[X.index >= start_test]
        y_train = y[y.index < start_test]
        # y_test = y[y.index >= start_test]
        model = RandomForestClassifier(
          n_estimators=1000, criterion='gini',
          max_depth=None, min_samples_split=2,
        min_samples_leaf=1, max_features='auto',
                    bootstrap=True, oob_score=False, n_jobs=1,
                    random_state=None, verbose=0)
        model.fit(X_train, y_train)
        return model

    def calculate_signals(self, event):
        """
        Calculate the SignalEvents based on market data.
        """
        sym = self.symbol_list[0]
        dt = self.datetime_now
        if event.type == 'MARKET':
            self.bar_index += 1
            if self.bar_index > 5:
                lags = self.bars.get_latest_bars_values(
                    self.symbol_list[0], "pct_chg", N=3
                )

                pred_series = pd.DataFrame({
                    'lag1': lags[1] * 100.0,
                    'lag2': lags[2] * 100.0},index=[0])
                pred = self.model.predict(pred_series)
                if pred < 0 and not self.long_market:
                    self.long_market = True
                    signal = SignalEvent(1, sym, dt, 'LONG', 1.0)
                    self.events.put(signal)
                if pred > 0 and self.long_market:
                    self.long_market = False
                    signal = SignalEvent(1, sym, dt, 'EXIT', 1.0)
                    self.events.put(signal)

from config import stk_lst
if __name__ == '__main__':
    bt = Backtest(csv_dir='/Users/pzh/Documents/华兴资管/DINAH/TestData',
                           symbol_list=['601166.SH'],
                           data_handler=TushareDataHandler,
                           portfolio=BenchmarkPortfolio,
                           strategy=SZ500DailyForecastStrategy,
                           execution_handler=SimulatedExecutionHandler,
                           initial_capital=200000,
                           start_date='2019-01-01',
                           heartbeat=0)
    bt.simulate_trading()
    bt.portfolio.equity_curve.to_csv('bt_result.csv')
    # bt = backtest.Backtest(csv_dir='/Users/pzh/Documents/华兴资管/DINAH/TestData',
    #                        symbol_list=stk_lst,
    #                        data_handler=TushareDataHandler,
    #                        portfolio=BenchmarkPortfolio,
    #                        strategy=MovingAverageCrossStrategy,
    #                        execution_handler=SimulatedExecutionHandler,
    #                        initial_capital=200000,
    #                        start_date='2019-01-02',
    #                        heartbeat=0)
    # bt.simulate_trading()
    # bt.portfolio.equity_curve.to_csv('bt_result.csv')
    #

