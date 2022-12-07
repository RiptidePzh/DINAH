# A toy example to test the effectiveness of Moving Average Crossing

from __future__ import print_function
import datetime
import numpy as np
import pandas as pd
from strategy import Strategy
from event import SignalEvent
from backtest import Backtest
from data import TushareDataHandler
from execution import SimulatedExecutionHandler
from portfolio import BenchmarkPortfolio

class MovingAverageCrossStrategy(Strategy):
    """
    Carries out a basic Moving Average Crossover strategy
    with a short/long simple weighted moving average.
    Default short/long windows are 5/30 periods respectively.
    """
    def __init__(self, bars, events, short_window=30, long_window=400):
        """
        Initialises the Moving Average Cross Strategy.
        Parameters:
        bars - The DataHandler object that provides bar information
        events - The Event Queue object.
        short_window - The short moving average lookback.
        long_window - The long moving average lookback.
        """
        self.bars = bars
        self.events = events
        self.short_window = short_window
        self.long_window = long_window
        self.symbol_list = self.bars.symbol_list
        # Set to True is a symbol is in the market
        # self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        '''
        Adds keys to the bought dictionary for all symbols,
        set them to 'OUT'
        :return:
        '''
        bought = {}
        for s in self.symbol_list:
            bought[s] = 'OUT'
        return bought

    def calculate_signals(self, event):
        """
        Generates a new set of signals based on the MAC
        SMA with the short window crossing the long window
        meaning a long entry and vice versa for a short entry.
        Parameters
        event - A MarketEvent object.
        """
        if event.type == 'MARKET':
            for s in self.symbol_list:
                bars = self.bars.get_latest_bars_values(
                    s, "pre_close", N=self.long_window
                )
                bar_date = self.bars.get_latest_bars(s, N=1)[0]['datetime']
                if not (True in np.isnan(bars)):
                    short_sma = np.mean(bars[-self.short_window:])
                    long_sma = np.mean(bars[-self.long_window:])
                    symbol = s
                    dt = datetime.datetime.utcnow()
                    sig_dir = ''

                    if short_sma > long_sma and self.bars.bought[s] == 'OUT':
                        print("LONG: %s" % bar_date)
                        sig_dir = 'LONG'
                        signal = SignalEvent(strategy_id='MAC', symbol=symbol, datetime=dt, signal_type=sig_dir, strength=1.0)
                        self.events.put(signal)
                        # self.bought[s] = 'LONG'
                    if short_sma < long_sma and self.bars.bought[s] == 'LONG':
                        print("SHORT: %s" % bar_date)
                        sig_dir = 'EXIT'
                        signal = SignalEvent(strategy_id='MAC', symbol=symbol, datetime=dt, signal_type=sig_dir, strength=1.0)
                        self.events.put(signal)
                        # self.bought[s] = 'LONG'

from config import stk_lst
import backtest

if __name__=='__main__':
    bt = backtest.Backtest(csv_dir='/Users/pzh/Documents/华兴资管/DINAH/TestData',
                           symbol_list=stk_lst,
                           data_handler=TushareDataHandler,
                           portfolio=BenchmarkPortfolio,
                           strategy=MovingAverageCrossStrategy,
                           execution_handler=SimulatedExecutionHandler,
                           initial_capital=200000,
                           start_date='2019-01-02',
                           heartbeat=0)
    bt.simulate_trading()
    bt.portfolio.equity_curve.to_csv('bt_result.csv')



