# data.py

import datetime
import os, os.path
import pandas as pd
import Quotation
import numpy as np

from event import MarketEvent

from abc import ABCMeta, abstractmethod

class DataHandler(metaclass=ABCMeta):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).

    The goal of a (derived) DataHandler object is to output a generated
    set of bars (OLHCVI) for each symbol requested.

    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.
    """

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or fewer if less bars are available.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bar to the latest symbol structure
        for all symbols in the symbol list.
        """
        raise NotImplementedError("Should implement update_bars()")

class LocalCSVDataHandler(DataHandler):
    """
    HistoricCSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface.
    """

    def __init__(self, events, csv_dir, symbol_list):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.

        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.

        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list

        self.fields = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True

        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        """
        comb_index = None
        for s in self.symbol_list:
            # Load the CSV file with no header information, indexed on date
            self.symbol_data[s] = pd.io.parsers.read_csv(
                os.path.join(self.csv_dir, '%s.csv' % s),
                header=0, index_col=0,
                names=self.fields
            )

            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index)

            # Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

            # Reindex the dataframes
            for s in self.symbol_list:
                self.symbol_data[s] = self.symbol_data[s].reindex(index=comb_index, method='pad').iterrows()

    def _get_new_series(self, symbol):
        """
        Returns the latest series from the data feed as a pd.series of
        [symbol] + [fields].
        """
        for b in self.symbol_data[symbol]:
            yield pd.Series(data=[symbol, datetime.datetime.strptime(b[0], '%Y-%m-%d'),
                                  ]+b[1].values.tolist(),
                            index=['symbol'] + self.fields)

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
        else:
            return bars_list[-N:]

    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            try:
                series = self._get_new_series(s).__next__()
            except StopIteration:
                self.continue_backtest = False
            else:
                if series is not None:
                    self.latest_symbol_data[s].append(series)
        self.events.put(MarketEvent())

class TushareDataHandler(DataHandler):
    """
    TushareCSVDataHandler is designed to read CSV files for
    each requested symbol from tushare format csv and provide an
    interface to obtain the "latest" bar in a manner identical to
    a live trading interface.
    """

    def __init__(self, events, csv_dir, symbol_list):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.

        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.

        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        # events: queue.Queue(), store the events to be done
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list

        # bought: dic,  record the status(LONG,SHORT,OUT) of each symbol
        self.bought = dict(zip(symbol_list,len(symbol_list)*['OUT']))
        self.fields = ['ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'datetime']
        # symbol_data: dic, record the data for each symbol on a rolling basis
        self.symbol_data = {}
        self.latest_symbol_data = {}
        # continue_backtest: bool, an indicator for continuing backtesting
        self.continue_backtest = True

        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        """
        comb_index = None
        for s in self.symbol_list:
            # Load the CSV file with no header information, indexed on date
            self.symbol_data[s] = pd.io.parsers.read_csv(
                os.path.join(self.csv_dir, '%s.csv' % s),
                header=0, index_col=0,
                names=self.fields
            )
            self.symbol_data[s].index = self.symbol_data[s]['datetime']
            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index)

            # Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

        # Reindex the dataframes
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].reindex(index=comb_index, method='pad').iterrows()


    def _get_new_series(self, symbol):
        """
        Returns the latest series from the data feed as a pd.series of
        [fields].
        """
        for b in self.symbol_data[symbol]:
            yield b[1]

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
        else:
            return bars_list[-N:]

    def get_latest_bars_values(self,symbol, val_name, N=1):
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
        else:
            return np.array([bar[val_name] for bar in bars_list[-N:]])


    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            try:
                series = self._get_new_series(s).__next__()
            except StopIteration:
                self.continue_backtest = False
            else:
                if series is not None:
                    self.latest_symbol_data[s].append(series)
        self.events.put(MarketEvent())

class TopicIndexDataHandler(DataHandler):
    """
    Topic Index DataHandler is designed to read CSV files for
    each requested Indexes from local csv and provide an
    interface to obtain the "latest" bar in a manner identical to
    a live trading interface.
    """
    def __init__(self, events, csv_dir, symbol_list):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.

        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.

        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        # events: queue.Queue(), store the events to be done
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list

        # bought: dic,  record the status(LONG,SHORT,OUT) of each symbol
        self.bought = dict(zip(symbol_list,len(symbol_list)*['OUT']))
        self.fields = ['ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'datetime']
        # symbol_data: dic, record the data for each symbol on a rolling basis
        self.symbol_data = {}
        self.latest_symbol_data = {}
        # continue_backtest: bool, an indicator for continuing backtesting
        self.continue_backtest = True

        self._open_convert_csv_files()




# TODO:
class RealtimeQuotationDataHandler(DataHandler):
    """
    RealtimeQuotationDataHandler is designed to get realtime quote
    from  web pages for each requested symbol and provide an interface
    to obtain the "latest" bar in a manner identical to a live trading interface.
    """
    # Under construction :)
    pass

