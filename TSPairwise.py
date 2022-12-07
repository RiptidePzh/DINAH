# Conduct Time Series Analysis On two selected stocks
# Stk1 denotes the benchmark, which will be our x
# Stk2 denotes the response y

from __future__ import print_function
import os
# Import the Time Series library
import statsmodels.tsa.stattools as ts
# Import Datetime and the Pandas DataReader
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
plt.style.use('dark_background')

def hurst_test(ts):
    """
    :param ts: a pd.Series or list object
    :return: ADF test results
    """
    """Returns the Hurst Exponent of the time series vector ts""" # Create the range of lag values
    lags = range(2, 100)
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags] # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)
    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

def ADF_test(test_data, N=None):
    """
    :param test_data: a pd.Series or list object
    :param N: num of lags
    :return: ADF test results
    """
    return ts.adfuller(test_data, N)

def plot_price_series(df, ts1, ts2):
    """
    plot the price line chart for the selected stocks
    :param df: dataframe
    :param ts1: name of the columns for stock1
    :param ts2: name of the columns for stock1
    """
    import matplotlib.dates as mdates
    df.index = pd.to_datetime(df.index)
    months = mdates.MonthLocator() # set index to be every month
    fig, ax = plt.subplots(figsize=(20,5))
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2], label=ts2)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m %Y')) # set display format
    ax.grid(True)
    fig.autofmt_xdate() # auto format the rotation
    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Prices' % (ts1, ts2))
    plt.legend()
    plt.show()

def plot_residuals(df):
    """
    plot the residuals plot
    :param df: a pd.Series of residuals, indexed on timedate
    """
    import matplotlib.dates as mdates
    df.index = pd.to_datetime(df.index)
    months = mdates.MonthLocator() # set index to be every month
    fig, ax = plt.subplots(figsize=(20,5))
    ax.plot(df.index, df['residuals'], label='residuals')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m %Y')) # set display format
    ax.grid(True)
    fig.autofmt_xdate() # auto format the rotation
    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('Residuals Plot')
    plt.legend()
    plt.show()

def calculate_residuals(df,stk1,stk2):
    """
    :param df: dataframe
    :param stk1: name of the columns for stock1
    :param stk2: name of the columns for stock2
    :return: dataframe for residuals, indexed on datatime
    """
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X=df[stk1].values.reshape(-1, 1), y=df[stk2].values.reshape(-1, 1))
    residuals = df[stk2].values - df[stk1].values * lr.coef_ - lr.intercept_
    res_df = pd.DataFrame(residuals, index=['residuals']).T
    res_df.index = df.index
    return res_df

def output_analysis_result(csv_dir, stk_1, stk_lst, plot=False):
    """
    Print out the TS analysis results
    :param csv_dir: directory for csv files, csv files should be named
                    in the form '000001.SZ.csv'
    :param stk_1: name for benchmark stk
    :param stk_lst: list for stks want to compare
    :param plot: True/False
    """
    df1 = pd.read_csv(os.path.join(csv_dir, '%s.csv' % stk_1), index_col=0).set_index('datetime')
    for stk_2 in stk_lst:
        df2 = pd.read_csv(os.path.join(csv_dir, '%s.csv' % stk_2), index_col=0).set_index('datetime')
        plt_df = pd.concat([df1['close'], df2['close']], axis=1)
        plt_df = plt_df.fillna(method='pad').dropna().sort_index()  # 保证不存在缺失，以二者的交集测试
        plt_df.columns = [stk_1, stk_2]
        # print(plt_df)
        if plot == True:
            plot_price_series(plt_df, stk_1, stk_2)
        res_df = calculate_residuals(plt_df, stk_1, stk_2)
        if plot == True:
            plot_residuals(res_df)
        adf_result = ADF_test(res_df)
        print('--------------------------------------------------------------')
        print('The ADF test result for %s and %s is Given below:' % (stk_1, stk_2))
        print('The tested Statistics is %s' % adf_result[0])
        print('The Total number of samples is %s' % adf_result[3])
        print('The 1% critical value is {}'.format(adf_result[4]['1%']))
        print('The 5% critical value is {}'.format(adf_result[4]['5%']))
        print('The 10% critical value is {}'.format(adf_result[4]['10%']))
        print('The p value is %s' % adf_result[1])
        print('The Hurst Exponent for Residuals is %s' % hurst_test(res_df.values))
        print('--------------------------------------------------------------')
        if adf_result[1] < 0.05:
            print(stk_1, stk_2, 'Possess a cointegrating relationship, at least for the time period sample considered.')


if __name__ == '__main__':
    # from config import stk_lst
    csv_dir = '/Users/pzh/Documents/华兴资管/DINAH/TestData/'
    stk_1 = '000100.SZ'
    stk_lst = ['600887.SH','601166.SH','603259.SH']
    output_analysis_result(csv_dir = csv_dir,
                           stk_1 = stk_1,
                           stk_lst = stk_lst,
                           plot = True)