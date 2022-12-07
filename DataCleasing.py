'''
This is a script for data cleasing
The Tushare DataHandler Feed in the data in format like:
(index) ts_code	open	high	low	close	pre_close	change	pct_chg	vol	amount	datetime
where the datetime column takes in date-time object yyyy-mm-dd
'''

from datetime import datetime

def convert_datetime(df, pattern='%Y%m%d'):
    """
    convert 20201118 to 2020-11-18 in datetime
    :param df: dataframe
    :param pattern: eg: '%Y%m%d'
    :return: pd.Series with original index
    """
    return df['datetime'].apply(lambda x:str(datetime.strptime(str(x), pattern).date()))