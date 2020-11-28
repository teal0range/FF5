import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

data_path = './data/preprocessed'


# Utils工具类
def convertCode(code) -> str:
    """
    将股票代码转换为标准6位
    :param code: 股票代码
    :return: 6位标准代码
    """
    return "{:06d}".format(int(code))


def generateDate(start, end) -> list:
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    current = start
    res = []
    while current < end:
        res.append(current.strftime("%Y-%m"))
        current = current.replace(day=28) + timedelta(days=4)
    return res


# IO 输入输出
def excel2Df(directory: str, **kwargs) -> pd.DataFrame:
    """
    合并excel表并且输出dataframe
    :rtype: pd.DataFrame
    :param directory: xlsx path
    :return: None
    """
    # 储存dataframe的临时变量
    df_ls = []
    # 遍历文件夹下的所有excel
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx"):
                df_ls.append(pd.read_excel(os.path.join(root, file), **kwargs))
    # 合并输出
    return pd.DataFrame().append(df_ls, sort=False)


# Filters 过滤器
def dateFilter(df: pd.DataFrame, date_col: str, date_format: str = None) -> pd.DataFrame:
    """
    过滤非6,12月的数据
    :param df: 数据表
    :param date_col: 表示时间的列名
    :param date_format: 日期格式
    :return: 过滤后的dataframe
    """
    if date_format is None:
        # 默认日期格式
        date_format = "%Y-%m-%d"
    return df[df[date_col].apply(lambda x: datetime.strptime(x, date_format).month in [6, 12])]


def MainBoardFilter(df: pd.DataFrame, code_col: str) -> pd.DataFrame:
    """
    过滤出A股数据
    :param df:数据表
    :param code_col:股票代码列名
    :return: 过滤后的数据
    """
    return df[np.isin(df[code_col].apply(convertCode).str[:3], ['000', '600', '601', '603', '605'])]


def FinanceFrames():
    # 合并&预处理财务报表数据
    balanceSheet = excel2Df("./data/raw/Balance sheet", index_col=0)
    IncomeSheet = excel2Df("./data/raw/Income statement")
    balanceSheet = balanceSheet[balanceSheet['Typrep'] == 'A'].sort_values(by=['Stkcd', 'Accper'])
    IncomeSheet = IncomeSheet[IncomeSheet['Typrep'] == 'A'].sort_values(by=['Stkcd', 'Accper'])
    # 日期过滤&板块过滤
    balanceSheet = MainBoardFilter(dateFilter(balanceSheet, 'Accper'), 'Stkcd').drop(['Typrep'], axis=1)
    IncomeSheet = MainBoardFilter(dateFilter(IncomeSheet, 'Accper'), 'Stkcd').drop(['Typrep'], axis=1)
    pd.merge(balanceSheet, IncomeSheet, on=['Stkcd', 'Accper'], how='outer'). \
        to_csv("./data/preprocessed/finance.csv", index=False)


def StockReturnFrames():
    # 合并&预处理
    # 市场收益率&无风险收益率
    mktmnth = excel2Df("./data/raw/mktmnth")
    mktmnth = mktmnth[mktmnth['Markettype'] == 5][['Trdmnt', 'Cmretwdos']]
    rf = excel2Df("./data/raw/rf")[['Clsdt', 'Nrrmtdt']]
    rf['month'] = rf['Clsdt'].str[:7]
    rf = rf.groupby('month').apply(lambda x: x.iloc[np.argmax(x['Clsdt'].values)])
    rf['Nrrmtdt'] = rf['Nrrmtdt'] / 100
    marketRf = pd.merge(mktmnth.rename(columns={"Trdmnt": "date"}),
                        rf[['month', 'Nrrmtdt']].rename(columns={"month": "date"}),
                        on='date', how='outer').sort_values(by=['date'])
    # 个股收益率
    stockmnth = excel2Df("./data/raw/stockmnth")[['Stkcd', 'Trdmnt', 'Msmvosd', 'Mretwd', 'Markettype']]
    stockmnth = stockmnth[np.isin(stockmnth['Markettype'], [1, 4])]. \
        rename(columns={'Trdmnt': 'date'}).drop('Markettype', axis=1)
    pd.merge(stockmnth, marketRf, on='date').sort_values(by=['Stkcd', 'date']).dropna(). \
        to_csv("data/preprocessed/stockReturns.csv", index=None)


def extraFactors():
    finance = pd.read_csv(os.path.join(data_path, 'finance.csv')).rename(columns={'Accper': "date"})
    finance['date'] = finance['date'].str[:7]

    stockReturns = pd.read_csv(os.path.join(data_path, 'stockReturns.csv'))

    df = pd.merge(stockReturns, finance, on=['Stkcd', 'date'])
    #   Size
    Size = stockReturns.groupby(['Stkcd']).apply(lambda x: pd.DataFrame(
        {
            'phase': [2016, 2017],
            'Size': [
                x[x['date'] == '2016-06']['Msmvosd'].iat[0] if len(x[x['date'] == '2016-06']) > 0 else np.NAN,
                x[x['date'] == '2017-06']['Msmvosd'].iat[0] if len(x[x['date'] == '2017-06']) > 0 else np.NAN
            ]
        }).dropna()).reset_index().drop(['level_1'], axis=1)
    #   B/M ratio
    BM = df.groupby(['Stkcd']).apply(lambda x: pd.DataFrame({
        'phase': [2016, 2017],
        'BM': [
            x[x['date'] == '2015-12']['total_equity'].iat[0] / x[x['date'] == '2015-12']['Msmvosd'].iat[0]
            if len(x[x['date'] == '2015-12']) > 0 else np.NAN,
            x[x['date'] == '2016-12']['total_equity'].iat[0] / x[x['date'] == '2016-12']['Msmvosd'].iat[0]
            if len(x[x['date'] == '2016-12']) > 0 else np.NAN
        ]
    }).dropna()).reset_index().drop(['level_1'], axis=1)

    #   Inv
    Inv = finance.groupby(['Stkcd']).apply(lambda x: pd.DataFrame({
        'phase': [2016, 2017],
        'Inv': [
            (x[x['date'] == '2015-12']['total_assets'].iat[0] - x[x['date'] == '2014-12']['total_assets'].iat[0])
            / x[x['date'] == '2014-12']['total_assets'].iat[0]
            if len(x[x['date'] == '2015-12']) > 0 and len(x[x['date'] == '2014-12']) > 0 else np.NAN,
            (x[x['date'] == '2016-12']['total_assets'].iat[0] - x[x['date'] == '2015-12']['total_assets'].iat[0])
            / x[x['date'] == '2015-12']['total_assets'].iat[0]
            if len(x[x['date'] == '2016-12']) > 0 and len(x[x['date'] == '2015-12']) > 0 else np.NAN
        ]
    }).dropna()).reset_index().drop(['level_1'], axis=1)

    #   OP

    OP = df.groupby(['Stkcd']).apply(lambda x: pd.DataFrame({
        'phase': [2016, 2017],
        'OP': [
            x[x['date'] == '2015-12']['operating profit'].iat[0] / x[x['date'] == '2015-12']['total_equity'].iat[0]
            if len(x[x['date'] == '2015-12']) > 0 else np.NAN,
            x[x['date'] == '2016-12']['operating profit'].iat[0] / x[x['date'] == '2016-12']['total_equity'].iat[0]
            if len(x[x['date'] == '2016-12']) > 0 and len(x[x['date'] == '2016-12']) > 0 else np.NAN
        ]
    }).dropna()).reset_index().drop(['level_1'], axis=1)

    pd.merge(pd.merge(BM, Inv), pd.merge(OP, Size), on=['Stkcd', 'phase']). \
        to_csv(os.path.join(data_path, "SortCols.csv"), index=False)


if __name__ == '__main__':
    FinanceFrames()
    StockReturnFrames()
    extraFactors()
