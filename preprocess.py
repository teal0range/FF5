import pandas as pd
import os
from datetime import datetime
import numpy as np
from numpy import vectorize


# Utils工具类
def convertCode(code) -> str:
    """
    将股票代码转换为标准6位
    :param code: 股票代码
    :return: 6位标准代码
    """
    return "{:06d}".format(int(code))


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
        to_csv("./data/preprocessed/finance.csv", index=None)


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
    stockmnth = stockmnth[np.isin(stockmnth['Markettype'], [1, 4])].rename(columns={'Trdmnt': 'date'})
    pd.merge(stockmnth, marketRf, on='date').sort_values(by=['Stkcd', 'date']). \
        to_csv("data/preprocessed/stockReturns.csv", index=None)


if __name__ == '__main__':
    StockReturnFrames()
