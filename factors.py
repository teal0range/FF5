import pandas as pd
import numpy as np
import os
from datetime import datetime
from numpy import vectorize

from preprocess import generateDate

data_path = 'data/preprocessed'
phase = {
    "2016": generateDate("2016-07-31", "2017-06-30"),
    "2017": generateDate("2017-07-31", "2018-01-31")
}

currentPhase = phase["2016"]


class Grouping:
    # 分组类
    def __init__(self, t):
        """形如[[('h',df0),('l',df1)],[('b',df0),('s',df1)]]
        用于存储交集之前的数据
        """
        self.groups = []

        """pd.DataFrame用于存储方便查询的格式"""
        self.frame = pd.DataFrame()
        self.view = pd.DataFrame()

        """标记类有没有准备好被查询"""
        self.isPrepared = False

        """区间"""
        self.t = t

        """BM Inv OP Size"""
        self.factorList = pd.read_csv(os.path.join(data_path, 'SortCols.csv'))
        self.factorList = self.factorList[self.factorList['phase'] == int(self.t)]
        self.factorList['MktSize'] = self.factorList['Size']

        """回报率文件"""
        self.stockReturn = self.date2Phase(
            pd.read_csv(os.path.join(data_path, 'stockReturns.csv'))[['Stkcd', 'date', 'Mretwd', 'Msmvosd', 'Nrrmtdt']]
        )

        """向量化getVMReturn函数"""
        self.getVMReturn = np.vectorize(self.getVMReturn, excluded=['section'])
        self.getVMExcessReturn = np.vectorize(self.getVMExcessReturn, excluded=['section'])

    def append(self, breakpoints, col_name):
        """
        用于向Group中添加维度
        :param df: 添加维度的数据表
        :param breakpoints: 断点，形如[('H',0),('L',0.5)],意味[0,0.5]分位点为H,[0.5,1]为L
        :param col_name: 数据列名
        :return: None
        """
        self.groups.append(self.split(breakpoints, col_name))
        self.isPrepared = False

    def split(self, breakpoints, col_name):
        """
        根据数据列分割factorList
        :param breakpoints: 断点，形如[('H',0),('L',0.5)],意味[0,0.5]分位点为H,[0.5,1]为L
        :param col_name: 数据列名
        :return: 形如[('h',df0),('l',df1)]
        """
        df = self.factorList
        df = df[['Stkcd', 'phase', 'MktSize', col_name]].sort_values(by=[col_name])
        res = []
        for idx, _breakpoint in enumerate(breakpoints):
            upper = 1 if idx == len(breakpoints) - 1 else breakpoints[idx + 1][1]
            tag, ratio = _breakpoint
            res.append((tag, df.iloc[int(ratio * len(df)):int(upper * len(df))].drop([col_name], axis=1)))
        return res

    def prepare(self):
        """
        准备类以备查询
        :return:None
        """
        self.isPrepared = True
        self.intersections()
        for i in range(len(self.frame)):
            data = self.frame.iat[i, self.getAxis()]
            self.frame.iat[i, self.getAxis()] = pd.merge(data, self.stockReturn, on=['Stkcd', 'phase'])

    def intersections(self, axis=0, cols=None, df=None):
        """
        根据数值做出排序交集
        :param axis:
        :param cols:
        :param df:
        :return:
        """
        if cols is None:
            cols = {}
        if axis == len(self.groups):
            cols['data'] = [df]
            self.frame = self.frame.append(pd.DataFrame(cols, index=[len(self.frame)]), sort=True)
            return

        for group in self.groups[axis]:
            if axis == 0:
                df = group[1]
            tmp = df
            if axis != 0:
                df = pd.merge(df, group[1], on=['Stkcd', 'phase', 'MktSize'])
            cols[axis] = [group[0]]
            self.intersections(axis + 1, cols, df)
            cols.pop(axis)
            df = tmp

    def getAxis(self):
        return len(self.groups)

    def __getitem__(self, item):
        if not self.isPrepared:
            self.prepare()
        res = self.frame
        for axis in range(self.getAxis()):
            res = res[res[axis] == item[axis]]
        return res.iat[0, self.getAxis()].reset_index().drop(['index'], axis=1)

    def getVMReturn(self, date, section):
        """
        获取某分组的VM加权回报，此函数在类初始化时会被向量化
        :param date: 时间 YYYY-MM-DD
        :param section: 类别
        :return:
        """
        df = self.__getitem__(section)
        df = df[df['date'] == date]
        return np.sum(df['Mretwd'] * df['MktSize']) / np.sum(df['MktSize'])

    def getVMExcessReturn(self, date, section):
        """
        获取某分组的VM加权回报，此函数在类初始化时会被向量化
        :param date: 时间 YYYY-MM-DD
        :param section: 类别
        :return:
        """
        df = self.__getitem__(section)
        df = df[df['date'] == date]
        return np.sum((df['Mretwd'] - df['Nrrmtdt']) * df['MktSize']) / np.sum(df['MktSize'])

    @staticmethod
    def date2Phase(df):
        """
        将时间转化为t, t年7月至t+1年6月
        :param df: 数据表
        :return:
        """
        df['phase'] = df['date'].apply(lambda x: datetime.strptime(x, "%Y-%m").year
        if datetime.strptime(x, "%Y-%m").month >= 7 else datetime.strptime(x, "%Y-%m").year - 1)
        return df


def Mktrf_MethodOne():
    df = pd.read_csv(os.path.join(data_path, 'stockReturns.csv'))
    df = df[['date', 'Cmretwdos', 'Nrrmtdt']].drop_duplicates().sort_values(by='date')
    df['Mktrf'] = df['Cmretwdos'] - df['Nrrmtdt']
    return df[['date', 'Mktrf']]


def SMB_MethodOne(t):
    # SMB BM
    g = Grouping(t)
    g.append([('S', 0), ('B', 0.5)], 'Size')
    g.append([('L', 0), ('N', 0.3), ('H', 0.7)], 'BM')
    SMB_BM = (g.getVMReturn(currentPhase, section=['S', 'L']) + g.getVMReturn(currentPhase, section=['S', 'N'])
              + g.getVMReturn(currentPhase, section=['S', 'H'])) / 3 - \
             (g.getVMReturn(currentPhase, section=['B', 'L']) + g.getVMReturn(currentPhase, section=['B', 'N'])
              + g.getVMReturn(currentPhase, section=['B', 'H'])) / 3

    g = Grouping(t)
    g.append([('S', 0), ('B', 0.5)], 'Size')
    g.append([('W', 0), ('N', 0.3), ('R', 0.7)], 'OP')
    SMB_OP = (g.getVMReturn(currentPhase, section=['S', 'W']) + g.getVMReturn(currentPhase, section=['S', 'N'])
              + g.getVMReturn(currentPhase, section=['S', 'R'])) / 3 - \
             (g.getVMReturn(currentPhase, section=['B', 'W']) + g.getVMReturn(currentPhase, section=['B', 'N'])
              + g.getVMReturn(currentPhase, section=['B', 'R'])) / 3

    g = Grouping(t)
    g.append([('S', 0), ('B', 0.5)], 'Size')
    g.append([('C', 0), ('N', 0.3), ('A', 0.7)], 'Inv')
    SMB_Inv = (g.getVMReturn(currentPhase, section=['S', 'C']) + g.getVMReturn(currentPhase, section=['S', 'N'])
               + g.getVMReturn(currentPhase, section=['S', 'A'])) / 3 - \
              (g.getVMReturn(currentPhase, section=['B', 'C']) + g.getVMReturn(currentPhase, section=['B', 'N'])
               + g.getVMReturn(currentPhase, section=['B', 'A'])) / 3

    return pd.DataFrame({'date': currentPhase, 'SMB': (SMB_BM + SMB_OP + SMB_Inv) / 3})


def HML_MethodOne(t):
    g = Grouping(t)
    g.append([('S', 0), ('B', 0.5)], 'Size')
    g.append([('L', 0), ('N', 0.3), ('H', 0.7)], 'BM')
    HML = (g.getVMReturn(currentPhase, section=['S', 'H']) + g.getVMReturn(currentPhase, section=['B', 'H'])) / 2 - \
          (g.getVMReturn(currentPhase, section=['S', 'L']) + g.getVMReturn(currentPhase, section=['B', 'L'])) / 2
    return pd.DataFrame({'date': currentPhase, 'HML': HML})


def RMW_MethodOne(t):
    g = Grouping(t)
    g.append([('S', 0), ('B', 0.5)], 'Size')
    g.append([('W', 0), ('N', 0.3), ('R', 0.7)], 'OP')
    RMW = (g.getVMReturn(currentPhase, section=['S', 'R']) + g.getVMReturn(currentPhase, section=['B', 'R'])) / 2 - \
          (g.getVMReturn(currentPhase, section=['S', 'W']) + g.getVMReturn(currentPhase, section=['B', 'W'])) / 2

    return pd.DataFrame({'date': currentPhase, 'RMW': RMW})


def CMA_MethodOne(t):
    g = Grouping(t)
    g.append([('S', 0), ('B', 0.5)], 'Size')
    g.append([('C', 0), ('N', 0.3), ('A', 0.7)], 'Inv')
    CMA = (g.getVMReturn(currentPhase, section=['S', 'C']) + g.getVMReturn(currentPhase, section=['B', 'C'])) / 2 - \
          (g.getVMReturn(currentPhase, section=['S', 'A']) + g.getVMReturn(currentPhase, section=['B', 'A'])) / 2

    return pd.DataFrame({'date': currentPhase, 'CMA': CMA})


@vectorize
def FF5(t):
    # 输出t期FF5因子的列表
    global currentPhase
    currentPhase = phase[t]
    res = pd.merge(Mktrf_MethodOne(),
                   pd.merge(SMB_MethodOne(t),
                            pd.merge(HML_MethodOne(t),
                                     pd.merge(RMW_MethodOne(t), CMA_MethodOne(t), on=['date']), on=['date']),
                            on=['date']), on=['date'])
    res.to_csv(os.path.join("result", t + "FF5.csv"), index=False)


def PortfolioExcessReturn(t, row_type, row_num, col_type='Size', col_num=5, csv=True):
    """
    指标分组计算组合
    :param t: 期号
    :param row_type: 行名
    :param row_num: 行分裂数
    :param col_type: 列名（默认Size）
    :param col_num: 列分裂数（默认5）
    :param csv: 是否输出csv
    :return:
    """
    global currentPhase
    currentPhase = phase[t]
    g = Grouping(t)
    g.append([(i, i * 1 / row_num) for i in range(row_num)], row_type)
    g.append([(i, i * 1 / row_num) for i in range(col_num)], col_type)

    @vectorize
    def formatDataframe(row_count, col_count):
        return pd.DataFrame({"date": currentPhase,
                             "row_type": [row_type] * len(currentPhase),
                             "row_count": [row_count + 1] * len(currentPhase),
                             "col_type": [col_type] * len(currentPhase),
                             "col_count": [col_count + 1] * len(currentPhase),
                             "ExcessReturn": g.getVMExcessReturn(currentPhase, section=[row_count, col_count])}
                            , index=[i for i in range(len(currentPhase))])

    df = pd.DataFrame().append(
        list(formatDataframe(np.arange(0, row_num * col_num, 1) % row_num,
                             np.arange(0, row_num * col_num, 1) // col_num))
    )
    if csv:
        df.to_csv(os.path.join("result", t + "_" + row_type + "_" + col_type + ".csv"), index=False)
    return df


if __name__ == '__main__':
    FF5(['2016', '2017'])
    PortfolioExcessReturn('2016', 'OP', 5)
    PortfolioExcessReturn('2017', 'OP', 5)
    PortfolioExcessReturn('2016', 'Inv', 5)
    PortfolioExcessReturn('2017', 'Inv', 5)
    PortfolioExcessReturn('2016', 'BM', 5)
    PortfolioExcessReturn('2017', 'BM', 5)
