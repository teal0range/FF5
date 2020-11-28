import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

# 读取并合并ff5数据
data1 = pd.read_csv("result/2016FF5.csv")
data2 = pd.read_csv("result/2017FF5.csv")
data_ff5 = data1.append(data2)
data_ff5 = data_ff5.reset_index(drop=True)
# 读取分组超额收益率数据
for fac in ['BM', 'Inv', 'OP']:
    data1 = pd.read_csv("result/2016_" + fac + "_Size.csv")
    data2 = pd.read_csv("result/2017_" + fac + "_Size.csv")
    if fac == 'BM':
        Size_BM = data1.append(data2)
    elif fac == 'Inv':
        Size_Inv = data1.append(data2)
    elif fac == 'OP':
        Size_OP = data1.append(data2)

regression_BM = pd.DataFrame(columns=['Size', 'BM', 'Const', 'Mktrf', 'SMB', 'HML', 'RMW', 'CMA', 'R-squared'])
regression_BM_pvalue = pd.DataFrame(columns=['Size', 'BM', 'Const_p', 'Mktrf_p', 'SMB_p', 'HML_p', 'RMW_p', 'CMA_p'])
x_ = Size_BM['row_count'].value_counts().shape[0]
y_ = Size_BM['col_count'].value_counts().shape[0]
for i in range(1, x_ + 1):
    for j in range(1, y_ + 1):
        X = data_ff5.iloc[0:19, 1:6]
        X = sm.add_constant(X)
        y = Size_BM[(Size_BM['row_count'] == i) & (Size_BM['col_count'] == j)].iloc[:, 5]
        y = y.reset_index(drop=True)
        model = sm.OLS(y, X)
        results = model.fit()
        regression_BM = regression_BM.append([{'Size': i, 'BM': j, 'Const': results.params[0],
                                               'Mktrf': results.params[1], 'SMB': results.params[2],
                                               'HML': results.params[3], 'RMW': results.params[4],
                                               'CMA': results.params[5], 'R-squared': results.rsquared}])
        regression_BM_pvalue = regression_BM_pvalue.append([{'Size': i, 'BM': j, 'Const_p': results.pvalues[0],
                                                             'Mktrf_p': results.pvalues[1], 'SMB_p': results.pvalues[2],
                                                             'HML_p': results.pvalues[3], 'RMW_p': results.pvalues[4],
                                                             'CMA_p': results.pvalues[5]}])

regression_Inv = pd.DataFrame(columns=['Size', 'Inv', 'Const', 'Mktrf', 'SMB', 'HML', 'RMW', 'CMA', 'R-squared'])
regression_Inv_pvalue = pd.DataFrame(columns=['Size', 'Inv', 'Const_p', 'Mktrf_p', 'SMB_p', 'HML_p', 'RMW_p', 'CMA_p'])
x_ = Size_Inv['row_count'].value_counts().shape[0]
y_ = Size_Inv['col_count'].value_counts().shape[0]
for i in range(1, x_ + 1):
    for j in range(1, y_ + 1):
        X = data_ff5.iloc[0:19, 1:6]
        X = sm.add_constant(X)
        y = Size_Inv[(Size_Inv['row_count'] == i) & (Size_Inv['col_count'] == j)].iloc[:, 5]
        y = y.reset_index(drop=True)
        model = sm.OLS(y, X)
        results = model.fit()
        regression_Inv = regression_Inv.append([{'Size': i, 'Inv': j, 'Const': results.params[0],
                                                 'Mktrf': results.params[1], 'SMB': results.params[2],
                                                 'HML': results.params[3], 'RMW': results.params[4],
                                                 'CMA': results.params[5], 'R-squared': results.rsquared}])
        regression_Inv_pvalue = regression_Inv_pvalue.append([{'Size': i, 'Inv': j, 'Const_p': results.pvalues[0],
                                                               'Mktrf_p': results.pvalues[1],
                                                               'SMB_p': results.pvalues[2], 'HML_p': results.pvalues[3],
                                                               'RMW_p': results.pvalues[4],
                                                               'CMA_p': results.pvalues[5]}])

regression_OP = pd.DataFrame(columns=['Size', 'OP', 'Const', 'Mktrf', 'SMB', 'HML', 'RMW', 'CMA', 'R-squared'])
regression_OP_pvalue = pd.DataFrame(columns=['Size', 'OP', 'Const_p', 'Mktrf_p', 'SMB_p', 'HML_p', 'RMW_p', 'CMA_p'])
x_ = Size_OP['row_count'].value_counts().shape[0]
y_ = Size_OP['col_count'].value_counts().shape[0]
for i in range(1, x_ + 1):
    for j in range(1, y_ + 1):
        X = data_ff5.iloc[0:19, 1:6]
        X = sm.add_constant(X)
        y = Size_OP[(Size_OP['row_count'] == i) & (Size_OP['col_count'] == j)].iloc[:, 5]
        y = y.reset_index(drop=True)
        model = sm.OLS(y, X)
        results = model.fit()
        regression_OP = regression_OP.append([{'Size': i, 'OP': j, 'Const': results.params[0],
                                               'Mktrf': results.params[1], 'SMB': results.params[2],
                                               'HML': results.params[3], 'RMW': results.params[4],
                                               'CMA': results.params[5], 'R-squared': results.rsquared}])
        regression_OP_pvalue = regression_OP_pvalue.append([{'Size': i, 'OP': j, 'Const_p': results.pvalues[0],
                                                             'Mktrf_p': results.pvalues[1], 'SMB_p': results.pvalues[2],
                                                             'HML_p': results.pvalues[3], 'RMW_p': results.pvalues[4],
                                                             'CMA_p': results.pvalues[5]}])

if not os.path.exists("regression_result/"):
    os.mkdir("regression_result/")

regression_BM.to_csv('regression_result/regression_BM.csv', index=False)
regression_BM_pvalue.to_csv('regression_result/regression_BM_pvalue.csv', index=False)
regression_Inv.to_csv('regression_result/regression_Inv.csv', index=False)
regression_Inv_pvalue.to_csv('regression_result/regression_Inv_pvalue.csv', index=False)
regression_OP.to_csv('regression_result/regression_OP.csv', index=False)
regression_OP_pvalue.to_csv('regression_result/regression_OP_pvalue.csv', index=False)
