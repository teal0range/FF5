## 因子处理
我们使用pandas + numpy进行数据预处理，回归分析使用statsmodel的api。

主要分为三个文件，每个文件为一个模块，具体解释如下:

- preprocess.py 负责数据预处理
  
  1. 首先将原始数据文件夹下的若干文件按文件夹合并为csv
  2. 其次将资产负债表和现金流量表合成finance.csv输出
  3. 其次将个股return、无风险收益率、市场收益率合并为stockReturns.csv输出
  4. 根据finance.csv和stockReturns.csv分别计算出BM SIZE INV OP 指标表,其中需要跨表操作时才将两表合并,否则依然分开使用两表.

- factors.py 负责计算Fama-French五因子
   
   我们选定的是论文中的2*3模型
   
   该文件下存在一个Grouping类，其主要功能是将股票根据BM SIZE INV OP大小进行分组。并通过getVMReturn函数返回对应组合的收益率，具体使用方法见类的说明
   
   后缀为_MethodOne的函数均为2*3法计算单个因子的函数
   
   FF5函数合并单个因子并输出2016FF5.csv和2017FF5.csv两个csv
   
- PortfolioExcessReturn是根据特定指标分出组合,并输出组合市值加权回报率的函数(即我们选定的应变量).

    我们按INV-SIZE,OP-SIZE,BM-SIZE大小排序分别分成5*5的组合，最终输出得到group_result文件夹下的6个csv
## 程序说明
要运行程序首先安装依赖
```commandline
pip install -r requirements.txt
```
然后按以下顺序执行命令
```commandline
python preprocess.py
python factors.py
python regeression.py
```

## 程序结构
```java
│  factors.py # 计算指标的py文件
│  preprocess.py # 预处理的py文件
│  regression.py # 回归处理py文件
│  requirements.txt # 依赖文件
│
├─data
│  ├─preprocessed # 初步预处理后的数据
│  │      finance.csv # 合并后的资产负债表
│  │      SortCols.csv # BM SIZE INV OP 指标表
│  │      stockReturns.csv # 合并后的回报表
│  │
│  └─raw # 原始数据
│      │  字段说明.txt
│      │
│      ├─Balance sheet
│      │      balance sheet.csv
│      │      FS_Combas1.xlsx
│      │      FS_Combas2.xlsx
│      │      FS_Combas3.xlsx
│      │
│      ├─Income statement
│      │      FS_Comins1.xlsx
│      │      FS_Comins2.xlsx
│      │      FS_Comins3.xlsx
│      │
│      ├─mktmnth
│      │      TRD_Cnmont.xlsx
│      │      TRD_Cnmont1.xlsx
│      │
│      ├─rf
│      │      TRD_Nrrate.xlsx
│      │
│      └─stockmnth
│              TRD_Mnth0.xlsx
│              TRD_Mnth1.xlsx
│              TRD_Mnth2.xlsx
│
├─factor_result # 2*3 FF五因子结果（2015、2016期）
│      2016FF5.csv
│      2017FF5.csv
│
├─group_result # 5*5分组 组合超额回报率
│      2016_BM_Size.csv
│      2016_Inv_Size.csv
│      2016_OP_Size.csv
│      2017_BM_Size.csv
│      2017_Inv_Size.csv
│      2017_OP_Size.csv
│
└─regression_result # 回归结果
       regression_BM.csv
       regression_BM_pvalue.csv
       regression_Inv.csv
       regression_Inv_pvalue.csv
       regression_OP.csv
       regression_OP_pvalue.csv

```

