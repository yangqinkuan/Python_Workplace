import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


datafile = 'C:/Users/Administrator/Desktop/500.csv.'
data = pd.read_csv(datafile, index_col='时间', encoding='gbk')
data = data.iloc[0:100]
# data.plot()
# plt.show()
#自相关系数


#进行查分
D_data = data.diff(1).dropna()
D_data.columns = ['收盘差分']
# D_data.plot()
#差分平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
#print(ADF(D_data['收盘'])[1])
#检查平稳序列的自相关和偏自相关图
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# plot_acf(D_data['收盘差分'], lags=40)
# plot_pacf(D_data['收盘差分'], lags=40)
data_time = data.index
data_close = data['收盘'].values
print(data_time)
print('-----')
print(data_close)
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateLocator
data_time_translation = [datetime.strptime(d, '%Y-%m-%d').date()
                         for d in data_time]
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
autodates = AutoDateLocator()
plt.gca().xaxis.set_major_locator(autodates)
plt.plot(data_time_translation, data_close, 'b', lw=2.5)
plt.show()



print()
