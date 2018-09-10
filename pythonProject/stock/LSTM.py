import tushare as tusha
import pandas as pd
from sklearn.preprocessing import scale
import pickle
# datas = {}
codes = ['399390', '399387', '399381', '399385', '399382', '399383', '399311', '399631', '399384', '399630', '399386', '399388']
# for code in codes:
#     datas[code] = pd.DataFrame(tusha.get_k_data(code))
#     print(datas[code])
seq_len = 3
code = '399390'
fields = ['close', 'open', 'high', 'low', 'volume']
split_date = '2017-01-01'
for code in codes:
    data = pd.DataFrame(tusha.get_k_data(code, start='2016-12-28', end='2017-12-30'))
    data.to_csv('C:/Users/Administrator/Desktop/stockpredict/LSTM输入数据/' + code + '.csv')


print(data)
def pailie(df):
    temp = df['close']
    df.drop(labels=['close', 'code'], axis=1, inplace=True)
    df.insert(1, 'close', temp)
    return df
# data = pailie(data)
# datatime = data['date'][data.date >=split_date]
# print(datatime)
# scaledata = data[fields]
# testinput = []
# for i in range(3,len(scaledata)):
#      a = scale(scaledata[i+1-seq_len:i+1])
#      testinput.append(a)
#
# print(testinput)