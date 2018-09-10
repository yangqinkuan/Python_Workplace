import pandas as pd
import tushare as tusha
import pickle
import numpy as np
# codes = ['399390', '399387', '399381', '399385', '399382', '399383', '399311', '399631', '399384', '399630', '399386', '399388']
# datas = {}
# for code in codes:
#     datas[code] = pd.DataFrame(tusha.get_k_data(code).loc[:, ["date", "close"]])
#     print(code)
# data1 = pd.DataFrame(tusha.get_k_data(code).loc[:, ["date", "close"]])
# data2 = pd.DataFrame(tusha.get_k_data('399387').loc[:, ["date", "close"]])
# arr = {}
# arr[code] = data1
# arr['399387'] = data2
# print(arr)

# arr = {'a': 1, 'b': 3, 'c': 2, 'd': 8, 'dasd': 46}
# desc_arr = sorted(arr.items(), key=lambda item: item[1], reverse=True)[0:2]
# arr1 = []
# for key in desc_arr:
#     print(key[0])
#     arr1.append(key[0])
# arrs = []
# arrs.append(arr1)
#
#
# print(arrs)
# print(desc_arr)
with open('C:/Users/Administrator/Desktop/stockpredict/1000指数参数保存/30000.txt', 'rb') as g:
        b = pickle.load(g)
        g.flush()
        g.close()
print(b)
for i in range(0, len(b)):
    for j in range(0, 3):
        b[i][j] = int(b[i][j])
data = pd.DataFrame(tusha.get_k_data('399390').loc[:, ["date", "close"]])
original_ts = pd.DataFrame(data.dropna())
original_ts = original_ts.set_index(["date", ])
predict_ts = original_ts['2017-01-01':'2018-01-01']
df = pd.DataFrame({'rate': b, }, index=predict_ts.index)
print(df)
print(type(df.loc['2017-01-03', 'rate'][0]))
df.to_csv('C:/Users/Administrator/Desktop/stockpredict/1000指数参数保存/30000.csv')
# print(predict_ts)
# print(len(predict_ts))

# arr = ['a', 'b', 'c']
# print(type(arr))
# for i in range(0, len(arr)):
#     arr[i] = arr[i] + '.xhsx'
# print(arr)
