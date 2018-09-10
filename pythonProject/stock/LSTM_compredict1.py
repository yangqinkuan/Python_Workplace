import pandas as pd
import pickle

codes = ['399390', '399387', '399381', '399385', '399382', '399383', '399311', '399631', '399384', '399630', '399386', '399388']

datas_LSTM = {}
datas_ARIMA = {}
result = {}
for code in codes:
    data_LSTM = pd.read_csv('C:/Users/Administrator/Desktop/stockpredict/LSTMreturn/return' + code + '.csv',parse_dates=[1], index_col=0, encoding='gbk')
    data_ARIMA = pd.read_csv('C:/Users/Administrator/Desktop/stockpredict/ARIMA1000rate/' + code + '.csv',parse_dates=[0], index_col=0, encoding='gbk')
    # data = data.drop(labels=['Unnamed: 0', ], axis=1, inplace=True)
    data_LSTM = data_LSTM .set_index(["date"])
    datas_LSTM[code] = data_LSTM
    datas_ARIMA[code] = data_ARIMA
length = len(datas_LSTM['399390'])
index = datas_LSTM['399390'].index
predict_arrs = []

for i in range(0, length):
    predict_dict = {}
    predict_arr = []
    for code in codes:
        predict_dict[code] = datas_LSTM[code].ix[i, 0]/10*0.5 + datas_ARIMA[code].ix[i, 0]*10*0.5
    desc_predict_dict = sorted(predict_dict.items(), key=lambda item: item[1], reverse=True)[0:3]
    for arr in desc_predict_dict:
       predict_arr.append(arr[0])
    predict_arrs.append(predict_arr)
print(predict_arrs)
with open('C:/Users/Administrator/Desktop/stockpredict/1000指数参数保存/30000.txt', 'wb') as f:
    pickle.dump(predict_arrs, f, 2)
    f.flush()
    f.close()
    print('成功搞定')