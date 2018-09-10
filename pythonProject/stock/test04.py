import pandas as pd
import tushare as tusha
import matplotlib.pyplot as plt

code = 'hs300'
data = pd.DataFrame(tusha.get_k_data(code).loc[:, ["date", "close"]])
original_ts = pd.DataFrame(data.dropna())
original_ts = original_ts.set_index(["date", ])
predict_ts = original_ts['2017-01-01':'2018-01-01']
predict_ts_list = predict_ts['close'].tolist()
predict = pd.read_csv('C:/Users/Administrator/Desktop/stockpredict/ARIMA/Nhs300.csv', parse_dates=[0], index_col=0, encoding='gbk')
predict_list = predict['rate'].tolist()
print(len(predict_list))
for i in range(1, len(predict_list)):
    predict_list[i] = predict_list[i] + predict_ts_list[i-1];
predict_ts_list = predict_ts_list[1:]
predict_list = predict_list[1:]
sum = 0
for i in range(0,len(predict_list)):
    sum += abs(predict_list[i]-predict_ts_list[i])/predict_ts_list[i]
print(sum/len(predict_list))
plt.plot(predict_ts_list, label='original series')
plt.plot(predict_list, label='predict series')
plt.legend()
plt.show()
print(len(predict_ts_list))
print(len(predict_list))