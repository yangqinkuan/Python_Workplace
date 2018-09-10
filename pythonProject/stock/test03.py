import pandas as pd
import tushare as tusha
code = '601919'
data = pd.DataFrame(tusha.get_k_data(code).loc[:, ["date", "close"]])
original_ts = pd.DataFrame(data.dropna())
original_ts = original_ts.set_index(["date", ])
predict_ts = original_ts['2017-01-01':'2018-01-01']
predict_ts_list = predict_ts['close'].tolist()
predict = pd.read_csv('C:/Users/Administrator/Desktop/601919.csv', parse_dates=[0], index_col=0, encoding='gbk')
predict_list = predict['rate'].tolist()
print(len(predict_list))
count1 = 0
count2 = 0
for i in range(1, len(predict_list)):
    if (predict_ts_list[i]-predict_ts_list[i-1]) > 0 and predict_list[i] > 0.0004:
        count1 += 1
    # else:
    #     count -= 1
for i in range(1, len(predict_list)):
    if (predict_ts_list[i]-predict_ts_list[i-1]) < 0 and predict_list[i] > 0.0004:
        count2 += 1
print(count1)
print(count2)

print(predict_ts_list)
print(predict_list)