import pandas as pd

data_lstm = pd.read_csv('C:/Users/Administrator/Desktop/stockpredict/LSTMsingle/000300.csv', parse_dates=[1], index_col=0, encoding='gbk')
data_lstm = data_lstm .set_index(["date"])
# data_lstm.to_csv('C:/Users/Administrator/Desktop/stockpredict/LSTMsingle/lstmhs300.csv')
data_arima = pd.read_csv('C:/Users/Administrator/Desktop/stockpredict/new/newhs300.csv', parse_dates=[0], index_col=0, encoding='gbk')
print(len(data_lstm))
print(data_lstm)
print('-------------------------------------')
print(len(data_arima))
print(data_arima)
for i in range(0, 242):
    data_arima.iloc[i, 0] = data_lstm.ix[i, 0]/10*0.5+data_arima.ix[i, 0]*0.5
print('*********************************************')
data_arima.to_csv('C:/Users/Administrator/Desktop/stockpredict/LSTMANDARIMA/LSTMANDARIMAhs300.csv')
print(data_arima)