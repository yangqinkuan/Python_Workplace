import pandas as pd
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
import statsmodels.tsa.stattools as st
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('C:/Users/Administrator/Desktop/500.csv', parse_dates=[0], index_col=0, encoding='gbk')
print(data.dtypes)
# temp = data.copy()
# temp['diff'] = data[data.columns[0]].diff(1).dropna()
# print(temp)
#print(data[data.columns[0]])
# print(data.columns[0])
# print(data.ix[0:10, ['stocknum']])
# 进行ADF平稳性检验下小于0.05则认为稳定 参数timeseries为时间序列


def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1]
# print(test_stationarity(data['stocknum']))

#寻找合适的差分阶数 df 为原始数据,maxdiff 为最大差分阶数


def best_diff(df, maxdiff = 8):
    p_set = {}
    for i in range(0, maxdiff):
        temp = df.copy() #每次循环前,重置
        if i == 0:
            temp['diff'] = temp[temp.columns[0]]
        else:
            temp['diff'] = temp[temp.columns[0]].diff(i)
            temp = temp.drop(temp.iloc[:i].index) #差分后,前几行的数据会变成NaN，所以删掉

        pvalue = test_stationarity(temp['diff'])
        p_set[i] = pvalue
        p_df = pd.DataFrame.from_dict(p_set, orient="index")
        p_df.columns = ['p_value']
    i = 0
    while i < len(p_df):
        if p_df['p_value'][i] < 0.01:
            bestdiff = i
            break
        i += 1
    return bestdiff
#print(best_diff(data))


#利用已经找到的合适差分进行差分 df为时间序列 diffn为差分阶数


def produce_diffed_timeseries(df, diffn):
    if diffn != 0:
        df['diff'] = df[df.columns[0]].diff(diffn)
    else:
        df['diff'] = df[df.columns[0]]
    df.dropna(inplace=True)    #差分之后的NaN去掉
    return df
# print(produce_diffed_timeseries(data,best_diff(data)))

# 寻找合适的p,q,  ts为传入差分后的时间序列,maxar, maxma为pq的最大备选值


# def choose_order(ts, maxar, maxma):
#     order = st.arma_order_select_ic(ts, maxar, maxma, ic=['aic', 'bic', 'hqic'])
#     return order.bic_min_order
# print('chafen',best_diff(pd.DataFrame(np.log(data[data.columns[0]]))))
# print(data[data.columns[0]])
# ts = produce_diffed_timeseries(pd.DataFrame(data[data.columns[0]]), best_diff(pd.DataFrame(data[data.columns[0]])))
# print(ts.ix[:, 1])
# 寻找合适的p,q,  ts为传入差分后的时间序列,maxar, maxma为pq的最大备选值


from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
def choose_order(data_ts, maxar, maxma):
    init_bic = float("inf")
    init_p = 0
    init_q = 0
    #init_properModel = None
    for p in np.arange(maxar):
        for q in np.arange(maxma):
            model = ARMA(data_ts, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                # init_properModel = results_ARMA
                init_bic = bic
    return init_p, init_q
# order = proper_model(ts, 7)
# print(order[0:3])
#
# order1 = proper_model(ts.ix[:, 1], 7)

# 对差分后的数据进行还原,ts为预测得到时间序列,df为原始时间序列,diffn为滞后的阶数


def predict_recover(ts, df, diffn):
    if diffn != 0:
        for i in range(0, diffn):
            ts.iloc[i] = ts.iloc[i] + df.ix[-diffn+i, 0]
        for i in range(diffn, ts.shape[0]):
            ts.iloc[i] = ts.iloc[i] + ts.iloc[i-diffn]
    print('还原完成')
    return ts


# print(data.ix[:, 0])
# chafen = produce_diffed_timeseries(pd.DataFrame(data.ix[:, 0]), 3)
# print(chafen.ix[:, 1])
# predict = chafen.ix[-3:, 1]
# print(predict.iloc[0], '--------')
# print('dddd', data.ix[:-3, 0])
# rcv = predict_recover(predict, pd.DataFrame(data.ix[:-3, 0]), 3)
# print(rcv)
# data['diff1'] = data.diff()
# data['diff2'] = data.ix[:, 0].diff(2)

# def run_arma(df, maxar, maxma, test_size = 10):
#     df = df.dropna()
#     train_size = len(df)-int(test_size)
#     train, test = df[:train_size], df[train_size:]
#     if test_stationarity(train.ix[:, 0]) < 0.01:
#         print('平稳,不需要差分')
#     else:
#         diffn = best_diff(train, maxdiff=8)
#         train = produce_diffed_timeseries(train, diffn)
#         print('差分阶数为'+str(diffn)+',已完成差分')
#     print('开始进行ARMA')
#     order = choose_order(train.ix[:, 1], maxar, maxma)
#     print('模型的阶数为: ' + str(order))
#     _ar = order[0]
#     _ma = order[1]
#     arma_mod = ARMA(train, order=order).fit()
# train_size = len(data)-int(10)
# train, test = data[:train_size], data[train_size:]
# print(train.ix[:, 0])
# train = produce_diffed_timeseries(train, 1)
# order = choose_order(train.ix[:, 1], 7, 7)
# print(order)
# print(train.ix[:, 1])
# print()
# model = ARMA(train.ix[:, 1], order)
# result_arma = model.fit(disp=-1, method='css')
# p = result_arma.forecast(steps=20, alpha=5)

model = ARIMA(data.ix[:-2, 0], order=(0, 1, 0)).fit(disp=0)
output = model.forecast(steps=2)
print('原始数据',data.ix[-2:, 0])
print(output[0])


plt.show()


print(data.dtypes)




