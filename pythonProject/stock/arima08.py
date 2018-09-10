import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt

import warnings




def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1]
# print(test_stationarity(data['stocknum']))

#　进行随机性检验


from statsmodels.stats.diagnostic import acorr_ljungbox


def test_stochastic(ts):
    p_value = acorr_ljungbox(ts, lags=1)[1]
    return p_value
#寻找合适的差分阶数 df 为原始数据,maxdiff 为最大差分阶数


def best_diff(df, maxdiff = 8):
    p_set = {}
    for i in range(0, maxdiff):
        temp = df.copy() #每次循环前,重置
        if i == 0:
            temp['diff'] = temp.ix[:, 0]
        else:
            temp['diff'] = temp.ix[:, 0].diff(i)
            temp = temp.drop(temp.iloc[:i].index) #差分后,前几行的数据会变成NaN，所以删掉
        pvalue = []
        pvalue.append(i)
        sta_pvalue = test_stationarity(temp['diff'])
        sto_pvalue = test_stochastic(temp['diff'])
        pvalue.append(sta_pvalue)
        pvalue.append(sto_pvalue)
        p_set[i] = pvalue
    i = 0
    while i < len(p_set):
        if p_set[i][1] < 0.05 and p_set[i][2] < 0.05:
            bestdiff = p_set[i][0]
            break
        i += 1
    return bestdiff





#利用已经找到的合适差分进行差分 df为时间序列 diffn为滞后差分阶数


def produce_diffed_timeseries(df, diffn):
    temp = df.copy()
    if diffn != 0:
        temp.ix[:, 0] = temp.ix[:, 0].diff(diffn)

    temp.dropna(inplace=True)    #差分之后的NaN去掉
    return temp




from statsmodels.tsa.arima_model import ARMA



def choose_order(data_ts, maxar, maxma):
    init_bic = float("inf")
    init_p = 0
    init_q = 0
    # init_properModel = None
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




# 对差分后的数据进行还原,ts为预测得到时间序列,df为原始时间序列,diffn为滞后的阶数


def predict_recover(ts, df, diffn, way):
    if diffn != 0:
        if way == 'forecast':
            if(len(ts)<=diffn):
                for i in range(0, len(ts)):
                    ts[i] = ts[i] + df.ix[-diffn + i, 0]

            if(len(ts)>diffn):
                for i in range(0, diffn):
                    ts[i] = ts[i] + df.ix[-diffn + i, 0]
                for i in range(diffn, ts.shape[0]):
                    ts[i] = ts[i] + ts[i-diffn]
        if way == 'predict':
            diff_shift_ts = df.shift(diffn).ix[:, 0]
            ts = ts.add(diff_shift_ts)
    ts = np.exp(ts)
    print('还原完成')
    return ts
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
def run_arma(original_ts, maxar=7, maxma=7):
    print(original_ts.columns[0], 'start arma')
    original_ts_log = np.log(original_ts)
    if test_stationarity(original_ts_log.ix[:, 0]) < 0.01 and test_stochastic(original_ts_log.ix[:, 0]) < 0.05:
        diffn = 0
        diff_original_ts_log = original_ts_log
        print('平稳,不需要差分')
    else:
        diffn = best_diff(original_ts_log, maxdiff=8)
        print(diffn)
        diff_original_ts_log = produce_diffed_timeseries(original_ts_log, diffn)
        plot_acf(diff_original_ts_log, lags=40)
        plot_pacf(diff_original_ts_log, lags=40)
        plt.show()
        print('差分滞后阶数为'+str(diffn)+',已完成差分')
    print('开始进行ARMA')
    order = choose_order(diff_original_ts_log, maxar, maxma)
    print('模型的阶数为: ' + str(order))
    model = ARMA(diff_original_ts_log.ix[:, 0], order).fit(disp='-1', method='css')

    f = model.forecast(steps=5, alpha=0.05)[0]
    p = model.predict()
    plt.plot(p, label='fitting series')
    plt.plot(diff_original_ts_log, label='original series')
    plt.legend()
    plt.show()
    predict = predict_recover(p, original_ts_log, diffn, 'predict')
    forecast = predict_recover(f, original_ts_log, diffn, 'forecast')

    #查看niheqingkuang
    # p = model.predict()
    # p = predict_recover(p, original_ts_log, diffn, 'predict')
    # plt.plot(p)
    # plt.plot(original_ts)
    # plt.show()
    return predict, forecast


if __name__ == '__main__':
    data = pd.read_csv('C:/Users/Administrator/Desktop/500.csv', parse_dates=[0], index_col=0, encoding='gbk')
    print(data.shape)


    warnings.filterwarnings("ignore")
    forecast_price = {}
    for i in range(0, data.shape[1]):
        original_ts = pd.DataFrame(data.ix[:-5, 0].dropna())
        print(original_ts)
        plt.plot(original_ts)
        plt.show()
        original_ts_log = np.log(original_ts)
        predict, forecast = run_arma(original_ts)
        print(forecast)
        print(predict)
        forecast_price[data.columns[i]] = forecast[0]
        print('预测新一天的值', forecast_price)
        plt.plot(predict, label='fitting series')
        plt.plot(original_ts, label='original series')
        plt.legend()
        plt.show()


    # print(forecast_price)
    # original_ts = pd.DataFrame(data.ix[:, 0].dropna())
    # forecast = run_arma(original_ts)
    # print(forecast)
    # value = {}
    # pvalue = []
    # pvalue.append(1)
    # pvalue.append(2)
    # value[0] = pvalue
    # print(value)







