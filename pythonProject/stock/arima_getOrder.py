import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pickle
import datetime
import warnings
from statsmodels.stats.diagnostic import acorr_ljungbox
import tushare as tusha
from statsmodels.tsa.arima_model import ARMA
'''
环境版本:
statsmodels 0.8.0
pandas 0.19.2
matplotlib 2.0.0
tushare 1.1.6
numpy 1.12.1
'''




def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1]

#　进行随机性检验


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
        # p_df = pd.DataFrame.from_dict(p_set, orient="index")
        # p_df.columns = ['p_value']
    i = 0
    while i < len(p_set):
        if p_set[i][1] < 0.01 and p_set[i][2] < 0.05:
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


# 模型参数选择
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

# 对时间序列进行建模
def run_arma(original_ts, maxar=7, maxma=7):
    print(original_ts.columns[0], 'start arma')
    original_ts_log = np.log(original_ts)
    if test_stationarity(original_ts_log.ix[:, 0]) < 0.01:
        diffn = 0
        diff_original_ts_log = original_ts_log
        print('平稳,不需要差分')
    else:
        diffn = best_diff(original_ts_log, maxdiff=8)
        diff_original_ts_log = produce_diffed_timeseries(original_ts_log, diffn)
        print('差分滞后阶数为'+str(diffn)+',已完成差分')

    order = choose_order(diff_original_ts_log, maxar, maxma)
    print('模型的阶数为: ' + str(order))
    model = ARMA(diff_original_ts_log.ix[:, 0], order).fit(disp='-1', method='css')

    f = model.forecast(steps=5, alpha=0.05)[0]

    p = model.predict()
    predict = predict_recover(p, original_ts_log, diffn, 'predict')
    forecast = predict_recover(f, original_ts_log, diffn, 'forecast')

    #查看niheqingkuang
    # p = model.predict()
    # p = predict_recover(p, original_ts_log, diffn, 'predict')
    # plt.plot(p)
    # plt.plot(original_ts)
    # plt.show()
    return predict, forecast, order


def serialization(filename, object):
    with open(filename, 'wb') as f:
        pickle.dump(object, f, 2)
        f.flush()
        f.close()


def deserialization(filename):
    with open(filename, 'rb') as g:
        b = pickle.load(g)
        g.flush()
        g.close()
        return b


if __name__ == '__main__':
    start = datetime.datetime.now()

    code = tusha.get_hs300s().ix[:, 1]
    stock_order = {}
    forecast_price = {}
    gains = {}
    warnings.filterwarnings("ignore")

    for i in range(0, 1):
        print(code.ix[i])
        data = tusha.get_k_data('601998').loc[:, ["date", "close"]]
        original_ts = pd.DataFrame(data.dropna())
        original_ts = original_ts.set_index(["date", ])
        predict, forecast,order = run_arma(original_ts)
        stock_order[code.ix[i]] = order
        forecast_price[code.ix[i]] = forecast
        gains[code.ix[i]] = (forecast[0]-original_ts.ix[-1, 0])/original_ts.ix[-1, 0]
        print(gains)
    # serialization("C:\\Users\\Administrator\\Desktop\\stockpredict\\ARIMA模型参数保存\\order.txt", stock_order)
    # b = deserialization("C:\\Users\\Administrator\\Desktop\\stockpredict\\ARIMA模型参数保存\\order.txt")
    # print(b)
    # print('预测新一天的值', forecast_price)
    # print(gains)
    # desc_gains = sorted(gains.items(), key=lambda item: item[1], reverse=True)
    # print("排序后的", desc_gains)
    # 将涨幅前100序列化到文件,方便下次使用
    # with open('C:\\Users\\Administrator\\Desktop\\desc100.txt', 'wb') as f:
    #     pickle.dump(desc_gains[0:100], f, 2)
    #     f.flush()
    #     f.close()

    #读取文件
    # with open('C:\\Users\\Administrator\\Desktop\\desc100.txt', 'rb') as g:
    #     b = pickle.load(g)
    #     g.flush()
    #     g.close()
    end = datetime.datetime.now()
    print(end-start)






