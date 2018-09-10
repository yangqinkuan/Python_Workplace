# 导入包
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from keras.layers import Input, Dense, LSTM, merge
from keras.models import Model
from keras.models import load_model
import numpy as np
import keras.layers
import pandas as pd

# 基础参数配置
class conf:
    instrument = '000300.SHA'  # 股票代码
    # 设置用于训练和回测的开始/结束日期

    start_date = '2005-01-01'
    split_date = '2017-01-01'
    end_date = '2018-01-01'
    fields = ['close', 'open', 'high', 'low', 'amount', 'volume']  # features
    seq_len = 3  # 每个input的长度
    batch = 50  # 整数，指定进行梯度下降时每个batch包含的样本数,训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步


# 数据导入以及初步处理
count = 0
# D.history_data为bigquant平台获取数据方法,可以自己改掉用别的方法获取数据，只需要把数据调成符合定义的格式

data = D.history_data(conf.instrument, conf.start_date, conf.end_date, conf.fields)

data['return'] = data['close'].shift(-2) / data['open'].shift(-1) - 1  # 计算未来1日收益率（未来第一日的收盘价/明日的开盘价）
data = data[data.amount > 0]
data.dropna(inplace=True)
datatime = data['date'][data.date >= conf.split_date]  # 记录predictions的时间，回测要用
data['return'] = data['return'].apply(lambda x: np.where(x >= 0.2, 0.2, np.where(x > -0.2, x, -0.2)))  # 去极值
data['return'] = data['return'] * 10  # 适当增大return范围，利于LSTM模型训练
data.reset_index(drop=True, inplace=True)
scaledata = data[conf.fields]
traindata = data[data.date < conf.split_date]

# 数据处理：设定每个input（30time series×6features）以及数据标准化
train_input = []
train_output = []
test_input = []
test_output = []
for i in range(conf.seq_len - 1, len(traindata)):
    a = scale(scaledata[i + 1 - conf.seq_len:i + 1])
    train_input.append(a)
    c = data['return'][i]
    train_output.append(c)
for j in range(len(traindata), len(data)):
    b = scale(scaledata[j + 1 - conf.seq_len:j + 1])
    test_input.append(b)
    c = data['return'][j]
    test_output.append(c)

# LSTM接受数组类型的输入
train_x = np.array(train_input)
train_y = np.array(train_output)
test_x = np.array(test_input)
test_y = np.array(test_output)
# 自定义激活函数
import tensorflow as tf


def atan(x):
    return tf.atan(x)


# 构建神经网络层 1层LSTM层+3层Dense层
# 用于1个输入情况
lstm_input = Input(shape=(3, 6), name='lstm_input')
Dense_output_1 = Dense(32, activation='tanh')(lstm_input)
keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)
Dense_output_11 = Dense(64, activation='tanh')(Dense_output_1)
keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)
Dense_output_111 = Dense(128, activation='tanh')(Dense_output_11)
keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)
lstm_output = LSTM(256, activation=atan, dropout_W=0.2, dropout_U=0.1)(Dense_output_111)
keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)
Dense_output_222 = Dense(128, activation='tanh')(lstm_output)
keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)
Dense_output_22 = Dense(64, activation='tanh')(Dense_output_222)
Dense_output_2 = Dense(32, activation='tanh')(Dense_output_22)
predictions = Dense(1, activation=atan)(Dense_output_2)

model = Model(input=lstm_input, output=predictions)
model.save('my_model.h5')
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

history = model.fit(train_x, train_y, batch_size=conf.batch, nb_epoch=400, verbose=2, )
predictions = model.predict(test_x)

iters = range(len(history.history['loss']))
plt.plot(iters, history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
print(history.history['loss'])
plt.show()

for i in range(0, len(predictions)):
    if (predictions[i] * test_y[i] > 0):
        count += 1
print(count / len(predictions))
# plt.plot(predictions)
# plt.plot(test_y)
# plt.show()
# 预测值和真实值的关系
data1 = test_y
data2 = predictions
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data2, data1, 'o', label="data")
ax.legend(loc='best')
# 如果预测值>0,取为1；如果预测值<=0,取为-1.为回测做准备
# for i in range(len(predictions)):
#     if predictions[i]>0:
#         predictions[i]=1
#     elif predictions[i]<=0:
#         predictions[i]=-1
# 将预测值与时间整合作为回测数据
cc = np.reshape(predictions, len(predictions), 1)
databacktest = pd.DataFrame()
databacktest['date'] = datatime
databacktest['direction'] = cc
# databacktest是预测好的判别数据,可以自己输出和处理得到可以使用的csv,然后在各个平台进行回测
# databacktest.to_csv('userlib/524000300.csv')





#   下面的代码是bigquant平台回测代码
# # 在沪深300上回测
# def initialize(context):
#     # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
#     context.set_commission(PerOrder(buy_cost=0.0000, sell_cost=0.000, min_cost=0))
#     # 传入预测数据和真实数据
#     context.predictions = databacktest
#
#     context.hold = conf.split_date
#
#
# # 回测引擎：每日数据处理函数，每天执行一次
# def handle_data(context, data):
#     current_dt = data.current_dt.strftime('%Y-%m-%d')
#     sid = context.symbol(conf.instrument)
#     cur_position = context.portfolio.positions[sid].amount  # 持仓
#     if cur_position == 0:
#         if databacktest['direction'].values[databacktest.date == current_dt] > 0.1:
#             context.order_target_percent(sid, 0.9)
#             context.date = current_dt
#     else:
#         if databacktest['direction'].values[databacktest.date == current_dt] < 0:
#             if context.trading_calendar.session_distance(pd.Timestamp(context.date), pd.Timestamp(current_dt)) >= 1:
#                 context.order_target(sid, 0)
#
#
# # 调用回测引擎
# m8 = M.backtest.v5(
#     instruments=conf.instrument,
#     start_date=conf.split_date,
#     end_date=conf.end_date,
#     initialize=initialize,
#     handle_data=handle_data,
#     order_price_field_buy='open',  # 表示 开盘 时买入
#     order_price_field_sell='close',  # 表示 收盘 前卖出
#     capital_base=100000,
#     benchmark='000300.SHA',
#     m_cached=False
# )
