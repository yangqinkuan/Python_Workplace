import pickle
import tushare as tush
# f = open('C:\\Users\\Administrator\\Desktop\\stockpredict\\data2.txt', 'wb')
#
# a = [(1, 2), (2, 3)]
# pickle.dump(a, f, 2)
g = open('C:\\Users\\Administrator\\Desktop\\stockpredict\\the100.txt', 'rb')
b = pickle.load(g)
g.flush()
g.close()
print(b)
basic_c = 1000
basic_j = 0
for i in range(0, len(b)):
    basic_c *= (1+b[i][1])
for i in range(0, len(b)):
    basic_j += 1000*(1+b[i][1])
basic_j /= 100*1000
print("前100平均涨幅", basic_j-1)
print(basic_c)
basic_c = 1000
basic_j = 0
for i in range(0, 10):
    basic_c *= (1+b[i][1])
for i in range(0, 10):
    basic_j += 1000*(1+b[i][1])
basic_j /= 10*1000
print("前10平均涨幅", basic_j-1)
print(basic_c)



