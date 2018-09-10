import pandas as pd
import numpy as np

#append
df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[2,3,4])
#res = df1.append([df2,df3],ignore_index=True)
#res1 = df1.append([df2,df3])

s1 = pd.Series([1,2,3,4],index=['a','b','c','d'])
res = df1.append(s1,ignore_index=True)
#res2 = df1.append(s1,ignore_index=True)

print(res)