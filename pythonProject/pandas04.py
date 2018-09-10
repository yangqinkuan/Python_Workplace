import pandas as pd
import numpy as np

# join, ['inner','outer']
df1 =pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'],index=[1,2,3])
df2 =pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[2,3,4])

print(df1)
print(df2)

#res = pd.concat([df1,df2],join='inner',ignore_index=True)
res = pd.concat([df1,df2],axis=1,join_axes=[df1.index])
print(res)