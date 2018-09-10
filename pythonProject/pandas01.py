import pandas as pd
import numpy as np


dates = pd.date_range('20160101',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])

df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
