import pandas as pd
data = pd.read_csv('resource/test01.csv')
print(data)

data.to_pickle('resource/student.pickle')