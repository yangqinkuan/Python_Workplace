from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target
X_train,X_test,y_train,y_test = train_test_split(data_X,data_y,test_size=0.2)
model = LinearRegression()
model.fit(X_train,y_train)
print(model.predict(X_test)[:4])
print(y_test[:4])

X,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=1)
plt.scatter(X,y)
plt.show()
