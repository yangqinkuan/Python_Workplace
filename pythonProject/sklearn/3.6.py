from sklearn import svm
from sklearn import datasets

# clf = svm.SVC()
# iris = datasets.load_iris()
# X,y = iris.data,iris.target
# clf.fit(X,y)
#
# # method 1 : pickle
# import pickle
# with open('save/clf.pickle','wb') as f:
#     pickle.dump(clf,f)


# read
# import pickle
# iris = datasets.load_iris()
# X,y = iris.data,iris.target
# with open('save/clf.pickle','rb') as f:
#     clf2 = pickle.load(f)
#     print(clf2.predict(X[0:1]))


# method 2:joblib
# clf = svm.SVC()
# iris = datasets.load_iris()
# X,y = iris.data,iris.target
# clf.fit(X,y)
# from sklearn.externals import joblib
# # Save
# joblib.dump(clf,'save/clf.pkl')
# # restore
# clf3 = joblib.load('save/clf.pkl')
# print(clf3.predict(X[0:1]))
