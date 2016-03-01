from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import numpy as np
import pandas as pd


f = open("../Data/train.csv")
df = pd.read_csv(f, header = 0)

# put the original column names in a python list
original_headers = list(df.columns.values)

# remove the non-numeric columns
# df = df._get_numeric_data()

# put the numeric column names in a python list
numeric_headers = list(df.columns.values)
print(numeric_headers)
# create a numpy array with the numeric values for input into scikit-learn
data = df.as_matrix()


X = data[:, 1:]  # select columns 1 through end
y = data[:, 0]   # select column 0, the stock price
print(X)
print('------------------------------')
print(y)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X,y)

testf = open("../Data/test.csv")
dtestf = pd.read_csv(testf, header = 0)
numeric_headers = list(df.columns.values)
data = df.as_matrix()
testX = data
texty = rfc.predict(testX)
print(texty)