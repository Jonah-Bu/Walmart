from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import numpy as np
import pandas as pd
from datetime import date
from datetime import datetime
from dateutil.parser import parse


startdate = datetime(2000, 1, 1)

def date_column_to_days(X, n):
    days = []
    for dts in X[:,n]:
        days.append(to_days(to_date(dts)))
    daysnp = np.array(days)
    X[:,n] = daysnp
    return X

def to_date(dts):
    return parse(dts)

def to_days(dt):
    delta = dt - startdate
    return delta.days

if __name__ == "__main__":
    f = open("../Data/train.csv")
    df = pd.read_csv(f, header = 0)

    # put the original column names in a python list
    original_headers = list(df.columns.values)

    # remove the non-numeric columns
    # df = df._get_numeric_data()

    # put the numeric column names in a python list
    numeric_headers = list(df.columns.values)
    # print(numeric_headers)
    # create a numpy array with the numeric values for input into scikit-learn
    data = df.as_matrix()


    X = data[:, 1:]  # select columns 1 through end

    # i = 0
    # while i < len(X[:,2]):
    #     dts = X[:,2][i]
    #     X[:,2][i] = to_days(to_date(dts))
    print(X)
    days = []
    for dts in X[:,2]:
        days.append(to_days(to_date(dts)))
    daysnp = np.array(days)
    X[:,2] = daysnp

    y = data[:, 0]   # select column 0, the stock price

    print('------------------------------')
    print(X)
    print('------------------------------')
    print(y)
    print('------------------------------')

    num = 500-
    XX = X[1:num,:]
    print(XX)
    print('------------------------------')
    yy = y[1:num ]
    print(yy)
    print('------------------------------')
    rfc = RandomForestClassifier(n_jobs=2, n_estimators=50, max_depth=5)
    rfc.fit(XX.tolist(),yy.tolist())
    print('------------------------------!!')

    testf = open("../Data/test.csv")
    dtestf = pd.read_csv(testf, header = 0)
    numeric_headers = list(dtestf.columns.values)
    data = dtestf.as_matrix()
    testX = data

    # print testX

    days = []
    for dts in testX[:,2]:
        days.append(to_days(to_date(dts)))
    daysnp = np.array(days)
    testX[:,2] = daysnp
    # print testX
    texty = rfc.predict(testX.tolist())

    print('------------------------------')
    print(texty)
    print('------------------------------')