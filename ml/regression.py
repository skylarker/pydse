from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_boston
import numpy as np


boston = load_boston()
X_train, X_test, Y_train, Y_test = train_test_split(boston.data,
                                                    boston.target, test_size=0.2, random_state=0)

# Linear Regression
regr = LinearRegression()
regr.fit(X_train, Y_train)
Y_pred = regr.predict(X_test)
print "MAE", mean_absolute_error(Y_test, Y_pred)

# Logistic Regression -> is a classifier (Binary)
avg_price_house = np.average(boston.target)
high_priced_idx = (Y_train >= avg_price_house)
Y_train[high_priced_idx] = 1
Y_train[np.logical_not(high_priced_idx)] = 0
Y_train = Y_train.astype(np.int8)
print Y_train

high_priced_idx = (Y_test >= avg_price_house)
Y_test[high_priced_idx] = 1
Y_test[np.logical_not(high_priced_idx)] = 0
y_test = Y_test.astype(np.int8)

clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print classification_report(Y_test, Y_pred)
















