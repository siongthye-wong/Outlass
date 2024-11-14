import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


pd.options.display.max_columns = None
pd.options.display.max_rows = None

#df = pd.read_csv('Advertising.csv')

from sklearn.model_selection import train_test_split

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

#training and testing split using all feature
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#data type is time relevant

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

from sklearn.svm import SVR

modelSvr = SVR(kernel='poly', C=5).fit(X_train, y_train)
y_pred = modelSvr.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# The mean absolute error
print("Mean absolute error: {} ".format(mean_absolute_error(y_test, y_pred)))

# The mean squared error
print("Mean squared error: {} ".format(mean_squared_error(y_test, y_pred)))

# Root mean squared error
print("Root mean squared error: {} ".format(mean_squared_error(y_test, y_pred)**0.5))

# Explained variance score: 1 is perfect prediction
print('Variance score: {} '.format(r2_score(y_test,y_pred)))

import pickle

pickle.dump(modelSvr, open("modeladvertising.h5", "wb")) #wb: write binary

print("Model saved. Please download the file.")
