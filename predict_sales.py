# -*- coding: utf-8 -*-
# This script predicts sales based on several store features

import numpy as np
import pandas as pd
import pylab as py
import matplotlib.pyplot as plt
import math
import datetime
from time import time

#import sklearn modules
from sklearn.metrics import r2_score
from sklearn import neighbors, datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit

# initialize time 
t0 = time()

types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(int),
         'PromoInterval': np.dtype(str)}

train_tmp = pd.read_csv("data/train.csv", dtype = types)
# Save rows with Sales > 0 (ignore rows with zero sales) 
train_tmp = train_tmp[train_tmp.Sales > 0]
store = pd.read_csv("data/store.csv")
# Merge extra data from store with train set
train = pd.merge(train_tmp, store, on = 'Store')

# Merge store data with test set
test_tmp = pd.read_csv("data/test.csv", dtype = types)
test = pd.merge(test_tmp, store, on = 'Store')

#Remove NaNs
test.loc[(test.Open.isnull()), 'Open'] = 1
train.loc[(train.CompetitionDistance.isnull()), 'CompetitionDistance'] = train['CompetitionDistance'].mean()
test.loc[(test.CompetitionDistance.isnull()), 'CompetitionDistance'] = test['CompetitionDistance'].mean()

# Feature engineering 
# Training set
train['Year'] = pd.to_datetime(train['Date']).dt.year
#train['Month'] = pd.to_datetime(train['Date']).dt.month
#train['Day'] = pd.to_datetime(train['Date']).dt.day
#st_dic = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
#as_dic = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
#train['StoreType1'] = train['StoreType'].map(st_dic).astype(int)
#train['Assortment1'] = train['Assortment'].map(as_dic).astype(int)
     

# Test set
test['Year'] = pd.to_datetime(test['Date']).dt.year
#test['Month'] = pd.to_datetime(test['Date']).dt.month
#test['Day'] = pd.to_datetime(test['Date']).dt.day
#test['StoreType1'] = test['StoreType'].map(st_dic).astype(int)
#test['Assortment1'] = test['Assortment'].map(as_dic).astype(int)

      
#Use log of sales
train['Sales'] = np.log(train['Sales']+1)

average_sales = np.exp(train['Sales'].mean())
print ("Average sales : " + str(average_sales))

labels_train = train['Sales'].values
     
features_drop_store = ['StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'Promo2']
features_drop_train = ['StateHoliday', 'Open', 'SchoolHoliday', 'Date', 'Customers', 'Sales']
features_drop_test =  ['StateHoliday', 'Open', 'SchoolHoliday', 'Date', 'Id']

     
train = train.drop(features_drop_train, axis = 1)
train = train.drop(features_drop_store, axis = 1)

# Save Id before dropping
test_ID = test['Id']

test = test.drop(features_drop_test, axis = 1)
test = test.drop(features_drop_store, axis = 1)

print ("Following features will be used: " + str(test.columns.values))

#save values of train and test in an array of lists
features_train = train.values
features_test = test.values

# Try machine learning models
clf = RandomForestRegressor(n_estimators = 10, n_jobs = -1, max_depth = 2000)
#clf = LinearRegression()
#clf = GradientBoostingRegressor()
#clf = AdaBoostRegressor(n_estimators = 100)


#Lets do grid search to find best parameters
#comment this part when best parameters are found
#params = {"n_estimators":[10, 20, 50],
#          "min_samples_split":[5, 10, 50]}
#cv_rf = GridSearchCV(clf, param_grid=params, n_jobs=1, cv=3)
#cv_rf.fit(features_train, labels_train)
#print cv_rf.best_params_
#print cv_rf.best_estimator_


# Cross-validation
cv_ss = ShuffleSplit(n_splits=5, test_size=0.3, random_state=3)
cv_score = cross_val_score(clf,features_train, labels_train,cv=cv_ss, scoring = 'r2')

print ("Average accuracy: %0.4f +/- %0.4f" % (cv_score.mean(), cv_score.std()))

#fit and predict train set
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_train)
print ("Training time = " + str(time() - t0))

# Calculate Root Mean Squared Percentage Error for training set ;
# RMSPE = sqrt(sum(((yi-yi_hat)/yi)**2)/n)
rmspe = 0
for i in range(1,len(pred)):
    rmspe = rmspe + ((np.exp(labels_train[i]) - np.exp(pred[i]))/np.exp(labels_train[i]))**2
rmspe = math.sqrt(rmspe/len(pred))

print('RMSPE on train set = ' + str(round(rmspe,4)))

# predict sales from test set
pred = clf.predict(features_test)
pred = np.exp(pred) - 1 # we added 1 earlier to take care of log of zeros
print('Avg sales for the test period = ' + str(pred.mean()))

# prepare prediction file with store id and future sales
pred_sales = pd.DataFrame(dict(Sales = pred, Id = test_ID))
pred_sales = pred_sales.sort_values(by = 'Id')
pred_sales.to_csv('./sales.csv', index = False)

