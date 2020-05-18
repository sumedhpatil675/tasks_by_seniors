# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:42:20 2020

@author: Sumedh Patil
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
##reading dataset from csv

dataset = pd.read_csv("iphone_purchase_records.csv")


dataset['Gender']= dataset ['Gender'].map({'Male':0,'Female':1})


#getting all column names
columns_list = list(dataset.columns)

#getting only featurs from dataset 
features = list(set(columns_list)-set(['Purchase Iphone']))


#getting output values in y
y = dataset['Purchase Iphone'].values

#getting all values of features
x = dataset[features].values

#spliting data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=10)


#creating NB classifier

gnb = GaussianNB()

#training model using training set
gnb.fit(train_x,train_y)


#predicting response from test dataset

y_pred = gnb.predict(test_x)


#evaulating model and checking accuracy

from sklearn import metrics


#performance matrix check
confusion_met = confusion_matrix(test_y,y_pred)
confusion_met
#model accuracy
print("Accuracy",metrics.accuracy_score(test_y,y_pred))










