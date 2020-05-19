# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:09:01 2020

@author: Sumedh Patil
"""

from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier




mnist = fetch_openml('mnist_784')

x,y = mnist['data'],mnist['target']

some_digit = x[36002]
some_digit_image = some_digit.reshape(28,28)#reshape it to plot it

plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")

y[36002]

x_train,x_test = x[:6000],x[6000:7000]

y_train,y_test = y[:6000],y[6000:7000]

#shuffling data 

shuffle_index = np.random.permutation(6000)
x_train  = x_train[shuffle_index]
y_train = y_train[shuffle_index]


##Logistic Regression
###########################################################
#making instace of model
logistic = LogisticRegression()

#fitting values for x and y
logistic.fit(x_train,y_train)
logistic.coef_
logistic.intercept_

#prediction from test data
prediction = logistic.predict(x_test)

accuracy = cross_val_score(logistic,x_train,y_train,cv=3,scoring="accuracy")
accuracy = accuracy.mean()

accuracy
 

#########################################################



##DecisionTree Classifier
########################################################
#making instance of model
dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)


count =0 
for i in range(0,6000):
    count+=1 if y_pred[i] == y_test[i] else 0


count


















