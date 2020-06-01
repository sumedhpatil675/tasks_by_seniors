# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:09:01 2020

@author: Sumedh Patil Roll No. 51910008
"""

from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
import seaborn as sns



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
 

##############################################################################



##SVM Classifier
##############################################################################
#importinv neccesary libraries
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd



train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#printing shape of the data
train_data.shape

#checking for missing values
train_data.isnull().sum()

#seeing description of test data
test_data.describe()

#checking dimensions of dataset
print("Dimensions : ",test_data.shape, "\n")

#datatypes
test_data.info()


#checking columns of dataset
train_data.columns

#checking unique values in label columns(classes)
order = list(np.sort(train_data['label'].unique()))
print(order)

#visualizing number of class and counts in datasets
plt.plot(figure=(16,10))
g = sns.countplot(train_data['label'],palette='icefire')
plt.title('Number of digit classes')
train_data.label.astype('category').value_counts()

#ploting some sample and converting into matrix
four = train_data.iloc[3,1:]
four.shape
four = four.values.reshape(28,28)
plt.imshow(four,cmap='gray')
plt.title("Digit 4") 

 #Data Preparation#
 
#average feature values
round(train_data.drop('label',axis=1).mean(),2)

#Seperating X and Y variables
 
#labels
y = train_data['label']

#droping label from X var
X = train_data.drop(columns='label')

##Normalization
X = X/255.0
test_data = test_data/255.0


#Scaling the features
from sklearn.preprocessing import scale 
X_scaled = scale(X)

#spliting data to train model
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.3,train_size=0.2,random_state=10)


#Model building

model_linear = SVC(kernel='linear')
model_linear.fit(X_train,y_train)

#predict
y_pred = model_linear.predict(X_test)


#checking confusion matrix and accuracyy
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#accuracy
print("accuracy :", metrics.accuracy_score(y_true=y_test,y_pred=y_pred),"\n")   

#cm
print(metrics.confusion_matrix(y_true=y_test,y_pred=y_pred))

###############################################################################



####KNN--Classifier--#########################################################


#applying PCA to the data
from sklearn.decomposition import PCA
pca = PCA(.95)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

#loading knn model 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7,n_jobs=-1)

#training the model
knn.fit(X_train,y_train)

#making prediction
y_pred2 = knn.predict(X_test)

print("accuracy :", metrics.accuracy_score(y_true=y_test,y_pred=y_pred2),"\n")   


################################################################333














