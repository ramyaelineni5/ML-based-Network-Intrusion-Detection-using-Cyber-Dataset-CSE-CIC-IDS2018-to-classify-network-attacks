# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 11:48:05 2020

@author: ramya
"""
import numpy as pn
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix






traindata = pd.read_csv('preprocessed_train_multiclass(a)_new.csv', header=None)
testdata = pd.read_csv('preprocessed_test_multiclass(a)_new.csv', header=None)

X = traindata.iloc[0:,1:73]# 73 columns of data frame with all rows
Y = traindata.iloc[0:,73]# 

T = testdata.iloc[0:,1:73]
C = testdata.iloc[0:,73]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
scaler = Normalizer().fit(T)
testT = scaler.transform(T)


traindata = pn.array(trainX)
trainlabel = pn.array(Y)

testdata = pn.array(testT)
testlabel = pn.array(C)




print("-------------------------Random Forest multi_class----------------------------")
#create the model with 100 trees
model = RandomForestClassifier(n_estimators=100)
model = model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
## Actual class predictions
predicted = model.predict(testdata)
pn.savetxt('predictedRF.txt', predicted, fmt='%01d')
# summarize the fit of the model
accuracy = accuracy_score(expected, predicted)


cm = metrics.confusion_matrix(expected, predicted)
tpr = float(np.sum([cm[1][1],cm[2][2],cm[3][3],cm[4][4],cm[5][5],cm[6][6],cm[7][7],cm[8][8],cm[9][9],cm[10][10],cm[11][11],cm[12][12],cm[13][13]])/np.sum(cm[1]))
fpr = float(np.sum([cm[0][1],cm[0][2],cm[0][3],cm[0][4],cm[0][5],cm[0][6],cm[0][7],cm[0][8],cm[0][9],cm[0][10],cm[0][11],cm[0][12],cm[0][13]])/np.sum(cm[0]))
print("Accuracy: ")
print("%.3f" %accuracy)


print("FPR: ")
print("%.3f" %fpr)
print("TPR: ")
print("%.3f" %tpr)

print ("Confusion matrix: ")
print (metrics.confusion_matrix(expected, predicted))
# performance
print ("Classification report for %s",  model)
print ("\n")
print (metrics.classification_report(expected, predicted))






