# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 11:48:05 2020

@author: ramya
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA







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

#newcode pca train 
pca=PCA(n_components=50)
pca.fit(trainX )
x_train_pca=pca.transform(trainX)

#newcode pca test
pca.fit(testT)
x_test_pca=pca.transform(testT) 



traindata = np.array(x_train_pca)
trainlabel = np.array(Y)

testdata = np.array(x_test_pca)
testlabel = np.array(C)

#print("***************************************************************")

print("-------------------------SVM Linear Multiclass--------------------------------")

clf_SVM=SVC(kernel='linear', C=1.0, random_state=0)
clf_SVM.fit(traindata, trainlabel)#.astype(int))
Y_pred=clf_SVM.predict(testdata)
expected = testlabel
accuracy = accuracy_score(expected, Y_pred)
recall = recall_score(expected, Y_pred , average="macro")
precision = precision_score(expected, Y_pred , average="macro")
f1 = f1_score(expected, Y_pred, average="macro")

print("Accuracy: ")
print("%.3f" %accuracy)
print("Precision: ")
print("%.3f" %precision)
print("Recall: ")
print("%.3f" %recall)
print("f1-score: ")
print("%.3f" %f1)

print ("Confusion matrix: ")
print (metrics.confusion_matrix(expected, Y_pred))
# performance
print ("Classification report for %s",  clf_SVM)
print ("\n")
print (metrics.classification_report(expected, Y_pred))

#print("------------------------SVM Classifier--------------------")
model = svm.SVC(kernel='rbf', gamma = 'auto', probability=False)
model = model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
predicted = model.predict(testdata)


print("-----------------------SVMrbf multiclass(all features)--------------------------------------")

accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="macro")
precision = precision_score(expected, predicted , average="macro")
f1 = f1_score(expected, predicted, average="macro")

cm = metrics.confusion_matrix(expected, predicted)
tpr = float(np.sum([cm[1][1],cm[2][2],cm[3][3],cm[4][4],cm[5][5],cm[6][6],cm[7][7],cm[8][8],cm[9][9],cm[10][10],cm[11][11],cm[12][12],cm[13][13]])/np.sum(cm[1]))
fpr = float(np.sum([cm[0][1],cm[0][2],cm[0][3],cm[0][4],cm[0][5],cm[0][6],cm[0][7],cm[0][8],cm[0][9],cm[0][10],cm[0][11],cm[0][12],cm[0][13]])/np.sum(cm[0]))
print("Accuracy: ")
print("%.3f" %accuracy)
print("Precision: ")
print("%.3f" %precision)
print("Recall: ")
print("%.3f" %recall)
print("f1-score: ")
print("%.3f" %f1)
print("FPR: ")
print("%.3f" %fpr)

print ("Confusion matrix: ")
print (metrics.confusion_matrix(expected, predicted))
# performance
print ("Classification report for %s",  model)
print ("\n")
print (metrics.classification_report(expected, predicted))




