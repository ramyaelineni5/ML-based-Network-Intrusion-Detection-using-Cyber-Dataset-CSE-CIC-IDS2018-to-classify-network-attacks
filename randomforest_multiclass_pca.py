# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 11:48:05 2020

@author: ramya
"""
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
#from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
#from sklearn.metrics import auc





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
#SVM
#print("------------------------SVM Classifier--------------------")

#proba = model.predict_proba(testdata)
#np.savetxt('G:/Pramita/FALL 2018_2ND_SEM/MS_Thesis/NIDS/Network-Intrusion-Detection-master/NSL-KDD/traditional/binary/predictedlabelSVM-rbf.txt', predicted, fmt='%01d')
#np.savetxt('G:/Pramita/FALL 2018_2ND_SEM/MS_Thesis/NIDS/Network-Intrusion-Detection-master/NSL-KDD/traditional/binary/predictedprobaSVM-rbf.txt', proba)

print("-------------------------Random Forest multi_class----------------------------")
#create the model with 100 trees
model = RandomForestClassifier(n_estimators=100)
model = model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
## Actual class predictions
predicted = model.predict(testdata)
np.savetxt('predictedRF.txt', predicted, fmt='%01d')
# summarize the fit of the model
accuracy = accuracy_score(expected, predicted)
#recall = recall_score(expected, predicted, pos_label=1,average="binary")
#precision = precision_score(expected, predicted , pos_label=1,average="binary")
#f1 = f1_score(expected, predicted , pos_label=1,average="binary")

cm = metrics.confusion_matrix(expected, predicted)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
#print("%.3f", tpr)
#print("%.3f", fpr)
print("Accuracy: ")
print("%.3f" %accuracy)

#print("Precision: ")
#print("%.3f" %precision)
#print("Recall: ")
#print("%.3f" %recall)
##print("f-score: ")
#print("%.3f" %f1)
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






