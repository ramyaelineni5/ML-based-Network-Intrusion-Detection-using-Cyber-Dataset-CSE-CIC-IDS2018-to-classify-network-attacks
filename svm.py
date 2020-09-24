import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix




traindata = pd.read_csv('preprocessed_train_4a_new.csv', header=None)
testdata = pd.read_csv('preprocessed_test_4a_new.csv', header=None)
X = traindata.iloc[0:,1:71]# 71 columns of data frame with all rows
Y = traindata.iloc[0:,71]

T = testdata.iloc[0:,1:71]
C = testdata.iloc[0:,71]


scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
scaler = Normalizer().fit(T)
testT = scaler.transform(T)


traindata = np.array(trainX)
trainlabel = np.array(Y)

testdata = np.array(testT)
testlabel = np.array(C)
#SVM
print("------------------------SVM Classifier--------------------")



print("----------------------SVM linear binary---------------------------")
model = svm.SVC(kernel='linear', C=1000,probability=False)
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
#proba = model.predict_proba(testdata)



# summarize the fit of the model
accuracy = accuracy_score(expected, predicted)

cm = metrics.confusion_matrix(expected, predicted)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])

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


print("-----------------------SVMrbf Binary--------------------------------------")
model = svm.SVC(kernel='rbf', probability= False)
model = model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
predicted = model.predict(testdata)
accuracy = accuracy_score(expected, predicted)
#recall = recall_score(expected, predicted, pos_label=1,average="binary")
#precision = precision_score(expected, predicted, pos_label=1,average="binary")
#f1 = f1_score(expected, predicted, pos_label=1,average="binary")

cm = metrics.confusion_matrix(expected, predicted)
#print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])

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

