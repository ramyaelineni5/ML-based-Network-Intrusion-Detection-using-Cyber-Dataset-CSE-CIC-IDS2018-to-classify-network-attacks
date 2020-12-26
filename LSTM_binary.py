import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, SpatialDropout1D
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer

from keras import callbacks
from keras.callbacks import CSVLogger


traindata = pd.read_csv('preprocessed_train_4a_new.csv', header=None)
testdata = pd.read_csv('preprocessed_test_4a_new.csv', header=None)

X = traindata.iloc[0:,1:71]# 0 to 71 columns of data frame with all rows
Y = traindata.iloc[0:,71] 

T = testdata.iloc[0:,1:71]
C = testdata.iloc[0:,71]

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train1 = np.array(Y)
y_test1 = np.array(C)
scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
# trainX.shape[0]
#trainX.shape[1]

# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))

batch_size = 50

# 1. define the network
model = Sequential()
#model.add(Embedding(10, emb_dim = 2, input_length=X.shape[1]))
#model.add(SpatialDropout1D(0.7))
model.add(LSTM(80,input_shape=(1, 70), return_sequences= True )) 
model.add(Dropout(0.3))
model.add(LSTM(80,return_sequences=True))  
model.add(Dropout(0.3))
model.add(LSTM(80, return_sequences=True)) 
model.add(Dropout(0.3))
model.add(LSTM(80, return_sequences=True))  
model.add(Dropout(0.3))
model.add(LSTM(80, return_sequences=True))  
model.add(Dropout(0.3))
model.add(LSTM(80, return_sequences=False))  
model.add(Dropout(0.3))
model.add(Dense(1))# the no. of output classes
model.add(Activation('sigmoid'))

model.summary()

# try using different optimizers and different optimizer configs
#model.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = ['accuracy'])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

csv_logger = CSVLogger('Binlstm80.csv',separator=',', append=False)
history= model.fit(X_train, y_train1, batch_size=batch_size, epochs=500, validation_data=(X_test, y_test1),callbacks=[csv_logger])

loss, accuracy = model.evaluate(X_test, y_test1)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_pred = model.predict_classes(X_test)
#print(y_pred)
np.savetxt('Binlst80predicted.txt', y_pred, fmt='%01d')

from sklearn.metrics import confusion_matrix
expected = y_test1
predicted = y_pred
cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
print("%.3f" %fpr)
print("tpr:")
print("%.3f" %tpr)
print("Classification report for %s", model)
print(metrics.classification_report(expected,predicted))





