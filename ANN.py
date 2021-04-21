import numpy as np 
import pandas as pd 

dataset = pd.read_csv('DataAnal/диплом/Financial Distress.csv')

X = dataset.iloc[:,:]
y = pd.DataFrame(X.iloc[:,2])
X.drop(columns = 'Financial Distress', inplace=True)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
labelencoder_X_3 = LabelEncoder()

Companies = np.unique(X["Company"])
C = pd.DataFrame(labelencoder_X_1.fit_transform(X["Company"]))
columnTransformer = ColumnTransformer([("Country", OneHotEncoder(), [0])])
C = pd.DataFrame(columnTransformer.fit_transform(C).toarray())
C.columns = Companies


Time = np.unique(X["Time"])
T = pd.DataFrame(labelencoder_X_2.fit_transform(X["Time"]))
onehotencoder = ColumnTransformer([('encoder', OneHotEncoder(), [0])])
T = pd.DataFrame(onehotencoder.fit_transform(T).toarray())
T.columns = Time

Features = np.unique(X["x80"])
F = pd.DataFrame(labelencoder_X_3.fit_transform(X["x80"]))
onehotencoder = ColumnTransformer([('encoder', OneHotEncoder(), [0])])
F = pd.DataFrame(onehotencoder.fit_transform(F).toarray())
F.columns = Features

P = pd.DataFrame(pd.concat((C,T),axis = 1))

X = X.drop(["Company","Time"],axis=1)
X = pd.DataFrame(pd.concat((P,X,F),axis = 1))


y = (y<-0.5)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0,stratify = y)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras 
from keras.models import Sequential 
from keras.layers import Dense

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
     self.val_f1s = []
     self.val_recalls = []
     self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):  
     val_predict = self.model.predict_classes(self.validation_data[0])
     val_targ = self.validation_data[1]
     _val_f1 = f1_score(val_targ, val_predict)
     _val_recall = recall_score(val_targ, val_predict)
     _val_precision = precision_score(val_targ, val_predict)
     self.val_f1s.append(_val_f1)
     self.val_recalls.append(_val_recall)
     self.val_precisions.append(_val_precision)
     print(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
     return
 
metrics = Metrics()

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 30, kernel_initializer = "uniform", activation = 'relu', input_shape=(556,)))

# Adding the second hidden layer

classifier.add(Dense(units = 30, kernel_initializer = "uniform", activation = 'relu'))

# Adding the third hidden layer

classifier.add(Dense(units = 30, kernel_initializer = "uniform", activation = 'relu'))

# Adding the fourth hidden layer

classifier.add(Dense(units = 30, kernel_initializer = "uniform", activation = 'relu'))

# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = 'sigmoid'))

# Compiling the ANN

classifier.compile(loss='binary_crossentropy',
          optimizer= "adam",
          metrics = ['accuracy']
          )

from sklearn.utils import class_weight
class_weight_real = class_weight.compute_class_weight('balanced'
                                               ,np.unique(y_train['Financial Distress'])
                                               ,y_train['Financial Distress'])

classifier.fit(X_train, y_train, 
 validation_data=(X_test, y_test),
 epochs=200,
 batch_size=200
 )

y_pred = classifier.predict_classes(X_test)
y_pred2 = classifier.predict_classes(X_train)


cm = confusion_matrix(y_test , y_pred)

cm2 = confusion_matrix(y_train, y_pred2)