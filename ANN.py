import numpy as np 
import pandas as pd 
import streamlit as st

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

#balanced data
data1 = pd.read_csv('DataAnal/диплом/Financial Distress.csv')
from imblearn.over_sampling import SMOTE

def label_conv(x):
    if x > -0.5:
        return 0
    else:
        return 1 
    
sm = SMOTE (#sampling_strategy = 0.9,
	    random_state=0,
	    k_neighbors=4)

data1 = data1.drop(['x80'], axis=1)
labels = data1.iloc[:,2].apply(label_conv).values
df = data1.iloc[:,3:].values

X_train0, X_test0, y_train0, y_test0 = train_test_split(df, labels, test_size=0.25, stratify=labels, random_state=33897)

X_train0, X_val, y_train0, y_val = train_test_split(X_train0, y_train0, test_size=0.10, stratify=y_train0, random_state=33897)
X_train_res, y_train_res = sm.fit_sample(X_train0, y_train0)

sc = StandardScaler()
X_train_res = sc.fit_transform(X_train_res)
X_test0 = sc.transform(X_test0)


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
        st.text(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
        #st.text(_val_f1)
        st.text(logs)
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

from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight

def ANN(epo, bs):
    st.title("Модель прогнозирования - ANN")  

    if st.checkbox("Использовать несбалансированные данные"):
        bal = None
    else:
        bal = 'balanced'
        
    class_weight_real = class_weight.compute_class_weight(bal
                                               ,np.unique(y_train['Financial Distress'])
                                               ,y_train['Financial Distress'])
    history = classifier.fit(X_train, y_train, 
         validation_data=(X_test, y_test),
         epochs=epo,
         batch_size=bs    
         )
        
    h = history.history
    st.text(h)
        
    for a in range(epo):
        st.write(f"Epoch {a+1}/{epo}:")
        st.text(f"loss: {h['loss'][a]} | accuracy: {h['accuracy'][a]} | val_loss: {h['val_loss'][a]} | val_accuracy: {h['val_accuracy'][a]}")
        
    y_pred = classifier.predict_classes(X_test)            
    from sklearn.metrics import accuracy_score
    a = accuracy_score(y_pred,y_test)
    st.write(a)
        
    from keras.utils import plot_model
    plot_model(classifier, to_file='DataAnal/диплом/pic/model.png')
    st.image('DataAnal/диплом/pic/model.png')
        
    if st.checkbox("Вывести точность обучения"):
        import matplotlib.pyplot as plt
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
            
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig('DataAnal/диплом/pic/figure.png')
        st.image('DataAnal/диплом/pic/figure.png')
            
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig('DataAnal/диплом/pic/saved_figure.png')
        st.image('DataAnal/диплом/pic/saved_figure.png')