import sys
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import TimeSeriesSplit

companies = pd.read_csv('DataAnal/диплом/Financial Distress.csv')
companies.rename(index=str, columns={"Company": "company", "Time": "time", "Financial Distress": "financial_distress"}, inplace=True)

total_n = len(companies.groupby('company')['company'].nunique())
distress_companies = companies[companies['financial_distress'] < -0.5]
u_distress = distress_companies['company'].unique()
feature_names = list(companies.columns.values)[3:] # ignore first 3: company, time, financial_distress

f80 = list(companies.groupby('company')['x80'].agg('mean'))
f80 = [int(c) for c in f80]

datadict = {}
distress_dict = {}

for i in range (1, total_n+1):
    datadict[i] = {}
    distress_dict[i] = {}

for idx, row in companies.iterrows():
    company = row['company']
    time = int(row['time'])
    
    datadict[company][time] = {}
    
    if row['financial_distress'] < -0.5:
        distress_dict[company][time] = 1
    else:
        distress_dict[company][time] = 0
        
    for feat_idx, column in enumerate(row[3:]):
        feat = feature_names[feat_idx]
        datadict[company][time][feat] = column
        
label_binarizer = LabelBinarizer()
label_binarizer.fit(range(max(f80)))
f80_oh = label_binarizer.transform(f80)

def rolling_operation(time, train_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods):

    for company in range(1, total_n+1):
            
            all_periods_exist = True
            for j in range(0, lookback_periods):
                if not time-j in distress_dict[company]:
                    all_periods_exist = False
            if not all_periods_exist:
                continue
            
            distress_at_eop = distress_dict[company][time]
            new_row = [company]

            for feature in feature_names:
                if feature == 'x80':
                    continue
                feat_sum = 0.0
                variance_arr = []
                for j in range(0, lookback_periods):
                    feat_sum += datadict[company][time-j][feature]
                    variance_arr.append(datadict[company][time-j][feature])
                new_row.append(feat_sum)
                new_row.append(np.var(variance_arr))
                
            for j in range(0,len(f80_oh[0])):
                new_row.append(f80_oh[company-1][j])

            if len(new_row) == ((len(feature_names)-1)*2 + 1 + len(f80_oh[0])) : # we have a complete row
                new_row.append(distress_at_eop)
                new_row_np = np.asarray(new_row)
                train_array.append(new_row_np)
    

def custom_timeseries_cv(datadict, distress_dict, feature_names, total_n, val_time, test_time, 
                         lookback_periods, total_periods=14):

    # Train data
    train_array = []
    for _t in range(1, val_time+1):
        time = (val_time+1) -_t # Start from time period 10 and work backwards
        train_array_np = rolling_operation(time, train_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods)

    train_array_np = np.asarray(train_array)
    print(train_array_np.shape)
    # print(train_array_np[0])
    
    # Val data
    if val_time != test_time:
        val_array = []
        for time in range(val_time+1, test_time+1):
            val_array_np = rolling_operation(time, val_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods)

        val_array_np = np.asarray(val_array)
        print(val_array_np.shape)
        # print(val_array_np[0])
    else:
        val_array_np = None

    # Test data
    test_array = []
    # start from time period 11 and work forwards
    for time in range(test_time+1,total_periods+1):
        test_array_np = rolling_operation(time, test_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods)

    test_array_np = np.asarray(test_array)
    print(test_array_np.shape)
    # print(test_array_np[0])
    
    return train_array_np, val_array_np, test_array_np

# Generate our sets
train_array_np, val_array_np, test_array_np = custom_timeseries_cv(datadict, distress_dict, feature_names, total_n,
                                                     val_time=9, test_time=12, lookback_periods=3, total_periods=14)

X_train = train_array_np[:,0:train_array_np.shape[1]-1]
y_train = train_array_np[:,-1].astype(int)

X_val = val_array_np[:,0:val_array_np.shape[1]-1]
y_val = val_array_np[:,-1].astype(int)

X_test = test_array_np[:,0:test_array_np.shape[1]-1]
y_test = test_array_np[:,-1].astype(int)

np.set_printoptions(threshold=sys.maxsize)

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier

def KNN():
    st.title("Модель прогнозирования - KNN")
    knn = KNeighborsClassifier(n_neighbors=15)
    clf = knn.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc_knb_model=roc_auc_score(y_test, y_pred)*100
    acc_knb_model
    
def LR():
    lr = LogisticRegression(C = 0.2)
    clf1 = lr.fit(X_train, y_train)
    y_pred1 = clf1.predict(X_test)
    acc_log_reg=roc_auc_score(y_test, y_pred1)*100
    acc_log_reg
    
def GNB():
    clf2 = GaussianNB().fit(X_train, y_train)
    y_pred2 = clf2.predict(X_test)
    acc_nb=roc_auc_score(y_test, y_pred2)*100
    acc_nb
    
def DT():
    clf3 = tree.DecisionTreeClassifier().fit(X_train, y_train)
    y_pred3 = clf3.predict(X_test)
    acc_dt=roc_auc_score(y_test, y_pred3)*100
    acc_dt
    
def RF():
    clf3 = tree.DecisionTreeClassifier().fit(X_train, y_train)
    y_pred3 = clf3.predict(X_test)
    acc_dt=roc_auc_score(y_test, y_pred3)*100
    acc_dt
    
def SVC():
    clf5 = SVC(gamma='auto').fit(X_train, y_train)
    y_pred5 = clf5.predict(X_test)
    acc_svm_model=roc_auc_score(y_test, y_pred5)*100
    acc_svm_model
    
def SGD():
    sgd_model=SGDClassifier()
    sgd_model.fit(X_train,y_train)
    sgd_pred=sgd_model.predict(X_test)
    acc_sgd=round(sgd_model.score(X_train,y_train)*100,10)
    acc_sgd
    
def XGB():
    xgb_model=XGBClassifier()
    xgb_model.fit(X_train,y_train)
    xgb_pred=xgb_model.predict(X_test)
    acc_xgb=round(xgb_model.score(X_train,y_train)*100,10)
    acc_xgb
    
def LGBM():
    lgbm = LGBMClassifier()
    lgbm.fit(X_train,y_train)
    lgbm_pred=lgbm.predict(X_test)
    acc_lgbm=round(lgbm.score(X_train,y_train)*100,10)
    acc_lgbm
    
def LR():
    regr = linear_model.LinearRegression()
    regr.fit(X_train,y_train)
    regr_pred=regr.predict(X_test)
    acc_regr=round(regr.score(X_train,y_train)*100,10)
    acc_regr
    
    
def Results():
    results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest','Stochastic Gradient Decent','Linear Regression','Naive Bayes','XGBoost','LightGBM','Decision Tree'],
    'Score': [acc_svm_model, acc_knb_model, acc_log_reg, 
              acc_rmf_model,acc_sgd,acc_regr,acc_nb,acc_xgb,acc_lgbm,acc_dt]})
    result_df = results.sort_values(by='Score', ascending=False)
    result_df = result_df.set_index('Score')
    result_df