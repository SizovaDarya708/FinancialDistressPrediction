import streamlit as st

def PrepFor():
    st.write("Варианты предобработки")
    method = st.selectbox("Выбор предобработки", ["Выбор предобработки", 
                                                  "Для методов машинного обучения", 
                                                  "Для нейронной сети",
                                                  "Для регрессионных методов"])
    if method == "Выбор предобработки":
        st.write("Выбор вариантов предобработки")
    elif method == "Для методов машинного обучения":
        DataPrepForML()
    elif method == "Для нейронной сети":
        DataPrepForANN()
    elif method == "Для регрессионных методов":
        DataPrepForRegr()
        
def DataPrepForRegr():
    st.write("""Предобработка данных для регрессионных методов""")
    st.write("""    
    Первый шаг:
    \nУстановка библиотек
    \nPython 3 обладает многими полезными аналитическими библиотеками, например, вот несколько полезных пакетов для загрузки: numpy(линейная алгебра), pandas(обработка данных), sklearn(аналитика)""")
     if st.checkbox("Показать код 1"):
        st.code( """import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
import xgboost as xgb

from sklearn.metrics import r2_score""")
    st.write("""    
    Второй шаг:
    \nЗагрузка данных из файла 
    \nФайлы входных данных доступны в каталоге "../input/"., но можно также указать конкретный путь к файлу.
    \nЗдесь же происходит предобработка данных: проверка на пропущенные значения, удаления столбцов, не участвующих в обучении, удаление фиктивных переменных с помощью вспомогательной функции.
    """)
     if st.checkbox("Показать код 2"):
            st.code("""import os
print(os.listdir("../input"))
dataset = pd.read_csv('../input/Financial Distress.csv')")

data = data.drop(['Company', 'Time'], axis=1)

def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df
    
data = onehot_encode(data, column='x80', prefix='x80')
""")
    st.write("""    
    Третий шаг:
    \nРазделение выборки и ее масштабирование 
    \n С помощью библиотеки sklearn.preprocessing и ее функции StandartScaler() происходит стандартизация функций путем удаления среднего и масштабирования до единичной дисперсии. Вызов функции fit_transform позволяет подогнать к данным, а затем преобразовать выборку.
    """)
     if st.checkbox("Показать код 3"):
            st.code("""
            y = data['Financial Distress'].copy()
X = data.drop('Financial Distress', axis=1).copy()

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=100)
""")
    
    
def DataPrepForANN():
    st.write("""Предобработка данных для нейронной сети""")
    st.write("""    
    Первый шаг:
    \nУстановка библиотек
    \nPython 3 обладает многими полезными аналитическими библиотеками, например, вот несколько полезных пакетов для загрузки: numpy(линейная алгебра), pandas(обработка данных)""")
     if st.checkbox("Показать код 1"):
        st.code( """import numpy as np # linear algebra
import pandas as pd """)
    st.write("""    
    Второй шаг:
    \nЗагрузка данных из файла 
    \n Файлы входных данных доступны в каталоге "../input/"., но можно также указать конкретный путь к файлу.""")
     if st.checkbox("Показать код 2"):
            st.code("""import os
print(os.listdir("../input"))
dataset = pd.read_csv('../input/Financial Distress.csv')")""")
    st.write("""    
    Третий шаг:
    \nОбработка столбцов
    \n Необходимо разделить набор данных на зависимые и независимые переменные""")
     if st.checkbox("Показать код 3"):
            st.code("""X = dataset.iloc[:,:]
y = pd.DataFrame(X.iloc[:,2])
X.drop(columns = 'Financial Distress', inplace=True)""")
    st.write("""    
    Четвертый шаг:
    \nКодирование категориальных переменных""")
     if st.checkbox("Показать код 4"):
            st.code("""from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
labelencoder_X_3 = LabelEncoder()

Companies = np.unique(X["Company"])
C = pd.DataFrame(labelencoder_X_1.fit_transform(X["Company"]))
onehotencoder = OneHotEncoder(categorical_features = [0])
C = pd.DataFrame(onehotencoder.fit_transform(C).toarray())
C.columns = Companies

Time = np.unique(X["Time"])
T = pd.DataFrame(labelencoder_X_2.fit_transform(X["Time"]))
onehotencoder = OneHotEncoder(categorical_features = [0])
T = pd.DataFrame(onehotencoder.fit_transform(T).toarray())
T.columns = Time

Features = np.unique(X["x80"])
F = pd.DataFrame(labelencoder_X_3.fit_transform(X["x80"]))
onehotencoder = OneHotEncoder(categorical_features = [0])
F = pd.DataFrame(onehotencoder.fit_transform(F).toarray())
F.columns = Features

P = pd.DataFrame(pd.concat((C,T),axis = 1))

X = X.drop(["Company","Time"],axis=1)
X = pd.DataFrame(pd.concat((P,X,F),axis = 1))""")
    st.write("""    
    Пятый шаг:
    \nРазделение набора данных на обучающую и тестовую выборки""")
     if st.checkbox("Показать код 5"):
            st.code("""y = (y<-0.5)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0,stratify = y)""")
    st.write("""    
    Шестой шаг:
    \nМасштабирование функций""")
     if st.checkbox("Показать код 6"):
            st.code("""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)""")
    
                                                           
def DataPrepForML():
    st.write("Предобработка данных для методов машинного обучения")
    st.write("""
    Цель состоит в том, чтобы предсказать, станет ли компания убыточной.
    \nПервый шаг:
    \nЗагрузить данные из файла    
    """)
    if st.checkbox("Показать код 1"):
        st.code( """companies = pd.read_csv('../Financial Distress.csv')
companies.rename(index=str, columns={"Company": "company", "Time": "time", "Financial Distress": "financial_distress"}, inplace=True)

total_n = len(companies.groupby('company')['company'].nunique())
distress_companies = companies[companies['financial_distress'] < -0.5]
u_distress = distress_companies['company'].unique()
feature_names = list(companies.columns.values)[3:] """ )
    st.write("""    
    Второй шаг:
    \nНастройка столбцов 
    \nОписание Kaggle говорит нам, что если число в столбце financal_distress <-0.5, компанию следует считать проблемной(убыточной).
    Мы можем представить, что это может быть какое-то финансовое соотношение, например, отношение дохода к капиталу.
    """)
    if st.checkbox("Показать код 2"):
        st.code(""" if row['financial_distress'] < -0.5:
        distress_dict[company][time] = 1
    else:
        distress_dict[company][time] = 0 """ )
    st.write("""    
    Третий шаг:
    \nCross validation
    \nСледует ввести новый набор функций для каждой обучающей строки: сумма по каждой функции за время t, t-1, t-2 ... t-n.
    Цель обучения будет заключаться в том, чтобы выяснить, произошло ли бедствие компании в конце периода (t).    
    """)
    if st.checkbox("Показать код 3"):
        st.code( """
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
        f80_oh = label_binarizer.transform(f80)""" )
    st.write("""    
    Четвертый шаг:
    \nГенерация данных для обучения
    \nСоздаем новые функции в виде массива np. 
    
    """)
    if st.checkbox("Показать код 4"):
        st.code("""def rolling_operation(time, train_array, datadict, distress_dict, feature_names, total_n,
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
                                                     val_time=9, test_time=12, lookback_periods=3, total_periods=14)""")
        
    st.write("""    
    Пятый шаг:
    \nСоздаем метки
    \nВытягиваем последний столбец как метку     
    """)
    if st.checkbox("Показать код 5"):
        st.code("""X_train = train_array_np[:,0:train_array_np.shape[1]-1]
y_train = train_array_np[:,-1].astype(int)

X_val = val_array_np[:,0:val_array_np.shape[1]-1]
y_val = val_array_np[:,-1].astype(int)

X_test = test_array_np[:,0:test_array_np.shape[1]-1]
y_test = test_array_np[:,-1].astype(int)""")
        
    
        