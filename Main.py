import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    st.sidebar.header('Прогнозирование банкротства компаний')
    page = st.sidebar.selectbox("Навигатор", ["Теория", "Практика","О данных"])
    
    data = load_data()    
    st.set_option('deprecation.showPyplotGlobalUse', False)    
    if page == "Теория":
        st.header("Прогнозирование банкротства компаний")
        st.write("Please select a page on the left.")
        st.image('DataAnal/диплом/pic/main.PNG')
        st.header("О методах прогнозирования, используемых в программе")
        aboutModels()
        info()
    elif page == "О данных":
        st.title("Данные для прогнозирования")
        st.write(data.head())
        st.write("Данные взяты с сайта kaggle")
        st.write('Данные взяты https://www.kaggle.com/shebrahimi/financial-distress')
        st.write("Этот набор данных предназначен для прогнозирования финансовых бедствий для выборки компаний.")
        st.write(data.describe())
        if st.checkbox("О файле Financial Distress.csv"):
            st.write("Первый столбец: Компания представляет собой образцы компаний.")
            st.write("Второй столбец: Время показывает разные периоды времени, которым принадлежат данные. Длина временного ряда варьируется от 1 до 14 для каждой компании.")
            st.write("Третий столбец: целевая переменная обозначается как «Финансовый кризис», если она будет больше -0,50, компанию следует считать здоровой (0). В противном случае он был бы расценен как финансово неблагополучный (1).")
            st.write("От четвертого до последнего столбца: характеристики, обозначенные от x1 до x83, представляют собой некоторые финансовые и нефинансовые характеристики отобранных компаний. Эти характеристики относятся к предыдущему периоду времени, который следует использовать для прогнозирования того, будет ли компания испытывать финансовые затруднения или нет (классификация). Признак x80 - категориальная переменная")
        Visualize(data)
    elif page == "XGBoosting":
        xgb = XGBoosting(data)
        st.write(xgb)
    elif page == "Практика":
        st.title("Data Exploration")
        st.write(data.head())
        st.write("""# Прогноз финансовых бедствий различных компаний""")
        method = st.selectbox("Выбор метода прогнозирования", ["Случайный лес", "XGBoosting", "Логистическая регрессия", "ANN", "Вывод"])
        if method == "Случайный лес":
            st.title("Модель прогнозирования - случайный лес")       
            #tree = RandomForest(data)
            #st.write(tree)
            r = forest()
            st.write(r)
        elif method == "XGBoosting":
            st.title("Модель прогнозирования - бустинг") 
            b = boost()
            st.write(b)
        elif method == "Логистическая регрессия":
            st.title("Модель прогнозирования - Логистическая регрессия")
            r = regress()
            st.write(r)
        elif method == "ANN":
            st.title("Модель прогнозирования - ANN") 
            a = ann()                    
            st.write(a)       
        elif method == "Вывод":
            st.title("Сравнение моделей") 
            st.image('DataAnal/диплом/pic/Итог.PNG')
            
def aboutModels():
    st.header("Где можно использовать линейную регрессию?")
    st.write("Это очень мощный метод, и его можно использовать для понимания факторов, влияющих на прибыльность. Его можно использовать для прогнозирования продаж в ближайшие месяцы путем анализа данных о продажах за предыдущие месяцы. Он также может быть использован для получения различной информации о поведении клиентов. К концу блога мы создадим модель, которая выглядит как на картинке ниже, т.е. определим линию, которая наилучшим образом соответствует данным.")
    st.image('DataAnal/диплом/pic/linear.gif')
    st.image('DataAnal/диплом/pic/mathLinear.PNG')
    
    st.header("Логистическая регрессия")
    st.write("Логистическая регрессия обычно используется для целей классификации. В отличие от линейной регрессии, зависимая переменная может принимать ограниченное количество значений только, т. Е. Зависимая переменнаякатегорический, Когда число возможных результатов только два, это называетсяБинарная логистическая регрессия")
    st.write("В линейной регрессии выход является взвешенной суммой входных данных. Логистическая регрессия - это обобщенная линейная регрессия в том смысле, что мы не выводим взвешенную сумму входных данных напрямую, а пропускаем ее через функцию, которая может отображать любое действительное значение в диапазоне от 0 до 1.")
    st.image('DataAnal/диплом/pic/logistic.gif')
    
    st.header("Случайный лес")
    st.write("RF (random forest) — это множество решающих деревьев. В задаче регрессии их ответы усредняются, в задаче классификации принимается решение голосованием по большинству.")
    st.image('DataAnal/диплом/pic/random.gif')
    
    st.header("Нейронные сети")    
    st.write("ИНС представляет собой систему соединённых и взаимодействующих между собой простых процессоров (искусственных нейронов). Такие процессоры обычно довольно просты (особенно в сравнении с процессорами, используемыми в персональных компьютерах). Каждый процессор подобной сети имеет дело только с сигналами, которые он периодически получает, и сигналами, которые он периодически посылает другим процессорам. И, тем не менее, будучи соединёнными в достаточно большую сеть с управляемым взаимодействием, такие по отдельности простые процессоры вместе способны выполнять довольно сложные задачи.")
    st.image('DataAnal/диплом/pic/ann1.gif')
    


def info():
    st.header("Статьи и материалы, используемые в работе:")
    st.write("Использование Streamlit")
    st.write("https://blog.skillfactory.ru/nauka-o-dannyh-data-science/kak-napisat-veb-prilozhenie-dlya-demonstratsii-data-science-proekta-na-python/ https://docs.streamlit.io/en/stable/api.html")
    st.write("https://medium.com/nuances-of-programming")
    st.write("О работе с несбалансированными данными:")
    st.write("https://www.machinelearningmastery.ru/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/")
    st.write("https://coderoad.ru/40568254/")
    st.write("О моделях прогнозирования")
    st.write("Логистическая регрессия:")
    st.write("https://www.machinelearningmastery.ru/building-a-logistic-regression-in-python-301d27367c24/")
    st.write("https://medium.com/nuances-of-programming")
    st.write("Случайный лес:")
    st.write("https://dyakonov.org/2016/11/14/")
    st.write("https://ru.wikipedia.org/wiki/Random_forest")
    st.write("https://www.machinelearningmastery.ru/implement-random-forest-scratch-python/")
    st.write("XGBoosting:")
    st.write("https://xgboost.readthedocs.io/en/latest/python/python_intro.html")
    st.write("https://www.machinelearningmastery.ru/xgboost-python-mini-course/")
       
    st.write("ANN")
    st.write("https://en.wikipedia.org/wiki/Artificial_neural_network")
    st.write(" https://www.geeksforgeeks.org/implementing-ann-training-process-in-python/")
    st.write("https://www.coursera.org/projects/basic-artificial-neural-networks-in-python")
    
    st.write("Прогнозирование временных рядов с помощью рекуррентных нейронных сетей")
    st.write("https://habr.com/ru/post/495884/")
    st.write("Прогнозирование временных рядов с помощью рекуррентных нейронных сетей LSTM в Python с использованием Keras")
    st.write("https://www.machinelearningmastery.ru/time-series-prediction-lstm-recurrent-neural-networks-python-keras/")
    st.write("How to Configure XGBoost for Imbalanced Classification")
    st.write("https://machinelearningmastery.com/xgboost-for-imbalanced-classification/#:~:text=The%20XGBoost%20algorithm%20is%20effective,over%20the%20model%20training%20procedure.")
    st.write("Intro XGboost Classification")
    st.write("https://www.kaggle.com/babatee/intro-xgboost-classification")
    
    
    
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    figure = plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    st.pyplot(plt.show())
    
    
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') 
    df = df[[col for col in df if df[col].nunique() > 1]] 
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    st.pyplot(plt.show())
    
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    st.pyplot(plt.show(ax))
    
def Visualize(data):
    data.dataframeName = 'Financial Distress.csv'
    nRow, nCol = data.shape
    st.write(f' Данные содержат {nRow} строк и {nCol} столбцов')    
    st.write("Графики распределения:")
    plotPerColumnDistribution(data, 10, 5)
    st.write("Матрица корреляции:")
    plotCorrelationMatrix(data, 21)
    st.write("Графики разброса и плотности:")
    plotScatterMatrix(data, 20, 10)
   # st.write("Распределение по времени")    
   # sns.distplot(data['Time'])
   # st.pyplot(sns.show())
    
    

def ann():     
    return 0.9601    
    
def regress():                     
    st.image('DataAnal/диплом/pic/Логистическая регрессия.PNG')
    return 0.9395


def boost():                     
    st.image('DataAnal/диплом/pic/XGBoost.PNG')
    return 0.9717

def forest():
    st.image('DataAnal/диплом/pic/Случайный лес.PNG')
    return 0.9307

@st.cache
def load_data():
    data = pd.read_csv('DataAnal/диплом/Financial Distress.csv')
    return data

def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df).mark_circle(size=60).encode(
        x=x_axis,
        y=y_axis,
        color='Origin',
        tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
    ).interactive()

    st.write(graph)

@st.cache
def RandomForest(data):
    distressed = [1 if row['Financial Distress'] <= -0.5 else 0 for _, row in data.iterrows()]
    data_full = data
    data_full['Distressed'] = pd.Series(distressed)
    data_full.loc[data_full['Distressed'] == 1, ['Financial Distress', 'Distressed']].head(10)

    SSS = StratifiedShuffleSplit(random_state=10, test_size=.3, n_splits=1)
    X = data_full.iloc[:, 3:-1].drop('x80', axis=1)
    y = data_full['Distressed'] 
    for train_index, test_index in SSS.split(X, y):
        print("CV:", train_index, "HO:", test_index)
        X_cv, X_ho = X.iloc[train_index], X.iloc[test_index]
        y_cv, y_ho = y[train_index], y[test_index]
        
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 50)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(5, 55, num = 10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 3, 4]
    bootstrap = [True, False]
    class_weight = ['balanced', None]

    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'class_weight': class_weight}

    rf_clsf = RandomForestClassifier(random_state=10, class_weight='balanced')
    rf_random = RandomizedSearchCV(estimator = rf_clsf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=10, n_jobs = -1, refit='f1', scoring=['f1', 'precision', 'recall'])
    rf_random.fit(X_cv, y_cv)
    
    predict = rf_random.predict(X_ho)
    probs = rf_random.predict_proba(X_ho)[:,1]
    
    from sklearn.metrics import roc_auc_score

    # Рассчитываем roc auc
    roc_value = roc_auc_score(y_ho, probs)
    
    from sklearn import metrics
    fpr, tpr, _ = metrics.roc_curve(y_ho, probs)
    figure = plt.plot(fpr,tpr,label="data 1, auc="+str(roc_value))
    plt.legend(loc=4)
    st.write(roc_value)
    #вывести график
    st.pyplot(figure)

    best_rf_clsf = rf_random.best_estimator_
    best_rf_clsf.fit(X_cv, y_cv)    
    
    return best_rf_clsf;

@st.cache
def XGBoosting(data):
    import xgboost as xgb
    xgb_learning_rate = [x for x in np.linspace(start = 0.001, stop = 0.1, num = 10)]
    xgb_n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    xgb_booster = ['gbtree', 'dart']
    xgb_colsample_bytree = [0.4, 0.6, 0.8, 1.0]
    xgb_colsample_bylevel = [0.5, 0.75, 1.0]
    xgb_scale_pos_weight = [(len(y_cv) - sum(y_cv))/sum(y_cv)]
    xgb_min_child_weight = [1]
    xgb_subsample = [0.5, 1.0]


    random_grid = {'learning_rate': xgb_learning_rate,
               'n_estimators': xgb_n_estimators,
               'booster': xgb_booster,
               'colsample_bytree': xgb_colsample_bytree,
               'colsample_bylevel': xgb_colsample_bylevel,
               'scale_pos_weight': xgb_scale_pos_weight,
               'min_child_weight': xgb_min_child_weight,
               'subsample': xgb_subsample}
    xgb_clsf = xgb.XGBClassifier(random_state=10)
    xgb_random = RandomizedSearchCV(estimator = xgb_clsf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=10, n_jobs = -1, refit='f1', scoring=['f1', 'precision', 'recall'])
    xgb_random.fit(X_cv, y_cv)
    
    best_xgb_clsf = xgb_random.best_estimator_
    best_xgb_clsf.fit(X_cv, y_cv)
    
    return best_xgb_clsf

if __name__ == "__main__":
    main()



#основная панель

#data = pd.read_csv('DataAnal/диплом/Financial Distress.csv')
#st.write(data.head())

#methods = ["RandomForest", "XGBoosting", "Logistic Regression"]
#choosenMethod = st.selectbox("Выберете модель прогнозирования: ", methods)

#if choosenMethod == 'RandomForest': 
  #  st.write('Выбран случайный лес')
    #tree = RandomForest()
    #st.write(tree)
#if choosenMethod == 'RXGBoosting':
  #  st.write('Выбран градиентный бустинг')
  #  xgboost = XGBoosting()
  #  st.write(xgboost)

