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
import ML as ml
import DataConverter as dc
import RegressionFunc as rf
import ANN as ann
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from pprint import pprint
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    st.sidebar.header('Прогнозирование банкротства компаний')
    page = st.sidebar.selectbox("Навигатор", ["Теория", "Практика","О данных", "Предобработка данных"])
    data = load_data()    
    st.set_option('deprecation.showPyplotGlobalUse', False)    
    if page == "Теория":
        st.header("Прогнозирование банкротства компаний")
        st.write("Please select a page on the left.")
       # st.image('DataAnal/диплом/pic/main.PNG')
        st.header("О методах прогнозирования, используемых в программе")
        aboutModels()
        info()
    elif page == "О данных":
        AboutData(data)       
        Visualize(data)
    elif page == "Предобработка данных":
        st.title("Предобработка данных")
        st.write(data.head())
        dc.PrepFor()
    elif page == "Практика":
        st.title("Data Exploration")
        st.write(data.head())
        st.write("""# Прогноз финансовых бедствий различных компаний""")
        method = st.selectbox("Выбор метода прогнозирования", ["Выбор модели",
                                                               "LightGBM",
                                                               "Stochastic Gradient Decent",
                                                               "Decision Tree",
                                                               "Naive Bayes",
                                                               "Support Vector Machines",
                                                               "KNN",
                                                               "Logistic Regression",
                                                               "Random Forest",
                                                               "Linear Regression",
                                                               "XGBoost", 
                                                               "ANN",
                                                               "Вывод"])
        if method == "Выбор модели":
            st.write("Выбор метода")
        elif method == "LightGBM":
            params = add_parameter_ui(method)
            get_classifier(method, params)
        elif method == "Stochastic Gradient Decent":
            params = add_parameter_ui(method)
            get_classifier(method, params)
        elif method == "Decision Tree":
            params = add_parameter_ui(method)
            get_classifier(method, params)
        elif method == "Naive Bayes":
            params = add_parameter_ui(method)
            get_classifier(method, params)
        elif method == "Support Vector Machines":
            params = add_parameter_ui(method)
            get_classifier(method, params)
        elif method == "KNN":
            params = add_parameter_ui(method)
            get_classifier(method, params)
        elif method == "Logistic Regression":
            params = add_parameter_ui(method)
            get_classifier(method, params)
        elif method == "Random Forest":
            params = add_parameter_ui(method)
            get_classifier(method, params)
        elif method == "Linear Regression":
            params = add_parameter_ui(method)
            get_classifier(method, params)
        elif method == "XGBoost":
            params = add_parameter_ui(method)
            get_classifier(method, params)
        elif method == "ANN":
            params = add_parameter_ui(method)
            get_classifier(method, params)                  
        elif method == "Вывод":
            st.title("Сравнение моделей") 
            st.image('DataAnal/диплом/pic/Итог.PNG')


#добавить настраиваемые параметры в моделях
def get_classifier(clf_name, params):
    clf = None
    data = load_data() 
    if clf_name == "Случайный лес":
        clf = RandomForest(data, params['n_start'], params['n_stop'], params['n_num'])
    elif clf_name == "LightGBM":
        clf = ml.LGBM( params['num_leaves'], params['n_estimators'],  params['min_child_samples'])
    elif clf_name == "Stochastic Gradient Decent":
        clf = ml.SGD(params['al'], params['epsilon'], params['eta'], params['n_iter'])
    elif clf_name == "Decision Tree":
        clf = ml.DT(params['min_samples_splitint'], params['min_samples_leaf'], params['ccp_alphanon_negative'])    
    elif clf_name == "Naive Bayes":
        clf = ml.GNB()
    elif clf_name == "Support Vector Machines":
        clf = ml.SVM(params['С'], params['degree'], params['cache'])
    elif clf_name == "KNN":
        clf = ml.KNN(params['n_neighbors'], params['leaf_size'], params['p'])
    elif clf_name == "Logistic Regression":
        clf = ml.LOR(params['С'], params['max_iter'])
    elif clf_name == "Random Forest":
        clf = ml.RF(params['max_depth'], params['min_samples_split'], params['min_samples_leaf'])
    elif clf_name == "Linear Regression":
        clf = ml.LR()
    elif clf_name == "Logistic Regression":
        clf = ml.LOR(params['С'], params['max_iter'])
    elif clf_name == "XGBoost":
        clf = ml.XGB()
    elif clf_name == "ANN":
        clf = ann.ANN(params['epo'], params['batch_size'])
    return clf


#посмотреть какие параметры можно изменять и настраивать в моделях
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "Случайный лес":
        n_start = st.sidebar.slider('n_start', 10, 200)
        params['n_start'] = n_start
        n_stop = st.sidebar.slider('n_stop', 300, 500)
        params['n_stop'] = n_stop
        n_num = st.sidebar.slider('n_num', 50, 1000)
        params['n_num'] = n_num
    elif clf_name == "LightGBM":
        st.sidebar.markdown("num_leaves: int, default=31. Максимальное количество листьев на дереве для базового обучения.")
        num_leaves = st.sidebar.slider('num_leaves', 30, 100)
        params['num_leaves'] = num_leaves
        st.sidebar.markdown("n_estimators: int, default=100. Количество boosted trees для обучения.")
        n_estimators = st.sidebar.slider('n_estimators', 50, 200)
        params['n_estimators'] = n_estimators
        st.sidebar.markdown("min_child_samples: int, default=20. Минимальное количество данных, необходимых для потомка/листа.")
        min_child_samples = st.sidebar.slider('min_child_samples', 1, 10)
        params['min_child_samples'] = min_child_samples
    elif clf_name == "Stochastic Gradient Decent":
        st.sidebar.markdown("alpha: float, default=0.0001. Константа, умножающая член регуляризации.")
        al = st.sidebar.slider('alpha', 0.01, 0.1)
        params['al'] = al
        st.sidebar.markdown("epsilon: float, default=0.1. Эпсилон для функции потерь.")
        epsilon = st.sidebar.slider('epsilon', 0.1, 1.0)
        params['epsilon'] = epsilon
        st.sidebar.markdown("eta0: double, default=0.0. Начальная скорость обучения.")
        eta = st.sidebar.slider('eta', 0.0, 1.0)
        params['eta'] = eta
        st.sidebar.markdown("n_iter_no_change: int, default=5. Количество итераций без улучшений, чтобы дождаться досрочной остановки.");
        n_iter = st.sidebar.slider('n_iter', 1, 10)
        params['n_iter'] = n_iter
    elif clf_name == "Decision Tree":
        st.sidebar.markdown("min_samples_split: int or float, default=2. Минимальное количество выборок, необходимых для разделения внутреннего узла.")
        min_samples_splitint = st.sidebar.slider('min_samples_split', 2, 10)
        params['min_samples_splitint'] = min_samples_splitint
        st.sidebar.markdown("min_samples_leaf: int or float, default=1. Минимальное количество выборок, которое требуется для конечного узла.")
        min_samples_leaf = st.sidebar.slider('min_samples_leaf', 0.01, 0.5)
        params['min_samples_leaf'] = min_samples_leaf
        st.sidebar.markdown("ccp_alphanon-negative: float, default=0.0. Параметр сложности, используемый для обрезки с минимальными затратами и сложностью.")
        ccp_alphanon_negative = st.sidebar.slider('ccp_alphanon_negative', 0.0, 1.0)
        params['ccp_alphanon_negative'] = ccp_alphanon_negative    
    #elif clf_name == "Naive Bayes":    
    elif clf_name == "Support Vector Machines":
        st.sidebar.markdown("C: float, default=1.0. Параметр регуляризации.")
        С = st.sidebar.slider('С', 1.0, 2.0)
        params['С'] = С
        st.sidebar.markdown("degree: int, default=3. Степень полиномиальной функции ядра.")
        degree = st.sidebar.slider('degree', 1, 10)
        params['degree'] = degree
        st.sidebar.markdown("cache_size: float, default=200. Размер кеша ядра (в МБ).")
        cache = st.sidebar.slider('cache', 100, 300)
        params['cache'] = cache 
    elif clf_name == "KNN":
        st.sidebar.markdown("n_neighbors: int, default=5. Количество соседей для использования.")
        n_neighbors = st.sidebar.slider('n_neighbors', 1, 20)
        params['n_neighbors'] = n_neighbors
        st.sidebar.markdown("leaf_size: int, default=30. Размер листьев, влияет на скорость построения запроса, на объем памяти, необходимый для хранения дерева.")
        leaf_size = st.sidebar.slider('leaf_size', 10, 100)
        params['leaf_size'] = leaf_size
        st.sidebar.markdown("p: int, default=2. Степенный параметр для метрики Минковского.")
        p = st.sidebar.slider('p', 2, 10)
        params['p'] = p
    elif clf_name == "Logistic Regression":
        st.sidebar.markdown("C: float, default=1.0. Параметр регуляризации.")
        С = st.sidebar.slider('С', 1.0, 2.0)
        params['С'] = С
        st.sidebar.markdown("max_iter: int, default=100. Максимальное количество итераций.")
        max_iter = st.sidebar.slider('max_iter', 50, 200)
        params['max_iter'] = max_iter       
    elif clf_name == "Random Forest":
        st.sidebar.markdown("max_depth: int, default=None. Максимальная глубина дерева.")
        max_depth = st.sidebar.slider('max_depth', 1, 30)
        params['max_depth'] = max_depth
        st.sidebar.markdown("min_samples_split: int or float, default=1. Минимальное количество выборок, необходимое для разделения внутреннего узла.")
        min_samples_split = st.sidebar.slider('min_samples_split', 0.01, 1.0)
        params['min_samples_split'] = min_samples_split
        st.sidebar.markdown("min_samples_leaf: int or float, default=1. Минимальное количество выборок, которое требуется для конечного узла.")
        min_samples_leaf = st.sidebar.slider('min_samples_leaf', 0.01, 0.5)
        params['min_samples_leaf'] = min_samples_leaf
    elif clf_name == "ANN":
        st.sidebar.markdown("epochs: int. Эпоха - это одна итерация в процессе обучения. Так как компьютеру сложно справиться со всем обучением сразу, его делят на эпохи")
        epo = st.sidebar.slider('epochs', 5, 200)
        params['epo'] = epo
        st.sidebar.markdown("batch_size: int. Общее число тренировочных объектов, представленных в одном обучении")
        batch_size = st.sidebar.slider('batch_size', 100, 1000)
        params['batch_size'] = batch_size
   # elif clf_name == "Linear Regression":   
   # elif clf_name == "XGBoost":        
    return params


@st.cache(allow_output_mutation=True)
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


@st.cache(allow_output_mutation=True)
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

def AboutData(data):
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
    
if __name__ == "__main__":
    main()


