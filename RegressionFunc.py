import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
import xgboost as xgb

from sklearn.metrics import r2_score

data = pd.read_csv('DataAnal/диплом/Financial Distress.csv')

data = data.drop(['Company', 'Time'], axis=1)

def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

data = onehot_encode(data, column='x80', prefix='x80')

y = data['Financial Distress'].copy()
X = data.drop('Financial Distress', axis=1).copy()

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=100)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

params = {'learning_rate': 0.001, 'max_depth': 6, 'lambda': 0.01}

boost_model = xgb.train(
    params,
    dtrain,
    num_boost_round=10000,
    evals=[(dval, 'eval')],
    early_stopping_rounds=10,
    verbose_eval=False
)

print("XGB Model R^2 Score:", r2_score(y_test, boost_model.predict(dtest)))