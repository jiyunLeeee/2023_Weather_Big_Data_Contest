# 겨울 모델링 

# Library
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import missingno as msno

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder , RobustScaler
from sklearn.metrics import mean_absolute_error

## CV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn import metrics

## Model
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression
import xgboost
import lightgbm as lgb

## Tunning
import time
import re
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from optuna.integration import XGBoostPruningCallback
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')


train_fin = pd.read_csv('보간수정_train.csv',encoding = 'UTF8')
test_fin = pd.read_csv('보간수정_test.csv',encoding = 'UTF8')

## Train Data
# Label Encoding
encoder = LabelEncoder()
encoder.fit(train_fin['year'])
train_fin["year"] = encoder.transform(train_fin['year']) + 1

# month / day / hour
train_fin['mmddhh'] = train_fin['mmddhh'].astype('str')
train_fin['hour'] = train_fin['mmddhh'].str[-2:]
train_fin['day'] = train_fin['mmddhh'].str[-4:-2]
train_fin['month'] = train_fin['mmddhh'].str[:-4]

train_fin['datetime'] = '000' + train_fin['year'].astype(str) + '-' + train_fin['month'] + '-' + train_fin['day'] + ' ' + train_fin['hour'] + ':00:00'
train_fin.drop(['mmddhh','year'], axis = 1 , inplace = True)

train_fin['month'] = train_fin['month'].astype('int')

# Y 에 Na 존재하는거 제거
train_fin['month'] = train_fin['month'].astype(int)
train_fin.dropna(subset=['ts'], inplace = True)

# Date를 Axis로 지정
train_fin.drop(['hour','day','Unnamed: 0'], axis = 1 , inplace = True)
train_fin = train_fin.reset_index(drop = True).set_index('datetime')

## Test Data
# Label Encoding
encoder = LabelEncoder()
encoder.fit(test_fin['year'])
test_fin["year"] = encoder.transform(test_fin['year']) + 1

encoder = LabelEncoder()
encoder.fit(test_fin['stn'])
test_fin['stn'] = encoder.transform(test_fin['stn']) + 1

# month / day / hour
test_fin['mmddhh'] = test_fin['mmddhh'].astype('str')
test_fin['hour'] = test_fin['mmddhh'].str[-2:]
test_fin['day'] = test_fin['mmddhh'].str[-4:-2]
test_fin['month'] = test_fin['mmddhh'].str[:-4]

test_fin['datetime'] = '000' + test_fin['year'].astype(str) + '-' + test_fin['month'] + '-' + test_fin['day'] + ' ' + test_fin['hour'] + ':00:00'

# Date를 Axis로 지정
test_fin.drop(['Unnamed: 0','mmddhh','year','hour','day','month'], axis = 1 , inplace = True)
test_fin = test_fin.reset_index(drop = True).set_index('datetime')


train_hybrid = train_fin.copy()


def LGBM_params_fold_start(LGBM_params, X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    pred_list = []
    mae_list = []

    for train_index, val_index in tscv.split(X):

        x_train, x_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**LGBM_params)
        model.fit(x_train , y_train ,
                eval_set=[(x_val,y_val)],
                eval_metric = 'mae' , verbose = False , early_stopping_rounds = 100)

        pred = model.predict(x_val)
        result = mean_absolute_error(pred,y_val)
        mae_list.append(result)

    return np.mean(mae_list)

def objectiveLGB(trial: Trial, X, y):
    param_lgb = {
    'learning_rate': trial.suggest_float('learning_rate', 0.15, 0.2),
    'n_estimators': trial.suggest_categorical('n_estimators', [700 , 800 , 900]),
    'max_depth': trial.suggest_categorical('max_depth', [14, 15, 16, 17, 18]),
    'min_child_weight': trial.suggest_categorical('min_child_weight', [1 , 2, 3, 4]),
    'reg_lambda': trial.suggest_float('l2_regularization', 0, 1),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
    'num_leaves': trial.suggest_int('num_leaves', 20, 30,),
    'min_split_gain': trial.suggest_float('min_split_gain', 0, 1),
    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
    'random_state': 112,
    'objective': 'huber',
    'metric': 'mae'
  }
    score = LGBM_params_fold_start(param_lgb, X, y)

    return score

## STN 1
# Data
train_hybrid_stn1 = train_hybrid[train_hybrid.stn == 1].reset_index(drop = True)
train_hybrid_stn1 = pd.get_dummies(data = train_hybrid_stn1, drop_first = True)

# train-test split
target = ['ts'] ; f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F', 'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn1[f_col] , train_hybrid_stn1[target] , test_size = 0.2 , shuffle = False)

# Time series
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)

X = dp.in_sample()
lm1 = LinearRegression(fit_intercept=False)
lm1.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train = lm1.predict(X)
trend_test = lm1.predict(X_)

train_y_delta = train_y - trend_train
test_y_delta = test_y - trend_test

# Optuna tunning for tree model (LGBM)
study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=42))

timeout = 1800 ; start_time = time.time()

while (time.time() - start_time) < timeout:
    study.optimize(lambda trial : objectiveLGB(trial, train_x, train_y_delta), n_trials= 1)
    print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

    if (time.time() - start_time) >= timeout:
        break

lgb_param1 = study.best_trial.params
lgb_best1 = lgb.LGBMRegressor(**lgb_param1)
lgb_best1.fit(train_x, train_y_delta)

lgb_train = lgb_best1.predict(train_x)
lgb_test = lgb_best1.predict(test_x)

# Result
original_train = [i+j for i,j in zip(sum(trend_train.tolist(), []), lgb_train.tolist())]
original_test = [i+j for i,j in zip(sum(trend_test.tolist(), []), lgb_test.tolist())]

mean_absolute_error(original_test,test_y)


## STN 2
# Data
train_hybrid_stn2 = train_hybrid[train_hybrid.stn == 2].reset_index(drop = True)
train_hybrid_stn2 = pd.get_dummies(data = train_hybrid_stn2, drop_first = True)

# train-test split
target = ['ts'] ; f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F', 'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn2[f_col] , train_hybrid_stn2[target] , test_size = 0.2 , shuffle = False)

# Time series
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)

X = dp.in_sample()
lm2 = LinearRegression(fit_intercept=False)
lm2.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train2 = lm2.predict(X)
trend_test2 = lm2.predict(X_)

train_y_delta = train_y - trend_train2
test_y_delta = test_y - trend_test2

# Optuna tunning for tree model (LGBM)
study = optuna.create_study(direction='minimize',sampler = TPESampler(seed=42))
timeout = 1800 ; start_time = time.time()

while (time.time() - start_time) < timeout:
    study.optimize(lambda trial : objectiveLGB(trial, train_x, train_y_delta), n_trials=1)
    print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

    if (time.time() - start_time) >= timeout:
        break

lgb_param2 = study.best_trial.params
lgb_best2 = lgb.LGBMRegressor(**lgb_param2)
lgb_best2.fit(train_x, train_y_delta)

lgb_train2 = lgb_best2.predict(train_x)
lgb_test2 = lgb_best2.predict(test_x)

# Result
original_train2 = [i+j for i,j in zip(sum(trend_train2.tolist(), []), lgb_train2.tolist())]
original_test2 = [i+j for i,j in zip(sum(trend_test2.tolist(), []), lgb_test2.tolist())]

mean_absolute_error(original_test2 , test_y)

## STN 3
# Data
train_hybrid_stn3 = train_hybrid[train_hybrid.stn == 3].reset_index(drop = True)
train_hybrid_stn3 = pd.get_dummies(data = train_hybrid_stn3, drop_first = True)

# train-test split
target = ['ts'] ; f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F', 'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn3[f_col] , train_hybrid_stn3[target] , test_size = 0.2 , shuffle = False)

# Time series
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)

X = dp.in_sample()
lm3 = LinearRegression(fit_intercept=False)
lm3.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train3 = lm3.predict(X)
trend_test3 = lm3.predict(X_)

train_y_delta = train_y - trend_train3
test_y_delta = test_y - trend_test3

# Optuna tunning for tree model (LGBM)
study = optuna.create_study(direction='minimize',sampler = TPESampler(seed=42))
timeout = 1800 ; start_time = time.time()

while (time.time() - start_time) < timeout:
    study.optimize(lambda trial : objectiveLGB(trial, train_x, train_y_delta), n_trials=1)
    print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

    if (time.time() - start_time) >= timeout:
        break

lgb_param3 = study.best_trial.params
lgb_best3 = lgb.LGBMRegressor(**lgb_param3)
lgb_best3.fit(train_x, train_y_delta)

lgb_train3 = lgb_best3.predict(train_x)
lgb_test3 = lgb_best3.predict(test_x)

# Result
original_train3 = [i+j for i,j in zip(sum(trend_train3.tolist(), []), lgb_train3.tolist())]
original_test3 = [i+j for i,j in zip(sum(trend_test3.tolist(), []), lgb_test3.tolist())]

mean_absolute_error(original_test3 , test_y)

## STN 4
# Data
train_hybrid_stn4 = train_hybrid[train_hybrid.stn == 4].reset_index(drop = True)
train_hybrid_stn4 = pd.get_dummies(data = train_hybrid_stn4, drop_first = True)

# train-test split
target = ['ts'] ; f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F', 'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn4[f_col] , train_hybrid_stn4[target] , test_size = 0.2 , shuffle = False)

# Time series
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)

X = dp.in_sample()
lm4 = LinearRegression(fit_intercept=False)
lm4.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train4 = lm4.predict(X)
trend_test4 = lm4.predict(X_)

train_y_delta = train_y - trend_train4
test_y_delta = test_y - trend_test4

# Optuna tunning for tree model (LGBM)
study = optuna.create_study(direction='minimize',sampler = TPESampler(seed=42))
timeout = 1800 ; start_time = time.time()

while (time.time() - start_time) < timeout:
    study.optimize(lambda trial : objectiveLGB(trial, train_x, train_y_delta), n_trials=1)
    print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

    if (time.time() - start_time) >= timeout:
        break

lgb_param4 = study.best_trial.params
lgb_best4 = lgb.LGBMRegressor(**lgb_param4)
lgb_best4.fit(train_x, train_y_delta)

lgb_train4 = lgb_best4.predict(train_x)
lgb_test4 = lgb_best4.predict(test_x)

# Result
original_train4 = [i+j for i,j in zip(sum(trend_train4.tolist(), []), lgb_train4.tolist())]
original_test4 = [i+j for i,j in zip(sum(trend_test4.tolist(), []), lgb_test4.tolist())]

mean_absolute_error(original_test4 , test_y)


## STN 5
# Data
train_hybrid_stn5 = train_hybrid[train_hybrid.stn == 5].reset_index(drop = True)
train_hybrid_stn5 = pd.get_dummies(data = train_hybrid_stn5, drop_first = True)

# train-test split
target = ['ts'] ; f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F', 'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn5[f_col] , train_hybrid_stn5[target] , test_size = 0.2 , shuffle = False)

# Time series
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)

X = dp.in_sample()
lm5 = LinearRegression(fit_intercept=False)
lm5.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train5 = lm5.predict(X)
trend_test5 = lm5.predict(X_)

train_y_delta = train_y - trend_train5
test_y_delta = test_y - trend_test5

# Optuna tunning for tree model (LGBM)
study = optuna.create_study(direction='minimize',sampler = TPESampler(seed=42))
timeout = 1800 ; start_time = time.time()

while (time.time() - start_time) < timeout:
    study.optimize(lambda trial : objectiveLGB(trial, train_x, train_y_delta), n_trials=1)
    print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

    if (time.time() - start_time) >= timeout:
        break

lgb_param5 = study.best_trial.params
lgb_best5 = lgb.LGBMRegressor(**lgb_param5)
lgb_best5.fit(train_x, train_y_delta)

lgb_train5 = lgb_best5.predict(train_x)
lgb_test5 = lgb_best5.predict(test_x)

# Result
original_train5 = [i+j for i,j in zip(sum(trend_train5.tolist(), []), lgb_train5.tolist())]
original_test5 = [i+j for i,j in zip(sum(trend_test5.tolist(), []), lgb_test5.tolist())]

mean_absolute_error(original_test5 , test_y)

## STN 6
# Data
train_hybrid_stn6 = train_hybrid[train_hybrid.stn == 6].reset_index(drop = True)
train_hybrid_stn6 = pd.get_dummies(data = train_hybrid_stn6, drop_first = True)

# train-test split
target = ['ts'] ; f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F', 'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn6[f_col] , train_hybrid_stn6[target] , test_size = 0.2 , shuffle = False)

# Time series
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)

X = dp.in_sample()
lm6 = LinearRegression(fit_intercept=False)
lm6.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train6 = lm6.predict(X)
trend_test6 = lm6.predict(X_)

train_y_delta = train_y - trend_train6
test_y_delta = test_y - trend_test6

# Optuna tunning for tree model (LGBM)
study = optuna.create_study(direction='minimize',sampler = TPESampler(seed=42))
timeout = 1800 ; start_time = time.time()

while (time.time() - start_time) < timeout:
    study.optimize(lambda trial : objectiveLGB(trial, train_x, train_y_delta), n_trials=1)
    print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

    if (time.time() - start_time) >= timeout:
        break

lgb_param6 = study.best_trial.params
lgb_best6 = lgb.LGBMRegressor(**lgb_param6)
lgb_best6.fit(train_x, train_y_delta)

lgb_train6 = lgb_best6.predict(train_x)
lgb_test6 = lgb_best6.predict(test_x)

# Result
original_train6 = [i+j for i,j in zip(sum(trend_train6.tolist(), []), lgb_train6.tolist())]
original_test6 = [i+j for i,j in zip(sum(trend_test6.tolist(), []), lgb_test6.tolist())]

mean_absolute_error(original_test6 , test_y)

## STN 7
# Data
train_hybrid_stn7 = train_hybrid[train_hybrid.stn == 7].reset_index(drop = True)
train_hybrid_stn7 = pd.get_dummies(data = train_hybrid_stn7, drop_first = True)

# train-test split
target = ['ts'] ; f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F', 'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn7[f_col] , train_hybrid_stn7[target] , test_size = 0.2 , shuffle = False)

# Time series
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)

X = dp.in_sample()
lm7 = LinearRegression(fit_intercept=False)
lm7.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train7 = lm7.predict(X)
trend_test7 = lm7.predict(X_)

train_y_delta = train_y - trend_train7
test_y_delta = test_y - trend_test7

# Optuna tunning for tree model (LGBM)
study = optuna.create_study(direction='minimize',sampler = TPESampler(seed=42))
timeout = 1800 ; start_time = time.time()

while (time.time() - start_time) < timeout:
    study.optimize(lambda trial : objectiveLGB(trial, train_x, train_y_delta), n_trials=1)
    print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

    if (time.time() - start_time) >= timeout:
        break

lgb_param7 = study.best_trial.params
lgb_best7 = lgb.LGBMRegressor(**lgb_param7)
lgb_best7.fit(train_x, train_y_delta)

lgb_train7 = lgb_best7.predict(train_x)
lgb_test7 = lgb_best7.predict(test_x)

# Result
original_train7 = [i+j for i,j in zip(sum(trend_train7.tolist(), []), lgb_train7.tolist())]
original_test7 = [i+j for i,j in zip(sum(trend_test7.tolist(), []), lgb_test7.tolist())]

mean_absolute_error(original_test7 , test_y)

## STN 8
# Data
train_hybrid_stn8 = train_hybrid[train_hybrid.stn == 8].reset_index(drop = True)
train_hybrid_stn8 = pd.get_dummies(data = train_hybrid_stn8, drop_first = True)

# train-test split
target = ['ts'] ; f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F', 'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn8[f_col] , train_hybrid_stn8[target] , test_size = 0.2 , shuffle = False)

# Time series
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)

X = dp.in_sample()
lm8 = LinearRegression(fit_intercept=False)
lm8.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train8 = lm8.predict(X)
trend_test8 = lm8.predict(X_)

train_y_delta = train_y - trend_train8
test_y_delta = test_y - trend_test8

# Optuna tunning for tree model (LGBM)
study = optuna.create_study(direction='minimize',sampler = TPESampler(seed=42))
timeout = 1800 ; start_time = time.time()

while (time.time() - start_time) < timeout:
    study.optimize(lambda trial : objectiveLGB(trial, train_x, train_y_delta), n_trials=1)
    print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

    if (time.time() - start_time) >= timeout:
        break

lgb_param8 = study.best_trial.params
lgb_best8 = lgb.LGBMRegressor(**lgb_param8)
lgb_best8.fit(train_x, train_y_delta)

lgb_train8 = lgb_best8.predict(train_x)
lgb_test8 = lgb_best8.predict(test_x)

# Result
original_train8 = [i+j for i,j in zip(sum(trend_train8.tolist(), []), lgb_train8.tolist())]
original_test8 = [i+j for i,j in zip(sum(trend_test8.tolist(), []), lgb_test8.tolist())]

mean_absolute_error(original_test8 , test_y)

## STN 9
# Data
train_hybrid_stn9 = train_hybrid[train_hybrid.stn == 9].reset_index(drop = True)
train_hybrid_stn9 = pd.get_dummies(data = train_hybrid_stn9, drop_first = True)

# train-test split
target = ['ts'] ; f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F', 'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn9[f_col] , train_hybrid_stn9[target] , test_size = 0.2 , shuffle = False)

# Time series
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)

X = dp.in_sample()
lm9 = LinearRegression(fit_intercept=False)
lm9.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max() + test_x.shape[0])
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train9 = lm9.predict(X)
trend_test9 = lm9.predict(X_)

train_y_delta = train_y - trend_train9
test_y_delta = test_y - trend_test9

# Optuna tunning for tree model (LGBM)
study = optuna.create_study(direction='minimize',sampler = TPESampler(seed=42))
timeout = 1800 ; start_time = time.time()

while (time.time() - start_time) < timeout:
    study.optimize(lambda trial : objectiveLGB(trial, train_x, train_y_delta), n_trials=1)
    print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

    if (time.time() - start_time) >= timeout:
        break

lgb_param9 = study.best_trial.params
lgb_best9 = lgb.LGBMRegressor(**lgb_param9)
lgb_best9.fit(train_x, train_y_delta)

lgb_train9 = lgb_best9.predict(train_x)
lgb_test9 = lgb_best9.predict(test_x)

# Result
original_train9 = [i+j for i,j in zip(sum(trend_train9.tolist(), []), lgb_train9.tolist())]
original_test9 = [i+j for i,j in zip(sum(trend_test9.tolist(), []), lgb_test9.tolist())]

mean_absolute_error(original_test9 , test_y)

## STN 10
train_hybrid_stn10 = train_hybrid[train_hybrid.stn == 10].reset_index(drop = True)
train_hybrid_stn10 = pd.get_dummies(data = train_hybrid_stn10, drop_first = True)

target = ['ts'] ; f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F', 'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn10[f_col] , train_hybrid_stn10[target] , test_size = 0.2 , shuffle = False)

dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)

X = dp.in_sample()
lm10 = LinearRegression(fit_intercept=False)
lm10.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train10 = lm10.predict(X)
trend_test10 = lm10.predict(X_)

train_y_delta = train_y - trend_train10
test_y_delta = test_y - trend_test10

study = optuna.create_study(direction='minimize',sampler = TPESampler(seed=42))
timeout = 1800 ; start_time = time.time()

while (time.time() - start_time) < timeout:
    study.optimize(lambda trial : objectiveLGB(trial, train_x, train_y_delta), n_trials=1)
    print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

    if (time.time() - start_time) >= timeout:
        break

lgb_param10 = study.best_trial.params
lgb_best10 = lgb.LGBMRegressor(**lgb_param10)
lgb_best10.fit(train_x, train_y_delta)

lgb_train10 = lgb_best10.predict(train_x)
lgb_test10 = lgb_best10.predict(test_x)

original_train10 = [i+j for i,j in zip(sum(trend_train.tolist(), []), lgb_train10.tolist())]
original_test10 = [i+j for i,j in zip(sum(trend_test.tolist(), []), lgb_test10.tolist())]

mean_absolute_error(original_test10 , test_y)




surface_tp_test_hybrid = test_fin.copy()



# 기상청 설명자료 속 선행지식 활용 (지점 abc)
test_hybrid_stn1 = surface_tp_test_hybrid[surface_tp_test_hybrid.stn == 1].reset_index(drop=True)
test_hybrid_stn2 = surface_tp_test_hybrid[surface_tp_test_hybrid.stn == 2].reset_index(drop=True)
test_hybrid_stn3 = surface_tp_test_hybrid[surface_tp_test_hybrid.stn == 3].reset_index(drop=True)

test_hybrid_stn1 = pd.get_dummies(data=test_hybrid_stn1, drop_first=True)
test_hybrid_stn2 = pd.get_dummies(data=test_hybrid_stn2, drop_first=True)
test_hybrid_stn3 = pd.get_dummies(data=test_hybrid_stn3, drop_first=True)




# dummy encoding을 했을 때 train에는 존재하지만, test에는 없는 열은 0으로 설정
test_hybrid_stn1['ww_X'] = 0
test_hybrid_stn2['ww_X'] = 0
test_hybrid_stn3['ww_X'] = 0
test_hybrid_stn3['ww_H'] = 0


### Time Series
target = ['ts'] ; f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F', 'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']

## STN 1
train_hybrid_stn1 = train_hybrid[train_hybrid.stn == 1].reset_index(drop = True)
train_hybrid_stn1 = pd.get_dummies(data = train_hybrid_stn1, drop_first = True)
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn1[f_col] , train_hybrid_stn1[target] , test_size = 0.2 , shuffle = False)
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)
X1 = dp.in_sample()
lm1 = LinearRegression(fit_intercept=False)

## STN 2
train_hybrid_stn2 = train_hybrid[train_hybrid.stn == 2].reset_index(drop = True)
train_hybrid_stn2 = pd.get_dummies(data = train_hybrid_stn2, drop_first = True)
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn2[f_col] , train_hybrid_stn2[target] , test_size = 0.2 , shuffle = False)
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)
X2 = dp.in_sample()
lm2 = LinearRegression(fit_intercept=False)

## STN 3
train_hybrid_stn3 = train_hybrid[train_hybrid.stn == 3].reset_index(drop = True)
train_hybrid_stn3 = pd.get_dummies(data = train_hybrid_stn3, drop_first = True)
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn3[f_col] , train_hybrid_stn3[target] , test_size = 0.2 , shuffle = False)
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)
X3 = dp.in_sample()
lm3 = LinearRegression(fit_intercept=False)

## STN 4
train_hybrid_stn4 = train_hybrid[train_hybrid.stn == 4].reset_index(drop = True)
train_hybrid_stn4 = pd.get_dummies(data = train_hybrid_stn4, drop_first = True)
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn4[f_col] , train_hybrid_stn4[target] , test_size = 0.2 , shuffle = False)
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)
X4 = dp.in_sample()
lm4 = LinearRegression(fit_intercept=False)

## STN 5
train_hybrid_stn5 = train_hybrid[train_hybrid.stn == 5].reset_index(drop = True)
train_hybrid_stn5 = pd.get_dummies(data = train_hybrid_stn2, drop_first = True)
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn5[f_col] , train_hybrid_stn5[target] , test_size = 0.2 , shuffle = False)
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)
X5 = dp.in_sample()
lm5 = LinearRegression(fit_intercept=False)

## STN 6
train_hybrid_stn6 = train_hybrid[train_hybrid.stn == 6].reset_index(drop = True)
train_hybrid_stn6 = pd.get_dummies(data = train_hybrid_stn6, drop_first = True)
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn6[f_col] , train_hybrid_stn6[target] , test_size = 0.2 , shuffle = False)
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)
X6 = dp.in_sample()
lm6 = LinearRegression(fit_intercept=False)

## STN 7
train_hybrid_stn7 = train_hybrid[train_hybrid.stn == 7].reset_index(drop = True)
train_hybrid_stn7 = pd.get_dummies(data = train_hybrid_stn7, drop_first = True)
train_hybrid_stn7['ww_X'] = 0
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn7[f_col] , train_hybrid_stn7[target] , test_size = 0.2 , shuffle = False)
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)
X7 = dp.in_sample()
lm7 = LinearRegression(fit_intercept=False)

## STN 8
train_hybrid_stn8 = train_hybrid[train_hybrid.stn == 8].reset_index(drop = True)
train_hybrid_stn8 = pd.get_dummies(data = train_hybrid_stn8, drop_first = True)
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn8[f_col] , train_hybrid_stn8[target] , test_size = 0.2 , shuffle = False)
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)
X8 = dp.in_sample()
lm8 = LinearRegression(fit_intercept=False)

## STN 9
train_hybrid_stn9 = train_hybrid[train_hybrid.stn == 9].reset_index(drop = True)
train_hybrid_stn9 = pd.get_dummies(data = train_hybrid_stn9, drop_first = True)
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn9[f_col] , train_hybrid_stn9[target] , test_size = 0.2 , shuffle = False)
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)
X9 = dp.in_sample()
lm9 = LinearRegression(fit_intercept=False)

## STN 10
train_hybrid_stn10 = train_hybrid[train_hybrid.stn == 10].reset_index(drop = True)
train_hybrid_stn10 = pd.get_dummies(data = train_hybrid_stn2, drop_first = True)
train_x, test_x, train_y, test_y = train_test_split(train_hybrid_stn10[f_col] , train_hybrid_stn10[target] , test_size = 0.2 , shuffle = False)
dp = DeterministicProcess(index = train_x.index , constant = True , order = 1 , drop = True)
X10 = dp.in_sample()
lm10 = LinearRegression(fit_intercept=False)


# dp
dp1 = DeterministicProcess(index = train_hybrid_stn1.index, constant = True, order = 1,drop = True)
dp2 = DeterministicProcess(index = train_hybrid_stn2.index, constant = True, order = 1,drop = True)
dp3 = DeterministicProcess(index = train_hybrid_stn3.index, constant = True, order = 1,drop = True)
dp4 = DeterministicProcess(index = train_hybrid_stn4.index, constant = True, order = 1,drop = True)
dp5 = DeterministicProcess(index = train_hybrid_stn5.index, constant = True, order = 1,drop = True)
dp6 = DeterministicProcess(index = train_hybrid_stn6.index, constant = True, order = 1,drop = True)
dp7 = DeterministicProcess(index = train_hybrid_stn7.index, constant = True, order = 1,drop = True)
dp8 = DeterministicProcess(index = train_hybrid_stn8.index, constant = True, order = 1,drop = True)
dp9 = DeterministicProcess(index = train_hybrid_stn9.index, constant = True, order = 1,drop = True)
dp10 = DeterministicProcess(index = train_hybrid_stn10.index, constant = True, order = 1,drop = True)

# X
X1 = dp1.in_sample() ; X2 = dp2.in_sample()
X3 = dp3.in_sample() ; X4 = dp4.in_sample()
X5 = dp5.in_sample() ; X6 = dp6.in_sample()
X7 = dp7.in_sample() ; X8 = dp8.in_sample()
X9 = dp9.in_sample() ; X10 = dp10.in_sample()


lm1.fit(X1, train_hybrid_stn1[target])
lm2.fit(X2, train_hybrid_stn2[target])
lm3.fit(X3, train_hybrid_stn3[target])
lm4.fit(X4, train_hybrid_stn4[target])
lm5.fit(X5, train_hybrid_stn5[target])
lm6.fit(X6, train_hybrid_stn6[target])
lm7.fit(X7, train_hybrid_stn7[target])
lm8.fit(X8, train_hybrid_stn8[target])
lm9.fit(X9, train_hybrid_stn9[target])
lm10.fit(X10, train_hybrid_stn10[target])


# 지점별 train data 추세 추출
trend_stn1 = lm1.predict(X1)
trend_stn2 = lm2.predict(X2)
trend_stn3 = lm3.predict(X3)
trend_stn4 = lm4.predict(X4)
trend_stn5 = lm5.predict(X5)
trend_stn6 = lm6.predict(X6)
trend_stn7 = lm7.predict(X7)
trend_stn8 = lm8.predict(X8)
trend_stn9 = lm9.predict(X9)
trend_stn10 = lm10.predict(X10)


# Best Parameter
lgb_param1 = {'learning_rate': 0.12829594410470951, 'n_estimators': 500, 'max_depth': 16, 'min_child_weight': 2, 'l2_regularization': 0.5336210596747876}
lgb_param2 = {'learning_rate': 0.16422823124652633, 'n_estimators': 800, 'max_depth': 16, 'min_child_weight': 2, 'l2_regularization': 0.48628837398667374,
              'colsample_bytree': 0.5877431614700145, 'num_leaves': 20, 'min_split_gain': 0.6322304426546818, 'bagging_freq': 10}
lgb_param3 = {'learning_rate': 0.18889534485982132, 'n_estimators': 500, 'max_depth': 17, 'min_child_weight': 4, 'l2_regularization': 0.32482046710162793}
lgb_param4 = {'learning_rate': 0.13046182446397234, 'n_estimators': 400, 'max_depth': 16, 'min_child_weight': 4, 'l2_regularization': 0.8668413429347246}
lgb_param5  = {'learning_rate': 0.1444166093465713,'n_estimators': 500,'max_depth': 17,'min_child_weight': 4, 'l2_regularization': 0.14729681542445355}
lgb_param6  = {'learning_rate': 0.16680093590253176 ,'n_estimators': 500, 'max_depth': 16, 'min_child_weight': 3,'l2_regularization': 0.5478551432621794}
lgb_param7  = {'learning_rate': 0.11803343495390418, 'n_estimators': 500, 'max_depth': 15, 'min_child_weight': 4, 'l2_regularization': 0.19592898855184526}
lgb_param8  ={'learning_rate': 0.12829594410470951, 'n_estimators': 500, 'max_depth': 16, 'min_child_weight': 2, 'l2_regularization': 0.5336210596747876}
lgb_param9  = {'learning_rate': 0.12829594410470951, 'n_estimators': 500, 'max_depth': 16, 'min_child_weight': 2, 'l2_regularization': 0.5336210596747876}
lgb_param10 = {'learning_rate': 0.11803433443831565,'n_estimators': 500,'max_depth': 17,'min_child_weight': 4, 'l2_regularization': 0.7771696259733507}

# Fit the model using best Parameter
lgb_best1 = lgb.LGBMRegressor(**lgb_param1)
lgb_best2 = lgb.LGBMRegressor(**lgb_param2)
lgb_best3 = lgb.LGBMRegressor(**lgb_param3)
lgb_best4 = lgb.LGBMRegressor(**lgb_param4)
lgb_best5 = lgb.LGBMRegressor(**lgb_param5)
lgb_best6 = lgb.LGBMRegressor(**lgb_param6)
lgb_best7 = lgb.LGBMRegressor(**lgb_param7)
lgb_best8 = lgb.LGBMRegressor(**lgb_param8)
lgb_best9 = lgb.LGBMRegressor(**lgb_param9)
lgb_best10 = lgb.LGBMRegressor(**lgb_param10)

lgb_best1.fit(train_hybrid_stn1[f_col], [i-j for i,j in zip(train_hybrid_stn1[target].ts.tolist(),  sum(trend_stn1.tolist(), []))])
lgb_best2.fit(train_hybrid_stn2[f_col], [i-j for i,j in zip(train_hybrid_stn2[target].ts.tolist(),  sum(trend_stn2.tolist(), []))])
lgb_best3.fit(train_hybrid_stn3[f_col], [i-j for i,j in zip(train_hybrid_stn3[target].ts.tolist(),  sum(trend_stn3.tolist(), []))])
lgb_best4.fit(train_hybrid_stn4[f_col], [i-j for i,j in zip(train_hybrid_stn4[target].ts.tolist(),  sum(trend_stn4.tolist(), []))])
lgb_best5.fit(train_hybrid_stn5[f_col], [i-j for i,j in zip(train_hybrid_stn5[target].ts.tolist(),  sum(trend_stn5.tolist(), []))])
lgb_best6.fit(train_hybrid_stn6[f_col], [i-j for i,j in zip(train_hybrid_stn6[target].ts.tolist(),  sum(trend_stn6.tolist(), []))])
lgb_best7.fit(train_hybrid_stn7[f_col], [i-j for i,j in zip(train_hybrid_stn7[target].ts.tolist(),  sum(trend_stn7.tolist(), []))])
lgb_best8.fit(train_hybrid_stn8[f_col], [i-j for i,j in zip(train_hybrid_stn8[target].ts.tolist(),  sum(trend_stn8.tolist(), []))])
lgb_best9.fit(train_hybrid_stn9[f_col], [i-j for i,j in zip(train_hybrid_stn9[target].ts.tolist(),  sum(trend_stn9.tolist(), []))])
lgb_best10.fit(train_hybrid_stn10[f_col], [i-j for i,j in zip(train_hybrid_stn10[target].ts.tolist(),  sum(trend_stn10.tolist(), []))])

# a년도의 지면온도를 stn 별로 예측
a_seasonality_lgb_best1 = lgb_best1.predict(test_hybrid_stn1[f_col])
a_seasonality_lgb_best2 = lgb_best2.predict(test_hybrid_stn1[f_col])
a_seasonality_lgb_best3 = lgb_best3.predict(test_hybrid_stn1[f_col])
a_seasonality_lgb_best4 = lgb_best4.predict(test_hybrid_stn1[f_col])
a_seasonality_lgb_best5 = lgb_best5.predict(test_hybrid_stn1[f_col])
a_seasonality_lgb_best6 = lgb_best6.predict(test_hybrid_stn1[f_col])
a_seasonality_lgb_best7 = lgb_best7.predict(test_hybrid_stn1[f_col])
a_seasonality_lgb_best8 = lgb_best8.predict(test_hybrid_stn1[f_col])
a_seasonality_lgb_best9 = lgb_best9.predict(test_hybrid_stn1[f_col])
a_seasonality_lgb_best10 = lgb_best10.predict(test_hybrid_stn1[f_col])

# b년도의 지면온도를 stn 별로 예측
b_seasonality_lgb_best1 = lgb_best1.predict(test_hybrid_stn2[f_col])
b_seasonality_lgb_best2 = lgb_best2.predict(test_hybrid_stn2[f_col])
b_seasonality_lgb_best3 = lgb_best3.predict(test_hybrid_stn2[f_col])
b_seasonality_lgb_best4 = lgb_best4.predict(test_hybrid_stn2[f_col])
b_seasonality_lgb_best5 = lgb_best5.predict(test_hybrid_stn2[f_col])
b_seasonality_lgb_best6 = lgb_best6.predict(test_hybrid_stn2[f_col])
b_seasonality_lgb_best7 = lgb_best7.predict(test_hybrid_stn2[f_col])
b_seasonality_lgb_best8 = lgb_best8.predict(test_hybrid_stn2[f_col])
b_seasonality_lgb_best9 = lgb_best9.predict(test_hybrid_stn2[f_col])
b_seasonality_lgb_best10 = lgb_best10.predict(test_hybrid_stn2[f_col])


# c년도의 지면온도를 stn 별로 예측
c_seasonality_lgb_best1 = lgb_best1.predict(test_hybrid_stn3[f_col])
c_seasonality_lgb_best2 = lgb_best2.predict(test_hybrid_stn3[f_col])
c_seasonality_lgb_best3 = lgb_best3.predict(test_hybrid_stn3[f_col])
c_seasonality_lgb_best4 = lgb_best4.predict(test_hybrid_stn3[f_col])
c_seasonality_lgb_best5 = lgb_best5.predict(test_hybrid_stn3[f_col])
c_seasonality_lgb_best6 = lgb_best6.predict(test_hybrid_stn3[f_col])
c_seasonality_lgb_best7 = lgb_best7.predict(test_hybrid_stn3[f_col])
c_seasonality_lgb_best8 = lgb_best8.predict(test_hybrid_stn3[f_col])
c_seasonality_lgb_best9 = lgb_best9.predict(test_hybrid_stn3[f_col])
c_seasonality_lgb_best10 = lgb_best10.predict(test_hybrid_stn3[f_col])



# 10개 지점 추세 평균값을 사용
avg_trend = [(a+b+c+d+e+f+g+h+i+j)/10 for a,b,c,d,e,f,g,h,i,j in zip(sum(trend_stn1.tolist(), []), sum(trend_stn2.tolist(), []), sum(trend_stn3.tolist(), []),
                                    sum(trend_stn4.tolist(), []), sum(trend_stn5.tolist(), []), sum(trend_stn6.tolist(), []),
                                    sum(trend_stn7.tolist(), []), sum(trend_stn8.tolist(), []), sum(trend_stn9.tolist(), []),
                                    sum(trend_stn10.tolist(), []))]


a_seasonality = [(a+b+c+d+e+f+g+h+i+j)/10 for a,b,c,d,e,f,g,h,i,j in zip(a_seasonality_lgb_best1.tolist(),a_seasonality_lgb_best2.tolist(), a_seasonality_lgb_best3.tolist(),
                                                                         a_seasonality_lgb_best4.tolist(),a_seasonality_lgb_best5.tolist(),a_seasonality_lgb_best6.tolist(),
                                                                         a_seasonality_lgb_best7.tolist(),a_seasonality_lgb_best8.tolist(),a_seasonality_lgb_best9.tolist(),
                                                                         a_seasonality_lgb_best10.tolist())]

a_pred = [i+j for i,j in zip(avg_trend, a_seasonality)]



b_seasonality = [(a+b+c+d+e+f+g+h+i+j)/10 for a,b,c,d,e,f,g,h,i,j in zip(b_seasonality_lgb_best1.tolist(),b_seasonality_lgb_best2.tolist(), b_seasonality_lgb_best3.tolist(),
                                                                         b_seasonality_lgb_best4.tolist(),b_seasonality_lgb_best5.tolist(), b_seasonality_lgb_best6.tolist(),
                                                                         b_seasonality_lgb_best7.tolist(),b_seasonality_lgb_best8.tolist(), b_seasonality_lgb_best9.tolist(),
                                                                         b_seasonality_lgb_best10.tolist())]

b_pred = [i+j for i,j in zip(avg_trend, b_seasonality)]



c_seasonality = [(a+b+c+d+e+f+g+h+i+j)/10 for a,b,c,d,e,f,g,h,i,j in zip(c_seasonality_lgb_best1.tolist(),c_seasonality_lgb_best2.tolist(), c_seasonality_lgb_best3.tolist(),
                                                                         c_seasonality_lgb_best4.tolist(),c_seasonality_lgb_best5.tolist(), c_seasonality_lgb_best6.tolist(),
                                                                         c_seasonality_lgb_best7.tolist(),c_seasonality_lgb_best8.tolist(), c_seasonality_lgb_best9.tolist(),
                                                                         c_seasonality_lgb_best10.tolist())]

c_pred = [i+j for i,j in zip(avg_trend, c_seasonality)]



# 예측한 ts 저장
a_pred.extend(b_pred)
a_pred.extend(c_pred)
test_fin['ts'] = a_pred


# datetime에서 month 분리하기
test_fin.reset_index(inplace = True)
test_fin['month'] = test_fin['datetime'].str.split("-").str.get(1)
test_fin['month'] = test_fin['month'].astype(int)
test_fin.drop(['datetime'],axis = 1 , inplace = True)



# 월 별로 나누기
winter = [11 , 12, 1] ; spring = [2, 3, 4] ; summer = [5, 6, 7]  ; autumn = [8 , 9, 10]

# Test
winter_test = test_fin[test_fin['month'].isin(winter)]
spring_test = test_fin[test_fin['month'].isin(spring)]
summer_test = test_fin[test_fin['month'].isin(summer)]
autumn_test = test_fin[test_fin['month'].isin(autumn)]

