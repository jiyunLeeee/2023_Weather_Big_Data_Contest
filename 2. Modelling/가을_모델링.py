# 가을 모델링 

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
from sklearn.model_selection import KFold

## Model
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor , HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb

## Tunning
import re
import optuna
from optuna.integration import XGBoostPruningCallback
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

train_fin = pd.read_csv('보간수정_train.csv',encoding = 'UTF8')
test_fin = pd.read_csv('보간수정_test.csv',encoding = 'UTF8')

# train data

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

# 불필요 열 제거 및 인덱스 설정
train_fin.drop(['hour','day','Unnamed: 0'], axis = 1 , inplace = True)
train_fin = train_fin.reset_index(drop = True).set_index('datetime')

# test data

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

test_fin.drop(['Unnamed: 0','mmddhh','year','hour','day','month'], axis = 1 , inplace = True)
test_fin = test_fin.reset_index(drop = True).set_index('datetime')


# ## 지점별 적합

# ### 지점 1

surface_tp_train_hybrid = surface_tp_train.copy()

surface_tp_train_hybrid_stn1 = surface_tp_train_hybrid[surface_tp_train_hybrid.stn == 1].reset_index(drop=True)

start_date = "1900-02-01 00:00:00"  # 시작 날짜
end_date = "1905-01-31 23:00:00"  # 종료 날짜
freq = "H"  # 월 단위로 설정

surface_tp_train_hybrid_stn1.index = pd.date_range(start=start_date, end=end_date, freq=freq)

surface_tp_train_hybrid_stn1.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)

surface_tp_train_hybrid_stn1 = pd.get_dummies(data=surface_tp_train_hybrid_stn1, drop_first=True)

# surface_tp_train_hybrid_stn1 = surface_tp_train_hybrid[surface_tp_train_hybrid.stn == 1].reset_index(drop=True)

# # 로버스트 스케일링
# train_X_num = surface_tp_train_hybrid_stn1[['ta', 'td', 'hm', 'ws', 'rn', 're', 'si',
#        'ss', 'sn']]

# # scikit-learn 패키지의 RobustScaler 클래스를 불러옵니다.
# from sklearn.preprocessing import RobustScaler
# # RobustScaler 객체를 생성합니다.
# robustScaler = RobustScaler()

# # fit_transform()을 사용해서 학습과 스케일링을 한 번에 적용합니다.
# X_train_robust = robustScaler.fit_transform(train_X_num)

# surface_tp_train_hybrid_stn1[['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn']] = X_train_robust

# start_date = "1900-02-01 00:00:00"  # 시작 날짜
# end_date = "1905-01-31 23:00:00"  # 종료 날짜
# freq = "H"  # 월 단위로 설정

# surface_tp_train_hybrid_stn1.index = pd.date_range(start=start_date, end=end_date, freq=freq)

# surface_tp_train_hybrid_stn1.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)

# surface_tp_train_hybrid_stn1 = pd.get_dummies(data=surface_tp_train_hybrid_stn1, drop_first=True)

from sklearn.model_selection import train_test_split

target = ['ts']
f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F',
       'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']

train_x, test_x, train_y, test_y = train_test_split(
                                                    surface_tp_train_hybrid_stn1[f_col],
                                                    surface_tp_train_hybrid_stn1[target],
                                                    test_size=0.2,
                                                    shuffle=False,
                                                    )




# from statsmodels.tsa.deterministic import DeterministicProcess
# from sklearn.linear_model import LinearRegression

# dp = DeterministicProcess(
#                           index = train_x.index,
#                           constant = True,
#                           order = 2,
#                           drop = True,
#                           )

# X = dp.in_sample()
# lm = LinearRegression(fit_intercept=False)
# lm.fit(X, train_y)

# test_const = [1 for i in range(test_x.shape[0])]
# test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
# test_trend_squared = [i**2 for i in test_trend]
# X_ = pd.DataFrame([test_const, test_trend, test_trend_squared]).T
# X_.columns = ['const', 'trend', 'trend_squared']
# X_.index = test_x.index

# trend_train = lm.predict(X)
# trend_test = lm.predict(X_)


from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

dp = DeterministicProcess(
                          index = train_x.index,
                          constant = True,
                          order = 1,
                          drop = True,
                          )

X = dp.in_sample()
lm1 = LinearRegression(fit_intercept=False)
lm1.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
# test_trend_squared = [i**2 for i in test_trend]
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train = lm1.predict(X)
trend_test = lm1.predict(X_)

import xgboost

train_y_delta = train_y - trend_train
test_y_delta = test_y - trend_test

# xgb= xgboost.XGBRegressor()
# xgb.fit(train_x, train_y_delta)

# xgb_train = xgb.predict(train_x)
# xgb_test = xgb.predict(test_x)




# model
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

# misc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.metrics import mean_absolute_error



def XGB_params_fold_start(XGB_params, X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    pred_list = []
    mae_list = []

    for train_index, val_index in tscv.split(X):

        x_train, x_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]

        model = xgboost.XGBRegressor(**XGB_params)

        model.fit(x_train, y_train,
                eval_set=[(x_val,y_val)],
                eval_metric='mae', verbose=False, early_stopping_rounds=100)

        pred = model.predict(x_val)
        result = mean_absolute_error(pred,y_val)

        mae_list.append(result)

    return np.mean(mae_list)



def objectiveXGB(trial: Trial, X, y):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : 1127,
        'objective' : 'reg:absoluteerror'
    }

    score = XGB_params_fold_start(param, X, y)

    return score



study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=42))

study.optimize(lambda trial : objectiveXGB(trial, train_x, train_y_delta), n_trials=30)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))



xgb_best1 = xgboost.XGBRegressor(**study.best_trial.params, **{'tree_method' : 'gpu_hist', 'predictor' : 'gpu_predictor'})
xgb_best1.fit(train_x, train_y_delta)

xgb_train = xgb_best1.predict(train_x)
xgb_test = xgb_best1.predict(test_x)

plt.plot(surface_tp_train_hybrid_stn1.index, surface_tp_train_hybrid_stn1.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, xgb_train, label="train")

plt.plot(test_x.index, xgb_test, label="train")



original_train = [i+j for i,j in zip(sum(trend_train.tolist(), []), xgb_train.tolist())]

original_test = [i+j for i,j in zip(sum(trend_test.tolist(), []), xgb_test.tolist())]

plt.plot(surface_tp_train_hybrid_stn1.index, surface_tp_train_hybrid_stn1.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, original_train, label="train")

plt.plot(test_x.index, original_test, label="train")


# ### 지점2


surface_tp_train_hybrid = surface_tp_train.copy()

surface_tp_train_hybrid_stn2 = surface_tp_train_hybrid[surface_tp_train_hybrid.stn == 2].reset_index(drop=True)

start_date = "1900-02-01 00:00:00"  # 시작 날짜
end_date = "1905-01-31 23:00:00"  # 종료 날짜
freq = "H"  # 월 단위로 설정

surface_tp_train_hybrid_stn2.index = pd.date_range(start=start_date, end=end_date, freq=freq)

surface_tp_train_hybrid_stn2.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)

surface_tp_train_hybrid_stn2 = pd.get_dummies(data=surface_tp_train_hybrid_stn2, drop_first=True)




# surface_tp_train_hybrid_stn2 = surface_tp_train_hybrid[surface_tp_train_hybrid.stn == 2].reset_index(drop=True)

# # 로버스트 스케일링
# train_X_num = surface_tp_train_hybrid_stn2[['ta', 'td', 'hm', 'ws', 'rn', 're', 'si',
#        'ss', 'sn']]

# # scikit-learn 패키지의 RobustScaler 클래스를 불러옵니다.
# from sklearn.preprocessing import RobustScaler
# # RobustScaler 객체를 생성합니다.
# robustScaler = RobustScaler()

# # fit_transform()을 사용해서 학습과 스케일링을 한 번에 적용합니다.
# X_train_robust = robustScaler.fit_transform(train_X_num)

# surface_tp_train_hybrid_stn2[['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn']] = X_train_robust

# start_date = "1900-02-01 00:00:00"  # 시작 날짜
# end_date = "1905-01-31 23:00:00"  # 종료 날짜
# freq = "H"  # 월 단위로 설정

# surface_tp_train_hybrid_stn2.index = pd.date_range(start=start_date, end=end_date, freq=freq)

# surface_tp_train_hybrid_stn2.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)

# surface_tp_train_hybrid_stn2 = pd.get_dummies(data=surface_tp_train_hybrid_stn2, drop_first=True)




from sklearn.model_selection import train_test_split

target = ['ts']
f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F',
       'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']

train_x, test_x, train_y, test_y = train_test_split(
                                                    surface_tp_train_hybrid_stn2[f_col],
                                                    surface_tp_train_hybrid_stn2[target],
                                                    test_size=0.2,
                                                    shuffle=False,
                                                    )




# from statsmodels.tsa.deterministic import DeterministicProcess
# from sklearn.linear_model import LinearRegression

# dp = DeterministicProcess(
#                           index = train_x.index,
#                           constant = True,
#                           order = 2,
#                           drop = True,
#                           )

# X = dp.in_sample()
# lm = LinearRegression(fit_intercept=False)
# lm.fit(X, train_y)

# test_const = [1 for i in range(test_x.shape[0])]
# test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
# test_trend_squared = [i**2 for i in test_trend]
# X_ = pd.DataFrame([test_const, test_trend, test_trend_squared]).T
# X_.columns = ['const', 'trend', 'trend_squared']
# X_.index = test_x.index

# trend_train = lm.predict(X)
# trend_test = lm.predict(X_)



from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

dp = DeterministicProcess(
                          index = train_x.index,
                          constant = True,
                          order = 1,
                          drop = True,
                          )

X = dp.in_sample()
lm2 = LinearRegression(fit_intercept=False)
lm2.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
# test_trend_squared = [i**2 for i in test_trend]
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train = lm2.predict(X)
trend_test = lm2.predict(X_)



plt.plot(surface_tp_train_hybrid_stn2.index, surface_tp_train_hybrid_stn2.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, trend_train, label="train")

plt.plot(test_x.index, trend_test, label="train")



import xgboost

train_y_delta = train_y - trend_train
test_y_delta = test_y - trend_test

# xgb= xgboost.XGBRegressor()
# xgb.fit(train_x, train_y_delta)

# xgb_train = xgb.predict(train_x)
# xgb_test = xgb.predict(test_x)



# model
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

# misc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.metrics import mean_absolute_error



def XGB_params_fold_start(XGB_params, X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    pred_list = []
    mae_list = []

    for train_index, val_index in tscv.split(X):

        x_train, x_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]

        model = xgboost.XGBRegressor(**XGB_params)

        model.fit(x_train, y_train,
                eval_set=[(x_val,y_val)],
                eval_metric='mae', verbose=False, early_stopping_rounds=100)

        pred = model.predict(x_val)
        result = mean_absolute_error(pred,y_val)

        mae_list.append(result)

    return np.mean(mae_list)




def objectiveXGB(trial: Trial, X, y):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : 1127,
        'objective' : 'reg:absoluteerror'
    }

    score = XGB_params_fold_start(param, X, y)

    return score




study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=42))

study.optimize(lambda trial : objectiveXGB(trial, train_x, train_y_delta), n_trials=30)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))




xgb_best2 = xgboost.XGBRegressor(**study.best_trial.params, **{'tree_method' : 'gpu_hist', 'predictor' : 'gpu_predictor'})
xgb_best2.fit(train_x, train_y_delta)

xgb_train = xgb_best2.predict(train_x)
xgb_test = xgb_best2.predict(test_x)




plt.plot(surface_tp_train_hybrid_stn2.index, surface_tp_train_hybrid_stn2.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, xgb_train, label="train")

plt.plot(test_x.index, xgb_test, label="train")





original_train = [i+j for i,j in zip(sum(trend_train.tolist(), []), xgb_train.tolist())]

original_test = [i+j for i,j in zip(sum(trend_test.tolist(), []), xgb_test.tolist())]





plt.plot(surface_tp_train_hybrid_stn2.index, surface_tp_train_hybrid_stn2.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, original_train, label="train")

plt.plot(test_x.index, original_test, label="train")


# ### 지점3 




surface_tp_train_hybrid = surface_tp_train.copy()

surface_tp_train_hybrid_stn3 = surface_tp_train_hybrid[surface_tp_train_hybrid.stn == 3].reset_index(drop=True)

start_date = "1900-02-01 00:00:00"  # 시작 날짜
end_date = "1905-01-31 23:00:00"  # 종료 날짜
freq = "H"  # 월 단위로 설정

surface_tp_train_hybrid_stn3.index = pd.date_range(start=start_date, end=end_date, freq=freq)

surface_tp_train_hybrid_stn3.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)

surface_tp_train_hybrid_stn3 = pd.get_dummies(data=surface_tp_train_hybrid_stn3, drop_first=True)

from sklearn.model_selection import train_test_split

target = ['ts']
f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F',
       'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']

train_x, test_x, train_y, test_y = train_test_split(
                                                    surface_tp_train_hybrid_stn3[f_col],
                                                    surface_tp_train_hybrid_stn3[target],
                                                    test_size=0.2,
                                                    shuffle=False,
                                                    )





from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

dp = DeterministicProcess(
                          index = train_x.index,
                          constant = True,
                          order = 1,
                          drop = True,
                          )

X = dp.in_sample()
lm3 = LinearRegression(fit_intercept=False)
lm3.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
# test_trend_squared = [i**2 for i in test_trend]
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train = lm3.predict(X)
trend_test = lm3.predict(X_)





plt.plot(surface_tp_train_hybrid_stn3.index, surface_tp_train_hybrid_stn3.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, trend_train, label="train")

plt.plot(test_x.index, trend_test, label="train")





import xgboost

train_y_delta = train_y - trend_train
test_y_delta = test_y - trend_test

# xgb= xgboost.XGBRegressor()
# xgb.fit(train_x, train_y_delta)

# xgb_train = xgb.predict(train_x)
# xgb_test = xgb.predict(test_x)





# model
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

# misc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.metrics import mean_absolute_error





def XGB_params_fold_start(XGB_params, X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    pred_list = []
    mae_list = []

    for train_index, val_index in tscv.split(X):

        x_train, x_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]

        model = xgboost.XGBRegressor(**XGB_params)

        model.fit(x_train, y_train,
                eval_set=[(x_val,y_val)],
                eval_metric='mae', verbose=False, early_stopping_rounds=100)

        pred = model.predict(x_val)
        result = mean_absolute_error(pred,y_val)

        mae_list.append(result)

    return np.mean(mae_list)




def objectiveXGB(trial: Trial, X, y):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : 1127,
        'objective' : 'reg:absoluteerror'
    }

    score = XGB_params_fold_start(param, X, y)

    return score





study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=42))

study.optimize(lambda trial : objectiveXGB(trial, train_x, train_y_delta), n_trials=30)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))





xgb_best3 = xgboost.XGBRegressor(**study.best_trial.params, **{'tree_method' : 'gpu_hist', 'predictor' : 'gpu_predictor'})
xgb_best3.fit(train_x, train_y_delta)

xgb_train = xgb_best3.predict(train_x)
xgb_test = xgb_best3.predict(test_x)





plt.plot(surface_tp_train_hybrid_stn3.index, surface_tp_train_hybrid_stn3.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, xgb_train, label="train")

plt.plot(test_x.index, xgb_test, label="train")





original_train = [i+j for i,j in zip(sum(trend_train.tolist(), []), xgb_train.tolist())]

original_test = [i+j for i,j in zip(sum(trend_test.tolist(), []), xgb_test.tolist())]





plt.plot(surface_tp_train_hybrid_stn3.index, surface_tp_train_hybrid_stn3.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, original_train, label="train")

plt.plot(test_x.index, original_test, label="train")


# ### 지점3




surface_tp_train_hybrid = surface_tp_train.copy()

surface_tp_train_hybrid_stn4 = surface_tp_train_hybrid[surface_tp_train_hybrid.stn == 4].reset_index(drop=True)

start_date = "1900-02-01 00:00:00"  # 시작 날짜
end_date = "1905-01-31 23:00:00"  # 종료 날짜
freq = "H"  # 월 단위로 설정

surface_tp_train_hybrid_stn4.index = pd.date_range(start=start_date, end=end_date, freq=freq)

surface_tp_train_hybrid_stn4.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)

surface_tp_train_hybrid_stn4 = pd.get_dummies(data=surface_tp_train_hybrid_stn4, drop_first=True)





from sklearn.model_selection import train_test_split

target = ['ts']
f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F',
       'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']

train_x, test_x, train_y, test_y = train_test_split(
                                                    surface_tp_train_hybrid_stn4[f_col],
                                                    surface_tp_train_hybrid_stn4[target],
                                                    test_size=0.2,
                                                    shuffle=False,
                                                    )





from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

dp = DeterministicProcess(
                          index = train_x.index,
                          constant = True,
                          order = 1,
                          drop = True,
                          )

X = dp.in_sample()
lm4 = LinearRegression(fit_intercept=False)
lm4.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
# test_trend_squared = [i**2 for i in test_trend]
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train = lm4.predict(X)
trend_test = lm4.predict(X_)





plt.plot(surface_tp_train_hybrid_stn4.index, surface_tp_train_hybrid_stn4.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, trend_train, label="train")

plt.plot(test_x.index, trend_test, label="train")





import xgboost

train_y_delta = train_y - trend_train
test_y_delta = test_y - trend_test

# xgb= xgboost.XGBRegressor()
# xgb.fit(train_x, train_y_delta)

# xgb_train = xgb.predict(train_x)
# xgb_test = xgb.predict(test_x)





# model
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

# misc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.metrics import mean_absolute_error





def XGB_params_fold_start(XGB_params, X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    pred_list = []
    mae_list = []

    for train_index, val_index in tscv.split(X):

        x_train, x_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]

        model = xgboost.XGBRegressor(**XGB_params)

        model.fit(x_train, y_train,
                eval_set=[(x_val,y_val)],
                eval_metric='mae', verbose=False, early_stopping_rounds=100)

        pred = model.predict(x_val)
        result = mean_absolute_error(pred,y_val)

        mae_list.append(result)

    return np.mean(mae_list)

def objectiveXGB(trial: Trial, X, y):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : 1127,
        'objective' : 'reg:absoluteerror'
    }

    score = XGB_params_fold_start(param, X, y)

    return score





study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=42))

study.optimize(lambda trial : objectiveXGB(trial, train_x, train_y_delta), n_trials=30)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))





xgb_best4 = xgboost.XGBRegressor(**study.best_trial.params, **{'tree_method' : 'gpu_hist', 'predictor' : 'gpu_predictor'})
xgb_best4.fit(train_x, train_y_delta)

xgb_train = xgb_best4.predict(train_x)
xgb_test = xgb_best4.predict(test_x)





plt.plot(surface_tp_train_hybrid_stn4.index, surface_tp_train_hybrid_stn4.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, xgb_train, label="train")

plt.plot(test_x.index, xgb_test, label="train")





original_train = [i+j for i,j in zip(sum(trend_train.tolist(), []), xgb_train.tolist())]

original_test = [i+j for i,j in zip(sum(trend_test.tolist(), []), xgb_test.tolist())]





plt.plot(surface_tp_train_hybrid_stn4.index, surface_tp_train_hybrid_stn4.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, original_train, label="train")

plt.plot(test_x.index, original_test, label="train")


# ### 지점5




surface_tp_train_hybrid = surface_tp_train.copy()

surface_tp_train_hybrid_stn5 = surface_tp_train_hybrid[surface_tp_train_hybrid.stn == 5].reset_index(drop=True)

start_date = "1900-02-01 00:00:00"  # 시작 날짜
end_date = "1905-01-31 23:00:00"  # 종료 날짜
freq = "H"  # 월 단위로 설정

surface_tp_train_hybrid_stn5.index = pd.date_range(start=start_date, end=end_date, freq=freq)

surface_tp_train_hybrid_stn5.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)

surface_tp_train_hybrid_stn5 = pd.get_dummies(data=surface_tp_train_hybrid_stn5, drop_first=True)





from sklearn.model_selection import train_test_split

target = ['ts']
f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F',
       'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']

train_x, test_x, train_y, test_y = train_test_split(
                                                    surface_tp_train_hybrid_stn5[f_col],
                                                    surface_tp_train_hybrid_stn5[target],
                                                    test_size=0.2,
                                                    shuffle=False,
                                                    )





from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

dp = DeterministicProcess(
                          index = train_x.index,
                          constant = True,
                          order = 1,
                          drop = True,
                          )

X = dp.in_sample()
lm5 = LinearRegression(fit_intercept=False)
lm5.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
# test_trend_squared = [i**2 for i in test_trend]
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train = lm5.predict(X)
trend_test = lm5.predict(X_)

plt.plot(surface_tp_train_hybrid_stn5.index, surface_tp_train_hybrid_stn5.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, trend_train, label="train")

plt.plot(test_x.index, trend_test, label="train")





import xgboost

train_y_delta = train_y - trend_train
test_y_delta = test_y - trend_test

# xgb= xgboost.XGBRegressor()
# xgb.fit(train_x, train_y_delta)

# xgb_train = xgb.predict(train_x)
# xgb_test = xgb.predict(test_x)





# model
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

# misc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.metrics import mean_absolute_error





def XGB_params_fold_start(XGB_params, X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    pred_list = []
    mae_list = []

    for train_index, val_index in tscv.split(X):

        x_train, x_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]

        model = xgboost.XGBRegressor(**XGB_params)

        model.fit(x_train, y_train,
                eval_set=[(x_val,y_val)],
                eval_metric='mae', verbose=False, early_stopping_rounds=100)

        pred = model.predict(x_val)
        result = mean_absolute_error(pred,y_val)

        mae_list.append(result)

    return np.mean(mae_list)

def objectiveXGB(trial: Trial, X, y):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : 1127,
        'objective' : 'reg:absoluteerror'
    }

    score = XGB_params_fold_start(param, X, y)

    return score

study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=42))

study.optimize(lambda trial : objectiveXGB(trial, train_x, train_y_delta), n_trials=30)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))





xgb_best5 = xgboost.XGBRegressor(**study.best_trial.params, **{'tree_method' : 'gpu_hist', 'predictor' : 'gpu_predictor'})
xgb_best5.fit(train_x, train_y_delta)

xgb_train = xgb_best5.predict(train_x)
xgb_test = xgb_best5.predict(test_x)





plt.plot(surface_tp_train_hybrid_stn5.index, surface_tp_train_hybrid_stn5.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, xgb_train, label="train")

plt.plot(test_x.index, xgb_test, label="train")




original_train = [i+j for i,j in zip(sum(trend_train.tolist(), []), xgb_train.tolist())]

original_test = [i+j for i,j in zip(sum(trend_test.tolist(), []), xgb_test.tolist())]

plt.plot(surface_tp_train_hybrid_stn5.index, surface_tp_train_hybrid_stn5.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, original_train, label="train")

plt.plot(test_x.index, original_test, label="train")


# ### 지점6




surface_tp_train_hybrid = surface_tp_train.copy()

surface_tp_train_hybrid_stn6 = surface_tp_train_hybrid[surface_tp_train_hybrid.stn == 6].reset_index(drop=True)

start_date = "1900-02-01 00:00:00"  # 시작 날짜
end_date = "1905-01-31 23:00:00"  # 종료 날짜
freq = "H"  # 월 단위로 설정

surface_tp_train_hybrid_stn6.index = pd.date_range(start=start_date, end=end_date, freq=freq)

surface_tp_train_hybrid_stn6.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)

surface_tp_train_hybrid_stn6 = pd.get_dummies(data=surface_tp_train_hybrid_stn6, drop_first=True)

from sklearn.model_selection import train_test_split

target = ['ts']
f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F',
       'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']

train_x, test_x, train_y, test_y = train_test_split(
                                                    surface_tp_train_hybrid_stn6[f_col],
                                                    surface_tp_train_hybrid_stn6[target],
                                                    test_size=0.2,
                                                    shuffle=False,
                                                    )

from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

dp = DeterministicProcess(
                          index = train_x.index,
                          constant = True,
                          order = 1,
                          drop = True,
                          )

X = dp.in_sample()
lm6 = LinearRegression(fit_intercept=False)
lm6.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
# test_trend_squared = [i**2 for i in test_trend]
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train = lm6.predict(X)
trend_test = lm6.predict(X_)





import xgboost

train_y_delta = train_y - trend_train
test_y_delta = test_y - trend_test

# xgb= xgboost.XGBRegressor()
# xgb.fit(train_x, train_y_delta)

# xgb_train = xgb.predict(train_x)
# xgb_test = xgb.predict(test_x)





# model
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

# misc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.metrics import mean_absolute_error





def XGB_params_fold_start(XGB_params, X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    pred_list = []
    mae_list = []

    for train_index, val_index in tscv.split(X):

        x_train, x_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]

        model = xgboost.XGBRegressor(**XGB_params)

        model.fit(x_train, y_train,
                eval_set=[(x_val,y_val)],
                eval_metric='mae', verbose=False, early_stopping_rounds=100)

        pred = model.predict(x_val)
        result = mean_absolute_error(pred,y_val)

        mae_list.append(result)

    return np.mean(mae_list)




def objectiveXGB(trial: Trial, X, y):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : 1127,
        'objective' : 'reg:absoluteerror'
    }

    score = XGB_params_fold_start(param, X, y)

    return score





study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=42))

study.optimize(lambda trial : objectiveXGB(trial, train_x, train_y_delta), n_trials=30)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))





xgb_best6 = xgboost.XGBRegressor(**study.best_trial.params, **{'tree_method' : 'gpu_hist', 'predictor' : 'gpu_predictor'})
xgb_best6.fit(train_x, train_y_delta)

xgb_train = xgb_best6.predict(train_x)
xgb_test = xgb_best6.predict(test_x)





original_train = [i+j for i,j in zip(sum(trend_train.tolist(), []), xgb_train.tolist())]

original_test = [i+j for i,j in zip(sum(trend_test.tolist(), []), xgb_test.tolist())]


# ### 지점 7 




surface_tp_train_hybrid = surface_tp_train.copy()

surface_tp_train_hybrid_stn7 = surface_tp_train_hybrid[surface_tp_train_hybrid.stn == 7].reset_index(drop=True)

start_date = "1900-02-01 00:00:00"  # 시작 날짜
end_date = "1905-01-31 23:00:00"  # 종료 날짜
freq = "H"  # 월 단위로 설정

surface_tp_train_hybrid_stn7.index = pd.date_range(start=start_date, end=end_date, freq=freq)

surface_tp_train_hybrid_stn7.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)

surface_tp_train_hybrid_stn7 = pd.get_dummies(data=surface_tp_train_hybrid_stn7, drop_first=True)

from sklearn.model_selection import train_test_split

target = ['ts']
f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F',
       'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']

train_x, test_x, train_y, test_y = train_test_split(
                                                    surface_tp_train_hybrid_stn7[f_col],
                                                    surface_tp_train_hybrid_stn7[target],
                                                    test_size=0.2,
                                                    shuffle=False,
                                                    )





from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

dp = DeterministicProcess(
                          index = train_x.index,
                          constant = True,
                          order = 1,
                          drop = True,
                          )

X = dp.in_sample()
lm7 = LinearRegression(fit_intercept=False)
lm7.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
# test_trend_squared = [i**2 for i in test_trend]
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train = lm7.predict(X)
trend_test = lm7.predict(X_)





import xgboost

train_y_delta = train_y - trend_train
test_y_delta = test_y - trend_test

# xgb= xgboost.XGBRegressor()
# xgb.fit(train_x, train_y_delta)

# xgb_train = xgb.predict(train_x)
# xgb_test = xgb.predict(test_x)





def XGB_params_fold_start(XGB_params, X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    pred_list = []
    mae_list = []

    for train_index, val_index in tscv.split(X):

        x_train, x_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]

        model = xgboost.XGBRegressor(**XGB_params)

        model.fit(x_train, y_train,
                eval_set=[(x_val,y_val)],
                eval_metric='mae', verbose=False, early_stopping_rounds=100)

        pred = model.predict(x_val)
        result = mean_absolute_error(pred,y_val)

        mae_list.append(result)

    return np.mean(mae_list)





def objectiveXGB(trial: Trial, X, y):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : 1127,
        'objective' : 'reg:absoluteerror'
    }

    score = XGB_params_fold_start(param, X, y)

    return score

study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=42))

study.optimize(lambda trial : objectiveXGB(trial, train_x, train_y_delta), n_trials=30)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))





xgb_best7 = xgboost.XGBRegressor(**study.best_trial.params, **{'tree_method' : 'gpu_hist', 'predictor' : 'gpu_predictor'})
xgb_best7.fit(train_x, train_y_delta)

xgb_train = xgb_best7.predict(train_x)
xgb_test = xgb_best7.predict(test_x)

plt.plot(surface_tp_train_hybrid_stn7.index, surface_tp_train_hybrid_stn7.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, xgb_train, label="train")

plt.plot(test_x.index, xgb_test, label="train")

original_train = [i+j for i,j in zip(sum(trend_train.tolist(), []), xgb_train.tolist())]

original_test = [i+j for i,j in zip(sum(trend_test.tolist(), []), xgb_test.tolist())]

plt.plot(surface_tp_train_hybrid_stn7.index, surface_tp_train_hybrid_stn7.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, original_train, label="train")

plt.plot(test_x.index, original_test, label="train")


# ### 지점 8 




surface_tp_train_hybrid = surface_tp_train.copy()

surface_tp_train_hybrid_stn8 = surface_tp_train_hybrid[surface_tp_train_hybrid.stn == 8].reset_index(drop=True)

start_date = "1900-02-01 00:00:00"  # 시작 날짜
end_date = "1905-01-31 23:00:00"  # 종료 날짜
freq = "H"  # 월 단위로 설정

surface_tp_train_hybrid_stn8.index = pd.date_range(start=start_date, end=end_date, freq=freq)

surface_tp_train_hybrid_stn8.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)

surface_tp_train_hybrid_stn8 = pd.get_dummies(data=surface_tp_train_hybrid_stn8, drop_first=True)

from sklearn.model_selection import train_test_split

target = ['ts']
f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F',
       'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']

train_x, test_x, train_y, test_y = train_test_split(
                                                    surface_tp_train_hybrid_stn8[f_col],
                                                    surface_tp_train_hybrid_stn8[target],
                                                    test_size=0.2,
                                                    shuffle=False,
                                                    )





from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

dp = DeterministicProcess(
                          index = train_x.index,
                          constant = True,
                          order = 1,
                          drop = True,
                          )

X = dp.in_sample()
lm8 = LinearRegression(fit_intercept=False)
lm8.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
# test_trend_squared = [i**2 for i in test_trend]
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train = lm8.predict(X)
trend_test = lm8.predict(X_)




plt.plot(surface_tp_train_hybrid_stn8.index, surface_tp_train_hybrid_stn8.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, trend_train, label="train")

plt.plot(test_x.index, trend_test, label="train")





import xgboost

train_y_delta = train_y - trend_train
test_y_delta = test_y - trend_test

# xgb= xgboost.XGBRegressor()
# xgb.fit(train_x, train_y_delta)

# xgb_train = xgb.predict(train_x)
# xgb_test = xgb.predict(test_x)





# model
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

# misc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.metrics import mean_absolute_error





def XGB_params_fold_start(XGB_params, X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    pred_list = []
    mae_list = []

    for train_index, val_index in tscv.split(X):

        x_train, x_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]

        model = xgboost.XGBRegressor(**XGB_params)

        model.fit(x_train, y_train,
                eval_set=[(x_val,y_val)],
                eval_metric='mae', verbose=False, early_stopping_rounds=100)

        pred = model.predict(x_val)
        result = mean_absolute_error(pred,y_val)

        mae_list.append(result)

    return np.mean(mae_list)




def objectiveXGB(trial: Trial, X, y):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : 1127,
        'objective' : 'reg:absoluteerror'
    }

    score = XGB_params_fold_start(param, X, y)

    return score





study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=42))

study.optimize(lambda trial : objectiveXGB(trial, train_x, train_y_delta), n_trials=30)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))





xgb_best8 = xgboost.XGBRegressor(**study.best_trial.params, **{'tree_method' : 'gpu_hist', 'predictor' : 'gpu_predictor'})
xgb_best8.fit(train_x, train_y_delta)

xgb_train = xgb_best8.predict(train_x)
xgb_test = xgb_best8.predict(test_x)





original_train = [i+j for i,j in zip(sum(trend_train.tolist(), []), xgb_train.tolist())]

original_test = [i+j for i,j in zip(sum(trend_test.tolist(), []), xgb_test.tolist())]




surface_tp_train_hybrid = surface_tp_train.copy()

surface_tp_train_hybrid_stn9 = surface_tp_train_hybrid[surface_tp_train_hybrid.stn == 9].reset_index(drop=True)

start_date = "1900-02-01 00:00:00"  # 시작 날짜
end_date = "1905-01-31 23:00:00"  # 종료 날짜
freq = "H"  # 월 단위로 설정

surface_tp_train_hybrid_stn9.index = pd.date_range(start=start_date, end=end_date, freq=freq)

surface_tp_train_hybrid_stn9.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)

surface_tp_train_hybrid_stn9 = pd.get_dummies(data=surface_tp_train_hybrid_stn9, drop_first=True)

from sklearn.model_selection import train_test_split

target = ['ts']
f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F',
       'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']

train_x, test_x, train_y, test_y = train_test_split(
                                                    surface_tp_train_hybrid_stn9[f_col],
                                                    surface_tp_train_hybrid_stn9[target],
                                                    test_size=0.2,
                                                    shuffle=False,
                                                    )

from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

dp = DeterministicProcess(
                          index = train_x.index,
                          constant = True,
                          order = 1,
                          drop = True,
                          )

X = dp.in_sample()
lm9 = LinearRegression(fit_intercept=False)
lm9.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
# test_trend_squared = [i**2 for i in test_trend]
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train = lm9.predict(X)
trend_test = lm9.predict(X_)

import xgboost

train_y_delta = train_y - trend_train
test_y_delta = test_y - trend_test

# xgb= xgboost.XGBRegressor()
# xgb.fit(train_x, train_y_delta)

# xgb_train = xgb.predict(train_x)
# xgb_test = xgb.predict(test_x)

# model
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

# misc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.metrics import mean_absolute_error

def XGB_params_fold_start(XGB_params, X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    pred_list = []
    mae_list = []

    for train_index, val_index in tscv.split(X):

        x_train, x_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]

        model = xgboost.XGBRegressor(**XGB_params)

        model.fit(x_train, y_train,
                eval_set=[(x_val,y_val)],
                eval_metric='mae', verbose=False, early_stopping_rounds=100)

        pred = model.predict(x_val)
        result = mean_absolute_error(pred,y_val)

        mae_list.append(result)

    return np.mean(mae_list)

def objectiveXGB(trial: Trial, X, y):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : 1127,
        'objective' : 'reg:absoluteerror'
    }

    score = XGB_params_fold_start(param, X, y)

    return score

study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=42))

study.optimize(lambda trial : objectiveXGB(trial, train_x, train_y_delta), n_trials=30)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

xgb_best9 = xgboost.XGBRegressor(**study.best_trial.params, **{'tree_method' : 'gpu_hist', 'predictor' : 'gpu_predictor'})
xgb_best9.fit(train_x, train_y_delta)

xgb_train = xgb_best9.predict(train_x)
xgb_test = xgb_best9.predict(test_x)

original_train = [i+j for i,j in zip(sum(trend_train.tolist(), []), xgb_train.tolist())]

original_test = [i+j for i,j in zip(sum(trend_test.tolist(), []), xgb_test.tolist())]

plt.plot(surface_tp_train_hybrid_stn9.index, surface_tp_train_hybrid_stn9.loc[:, 'ts'], label="원본")

plt.plot(train_x.index, original_train, label="train")

plt.plot(test_x.index, original_test, label="train")


# ### 지점 10 




surface_tp_train_hybrid = surface_tp_train.copy()

surface_tp_train_hybrid_stn10 = surface_tp_train_hybrid[surface_tp_train_hybrid.stn == 10].reset_index(drop=True)

start_date = "1900-02-01 00:00:00"  # 시작 날짜
end_date = "1905-01-31 23:00:00"  # 종료 날짜
freq = "H"  # 월 단위로 설정

surface_tp_train_hybrid_stn10.index = pd.date_range(start=start_date, end=end_date, freq=freq)

surface_tp_train_hybrid_stn10.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)

surface_tp_train_hybrid_stn10 = pd.get_dummies(data=surface_tp_train_hybrid_stn10, drop_first=True)

from sklearn.model_selection import train_test_split

target = ['ts']
f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F',
       'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']

train_x, test_x, train_y, test_y = train_test_split(
                                                    surface_tp_train_hybrid_stn10[f_col],
                                                    surface_tp_train_hybrid_stn10[target],
                                                    test_size=0.2,
                                                    shuffle=False,
                                                    )

from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

dp = DeterministicProcess(
                          index = train_x.index,
                          constant = True,
                          order = 1,
                          drop = True,
                          )

X = dp.in_sample()
lm10 = LinearRegression(fit_intercept=False)
lm10.fit(X, train_y)

test_const = [1 for i in range(test_x.shape[0])]
test_trend = np.arange(X['trend'].max(), X['trend'].max()+test_x.shape[0])
# test_trend_squared = [i**2 for i in test_trend]
X_ = pd.DataFrame([test_const, test_trend]).T
X_.columns = ['const', 'trend']
X_.index = test_x.index

trend_train = lm10.predict(X)
trend_test = lm10.predict(X_)

import xgboost

train_y_delta = train_y - trend_train
test_y_delta = test_y - trend_test

# model
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

# misc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.metrics import mean_absolute_error

def XGB_params_fold_start(XGB_params, X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    pred_list = []
    mae_list = []

    for train_index, val_index in tscv.split(X):

        x_train, x_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]

        model = xgboost.XGBRegressor(**XGB_params)

        model.fit(x_train, y_train,
                eval_set=[(x_val,y_val)],
                eval_metric='mae', verbose=False, early_stopping_rounds=100)

        pred = model.predict(x_val)
        result = mean_absolute_error(pred,y_val)

        mae_list.append(result)

    return np.mean(mae_list)

def objectiveXGB(trial: Trial, X, y):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : 1127,
        'objective' : 'reg:absoluteerror'
    }

    score = XGB_params_fold_start(param, X, y)

    return score

study = optuna.create_study(direction='minimize',sampler=TPESampler(seed=42))

study.optimize(lambda trial : objectiveXGB(trial, train_x, train_y_delta), n_trials=30)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

xgb_best10 = xgboost.XGBRegressor(**study.best_trial.params, **{'tree_method' : 'gpu_hist', 'predictor' : 'gpu_predictor'})
xgb_best10.fit(train_x, train_y_delta)

xgb_train = xgb_best10.predict(train_x)
xgb_test = xgb_best10.predict(test_x)

original_train = [i+j for i,j in zip(sum(trend_train.tolist(), []), xgb_train.tolist())]

original_test = [i+j for i,j in zip(sum(trend_test.tolist(), []), xgb_test.tolist())]


# ## 앙상블




surface_tp_test_hybrid = surface_tp_test.copy()

# 기상청 설명자료 속 선행지식 활용 (지점 abc)
surface_tp_test_hybrid_stn1 = surface_tp_test_hybrid[surface_tp_test_hybrid.stn == 'a'].reset_index(drop=True)
surface_tp_test_hybrid_stn2 = surface_tp_test_hybrid[surface_tp_test_hybrid.stn == 'b'].reset_index(drop=True)
surface_tp_test_hybrid_stn3 = surface_tp_test_hybrid[surface_tp_test_hybrid.stn == 'c'].reset_index(drop=True)

# 기상청 설명자료 속 선행지식 활용 (F~G년)
start_date = "1906-02-01 00:00:00"  # 시작 날짜
end_date = "1907-01-31 23:00:00"  # 종료 날짜
freq = "H"  # 월 단위로 설정

surface_tp_test_hybrid_stn1.index = pd.date_range(start=start_date, end=end_date, freq=freq)
surface_tp_test_hybrid_stn2.index = pd.date_range(start=start_date, end=end_date, freq=freq)
surface_tp_test_hybrid_stn3.index = pd.date_range(start=start_date, end=end_date, freq=freq)

surface_tp_test_hybrid_stn1.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)
surface_tp_test_hybrid_stn2.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)
surface_tp_test_hybrid_stn3.drop(['stn', 'mmddhh', 'year'], axis=1, inplace=True)

surface_tp_test_hybrid_stn1 = pd.get_dummies(data=surface_tp_test_hybrid_stn1, drop_first=True)
surface_tp_test_hybrid_stn2 = pd.get_dummies(data=surface_tp_test_hybrid_stn2, drop_first=True)
surface_tp_test_hybrid_stn3 = pd.get_dummies(data=surface_tp_test_hybrid_stn3, drop_first=True)




dp = DeterministicProcess(
                          index = surface_tp_train_hybrid_stn1.index,
                          constant = True,
                          order = 1,
                          drop = True,
                          )

X = dp.in_sample()





target = ['ts']
f_col = ['ta', 'td', 'hm', 'ws', 'rn', 're', 'si', 'ss', 'sn', 'ww_F',
       'ww_G', 'ww_H', 'ww_R', 'ww_S', 'ww_X']

lm1.fit(X, surface_tp_train_hybrid_stn1[target])
lm2.fit(X, surface_tp_train_hybrid_stn2[target])
lm3.fit(X, surface_tp_train_hybrid_stn3[target])
lm4.fit(X, surface_tp_train_hybrid_stn4[target])
lm5.fit(X, surface_tp_train_hybrid_stn5[target])
lm6.fit(X, surface_tp_train_hybrid_stn6[target])
lm7.fit(X, surface_tp_train_hybrid_stn7[target])
lm8.fit(X, surface_tp_train_hybrid_stn8[target])
lm9.fit(X, surface_tp_train_hybrid_stn9[target])
lm10.fit(X, surface_tp_train_hybrid_stn10[target])





# 지점별 train data 추세 추출
trend_stn1 = lm1.predict(X)
trend_stn2 = lm2.predict(X)
trend_stn3 = lm3.predict(X)
trend_stn4 = lm4.predict(X)
trend_stn5 = lm5.predict(X)
trend_stn6 = lm6.predict(X)
trend_stn7 = lm7.predict(X)
trend_stn8 = lm8.predict(X)
trend_stn9 = lm9.predict(X)
trend_stn10 = lm10.predict(X)





# 추세를 제거한 계절성만 남은 전체 train data로 학습
xgb_best1.fit(surface_tp_train_hybrid_stn1[f_col], [i-j for i,j in zip(surface_tp_train_hybrid_stn1[target].ts.tolist(),  sum(trend_stn1.tolist(), []))])
xgb_best2.fit(surface_tp_train_hybrid_stn2[f_col], [i-j for i,j in zip(surface_tp_train_hybrid_stn2[target].ts.tolist(),  sum(trend_stn2.tolist(), []))])
xgb_best3.fit(surface_tp_train_hybrid_stn3[f_col], [i-j for i,j in zip(surface_tp_train_hybrid_stn3[target].ts.tolist(),  sum(trend_stn3.tolist(), []))])
xgb_best4.fit(surface_tp_train_hybrid_stn4[f_col], [i-j for i,j in zip(surface_tp_train_hybrid_stn4[target].ts.tolist(),  sum(trend_stn4.tolist(), []))])
xgb_best5.fit(surface_tp_train_hybrid_stn5[f_col], [i-j for i,j in zip(surface_tp_train_hybrid_stn5[target].ts.tolist(),  sum(trend_stn5.tolist(), []))])
xgb_best6.fit(surface_tp_train_hybrid_stn6[f_col], [i-j for i,j in zip(surface_tp_train_hybrid_stn6[target].ts.tolist(),  sum(trend_stn6.tolist(), []))])
xgb_best7.fit(surface_tp_train_hybrid_stn7[f_col], [i-j for i,j in zip(surface_tp_train_hybrid_stn7[target].ts.tolist(),  sum(trend_stn7.tolist(), []))])
xgb_best8.fit(surface_tp_train_hybrid_stn8[f_col], [i-j for i,j in zip(surface_tp_train_hybrid_stn8[target].ts.tolist(),  sum(trend_stn8.tolist(), []))])
xgb_best9.fit(surface_tp_train_hybrid_stn9[f_col], [i-j for i,j in zip(surface_tp_train_hybrid_stn9[target].ts.tolist(),  sum(trend_stn9.tolist(), []))])
xgb_best10.fit(surface_tp_train_hybrid_stn10[f_col], [i-j for i,j in zip(surface_tp_train_hybrid_stn10[target].ts.tolist(),  sum(trend_stn10.tolist(), []))])




# 없는 열은 0으로 설정
surface_tp_test_hybrid_stn1['ww_X'] = 0
surface_tp_test_hybrid_stn2['ww_X'] = 0
surface_tp_test_hybrid_stn3['ww_X'] = 0

surface_tp_test_hybrid_stn3['ww_H'] = 0





# 지점 a
# a_trend_lm1 = lm1.predict(surface_tp_test_hybrid_stn1[f_col])
# a_trend_lm2 = lm2.predict(surface_tp_test_hybrid_stn1[f_col])
# a_trend_lm3 = lm3.predict(surface_tp_test_hybrid_stn1[f_col])
# a_trend_lm4 = lm4.predict(surface_tp_test_hybrid_stn1[f_col])
# a_trend_lm5 = lm5.predict(surface_tp_test_hybrid_stn1[f_col])
# a_trend_lm6 = lm6.predict(surface_tp_test_hybrid_stn1[f_col])
# a_trend_lm7 = lm7.predict(surface_tp_test_hybrid_stn1[f_col])
# a_trend_lm8 = lm8.predict(surface_tp_test_hybrid_stn1[f_col])
# a_trend_lm9 = lm9.predict(surface_tp_test_hybrid_stn1[f_col])
# a_trend_lm10 = lm10.predict(surface_tp_test_hybrid_stn1[f_col])

a_seasonality_xgb_best1 = xgb_best1.predict(surface_tp_test_hybrid_stn1[f_col])
a_seasonality_xgb_best2 = xgb_best2.predict(surface_tp_test_hybrid_stn1[f_col])
a_seasonality_xgb_best3 = xgb_best3.predict(surface_tp_test_hybrid_stn1[f_col])
a_seasonality_xgb_best4 = xgb_best4.predict(surface_tp_test_hybrid_stn1[f_col])
a_seasonality_xgb_best5 = xgb_best5.predict(surface_tp_test_hybrid_stn1[f_col])
a_seasonality_xgb_best6 = xgb_best6.predict(surface_tp_test_hybrid_stn1[f_col])
a_seasonality_xgb_best7 = xgb_best7.predict(surface_tp_test_hybrid_stn1[f_col])
a_seasonality_xgb_best8 = xgb_best8.predict(surface_tp_test_hybrid_stn1[f_col])
a_seasonality_xgb_best9 = xgb_best9.predict(surface_tp_test_hybrid_stn1[f_col])
a_seasonality_xgb_best10 = xgb_best10.predict(surface_tp_test_hybrid_stn1[f_col])

# 지점 b
# b_trend_lm1 = lm1.predict(surface_tp_test_hybrid_stn2[f_col])
# b_trend_lm2 = lm2.predict(surface_tp_test_hybrid_stn2[f_col])
# b_trend_lm3 = lm3.predict(surface_tp_test_hybrid_stn2[f_col])
# b_trend_lm4 = lm4.predict(surface_tp_test_hybrid_stn2[f_col])
# b_trend_lm5 = lm5.predict(surface_tp_test_hybrid_stn2[f_col])
# b_trend_lm6 = lm6.predict(surface_tp_test_hybrid_stn2[f_col])
# b_trend_lm7 = lm7.predict(surface_tp_test_hybrid_stn2[f_col])
# b_trend_lm8 = lm8.predict(surface_tp_test_hybrid_stn2[f_col])
# b_trend_lm9 = lm9.predict(surface_tp_test_hybrid_stn2[f_col])
# b_trend_lm10 = lm10.predict(surface_tp_test_hybrid_stn2[f_col])

b_seasonality_xgb_best1 = xgb_best1.predict(surface_tp_test_hybrid_stn2[f_col])
b_seasonality_xgb_best2 = xgb_best2.predict(surface_tp_test_hybrid_stn2[f_col])
b_seasonality_xgb_best3 = xgb_best3.predict(surface_tp_test_hybrid_stn2[f_col])
b_seasonality_xgb_best4 = xgb_best4.predict(surface_tp_test_hybrid_stn2[f_col])
b_seasonality_xgb_best5 = xgb_best5.predict(surface_tp_test_hybrid_stn2[f_col])
b_seasonality_xgb_best6 = xgb_best6.predict(surface_tp_test_hybrid_stn2[f_col])
b_seasonality_xgb_best7 = xgb_best7.predict(surface_tp_test_hybrid_stn2[f_col])
b_seasonality_xgb_best8 = xgb_best8.predict(surface_tp_test_hybrid_stn2[f_col])
b_seasonality_xgb_best9 = xgb_best9.predict(surface_tp_test_hybrid_stn2[f_col])
b_seasonality_xgb_best10 = xgb_best10.predict(surface_tp_test_hybrid_stn2[f_col])

# 지점 c
# c_trend_lm1 = lm1.predict(surface_tp_test_hybrid_stn3[f_col])
# c_trend_lm2 = lm2.predict(surface_tp_test_hybrid_stn3[f_col])
# c_trend_lm3 = lm3.predict(surface_tp_test_hybrid_stn3[f_col])
# c_trend_lm4 = lm4.predict(surface_tp_test_hybrid_stn3[f_col])
# c_trend_lm5 = lm5.predict(surface_tp_test_hybrid_stn3[f_col])
# c_trend_lm6 = lm6.predict(surface_tp_test_hybrid_stn3[f_col])
# c_trend_lm7 = lm7.predict(surface_tp_test_hybrid_stn3[f_col])
# c_trend_lm8 = lm8.predict(surface_tp_test_hybrid_stn3[f_col])
# c_trend_lm9 = lm9.predict(surface_tp_test_hybrid_stn3[f_col])
# c_trend_lm10 = lm10.predict(surface_tp_test_hybrid_stn3[f_col])

c_seasonality_xgb_best1 = xgb_best1.predict(surface_tp_test_hybrid_stn3[f_col])
c_seasonality_xgb_best2 = xgb_best2.predict(surface_tp_test_hybrid_stn3[f_col])
c_seasonality_xgb_best3 = xgb_best3.predict(surface_tp_test_hybrid_stn3[f_col])
c_seasonality_xgb_best4 = xgb_best4.predict(surface_tp_test_hybrid_stn3[f_col])
c_seasonality_xgb_best5 = xgb_best5.predict(surface_tp_test_hybrid_stn3[f_col])
c_seasonality_xgb_best6 = xgb_best6.predict(surface_tp_test_hybrid_stn3[f_col])
c_seasonality_xgb_best7 = xgb_best7.predict(surface_tp_test_hybrid_stn3[f_col])
c_seasonality_xgb_best8 = xgb_best8.predict(surface_tp_test_hybrid_stn3[f_col])
c_seasonality_xgb_best9 = xgb_best9.predict(surface_tp_test_hybrid_stn3[f_col])
c_seasonality_xgb_best10 = xgb_best10.predict(surface_tp_test_hybrid_stn3[f_col])





# 10개 지점 추세 평균값을 사용
avg_trend = [(a+b+c+d+e+f+g+h+i+j)/10 for a,b,c,d,e,f,g,h,i,j in zip(sum(trend_stn1.tolist(), []), sum(trend_stn2.tolist(), []), sum(trend_stn3.tolist(), []),
                                    sum(trend_stn4.tolist(), []), sum(trend_stn5.tolist(), []), sum(trend_stn6.tolist(), []),
                                    sum(trend_stn7.tolist(), []), sum(trend_stn8.tolist(), []), sum(trend_stn9.tolist(), []),
                                    sum(trend_stn10.tolist(), []))]

# a_trend = [(a+b+c+d+e+f+g+h+i+j)/10 for a,b,c,d,e,f,g,h,i,j in zip(sum(a_trend_lm1.tolist(), []), sum(a_trend_lm2.tolist(), []), sum(a_trend_lm3.tolist(), []),
#                                     sum(a_trend_lm4.tolist(), []), sum(a_trend_lm5.tolist(), []), sum(a_trend_lm6.tolist(), []),
#                                     sum(a_trend_lm7.tolist(), []), sum(a_trend_lm8.tolist(), []), sum(a_trend_lm9.tolist(), []),
#                                     sum(a_trend_lm10.tolist(), []))]

a_seasonality = [(a+b+c+d+e+f+g+h+i+j)/10 for a,b,c,d,e,f,g,h,i,j in zip(a_seasonality_xgb_best1.tolist(),a_seasonality_xgb_best2.tolist(), a_seasonality_xgb_best3.tolist(),
                                                                         a_seasonality_xgb_best4.tolist(),a_seasonality_xgb_best5.tolist(),a_seasonality_xgb_best6.tolist(),
                                                                         a_seasonality_xgb_best7.tolist(),a_seasonality_xgb_best8.tolist(),a_seasonality_xgb_best9.tolist(),
                                                                         a_seasonality_xgb_best10.tolist())]

a_pred = [i+j for i,j in zip(avg_trend, a_seasonality)]

# b_trend = [(a+b+c+d+e+f+g+h+i+j)/10 for a,b,c,d,e,f,g,h,i,j in zip(sum(b_trend_lm1.tolist(), []), sum(b_trend_lm2.tolist(), []), sum(b_trend_lm3.tolist(), []),
#                                     sum(b_trend_lm4.tolist(), []), sum(b_trend_lm5.tolist(), []), sum(b_trend_lm6.tolist(), []),
#                                     sum(b_trend_lm7.tolist(), []), sum(b_trend_lm8.tolist(), []), sum(b_trend_lm9.tolist(), []),
#                                     sum(b_trend_lm10.tolist(), []))]

b_seasonality = [(a+b+c+d+e+f+g+h+i+j)/10 for a,b,c,d,e,f,g,h,i,j in zip(b_seasonality_xgb_best1.tolist(),b_seasonality_xgb_best2.tolist(), b_seasonality_xgb_best3.tolist(),
                                                                         b_seasonality_xgb_best4.tolist(),b_seasonality_xgb_best5.tolist(), b_seasonality_xgb_best6.tolist(),
                                                                         b_seasonality_xgb_best7.tolist(),b_seasonality_xgb_best8.tolist(), b_seasonality_xgb_best9.tolist(),
                                                                         b_seasonality_xgb_best10.tolist())]

b_pred = [i+j for i,j in zip(avg_trend, b_seasonality)]

# c_trend = [(a+b+c+d+e+f+g+h+i+j)/10 for a,b,c,d,e,f,g,h,i,j in zip(sum(c_trend_lm1.tolist(), []), sum(c_trend_lm2.tolist(), []), sum(c_trend_lm3.tolist(), []),
#                                     sum(c_trend_lm4.tolist(), []), sum(c_trend_lm5.tolist(), []), sum(c_trend_lm6.tolist(), []),
#                                     sum(c_trend_lm7.tolist(), []), sum(c_trend_lm8.tolist(), []), sum(c_trend_lm9.tolist(), []),
#                                     sum(c_trend_lm10.tolist(), []))]

c_seasonality = [(a+b+c+d+e+f+g+h+i+j)/10 for a,b,c,d,e,f,g,h,i,j in zip(c_seasonality_xgb_best1.tolist(),c_seasonality_xgb_best2.tolist(), c_seasonality_xgb_best3.tolist(),
                                                                         c_seasonality_xgb_best4.tolist(),c_seasonality_xgb_best5.tolist(), c_seasonality_xgb_best6.tolist(),
                                                                         c_seasonality_xgb_best7.tolist(),c_seasonality_xgb_best8.tolist(), c_seasonality_xgb_best9.tolist(),
                                                                         c_seasonality_xgb_best10.tolist())]

c_pred = [i+j for i,j in zip(avg_trend, c_seasonality)]





surface_tp_test = pd.read_csv('/content/drive/MyDrive/기상청/데이터/지면온도/surface_tp_test.csv')

# 불필요 첫 2열 제거
surface_tp_test = surface_tp_test.iloc[:, 1:]

# 예측값 병합
a_pred.extend(b_pred)

a_pred.extend(c_pred)




