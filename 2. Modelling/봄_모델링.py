# 봄 모델링 


import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import numpy as np
import random
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from catboost import CatBoostRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import RobustScaler
from scipy.signal import savgol_filter

from sklearn.model_selection import train_test_split
from pygam import LinearGAM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score


# 데이터 읽어오기
data_na = pd.read_csv("surface_tp_train.csv")
data_imp = pd.read_csv("지면온도train_imputed.csv")

# 분석에 이용할 데이터 만들기 
data_na['surface_tp_train.mmddhh']=data_na['surface_tp_train.mmddhh'].astype('str')

data_na['시간']=data_na['surface_tp_train.mmddhh'].str[-2:]
data_na['일']=data_na['surface_tp_train.mmddhh'].str[-4:-2]
data_na['월']=data_na['surface_tp_train.mmddhh'].str[:-4]

data_na.rename(columns={'surface_tp_train.stn':'지점번호'},inplace=True)


data=pd.concat([data_na[['지점번호','시간','월','일']],
                data_imp],axis=1)

data=data.drop(['time'],axis=1).dropna(axis=0)
need_scale=['평균기온', '평균이슬점온도', '평균상대습도', 
         '평균풍속', '누적강수량', '누적강수유무']

data[['월','일','시간']]=data[['월','일','시간']].astype('int')

encoder=LabelEncoder()
data['현천계현천']=encoder.fit_transform(data['현천계현천'])

# 계절별로 나누어주기

spring=data[data['월'].isin([2,3,4])] ## 봄: 2-4월 
summer=data[data['월'].isin([5,6,7])] ## 여름: 5-7월
fall=data[data['월'].isin([8,9,10])] ## 가을: 8-10월
winter=data[data['월'].isin([11,12,1])] ## 겨울: 11-1월

    
# RobustScaler
spscaler = RobustScaler() 
suscaler = RobustScaler() 
fascaler = RobustScaler()
wiscaler = RobustScaler()


# Robust Scaling 적용
spring[need_scale] = spscaler.fit_transform(spring[need_scale])
summer[need_scale] = suscaler.fit_transform(summer[need_scale])
fall[need_scale] = fascaler.fit_transform(fall[need_scale])
winter[need_scale] = wiscaler.fit_transform(winter[need_scale])

train_spring=spring
train_summer= summer
train_fall = fall 
train_winter = winter 


# 데이터 읽어오기
test_na = pd.read_csv("surface_tp_test.csv")
test_imp = pd.read_csv("지면온도test_imputed.csv")

# 분석에 이용할 데이터 만들기 
test_na['surface_tp_test.mmddhh']=test_na['surface_tp_test.mmddhh'].astype('str')

test_na['시간']=test_na['surface_tp_test.mmddhh'].str[-2:]
test_na['일']=test_na['surface_tp_test.mmddhh'].str[-4:-2]
test_na['월']=test_na['surface_tp_test.mmddhh'].str[:-4]

test_na.rename(columns={'surface_tp_test.stn':'STN'},inplace=True)
test_na.rename(columns={'surface_tp_test.year':'YEAR'},inplace=True)
test_na.rename(columns={'surface_tp_test.mmddhh':'MMDDHH'},inplace=True)

test_na[['STN','YEAR','MMDDHH']]=test_na[['STN','YEAR','MMDDHH']].astype('str')  # 답안지 데이터와의 type 통일 

test=pd.concat([test_na[['STN','시간','월','일','YEAR','MMDDHH']],
                test_imp],axis=1)

test=test.drop(['time'],axis=1)

need_scale=['평균기온', '평균이슬점온도', '평균상대습도', 
         '평균풍속', '누적강수량', '누적강수유무']

test[['월','일','시간']]=test[['월','일','시간']].astype('int')

test['현천계현천']=encoder.transform(test['현천계현천'])

# 계절별로 나누어주기

test_spring=test[test['월'].isin([2,3,4])] ## 봄: 2-4월 
test_summer=test[test['월'].isin([5,6,7])] ## 여름: 5-7월
test_fall=test[test['월'].isin([8,9,10])] ## 가을: 8-10월
test_winter=test[test['월'].isin([11,12,1])] ## 겨울: 11-1월

# Robust Scaling 적용
test_spring[need_scale] = spscaler.transform(test_spring[need_scale])
test_summer[need_scale] = suscaler.transform(test_summer[need_scale])
test_fall[need_scale] = fascaler.transform(test_fall[need_scale])
test_winter[need_scale] = wiscaler.transform(test_winter[need_scale])


x_col=data.drop(['지점번호','지면온도','적설깊이'],axis=1).columns.to_list() # 겨울의 경우 적설깊이 변수를 써줘야 한다. 
x_winter=data.drop(['지점번호','지면온도'],axis=1).columns.to_list()

y_col='지면온도'
print(x_col)


def cv_MAE(data,model,x_col=x_col):
    
    MAE= np.array([])
    
    for i in tqdm(range(1,11)):
        idx=(data['지점번호']==i)
        
        train,test = data[~idx],data[idx]
        x_train,y_train = train[x_col],train[y_col]
        x_test,y_test = test[x_col],test[y_col]
    
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
    
        score=mean_absolute_error(y_test,y_pred)
        MAE=np.append(MAE,score)

    print("폴드 별 MAE",MAE)
    print("CV 평균 MAE",np.mean(MAE))
    
def score(data,model,x_col=x_col):
    
    MAE= np.array([])
    
    for i in tqdm(range(1,11)):
        idx=(data['지점번호']==i)
        
        train,test = data[~idx],data[idx]
        x_train,y_train = train[x_col],train[y_col]
        x_test,y_test = test[x_col],test[y_col]
    
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
    
        score=mean_absolute_error(y_test,y_pred)
        MAE=np.append(MAE,score)

    return(np.mean(MAE))


class regressor:
    def __init__(self, model1=None, model2=None):
        if model1 is None:
            self.model1 = CatBoostRegressor(verbose=False,has_time=True,random_state=6)
        else:
            self.model1 = model1

        if model2 is None:
            self.model2 = lgb.LGBMRegressor(random_state=6)
        else:
            self.model2 = model2

        self.model3 = None
        self.model4 = None

    def fit(self, x_train, y_train):
        # 1차 적합
        self.model1.fit(x_train,y_train)
        train_fitted_values = self.model1.predict(x_train)
        train_residual = y_train - train_fitted_values

        # 2차 적합
        self.model2.fit(x_train, train_residual)
        train_fitted_values2 = self.model2.predict(x_train)
        train_residual2 = train_residual - train_fitted_values2

        # 3차 적합
        self.model3 = lgb.LGBMRegressor(random_state=6)
        self.model3.fit(x_train, train_residual2)
        train_fitted_values3 = self.model3.predict(x_train)
        train_residual3 = train_residual2 - train_fitted_values3

        # 4차 적합
        self.model4 = lgb.LGBMRegressor(random_state=6)
        self.model4.fit(x_train, train_residual3)

    def predict(self, x_test):
        test_fitted_values = self.model1.predict(x_test)
        test_fitted_values2 = self.model2.predict(x_test)
        test_fitted_values3 = self.model3.predict(x_test)
        test_residual = self.model4.predict(x_test)

        return test_fitted_values + test_fitted_values2 + test_fitted_values3 + test_residual

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        score = mean_absolute_error(y_test, y_pred)
        return score

# 봄 

sp_param1= {'learning_rate': 0.04158500359538149, 'depth': 12, 'iterations': 419}
spring1=CatBoostRegressor(verbose=False,**sp_param1,has_time=True,random_state=6)

sp_param2 = {'max_depth': 15, 'learning_rate': 0.09950693202165563, 'n_estimators': 902, 'min_child_samples': 59, 'subsample': 0.8052368021730489}
spring2= lgb.LGBMRegressor(**sp_param2,random_state=6) 

# 여름 

su_param1= {'learning_rate': 0.12515542897654602, 'depth': 13, 'iterations': 401} 
summer1=CatBoostRegressor(verbose=False,**su_param1,has_time=True,random_state=6)

su_param2= {'max_depth': 14, 'learning_rate': 0.09990601824979606, 'n_estimators': 924, 'min_child_samples': 74, 'subsample': 0.30422223766873524}
summer2= lgb.LGBMRegressor(**su_param2,random_state=6) 


spr = regressor(model1=spring1,model2=spring2) #봄 
sumr= regressor(model1=summer1,model2=summer2) #여름 

spr.fit(train_spring[x_col],train_spring[y_col])
spring_predict=spr.predict(test_spring[x_col])

sumr.fit(train_summer[x_col],train_summer[y_col])
summer_predict=sumr.predict(test_summer[x_col])


# 옵튜나를 이용한 하이퍼 파라미터 튜닝 



import optuna
from optuna import Trial
from optuna.samplers import TPESampler



# 봄 모델 1차 적합 모델 튜닝 
def objective_cbr(trial):
    
    
    # 하이퍼파라미터 검색 범위
    learning_rate = trial.suggest_uniform('learning_rate', 0.01, 0.5)
    depth = trial.suggest_int('depth', 7, 16)
    iterations = trial.suggest_int('iterations', 250, 500)

    model = CatBoostRegressor(verbose=False,
                              learning_rate=learning_rate,
                              depth=depth,
                              iterations=iterations)
    
    MAE= np.array([])
    
    data=spring
    
    for i in range(1,11):
        idx=(data['지점번호']==i)
        
        train,test = data[~idx],data[idx]
        x_train,y_train = train[x_col],train[y_col]
        x_test,y_test = test[x_col],test[y_col]
    
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
    
        score=mean_absolute_error(y_test,y_pred)
        MAE=np.append(MAE,score)
        
    mae= np.mean(MAE)
    
    
    return mae

study = optuna.create_study(
        direction='minimize', 
        sampler=TPESampler())

study.optimize(objective_cbr,n_trials=15)

trial = study.best_trial
trial_params = trial.params
print('Best Trial: score {},\nparams {}'.format(trial.value, trial_params))

# Best Trial: score 1.6421336515585658,
# params {'learning_rate': 0.04158500359538149, 'depth': 12, 'iterations': 419}


# 봄 모델 2차 적합 모델 튜닝 

x_train, x_test, y_train, y_test = train_test_split(
    spring[x_col], spring[y_col], test_size=0.33,
    shuffle=True)

spring1.fit(x_train,y_train)
train_fitted_values = spring1.predict(x_train)
train_residual = y_train - train_fitted_values

test_fitted_values= spring1.predict(x_test)
test_residual = y_test - test_fitted_values

def objective_spring(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 10, 17),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 900, 970),
        'min_child_samples': trial.suggest_int('min_child_samples', 55, 75),
        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0)
    }

    model2 = lgb.LGBMRegressor(**param)
    model2.fit(x_train, train_residual)

    train_predictions2 = model2.predict(x_train)
    train_residuals2 = train_residual - train_predictions2
    y_pred = model2.predict(x_test)
    score = mean_absolute_error(test_residual, y_pred)

    return score

#study = optuna.create_study(direction='minimize')
study.optimize(objective_spring, n_trials=100)

best_params = study.best_params
print(best_params)

# {'max_depth': 12, 'learning_rate': 0.08334925654486108, 'n_estimators': 960, 'min_child_samples': 67, 'subsample': 0.5200456584507223}
# score= 1.3836848170174547
# {'max_depth': 15, 'learning_rate': 0.09950693202165563, 'n_estimators': 902, 'min_child_samples': 59, 'subsample': 0.8052368021730489}
# 1.378931408387576.





