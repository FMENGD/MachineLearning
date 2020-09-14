# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 17:06:14 2020

@author: FMENG
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('data/train1.csv')
test_df = pd.read_csv('data/test1.csv')
train_df = train[train.loc[:,'Time']<3]
val_df = train[train.loc[:,'Time']==3]

#print(train_df.head())
#print(val_df.head())


def cb_eval(train_df,val_df):
    train_df = train_df.copy()
    val_df = val_df.copy()
    """
    try:
        from sklearn.preprocessing import LabelEncoder
        lb_encoder = LabelEncoder()
        lb_encoder.fit(train_df.loc[:,'RoomDir'].append(val_df.loc[:,'RoomDir']))
        train_df.loc[:,'RoomDir'] = lb_encoder.transform(train_df.loc[:,'RoomDir'])
        val_df.loc[:,'RoomDir'] = lb_encoder.transform(val_df.loc[:,'RoomDir'])
    except Exception as e:
        print(e)
    """
    
    import catboost as cb
    from catboost import Pool
    X_train = train_df.drop(['Label'],axis=1)
    Y_train = train_df.loc[:,'Label'].values
    X_val = val_df.drop(['Label'],axis=1)
    Y_val = val_df.loc[:,'Label'].values
    
    from sklearn.metrics import mean_squared_error
    
    try:
        eval_df = val_df.copy().drop('Time',axis=1)
    except Exception as e:
        eval_df = val_df.copy()
    
    train_pool = Pool(X_train, Y_train, cat_features=None)
    val_pool = Pool(X_val, cat_features=None)
    cb_model = cb.CatBoostRegressor(depth=8, learning_rate=0.11, iterations=2750, l2_leaf_reg=0.1, model_size_reg=2, loss_function='RMSE')
    cb_model.fit(train_pool, verbose=True)
    
    y_pred = cb_model.predict(val_pool)
    print(np.sqrt(mean_squared_error(Y_val,y_pred)),end=' ')
    
    eval_df.loc[:,'Y_pred'] = y_pred
    eval_df.loc[:,'RE'] = eval_df.loc[:,'Y_pred']-eval_df.loc[:,'Label']
    
    print('')
    feature = X_train.columns
    fe_im = cb_model.feature_importances_
    print(pd.DataFrame({'fe':feature,'im':fe_im}).sort_values(by='im',ascending=False))
    
    import matplotlib.pyplot as plt
    plt.clf()
    plt.figure(figsize=(15,4))
    plt.plot([Y_train.min(),Y_train.max()],[0,0],color='red')
    plt.scatter(x=eval_df.loc[:'Label'],y=eval_df.loc[:'RE'])
    plt.show()
    
    return eval_df


#cb_eval = cb_eval(train_df,val_df)


def lgb_eval(train_df,val_df):
    train_df = train_df.copy()
    val_df = val_df.copy()
    
    import lightgbm as lgb
    X_train = train_df.drop(['Label'],axis=1)
    Y_train = train_df.loc[:,'Label'].values
    X_val = val_df.drop(['Label'],axis=1)
    Y_val = val_df.loc[:,'Label'].values
    
    from sklearn.metrics import mean_squared_error
    
    try:
        eval_df = val_df.copy().drop('Time',axis=1)
    except Exception as e:
        eval_df = val_df.copy()
    
    lgb_model = lgb.LGBMRegressor(objective='regression', num_leaves=900,
                              learning_rate=0.1, n_estimators=3141, bagging_fraction=0.7,
                              feature_fraction=0.6, reg_alpha=0.3, reg_lambda=0.3,
                              min_data_in_leaf=18, min_sum_hessian_in_leaf=0.001)

    lgb_model.fit(X_train, Y_train)
    y_pred = lgb_model.predict(X_val)
    print(np.sqrt(mean_squared_error(Y_val,y_pred)),end='')
    
    eval_df.loc[:,'Y_pred'] = y_pred
    eval_df.loc[:,'RE'] = eval_df.loc[:,'Y_pred'] - eval_df.loc[:,'Label']
    
    print('')
    feature = X_train.columns
    fe_im = lgb_model.feature_importances_
    print(pd.DataFrame({'fe':feature,'im':fe_im}).sort_values(by='im',ascending=False))
    
    import matplotlib.pyplot as plt
    plt.clf()
    plt.figure(figsize=(15,4))
    plt.plot([Y_train.min(),Y_train.max()],[0,0],color='red')
    plt.scatter(x=eval_df.loc[:'Label'],y=eval_df.loc[:'RE'])
    plt.show()
    
    return eval_df

train_data = pd.read_csv('data/train1.csv')
train_df = train_data[train_data.loc[:,'Time']<3]
val_df=train_data[train_data.loc[:,'Time']==3]

comb_train_df = train_df.copy()
comb_val_df = val_df.copy()

items = ['RoomArea','BedRoom','LivingRoom','BathRoom','TolHeight']
for item in items:
    Neigh = train_df.groupby('小区名',as_index=False)[item].agg({item+"Ne":'mean'})
    comb_train_df = comb_train_df.merge(Neigh,on='小区名',how='left')
#去掉小区名
comb_train_df.drop(['小区名'],axis=1,inplace=True)
for item in items:
    Neigh = comb_val_df.groupby('小区名',as_index=False)[item].agg({item+"Ne":'mean'})
    comb_val_df = comb_val_df.merge(Neigh,on='小区名',how='left')
#去掉小区名
comb_val_df.drop(['小区名'],axis=1,inplace=True)

#lgb_eval = lgb_eval(train_df=comb_train_df,val_df=comb_val_df)


def cb_pred():
    train_df = pd.read_csv('data/train1.csv')
    test_df = pd.read_csv('data/test1.csv')
    """
    try:
        from sklearn.preprocessing import LabelEncoder
        lb_encoder = LabelEncoder()
        lb_encoder.fit(train_df.loc[:,'RoomDir2'].append(test_df.loc[:,'RoomDir']))
        train_df.loc[:,'RoomDir'] = lb_encoder.transform(train_df.loc[:,'RoomDir'])
        test_df.loc[:,'RoomDir'] = lb_encoder.transform(test_df.loc[:,'RoomDir'])
    except Exception as e:
        print(e)
    """
    import catboost as cb
    from catboost import Pool
    X_train = train_df.drop(['Label','ID'],axis=1)
    Y_train = train_df.loc[:,'Label'].values
    test_id = test_df.loc[:,'ID']
    X_test = test_df.drop(['ID'],axis=1)
    print(X_train.shape)
    print(X_test.shape)
    
    from sklearn.metrics import mean_squared_error
    
    train_pool = Pool(X_train, Y_train, cat_features=None)
    test_pool = Pool(X_test, cat_features=None)
    cb_model = cb.CatBoostRegressor(depth=11, learning_rate=0.11, iterations=2750, l2_leaf_reg=0.1, model_size_reg=2, loss_function='RMSE')
    cb_model.fit(train_pool, verbose=True)
    
    y_pred = cb_model.predict(test_pool)
    
    test_lgb = pd.DataFrame({'ID': test_id, 'Label': y_pred})
    test_lgb.to_csv('./result/catboost_sia.csv', index=False)
    

    return None
"""
from time import *
begin_time = time()
cb_pred()
end_time = time()
run_time = end_time-begin_time
print ('catbooost运行时间：',run_time)
"""
def lgb_pred():
    train_df = pd.read_csv('data/train1.csv')
    test_df = pd.read_csv('data/test1.csv')
    items = ['RoomArea','BedRoom','LivingRoom','BathRoom','TolHeight']
    for item in items:
        Neigh = train_df.groupby('小区名',as_index=False)[item].agg({item+"Ne":'mean'})
        train_df = train_df.merge(Neigh,on='小区名',how='left')
    train_df.drop(['小区名'],axis=1,inplace=True)
    for item in items:
        Neigh = test_df.groupby('小区名',as_index=False)[item].agg({item+"Ne":'mean'})
        test_df = test_df.merge(Neigh,on='小区名',how='left')
    test_df.drop(['小区名'],axis=1,inplace=True)
    
    import lightgbm as lgb
    X_train = train_df.drop(['Label','ID'],axis=1)
    Y_train = train_df.loc[:,'Label'].values
    test_id = test_df.loc[:,'ID']
    X_test=test_df.drop(['ID'],axis=1)
    print(X_train.shape)
    print(X_test.shape)
    
    from sklearn.metrics import mean_squared_error
    
    lgb_model = lgb.LGBMRegressor(objective='regression', num_leaves=900,
                              learning_rate=0.1, n_estimators=3141, bagging_fraction=0.7,
                              feature_fraction=0.6, reg_alpha=0.3, reg_lambda=0.3,
                              min_data_in_leaf=18, min_sum_hessian_in_leaf=0.001)

    lgb_model.fit(X_train, Y_train)
    y_pred = lgb_model.predict(X_test)
    test_lgb = pd.DataFrame({'ID': test_id, 'Label': y_pred})
    test_lgb.to_csv('./result/lgb_sia.csv', index=False)

    return None

#print("lgb_pred")   
#lgb_pred()
   
#from time import *  
#begin_time = time()
#lgb_pred()
#end_time = time()
#run_time = end_time-begin_time
#print ('lightgbm运行时间：',run_time)

def xgb_eval(train_df,val_df):
    train_df = train_df.copy()
    val_df = val_df.copy()
    
    import xgboost as xgb
    X_train = train_df.drop(['Label'],axis=1)
    Y_train = train_df.loc[:,'Label'].values
    X_val = val_df.drop(['Label'],axis=1)
    Y_val = val_df.loc[:,'Label'].values
    
    from sklearn.metrics import mean_squared_error
    
    try:
        eval_df=val_df.copy().drop('Time',axis=1)
    except Exception as e:
        eval_df=val_df.copy()

    reg_model=xgb.XGBRegressor(max_depth=8,n_estimators=1000,n_jobs=-1)
    reg_model.fit(X_train,Y_train,eval_set=[(X_val,Y_val)],verbose=100,early_stopping_rounds=10)

    y_pred=reg_model.predict(X_val)
    print(np.sqrt(mean_squared_error(Y_val,y_pred)),end=' ')
    
    eval_df.loc[:,'Y_pred']=y_pred
    eval_df.loc[:,'RE']=eval_df.loc[:,'Y_pred']-eval_df.loc[:,'Label']

    print('')
    feature=X_train.columns
    fe_im=reg_model.feature_importances_
    print(pd.DataFrame({'fe':feature,'im':fe_im}).sort_values(by='im',ascending=False))

    import matplotlib.pyplot as plt
    plt.clf()
    plt.figure(figsize=(15,4))
    plt.plot([Y_train.min(),Y_train.max()],[0,0],color='red')
    plt.scatter(x=eval_df.loc[:,'Label'],y=eval_df.loc[:,'RE'])
    plt.show()
    
    return eval_df


comb_train_df = train_df.copy()
comb_val_df = val_df.copy()
"""
items = ['RoomArea','BedRoom','LivingRoom','BathRoom','TolHeight']
for item in items:
    Neigh = train_df.groupby('小区名',as_index=False)[item].agg({item+"Ne":'mean'})
    comb_train_df = comb_train_df.merge(Neigh,on='小区名',how='left')
#去掉小区名
comb_train_df.drop(['小区名'],axis=1,inplace=True)
for item in items:
    Neigh = comb_val_df.groupby('小区名',as_index=False)[item].agg({item+"Ne":'mean'})
    comb_val_df = comb_val_df.merge(Neigh,on='小区名',how='left')
#去掉小区名
comb_val_df.drop(['小区名'],axis=1,inplace=True)
  """  
#eval_df = xgb_eval(train_df=comb_train_df,val_df=comb_val_df)    
# 调参记录
# dep8
#     est1000:1.985,
#     est1880:1.943,1.94
#     earning_rate=0.05,est2643:1.9446
#     earning_rate=0.03,est4237:1.975
#     reg_alpha0.5,reg_lambda0.5,est1597:1.945
#         earning_rate=0.05,est2245:1.947
#     min_child_weight2,est1070:1.97
def xgb_pred():
    train_df=pd.read_csv('data/train1.csv')
    test_df=pd.read_csv('data/test1.csv')
    
    import xgboost as xgb
    X_train=train_df.drop(['Label','ID'],axis=1)
    Y_train=train_df.loc[:,'Label'].values
    test_id=test_df.loc[:,'ID']
    X_test=test_df.drop(['ID'],axis=1)
    

    from sklearn.metrics import mean_squared_error

    reg_model=xgb.XGBRegressor(max_depth=8,n_estimators=1880,n_jobs=-1)
    reg_model.fit(X_train,Y_train,eval_set=[(X_train,Y_train)],verbose=100,early_stopping_rounds=10)

    y_pred=reg_model.predict(X_test)

    sub_df=pd.DataFrame({
        'ID':test_id,
        'Label':y_pred
    })
    sub_df.to_csv('./result/fangmeng_sia.csv',index=False)
    
    return None

from time import *  
begin_time = time()
xgb_pred()
end_time = time()
run_time = end_time-begin_time
print ('xgboost运行时间：',run_time)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    