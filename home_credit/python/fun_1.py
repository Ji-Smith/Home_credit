import pandas as pd
import lightgbm as lgb
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV,LinearRegression
from sklearn.preprocessing import OneHotEncoder  
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

train = pd.read_csv('../data/application_train.csv')
test = pd.read_csv('../data/application_test.csv')

def transform_label(DFrame1,DFrame2):
    for i in DFrame1.columns:
        if isinstance(DFrame1.loc[0,i],str):
            print(i)
            train[i] = train[i].fillna('nan')
            test[i] = test[i].fillna('nan')
            tmp = list(DFrame1[i])
            tmp.extend(DFrame2[i])
            label = LabelEncoder().fit(tmp)
            DFrame1[i] = label.transform(DFrame1[i])
            DFrame2[i] = label.transform(DFrame2[i])
            tmp = list(DFrame1[i])
            tmp.extend(DFrame2[i])
            onehot = OneHotEncoder(sparse=False).fit(np.array(tmp).reshape(-1,1))
            DFrame1[i] = onehot.transform(DFrame1[i].reshape(-1,1))
            DFrame2[i] = onehot.transform(DFrame2[i].reshape(-1,1))
            print(i)

def lgb_train(train,test,target,csv):
    Y = train[target]
    X = train.drop([target,'SK_ID_CURR'],1)
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
    lgb_train = lgb.Dataset(x_train,y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    params = {  
        'boosting_type': 'gbdt',  
        'objective': 'binary',  
        #'metric': {'binary_logloss', 'auc'},
        'metric': 'auc',
        #'metric': 'l2',
        'num_leaves': 300,
        'max_depth': 30,  
        'min_data_in_leaf': 50,  
        'learning_rate': 0.01,  
        'feature_fraction': 0.8,  
        'bagging_fraction': 0.8,  
        'bagging_freq': 5,  
        'lambda_l1': 0.2,    
        'lambda_l2': 0.1,  # 越小l2正则程度越高  
        'min_gain_to_split': 0.2,  
        'verbose': 5,  
        'is_unbalance': True  
    }
    gbm = lgb.train(params,  
            lgb_train,  
            num_boost_round=10000,  
            valid_sets=lgb_eval,  
            early_stopping_rounds=500)
    

    df = pd.DataFrame()
    df['SK_ID_CURR'] = test['SK_ID_CURR']
    test = test.drop('SK_ID_CURR',1)
    pred = gbm.predict(test, num_iteration=gbm.best_iteration)
    df['TARGET'] = pred
    df.to_csv('../%s.csv'%csv,index=False)



def logit_regression(train,test,target,csv):
    #global pred
    train_0 = train[train.TARGET == 0]
    train_1 = train[train.TARGET == 1]
    train1,train2 = train_test_split(train_0)
    train1 = train1.iloc[0:3*train_1.shape[0],]
    train = pd.concat([train1,train_1])
    train = train.fillna(0)
    test = test.fillna(0)
    Y = train[target]
    X = train.drop([target,'SK_ID_CURR'],1)
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
    lr = LogisticRegressionCV(multi_class="ovr",fit_intercept=True,cv=2,penalty="l2",solver="lbfgs",class_weight='balance',tol=0.01)
    re = lr.fit(x_train,y_train)
    print("准确率: "+str(re.score(x_train,y_train)))

    re = lr.fit(X,Y)
    print("准确率: "+str(re.score(x_train,y_train)))
    df = pd.DataFrame()
    df['SK_ID_CURR'] = test['SK_ID_CURR']
    test = test.drop('SK_ID_CURR',1)
    pred = re.predict_proba(test)
    df['TARGET'] = pred[:,1]
    df.to_csv('../%s.csv'%csv,index=False)
    
    
"""
def logit_regression(train,test,target,csv):
    #global pred
    train = train.fillna(-1)
    test = test.fillna(-1)
    Y = train[target]
    X = train.drop([target,'SK_ID_CURR'],1)
    X['intercept'] = 1
    x_train,x_test,y_train,y_test = train_test_split(X,Y)
    logit = sm.Logit(y_train,x_train)
    result = logit.fit()
    pred = result.predict(x_test)
    #print(len(pred))
    err = []
    y_test = list(y_test)
    pred = list(pred)
    for i in range(len(x_test)):
        #print(i)
        err.append(pred[i] - y_test[i])
    print("RMSE: " + str(np.mean(np.array(err)**2)))
    logit = sm.Logit(Y,X)
    result = logit.fit()

    df = pd.DataFrame()
    df['SK_ID_CURR'] = test['SK_ID_CURR']
    test = test.drop('SK_ID_CURR',1)
    test['intercept'] = 1
    pred = result.predict(test)
    df['TARGET'] = pred
    df.to_csv('../%s.csv'%csv,index=False)
"""   
    

if __name__ == '__main__':
    transform_label(train,test)
    #lgb_train(train,test,'TARGET','result5')
    #logit_regression(train,test,'TARGET','result5')












    
