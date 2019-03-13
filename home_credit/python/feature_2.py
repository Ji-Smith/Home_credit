import pandas as pd
import numpy as np
from other import *
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

if __name__ == '__main__':
    print("开始时间: " + str(time.strftime("%m-%d %H:%M:%S",time.localtime())))
    """
    train = pd.read_csv('../data/application_train.csv')
    train = train[['SK_ID_CURR']]
    test = pd.read_csv('../data/application_test.csv')
    test = test[['SK_ID_CURR']]
    train2 = train
    test2 = test
    
    previous = pd.read_csv('../data/previous_application.csv')
    for i in previous.columns:
        if type(previous[i][0]) == np.int64:
            previous[i] = previous[i].astype(np.int32)
        elif type(previous[i][0]) == np.float64:
            previous[i] = previous[i].astype(np.float32)
    previous_agg = agg_numeric(previous.drop(columns = ['SK_ID_PREV']), group_var = 'SK_ID_CURR', df_name = 'previous_loans')
    previous_counts = count_categorical(previous, group_var = 'SK_ID_CURR', df_name = 'previous_loans')
    print('Previous aggregated shape: ', previous_agg.shape)
    print('Previous categorical counts shape: ', previous_counts.shape)
    train = train.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
    train = train.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')
    test = test.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
    test = test.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')
    train.to_csv('train1.csv',index=False)
    test.to_csv('test1.csv',index=False)
    gc.enable()
    del previous, previous_agg, previous_counts,train,test
    gc.collect()

    train = train2
    test = test2
    cash = pd.read_csv('../data/POS_CASH_balance.csv')
    for i in cash.columns:
        if type(cash[i][0]) == np.int64:
            cash[i] = cash[i].astype(np.int32)
        elif type(cash[i][0]) == np.float64:
            cash[i] = cash[i].astype(np.float32)
    cash_by_client = aggregate_client(cash, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['cash', 'client'])
    print('Cash by Client Shape: ', cash_by_client.shape)
    train = train.merge(cash_by_client, on = 'SK_ID_CURR', how = 'left')
    test = test.merge(cash_by_client, on = 'SK_ID_CURR', how = 'left')
    train.to_csv('train2.csv',index=False)
    test.to_csv('test2.csv',index=False)
    gc.enable()
    del cash, cash_by_client,train,test
    gc.collect()

    train = train2
    test = test2
    credit = pd.read_csv('../data/credit_card_balance.csv')
    for i in credit.columns:
        if type(credit[i][0]) == np.int64:
            credit[i] = credit[i].astype(np.int32)
        elif type(credit[i][0]) == np.float64:
            credit[i] = credit[i].astype(np.float32)
    credit_by_client = aggregate_client(credit, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['credit', 'client'])
    print('Credit by client shape: ', credit_by_client.shape)
    train = train.merge(credit_by_client, on = 'SK_ID_CURR', how = 'left')
    test = test.merge(credit_by_client, on = 'SK_ID_CURR', how = 'left')
    train.to_csv('train3.csv',index=False)
    test.to_csv('test3.csv',index=False)
    gc.enable()
    del credit, credit_by_client,train,test
    gc.collect()

    train = train2
    test = test2
    installments = pd.read_csv('../data/installments_payments.csv')
    for i in installments.columns:
        if type(installments[i][0]) == np.int64:
            installments[i] = installments[i].astype(np.int32)
        elif type(installments[i][0]) == np.float64:
            installments[i] = installments[i].astype(np.float32)
    installments_by_client = aggregate_client(installments, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['installments', 'client'])
    print('Installments by client shape: ', installments_by_client.shape)
    train = train.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left')
    test = test.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left')
    train.to_csv('train4.csv',index=False)
    test.to_csv('test4.csv',index=False)
    gc.enable()
    del installments, installments_by_client,train,test
    gc.collect()

    #print("特征工程2共构造了特征：",train.shape[1]-1)
    #train.to_csv('../feature/train_feature_2.csv',index=False)
    #test.to_csv('../feature/test_feature_2.csv',index=False)
    """
    train = pd.read_csv('train1.csv')
    for i in train.columns:
        if type(train[i][0]) == np.int64:
            train[i] = train[i].astype(np.int32)
        elif type(train[i][0]) == np.float64:
            train[i] = train[i].astype(np.float32)
    for j in range(2,5):
        train_new = pd.read_csv('train%s.csv'%j)
        for i in train_new.columns:
            if type(train_new[i][0]) == np.int64:
                train_new[i] = train_new[i].astype(np.int32)
            elif type(train_new[i][0]) == np.float64:
                train_new[i] = train_new[i].astype(np.float32)
        train = train.merge(train_new,on='SK_ID_CURR', how = 'left')
    train.to_csv('../feature/train_feature_2.csv',index=False)
    gc.enable()
    del train
    gc.collect()

    test = pd.read_csv('test1.csv')
    for i in test.columns:
        if type(test[i][0]) == np.int64:
            test[i] = test[i].astype(np.int32)
        elif type(test[i][0]) == np.float64:
            test[i] = test[i].astype(np.float32)
    for j in range(2,5):
        test_new = pd.read_csv('test%s.csv'%j)
        for i in test_new.columns:
            if type(test_new[i][0]) == np.int64:
                test_new[i] = test_new[i].astype(np.int32)
            elif type(test_new[i][0]) == np.float64:
                test_new[i] = test_new[i].astype(np.float32)
        test = test.merge(test_new,on='SK_ID_CURR', how = 'left')
    test.to_csv('../feature/test_feature_2.csv',index=False)
    gc.enable()
    del test
    gc.collect()
    
    
    print("结束时间: " + str(time.strftime("%m-%d %H:%M:%S",time.localtime())))
    
    

