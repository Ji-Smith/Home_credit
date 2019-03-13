import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from other import *
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

if __name__ == '__main__':
    bureau = pd.read_csv('../data/bureau.csv')
    train = pd.read_csv('../data/application_train.csv')
    test = pd.read_csv('../data/application_test.csv')
    bureau_balance = pd.read_csv('../data/bureau_balance.csv')

    bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
    bureau_agg = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')

    bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
    bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')

    bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')
    bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')

    bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')

    df = train[['SK_ID_CURR']]
    bureau_counts = bureau_counts.reset_index()
    df = df.merge(bureau_counts,on = 'SK_ID_CURR',how='left')
    df = df.merge(bureau_agg,on = 'SK_ID_CURR',how='left')
    df = df.merge(bureau_balance_by_client,on = 'SK_ID_CURR',how='left')
    df.to_csv('../feature/train_feature_1.csv',index=False)

    df = test[['SK_ID_CURR']]
    df = df.merge(bureau_counts,on = 'SK_ID_CURR',how='left')
    df = df.merge(bureau_agg,on = 'SK_ID_CURR',how='left')
    df = df.merge(bureau_balance_by_client,on = 'SK_ID_CURR',how='left')
    df.to_csv('../feature/test_feature_1.csv',index=False)
    

