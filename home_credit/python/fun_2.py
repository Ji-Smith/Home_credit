import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import gc
import matplotlib.pyplot as plt


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

            
def model(features, test_features, encoding = 'ohe', n_folds = 5):
    
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    labels = features['TARGET']
    
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        cat_indices = 'auto'

    elif encoding == 'le':
        label_encoder = LabelEncoder()
        
        cat_indices = []

        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))
                cat_indices.append(i)
    
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    feature_names = list(features.columns)
    
    features = np.array(features)
    test_features = np.array(test_features)
    
    k_fold = KFold(n_splits = n_folds, shuffle = False, random_state = 50)

    feature_importance_values = np.zeros(len(feature_names))

    test_predictions = np.zeros(test_features.shape[0])

    out_of_fold = np.zeros(features.shape[0])

    valid_scores = []
    train_scores = []
    
    for train_indices, valid_indices in k_fold.split(features):
        
        train_features, train_labels = features[train_indices], labels[train_indices]
        
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 5)
        
        best_iteration = model.best_iteration_

        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits

        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]

        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    valid_auc = roc_auc_score(labels, out_of_fold)

    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics

def plot_feature_importances(df):

    df = df.sort_values('importance', ascending = False).reset_index()
    
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))

    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df

def deal_dataframe(DFrame):
    for i in DFrame.columns:
        if type(DFrame[i][0]) == np.int64:
            DFrame[i] = DFrame[i].astype(np.int32)
        elif type(DFrame[i][0]) == np.float64:
            DFrame[i] = DFrame[i].astype(np.float16)
    return DFrame

if __name__ == '__main__':
    
    train_feature2 = pd.read_csv('../feature/train_feature_2.csv')
    train_feature2 = deal_dataframe(train_feature2)
    
    train_feature1 = pd.read_csv('../feature/train_feature_1.csv')
    train_feature1 = deal_dataframe(train_feature1)

    train = pd.read_csv('../data/application_train.csv')
    train = deal_dataframe(train)

    train = train.merge(train_feature2,on='SK_ID_CURR', how = 'left')
    gc.enable()
    del train_feature2
    gc.collect()
    train = train.merge(train_feature1,on='SK_ID_CURR', how = 'left')
    gc.enable()
    del train_feature1
    gc.collect()
    train = deal_dataframe(train)
    
    
    test_feature2 = pd.read_csv('../feature/test_feature_2.csv')
    test_feature2 = deal_dataframe(test_feature2)
    
    test_feature1 = pd.read_csv('../feature/test_feature_1.csv')
    test_feature1 = deal_dataframe(test_feature1)

    test = pd.read_csv('../data/application_test.csv')
    test = deal_dataframe(test)
    #transform_label(train,test)

    test = test.merge(test_feature2,on='SK_ID_CURR', how = 'left')
    gc.enable()
    del test_feature2
    gc.collect()
    test = test.merge(test_feature1,on='SK_ID_CURR', how = 'left')
    gc.enable()
    del test_feature1
    gc.collect()
    test = deal_dataframe(test)
    
    result, fi, metrics = model(train, test, encoding = 'le')
    result.to_csv('../提交/result6.csv',index=False)
    print(metrics)

    fi_sorted = plot_feature_importances(fi)
    top_300 = list(fi_sorted['feature'])[:300]
    top_300.append('SK_ID_CURR')
    test = test[top_300]
    top_300.append('TARGET')
    train = train[top_300]
    result, fi, metrics = model(train, test, encoding = 'le')
    result.to_csv('../提交/result7.csv',index=False)
    print(metrics)









































    
