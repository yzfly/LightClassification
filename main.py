import os
import lightgbm as lgb
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix



def data_info():
    # Structured dataset information
    # Reference: datasets/bank-additional-names.txt
    info = {}
    numeric_fea = ['age', 'duration', 'campaign', 'pdays', 
                'previous', 'emp.var.rate', 'cons.price.idx',
                'cons.conf.idx', 'euribor3m', 'nr.employed']
    categorical_fea = ['job','marital', 'education',
                'default', 'housing', 'loan', 
                'contact', 'month', 'day_of_week',
                'poutcome']
    jobs = ["admin.","blue-collar","entrepreneur",
            "housemaid","management","retired",
            "self-employed","services",
            "student","technician",
            "unemployed","unknown"]
    marital = ["divorced","married","single","unknown"]
    info['numeric'] = numeric_fea
    info['categorical'] = categorical_fea
    return info
    
def get_args():
    parser = argparse.ArgumentParser(description="LightGBM")
    parser.add_argument("--data_path", default="datasets/bank-additional.csv", type=str)
    parser.add_argument("--split_ratio", default=0.8, type=float)
    parser.add_argument("--thres", default=0.5, type=float)
    args = parser.parse_args()
    return args

def train_val_dataset(args):
    d_info = data_info()
    df = pd.read_csv(args.data_path, sep=';')

    # label
    df.loc[df['y'] == 'yes', 'y'] = 1
    df.loc[df['y'] == 'no', 'y'] = 0
    label = np.array(df['y'], dtype=int)

    # numeric features
    features = []
    for fea_name in d_info['numeric']:
        features.append(np.array(df[fea_name], dtype=float))
    features.append(label)
    dset = np.stack(features, axis=-1)

    # train val split
    np.random.seed(11)
    np.random.shuffle(dset)
    sep_ind = int(args.split_ratio * dset.shape[0])
    train_set = dset[:sep_ind]
    val_set = dset[sep_ind:]
    return train_set, val_set

def sigmoid(x):
    z = 1/(1 + np.exp(-x))
    return z

def main(args):
    train_set, val_set = train_val_dataset(args)
    train_x, train_y = train_set[:,:-1], train_set[:,-1]
    val_x, val_y = val_set[:,:-1], val_set[:,-1]

    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_val = lgb.Dataset(val_x, val_y)

    lgb_params = {
        'task': "train",
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'n_estimators': 100,
        'num_class': 1,
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'verbose': 1
    }
    print("Start training...")
    model = lgb.train(lgb_params,
                     lgb_train, 
                     num_boost_round=20,
                     valid_sets=lgb_val,
                     callbacks=[lgb.early_stopping(stopping_rounds=10)]
                     )

    print('Saving LightGBM model...')
    model.save_model('lightgbm_model.txt')

    print("Start predicting...")
    y_pred = model.predict(val_x, num_iteration=model.best_iteration)

    #y_pred = sigmoid(y_pred)
    val_y = val_y.astype(int)

    y_pred[y_pred >= args.thres] = 1
    y_pred[y_pred < args.thres] = 0
    cm = confusion_matrix(val_y, y_pred)
    recall = cm[1][1] / np.sum(val_y)
    precision = cm[1][1] / np.sum(y_pred)
    f1 = 2*recall*precision/(recall+precision)
    acc = (cm[0][0]+cm[1][1]) / val_y.shape[0]
    print('Acc: {}, F1: {}, Precision: {}, Recall: {}'.format(acc, f1, precision, recall))
    print("Confusion Matrix: ", cm)

if __name__ == '__main__':
    args = get_args()
    main(args)
