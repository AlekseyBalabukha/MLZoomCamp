#imports
import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

import bentoml

#data loading and preparation:
df = pd.read_csv('./data/CreditScoring.csv')
df.columns = df.columns.str.lower()
# map target variable:
df['status'] = df['status'].map({
    1: 'ok',
    2: 'default',
    0: 'unk'
})
# map other features:
home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}
df['home'] = df['home'].map(home_values)
# matrital:
marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}
df['marital'] = df['marital'].map(marital_values)
# records:
records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}
df['records'] = df['records'].map(records_values)
#jobs:
job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}
df['job'] = df['job'].map(job_values)

for column in ['income', 'assets', 'debt']:
    df[column] = df[column].replace(to_replace=99999999, value=np.nan)
df = df.fillna(0)
df = df[df['status'] != 'unk'].reset_index(drop=True)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)
#reset_indexes:
df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

# convert into binary format:
df_full_train['status'] = (df_full_train['status'] == 'default').astype(int)
df_train['status'] = (df_train['status'] == 'default').astype(int)
df_val['status'] = (df_val['status'] == 'default').astype(int)
df_test['status'] = (df_test['status'] == 'default').astype(int)

#assign target variables separately:
y_full_train = df_full_train['status'].values
y_train = df_train['status'].values
y_val = df_val['status'].values
y_test = df_test['status'].values

# remove target from dataset:
del df_full_train['status']
del df_train['status']
del df_val['status']
del df_test['status']

# turn data into Dictionaries to use One-hot encoding later
train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')
dicts_full_train = df_full_train.to_dict(orient='records')

# train DictVectorizer:
dv = DictVectorizer(sparse=False)
dv.fit(train_dicts)
X_train = dv.transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)
X_full_train = dv.transform(dicts_full_train)

#matrix for xgboost
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train)
dtest = xgb.DMatrix(X_test)

# xgboost:
xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
model = xgb.train(xgb_params, dfulltrain, num_boost_round=175) # final model
y_pred = model.predict(dtest)
auc = roc_auc_score(y_test, y_pred)
print(f'auc = {auc}')

bentoml.xgboost.save_model("credit_risk_model", model,
                           custom_objects={
                                "DictVectorizer": dv
                           },
                           signatures = { # models signatures for runner inference
                            'predict': {
                                'batchable': True,
                                'batch_dim': 0 # 0 means BentoML will concatenate request
                            }
                           }
                           )