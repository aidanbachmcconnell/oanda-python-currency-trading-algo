#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:51:58 2023

@author: aidanmcconnell
"""

import pandas as pd

print(pd.__version__)
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn
import random
#from sklearn.preprocessing import train_test_split
def one_hot_encode_fit_and_save(df):

    categorical_cols = self.categorical_cols
    print()
    print('One Hot Encoding columns:',categorical_cols)
    s = time.time()
    ohe = OneHotEncoder(handle_unknown = 'ignore')
    print('fitting OHE...')
    ohe = ohe.fit(df[categorical_cols])
    path = self.model_dir + 'one_hot_encoder.joblib'
    print('Saving OHE to:',path)
    joblib.dump(ohe, path )




    cat_df = ohe.transform(df[categorical_cols])
    try:
        feat_names = list(ohe.get_feature_names_out())
    except:
        feat_names = list(ohe.get_feature_names())
    cat_df = pd.DataFrame.sparse.from_spmatrix(cat_df,index = df.index,columns = feat_names)
    df = pd.concat([df.loc[:,~df.columns.isin(categorical_cols)],cat_df],axis = 1,join = 'inner')
    cat_df = 0
    e = time.time()
    print('one hot encoding time:',(e-s)/60,' minutes')


def one_hot_encode_transform(self,df):
    import joblib
    categorical_cols = self.categorical_cols
    print()
    s = time.time()
    path = self.model_dir + 'one_hot_encoder.joblib'
    print('Loading OHE from:',path)
    ohe = joblib.load(path)
    cat_df = ohe.transform(df[categorical_cols])
    try:
        feat_names = list(ohe.get_feature_names_out())
    except:
        feat_names = list(ohe.get_feature_names())
    cat_df = pd.DataFrame.sparse.from_spmatrix(cat_df,index = df.index,columns = feat_names)
    df = pd.concat([df.loc[:,~df.columns.isin(categorical_cols)],cat_df],axis = 1,join = 'inner')
    cat_df = 0
    e = time.time()
    print('one hot encoding time:',round((e-s)/60,2),' minutes')

    return df
cur = 'EUR_USD'
curs = ['AUD_CAD',
        'AUD_CHF',
        'AUD_JPY',
        'AUD_NZD',
        'AUD_USD',
        'CAD_CHF',
        'CAD_JPY',
        'CHF_JPY',
        'EUR_AUD',
        'EUR_CAD',
        'EUR_CHF',
        'EUR_JPY',
        'EUR_USD',
        'GBP_JPY',
        'GBP_USD'
        ]
final_df = pd.DataFrame([0])
for cur in curs:
    print(cur)
    file = 'SIM_CLEAN_MA_' + cur + '_M5_2016-01-01_2022-01-31.csv'
    print(file)
    df = pd.read_csv(file)
    print(df.shape)
    print(df.head())
    cols = df.columns
    for col in df.columns:
        print(col)
    variables = ['ending_val',
                 'instrument',
                 'buy_or_sell',
                 'RR',
                 'lookup_range',
                 'bullish_ma',
                 'bullish_candle',
                 'support',
                 'candle_size',
                 'delta_filter',
                 'delta_filter2',
                 'upper_wick',
                 'lower_wick'
                 
                 ]
    df = df[variables]
    if 'JPY' in cur:
        df['candle_size'] = df['candle_size'] / 100
        df['delta_filter'] = df['delta_filter'] / 100
        df['upper_wick'] = df['upper_wick'] / 100
        df['lower_wick'] = df['lower_wick'] / 100
        
    if final_df.shape[0] == 1:
        final_df = df
    else:
        final_df = final_df.append(df)
print()
print()
print()
print('FINAL_SHAPE',final_df.shape)
x_cols = ['instrument',
        'buy_or_sell',
        'RR',
        'lookup_range',
        'bullish_ma',
        'bullish_candle',
        'support',
        'candle_size',
        'delta_filter',
        'delta_filter2',
        'upper_wick',
        'lower_wick']


final_df['random'] = 0
arr = final_df.values
for i in range(arr.shape[0]):
    num = random.random()
    arr[i,-1] = num
final_df = pd.DataFrame(arr,columns = final_df.columns)
final_df = final_df.sort_values(by = 'random')
train = final_df.iloc[:1000000,:]
test = final_df.iloc[1000000:,:]
X_train = train[x_cols]
y_train = train['ending_val']
X_test = test[x_cols]
y_test = test['ending_val']

final_df = 0
train = 0
test = 0
print('ML DATA SHAPES')
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
for c in X_train.columns:
    print(c)
    

categorical_cols = ['instrument',
        'buy_or_sell',
        'bullish_ma',
        'bullish_candle',
        'support']
for c in categorical_cols:
    print()
    print(c,X_train[c].value_counts().index)
    
    
    
    
    
def fit_ohe2(df):
    new_cols = []
    df_pandas_one_hot = pd.get_dummies(df[categorical_cols])
    for col in df_pandas_one_hot.columns:
        print(col)
                
    
        
                
        
        
        
        
def fit_ohe(df,df2,categorical_cols):
    ohe = OneHotEncoder(handle_unknown = 'ignore')
    print('fitting OHE...')
    ohe = ohe.fit(df[categorical_cols])
    cat_df = ohe.transform(df[categorical_cols])
    cat_df2 = ohe.transform(df2[categorical_cols])
    try:
        feat_names = list(ohe.get_feature_names_out())
    except:
        feat_names = list(ohe.get_feature_names())
    cat_df = pd.DataFrame.sparse.from_spmatrix(cat_df,index = df.index,columns = feat_names)
    df = pd.concat([df.loc[:,~df.columns.isin(categorical_cols)],cat_df],axis = 1,join = 'inner')
    cat_df = 0
    
    
    
fit_ohe2(X_train)

    

    
    










