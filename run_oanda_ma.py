import oanda_ma_algo_v1
import pandas as pd
import numpy as np
import os
from datetime import datetime, date, timedelta
import time 
import oandapyV20
from oandapyV20 import API

accountID='101-001-20062555-002'
access_token='2ad78612c58e604890fd961550f73cfd-28190cc09d5cbc67b2c14503575d6132'

api = API(access_token)
client=API(access_token)




file = 'best_trade_params.csv'
param_df = pd.read_csv(file)


dir_name = os.getcwd() + '/trading_folder'

isdir = os.path.isdir(dir_name) 
if isdir:
    pass
else:
    print('MAKING DIRECTORY:',dir_name)
    os.mkdir(dir_name) 

    
    
 
while 1 == 1:

    today = date.today()
    dir_name = os.getcwd() + '/trading_folder/' + str(today)

    isdir = os.path.isdir(dir_name)
    if isdir:
        pass
    else:
        print('MAKING DIRECTORY:',dir_name)
        os.mkdir(dir_name)
    oanda_ma_algo_v1.run_trade_algo_once(dir_name,param_df)

    

    