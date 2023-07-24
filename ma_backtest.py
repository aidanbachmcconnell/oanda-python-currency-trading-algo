import pandas as pd
import numpy as np
import os 
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import scipy as sp
import time
from random import random
pio.renderers.default = "iframe"

import pandas as pd
import numpy as np
import os
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import scipy as sp
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import time 
import os



import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades 
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.transactions as trans
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.definitions.primitives as primitives
import pandas as pd
from dateutil.relativedelta import relativedelta
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import psycopg2
import os
import io
from io import StringIO 
import boto3
import pandas as pd
import numpy as np

import time 
import pygsheets
import datedelta
import calendar
import  csv
import json

from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.pricing import PricingStream

from scipy import stats


def create_spike_trigger(df,col = 'delta_max',lookback_threshold = 3600):
    print('CREATING SPIKE TRIGGER...')
    df[col + '_spike_trigger'] = 0
    col_loc = df.columns.get_loc(col)
    arr = df.values
    for i in range(lookback_threshold,arr.shape[0]):
        if i % 200000 == 0:
            print(i)
        try:
            cur_max = max(arr[i - lookback_threshold : i ,col_loc])
        except:
            print('ERROR',i)
            cur_max = 100
        if arr[i,col_loc] > cur_max and sum(arr[i - 100 : i,-1]) == 0:
            arr[i,-1] = 1
    return pd.DataFrame(arr,columns = df.columns)

def create_wick_trigger(df,col = 'wick',lookback_threshold = 3600):
    
    print('CREATING WICK TRIGGER...')
    df['wick'] = 0
    df.loc[df[df['delta_vector'] < 0].index,'wick'] = df.loc[df[df['delta_vector'] < 0].index,'c'] - df.loc[df[df['delta_vector'] < 0].index,'l']
    df.loc[df[df['delta_vector'] > 0].index,'wick'] = df.loc[df[df['delta_vector'] > 0].index,'h'] - df.loc[df[df['delta_vector'] > 0].index,'c']
    df['wick_trigger'] = 0


    col_loc = df.columns.get_loc(col)
    arr = df.values
    for i in range(lookback_threshold,arr.shape[0]):
        if i % 200000 == 0:
            print(i)
        try:
            cur_max = max(arr[i - lookback_threshold : i ,col_loc])
        except:
            print('ERROR',i)
            cur_max = 100
        if arr[i,col_loc] > cur_max and sum(arr[i - 100 : i,-1]) == 0:
            arr[i,-1] = 1
    return pd.DataFrame(arr,columns = df.columns)
def consecutive_candles(df,num = 10,col = 'delta_vector'):
    print('CREATING CONSECUTIVE CANDLES TRIGGER...')
    df[col + '_consecutive_trigger'] = 0
    col_loc = df.columns.get_loc(col)
    arr = df.values
    last = -1
    counter = 0
    for i in range(arr.shape[0]):
        
        if i % 200000 == 0:
            print(i)
            
        if arr[i-1,col_loc] < 0 and arr[i,col_loc] < 0:
            counter += 1
        elif arr[i-1,col_loc] > 0 and arr[i,col_loc] > 0:
            counter += 1  
        else:
            counter = 0
            
        if counter == num:
            arr[i,-1] = 1
            counter = 0
    df = pd.DataFrame(arr,columns = df.columns)
    print('SHAPE OF TRIGGER DF:',df[df[col + '_consecutive_trigger'] == 1].shape)
    return df

def get_pearsons_corr(df,lookback = 10):
    print('GETTING PEARSONS CORR')
    df['ind'] = list(range(df.shape[0]))
    df['pearsons_corr'] = 0
    ind_col = df.columns.get_loc('ind')
    o_col = df.columns.get_loc('o')
    
    arr = df.values
    for i in range(arr.shape[0]):
        if i % 200000 == 0:
            print(i)
        r, p = sp.stats.pearsonr(arr[(i - looback) : i,ind_col], arr[(i - looback) : i,o_col])
        arr[i,-1] = r
    return pd.DataFrame(arr,columns = df.columns)

def get_pearsons_corr(df,lookback = 10):
    print('GETTING PEARSONS CORR')
    df['ind'] = list(range(df.shape[0]))
    df['pearsons_corr'] = 0
    ind_col = df.columns.get_loc('ind')
    o_col = df.columns.get_loc('o')
    
    arr = df.values
    for i in range(lookback,arr.shape[0]):
        if i % 200000 == 0:
            print(i)
        r, p = sp.stats.pearsonr(arr[(i - lookback) : i,ind_col], arr[(i - lookback) : i,o_col])
        arr[i,-1] = r
    return pd.DataFrame(arr,columns = df.columns)
def get_pearsons_corr2(df,lookback = 10):
    print('GETTING PEARSONS CORR')
    df['ind'] = list(range(df.shape[0]))
    df['pearsons_lookup'] = 0
    df['pearsons_corr2'] = 0
    ind_col = df.columns.get_loc('ind')
    o_col = df.columns.get_loc('o')
    c_col = df.columns.get_loc('c')
    new_col = df.columns.get_loc('pearsons_lookup')
    
    arr = df.values
    for i in range(lookback,arr.shape[0]):
        if i % 200000 == 0:
            print(i)
        if i % 2 == 0:
            arr[i,new_col] = arr[i,o_col]
        else:
            arr[i,new_col] = arr[i,c_col]        
            
            
    for i in range(lookback,arr.shape[0]):
        if i % 200000 == 0:
            print(i)
        r, p = sp.stats.pearsonr(arr[(i - lookback) : i,ind_col], arr[(i - lookback) : i,new_col])
        arr[i,-1] = r
    return pd.DataFrame(arr,columns = df.columns)
def convert_timestamp(df):
    time_col = df.columns.get_loc('time')
    arr = df.values 
    for i in range(arr.shape[0]):
        arr[i,time_col] = datetime.strptime(arr[i,time_col][:-4], '%Y-%m-%dT%H:%M:%S.%f')
    return pd.DataFrame(arr,columns = df.columns)
def get_best_fit(df,lookback = 10):

    def get_slope(y1,y2,total_x = 10):
        """y = mx + b"""
        return (y2 - y1) / total_x
    def get_best_fit_vals(y1,y2,total_x = 10):
        m = get_slope(y1,y2,total_x = total_x)
        lst = []
        for i in range(total_x):
            lst.append((m * i) + y1)
        return lst
    def compare_vals(lst,open_list,close_list,total_x):

        for i in range(total_x):
            if lst[i] <= max(open_list[i],close_list[i]) and lst[i] >= min(open_list[i],close_list[i]):
                pass
            else:
                return 0
        return 1
    print('GETTING BEST FIT INDICATOR')
    df['best_fit'] = 0
    o_col = df.columns.get_loc('o')
    c_col = df.columns.get_loc('c')
    new_col = df.columns.get_loc('best_fit')
    
    arr = df.values
    total_x = lookback 
    for i in range(lookback,arr.shape[0]):
        if i % 200000 == 0:
            print(i)

        y1 = np.mean([arr[i - total_x,o_col],arr[i - total_x,c_col]])
        y2 = np.mean([arr[i ,o_col],arr[i ,c_col]])
        
        temp_lst = get_best_fit_vals(y1,y2,total_x)
        arr[i,new_col] = compare_vals(lst = temp_lst,open_list = list(arr[i - total_x:i,o_col]),close_list = list(arr[i - total_x:i,c_col]),total_x = total_x)
            
    
    return pd.DataFrame(arr,columns = df.columns)   

def print_example(df,ind_list,ind = 30,delta = 240,trendline_lookback = 20):
    i = ind_list[ind]   
    print('row loc:',i,' ind:',ind,' delta:',delta)
    fig = go.Figure(data=go.Candlestick(x=df.iloc[i - delta:i + (delta*2),:]['time'],
                        open=df.iloc[i - delta:i + (delta*2),:]['o'],
                        high=df.iloc[i - delta:i + (delta*2),:]['h'],
                        low=df.iloc[i - delta:i + (delta*2),:]['l'],
                        close=df.iloc[i - delta:i + (delta*2),:]['c']))


    fig.add_vrect(x0=df['time'].iloc[i - 1], x1=df['time'].iloc[i+1], 
                  annotation_text="trigger point" , annotation_position="top left",
                  fillcolor="green", opacity=0.25, line_width=0)
    
    fig.add_shape(type='line',
                    x0=df.iloc[i - trendline_lookback,:]['time'],
                    y0=df.iloc[i - trendline_lookback,:]['o'],
                    x1=df.iloc[i,:]['time'],
                    y1=df.iloc[i ,:]['o'],
                    line=dict(color='Red',),
                    xref='x',
                    yref='y'
    )
    
    
    try:
        plot(fig)
    except:
        from plotly.offline import plot
        plot(fig)
def get_max_min_open_or_close(df):
    o_col = df.columns.get_loc('o')
    c_col = df.columns.get_loc('c')
    df['max_o_c'] = 0
    df['min_o_c'] = 0
    max_col = df.columns.get_loc('max_o_c')
    min_col = df.columns.get_loc('min_o_c')    
    arr = df.values
    for i in range(arr.shape[0]):
        arr[i,max_col] = max(arr[i,o_col],arr[i,c_col])
        arr[i,min_col] = min(arr[i,o_col],arr[i,c_col])
    return pd.DataFrame(arr,columns = df.columns)

def channel_indicator(df,lookback = 20):
    """Take the max of a bunch of consecutive candles and find the slope of the line
    Take the min of a bunch of consecutive candles and find the slope of the line
    
    """
    def get_slope(y1,y2,total_x = 10):
        """y = mx + b"""
        return (y2 - y1) / total_x
    def get_best_fit_vals(y1,y2,total_x = 10):
        m = get_slope(y1,y2,total_x = total_x)
        lst = []
        for i in range(total_x):
            lst.append((m * i) + y1)
        return lst
    def compare_vals(lst,check_list,total_x,check_type = 'max'):

        for i in range(total_x):
            if check_type == 'max':
                if lst[i] >= check_list[i]:
                    pass
                else:
                    return 0
            if check_type == 'min':
                if lst[i] <= check_list[i]:
                    pass
                else:
                    return 0
        return 1
    
    
    print('GETTING CHANNEL INDICATOR')
    df['channel_indicator'] = 0
    max_col = df.columns.get_loc('max_o_c')
    min_col = df.columns.get_loc('min_o_c')
    new_col = df.columns.get_loc('channel_indicator')
    df['spread_indicator'] = 0
    spread_col = df.columns.get_loc('spread_indicator')
    
    arr = df.values
    total_x = lookback 
    for i in range(lookback,arr.shape[0]):
        if i % 200000 == 0:
            print(i)

        y1 = arr[i - total_x,max_col]
        y2 = arr[i ,max_col]
        
        temp_lst1 = get_best_fit_vals(y1,y2,total_x)
        check_1 = compare_vals(lst = temp_lst1,
                                      check_list = list(arr[i - total_x:i,min_col]),
                                      total_x = total_x,
                                      check_type = 'max'
                                     )
        
        
        y1 = arr[i - total_x,min_col]
        y2 = arr[i ,min_col]
        
        temp_lst2 = get_best_fit_vals(y1,y2,total_x)
        check_2 = compare_vals(lst = temp_lst2,
                                      check_list = list(arr[i - total_x:i,max_col]),
                                      total_x = total_x,
                                      check_type = 'min'
                                     )   
        spread = np.array(temp_lst1) - np.array(temp_lst2)     
        arr[i,spread_col] = max(spread)
        
        
        if check_1 == 1 and check_2 == 1:
            arr[i,new_col] = 1
    
    return pd.DataFrame(arr,columns = df.columns)  


def get_support(df,lookup_range = 60,stop_range = 20000,lookup_range2 = 200):
    print('GETTING SUPPORT INDICATOR')
    s = time.time()
    c_col = df.columns.get_loc('c')
    l_col = df.columns.get_loc('l')
    df['support_lookup'] = 0
    df['support_indicator'] = 0
    lookup_col = df.columns.get_loc('support_lookup')
    new_col = df.columns.get_loc('support_indicator')
    arr = df.values
    
    for i in range(arr.shape[0]):
        if i % 200000 == 0:
            print(i)
        try:
            if arr[i,c_col] == min(arr[i - lookup_range : i + lookup_range,c_col]):
                lookup_ind = i
                val = arr[i,c_col]
                lookup_check = 0
                for j in range(i + lookup_range2,i + stop_range):
                    if arr[j,l_col] <= val and lookup_check == 0:
                        arr[j,new_col] = 1
                        arr[j,lookup_col] = lookup_ind
                        lookup_check = 1
                        break
        except:
            pass
        

    e = time.time()
    print('TOTAL FUNCTION TIME:',(e-s)/60,' MINUTES')
    df = pd.DataFrame(arr,columns = df.columns)
    print('SHAPE',df[df['support_indicator'] == 1].shape)
    return df
def simulate_results(
    num_trades = 200,
    acc_val = 10000,
    rr = 6,
    risk_size = .02,
    win_rate = .2,
pr = True):
    start = acc_val
    for i in range(num_trades):
        rand = random()
        if rand > win_rate:
            acc_val = acc_val - (acc_val * risk_size)
        else:
            acc_val = acc_val + (acc_val * (risk_size * rr))
    if pr:
        print('STARTING VAL:',round(start),' ENDING VAL:',round(acc_val))
    return round(acc_val)


def strategy_tester_buy(df,col = 'support_indicator',sl = .1,tp = .5,num_trades = 300,risk_size = .01,entry = 'c',pr = True):
    print()
    print('RUNNING BUY SIMULATOR','SL',sl,'TP',tp)
    col_num = df.columns.get_loc(col)
    h_col = df.columns.get_loc('h')
    l_col = df.columns.get_loc('l')
    c_col = df.columns.get_loc('c')
    entry_col = df.columns.get_loc(entry)
    arr = df.values
    temp_df = df[df[col] == 1]
    print('SHAPE:',temp_df.shape)    
    trade_list = []
    trade_res = []
    wins = []
    losses = []
    for i in range(arr.shape[0]):

        if arr[i,col_num] == 1:
            val = arr[i,entry_col]
            try:
                for j in range(i + 1,i + 10000):
                    if i not in trade_list:
                        if arr[j,l_col] < val - sl:
                            trade_res.append(-10)
                            trade_list.append(i)
                            losses.append(i)
                            break
                        if arr[j,h_col] >= val + tp:
                            trade_res.append(50)
                            trade_list.append(i)
                            wins.append(i)
                            break
            except:
                pass
    win_rate = len(wins) / len(trade_list)
    rr = tp / sl
   # num_trades = len(trade_list)
  #  print('WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
    vals = []


    acc_val = 10000



    for i in range(30):

        final_val = simulate_results(num_trades = num_trades,acc_val = acc_val,rr = rr,risk_size = risk_size,win_rate = win_rate,pr = pr)
        vals.append(final_val)


    print('MEDIAN SIM:',np.median(vals),'AVG SIM:',sum(vals) / len(vals),'WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
    return np.median(vals),win_rate


def strategy_tester_sell(df,col = 'support_indicator',sl = .1,tp = .5,num_trades = 300,risk_size = .01,entry = 'c',pr = True):
    print()
    print('RUNNING SELL SIMULATOR','SL',sl,'TP',tp)
    col_num = df.columns.get_loc(col)
    h_col = df.columns.get_loc('h')
    l_col = df.columns.get_loc('l')
    c_col = df.columns.get_loc('c')
    entry_col = df.columns.get_loc(entry)
    arr = df.values
    temp_df = df[df[col] == 1]
    print('SHAPE:',temp_df.shape)    
    trade_list = []
    trade_res = []
    wins = []
    losses = []
    for i in range(arr.shape[0]):

        if arr[i,col_num] == 1:
            val = arr[i,entry_col]
            for j in range(i + 1,i + 10000):
                if i not in trade_list:
                    if arr[j,h_col] > val + sl:
                        trade_res.append(-10)
                        trade_list.append(i)
                        losses.append(i)
                        break
                    if arr[j,l_col] <= val - tp:
                        trade_res.append(50)
                        trade_list.append(i)
                        wins.append(i)
                        break
    win_rate = len(wins) / len(trade_list)
    rr = tp / sl
   # num_trades = len(trade_list)
  #  print('WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
    vals = []


    acc_val = 10000



    for i in range(30):

        final_val = simulate_results(num_trades = num_trades,acc_val = acc_val,rr = rr,risk_size = risk_size,win_rate = win_rate,pr = pr)
        vals.append(final_val)


    print('MEDIAN SIM:',np.median(vals),'AVG SIM:',sum(vals) / len(vals),'WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
    return np.median(vals),win_rate

def strategy_tester_buy(df,col = 'support_indicator',sl = .1,tp = .5,num_trades = 300,risk_size = .01,entry = 'c',pr = True):
    print()
    print('RUNNING BUY SIMULATOR','SL',sl,'TP',tp)
    col_num = df.columns.get_loc(col)
    h_col = df.columns.get_loc('h')
    l_col = df.columns.get_loc('l')
    c_col = df.columns.get_loc('c')
    entry_col = df.columns.get_loc(entry)
    arr = df.values
    temp_df = df[df[col] == 1]
    print('SHAPE:',temp_df.shape)    
    trade_list = []
    trade_res = []
    wins = []
    losses = []
    for i in range(arr.shape[0]):

        if arr[i,col_num] == 1:
            val = arr[i,entry_col]
            trade_id = 0
            try:
                for j in range(i + 1,i + 10000):
                    if i not in trade_list:
                        if trade_id == 0:
                            if arr[j,l_col] < val - sl:
                                trade_res.append(-10)
                                trade_list.append(i)
                                losses.append(i)
                                trade_id = 1

                                break
                            if arr[j,h_col] >= val + tp:
                                trade_res.append(50)
                                trade_list.append(i)
                                wins.append(i)
                                trade_id = 1
                                break
            except:
                pass
    win_rate = len(wins) / len(trade_list)
    rr = tp / sl
   # num_trades = len(trade_list)
  #  print('WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
    vals = []


    acc_val = 10000



    for i in range(30):

        final_val = simulate_results(num_trades = num_trades,acc_val = acc_val,rr = rr,risk_size = risk_size,win_rate = win_rate,pr = pr)
        vals.append(final_val)


    print('MEDIAN SIM:',np.median(vals),'AVG SIM:',sum(vals) / len(vals),'WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
    return np.median(vals),win_rate,wins,losses

def strategy_tester_sell(df,col = 'support_indicator',sl = .1,tp = .5,num_trades = 300,risk_size = .01,entry = 'c',pr = True):
    print()
    print('RUNNING SELL SIMULATOR','SL',sl,'TP',tp)
    col_num = df.columns.get_loc(col)
    h_col = df.columns.get_loc('h')
    l_col = df.columns.get_loc('l')
    c_col = df.columns.get_loc('c')
    entry_col = df.columns.get_loc(entry)
    arr = df.values
    temp_df = df[df[col] == 1]
    print('SHAPE:',temp_df.shape)    
    trade_list = []
    trade_res = []
    wins = []
    losses = []
    for i in range(arr.shape[0]):

        if arr[i,col_num] == 1:
            val = arr[i,entry_col]
            try:
                for j in range(i + 1,i + 10000):
                    if i not in trade_list:
                        if arr[j,h_col] > val + sl:
                            trade_res.append(-10)
                            trade_list.append(i)
                            losses.append(i)
                            break
                        if arr[j,l_col] <= val - tp:
                            trade_res.append(50)
                            trade_list.append(i)
                            wins.append(i)
                            break
            except:
                pass
    win_rate = len(wins) / len(trade_list)
    rr = tp / sl
   # num_trades = len(trade_list)
  #  print('WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
    vals = []


    acc_val = 10000



    for i in range(30):

        final_val = simulate_results(num_trades = num_trades,acc_val = acc_val,rr = rr,risk_size = risk_size,win_rate = win_rate,pr = pr)
        vals.append(final_val)


    print('MEDIAN SIM:',np.median(vals),'AVG SIM:',sum(vals) / len(vals),'WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
    return np.median(vals),win_rate,wins,losses
def ema(df,num = 14):
    print('GETTING EMA INDICATOR FOR:',num)
    close_col = df.columns.get_loc('c')
    df['ema_' + str(num)] = df['c']
    arr = df.values
    mult = 2/ (num + 1)
    for i in range(num,arr.shape[0]):
        sma = sum(arr[i - num + 1: i + 1,close_col]) / num
        arr[i,-1] = ((arr[i,close_col] - arr[i - 1,-1]) * mult) + arr[i - 1,-1]
    return pd.DataFrame(arr,columns = df.columns)
def strategy_tester_buy(df,col = 'support_indicator',sl = .1,tp = .5,num_trades = 300,risk_size = .01,entry = 'c',pr = True):
    print()
    print('RUNNING BUY SIMULATOR','SL',sl,'TP',tp)
    col_num = df.columns.get_loc(col)
    h_col = df.columns.get_loc('h')
    l_col = df.columns.get_loc('l')
    c_col = df.columns.get_loc('c')
    entry_col = df.columns.get_loc(entry)
    arr = df.values
    temp_df = df[df[col] == 1]
    print('SHAPE:',temp_df.shape)    
    trade_list = []
    trade_res = []
    wins = []
    losses = []
    for i in range(arr.shape[0]):

        if arr[i,col_num] == 1:
            val = arr[i,entry_col]
            trade_id = 0
            try:
                for j in range(i + 1,i + 10000):
                    if i not in trade_list:
                        if trade_id == 0:
                            if arr[j,l_col] < val - sl:
                                trade_res.append(-10)
                                trade_list.append(i)
                                losses.append(i)
                                trade_id = 1

                                break
                            if arr[j,h_col] >= val + tp:
                                trade_res.append(50)
                                trade_list.append(i)
                                wins.append(i)
                                trade_id = 1
                                break
            except:
                pass
    win_rate = len(wins) / len(trade_list)
    rr = tp / sl
   # num_trades = len(trade_list)
  #  print('WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
    vals = []


    acc_val = 10000



    for i in range(30):

        final_val = simulate_results(num_trades = num_trades,acc_val = acc_val,rr = rr,risk_size = risk_size,win_rate = win_rate,pr = pr)
        vals.append(final_val)


    print('MEDIAN SIM:',np.median(vals),'AVG SIM:',sum(vals) / len(vals),'WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
    return np.median(vals),win_rate,wins,losses

def strategy_tester_sell(df,col = 'support_indicator',sl = .1,tp = .5,num_trades = 300,risk_size = .01,entry = 'c',pr = True):
    print()
    print('RUNNING SELL SIMULATOR','SL',sl,'TP',tp)
    col_num = df.columns.get_loc(col)
    h_col = df.columns.get_loc('h')
    l_col = df.columns.get_loc('l')
    c_col = df.columns.get_loc('c')
    entry_col = df.columns.get_loc(entry)
    arr = df.values
    temp_df = df[df[col] == 1]
    print('SHAPE:',temp_df.shape)    
    trade_list = []
    trade_res = []
    wins = []
    losses = []
    for i in range(arr.shape[0]):

        if arr[i,col_num] == 1:
            val = arr[i,entry_col]
            try:
                for j in range(i + 1,i + 10000):
                    if i not in trade_list:
                        if arr[j,h_col] > val + sl:
                            trade_res.append(-10)
                            trade_list.append(i)
                            losses.append(i)
                            break
                        if arr[j,l_col] <= val - tp:
                            trade_res.append(50)
                            trade_list.append(i)
                            wins.append(i)
                            break
            except:
                pass
    win_rate = len(wins) / len(trade_list)
    rr = tp / sl
   # num_trades = len(trade_list)
  #  print('WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
    vals = []


    acc_val = 10000



    for i in range(30):

        final_val = simulate_results(num_trades = num_trades,acc_val = acc_val,rr = rr,risk_size = risk_size,win_rate = win_rate,pr = pr)
        vals.append(final_val)


    print('MEDIAN SIM:',np.median(vals),'AVG SIM:',sum(vals) / len(vals),'WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
    return np.median(vals),win_rate,wins,losses


def add_delta_cols(df):
    #df['delta_vector'] = 0
    #df['delta_max_vector'] = 0

    df['delta_vector'] = 0
    df['delta_max'] = 0
    df['upper_wick'] = 0
    df['lower_wick'] = 0
    
    dv_col = df.columns.get_loc('delta_vector')
    dm_col = df.columns.get_loc('delta_max')
    uw_col = df.columns.get_loc('upper_wick')
    lw_col = df.columns.get_loc('lower_wick')
    
    o_col = df.columns.get_loc('o')
    h_col = df.columns.get_loc('h')
    l_col = df.columns.get_loc('l')
    c_col = df.columns.get_loc('c')
    arr = df.values
    for i in range(arr.shape[0]):
        arr[i,dv_col] = arr[i,c_col] - arr[i,o_col]
        arr[i,dm_col] = arr[i,h_col] - arr[i,l_col]
        if arr[i,c_col] > arr[i,o_col]:
            arr[i,uw_col] = arr[i,h_col] - arr[i,c_col]
            arr[i,lw_col] = arr[i,o_col] - arr[i,l_col]
            
        elif arr[i,c_col] < arr[i,o_col]:
            arr[i,uw_col] = arr[i,h_col] - arr[i,o_col]
            arr[i,lw_col] = arr[i,c_col] - arr[i,l_col]   
            
    df = pd.DataFrame(arr,columns = df.columns)
    df['delta'] = abs(df['delta_vector'])
    
    return df
def load_df(pair = 'EUR_USD'
            ,granularity = 'M5'
            ,start = datetime(2016,1,1,0,0,0)
            ,end = datetime(2022,7,31,0,0,0)):
    dir_name = os.getcwd() + '/' + pair + '_' + granularity
    path = dir_name + '/' + str(date(start.year,start.month,start.day)) + '_' + str(date(end.year,end.month,end.day)) + '.csv'
    print('PATH:',path)
    df = pd.read_csv(path)
    df = add_delta_cols(df)
    print(df.shape)
    return df

def simulate_results(
    num_trades = 200,
    acc_val = 10000,
    rr = 6,
    risk_size = .02,
    win_rate = .2,
    pr = True):
    
    start = acc_val
    for i in range(num_trades):
        rand = random.random()
        if rand > win_rate:
            acc_val = acc_val - (acc_val * risk_size)
        else:
            acc_val = acc_val + (acc_val * (risk_size * rr))
    if pr:
        print('STARTING VAL:',round(start),' ENDING VAL:',round(acc_val))
    return round(acc_val)


def run_single_parameter_sim_loop(df,
                                  pair,
                    lookup_range,
                   bullish_ma ,
                    bullish_candle ,
                    candle_size ,
                delta_filter ,
                delta_filter2 ,
                upper_wick ,
                lower_wick ,
                 support ):
    
    def strategy_tester_buy(df,col = 'support_indicator',sl = .1,tp = .5,num_trades = 300,risk_size = .01,entry = 'c',pr = True):

        col_num = df.columns.get_loc(col)
        h_col = df.columns.get_loc('h')
        l_col = df.columns.get_loc('l')
        c_col = df.columns.get_loc('c')
        entry_col = df.columns.get_loc(entry)
        arr = df.values
        temp_df = df[df[col] == 1]

        trade_list = []
        trade_res = []
        wins = []
        losses = []
        for i in range(arr.shape[0]):

            if arr[i,col_num] == 1:
                val = arr[i,entry_col]
                trade_id = 0
                try:
                    for j in range(i + 1,i + 20000):
                        if i not in trade_list:
                            if trade_id == 0:
                                if arr[j,l_col] <= val - sl:
                                    trade_res.append(-10)
                                    trade_list.append(i)
                                    losses.append(i)
                                    trade_id = 1

                                    break
                                elif arr[j,h_col] >= val + tp:
                                    trade_res.append(50)
                                    trade_list.append(i)
                                    wins.append(i)
                                    trade_id = 1
                                    break
                except:
                    pass
        if len(trade_list) == 0:
            print('NO TRADES, SHAPE IS 0:')
            return 0,0,0,0

        win_rate = len(wins) / len(trade_list)
        rr = tp / sl
        vals = []


        acc_val = 10000

        for i in range(30):

            final_val = simulate_results(num_trades = num_trades,acc_val = acc_val,rr = rr,risk_size = risk_size,win_rate = win_rate,pr = pr)
            vals.append(final_val)


        #print('MEDIAN SIM:',np.median(vals),'AVG SIM:',sum(vals) / len(vals),'SL',sl,'TP',tp,'WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
        return np.median(vals),win_rate,wins,losses

    def strategy_tester_sell(df,col = 'support_indicator',sl = .1,tp = .5,num_trades = 300,risk_size = .01,entry = 'c',pr = True):

        col_num = df.columns.get_loc(col)
        h_col = df.columns.get_loc('h')
        l_col = df.columns.get_loc('l')
        c_col = df.columns.get_loc('c')
        entry_col = df.columns.get_loc(entry)
        arr = df.values
        temp_df = df[df[col] == 1]

        trade_list = []
        trade_res = []
        wins = []
        losses = []
        for i in range(arr.shape[0]):

            if arr[i,col_num] == 1:
                val = arr[i,entry_col]
                try:
                    for j in range(i + 1,i + 20000):
                        if i not in trade_list:
                            if arr[j,h_col] >= val + sl:
                                trade_res.append(-10)
                                trade_list.append(i)
                                losses.append(i)
                                break
                            elif arr[j,l_col] <= val - tp:
                                trade_res.append(50)
                                trade_list.append(i)
                                wins.append(i)
                                break
                except:
                    pass
        if len(trade_list) == 0:
            print('NO TRADES, SHAPE IS 0:')
            return 0,0,0,0
        win_rate = len(wins) / len(trade_list)
        rr = tp / sl
        vals = []

        acc_val = 10000

        for i in range(30):

            final_val = simulate_results(num_trades = num_trades,acc_val = acc_val,rr = rr,risk_size = risk_size,win_rate = win_rate,pr = pr)
            vals.append(final_val)


        #print('MEDIAN SIM:',np.median(vals),'AVG SIM:',sum(vals) / len(vals),'SL',sl,'TP',tp,'WIN RATE',win_rate,' RR:',rr, 'NUM TRADES:',num_trades,' RISK SIZE:',risk_size)
        return np.median(vals),win_rate,wins,losses
    
    def ma_strategy(df,
                        lookup_range = 1500,
                       bullish_ma = True,
                        bullish_candle = True,
                        candle_size = .001,
                    delta_filter = .001,
                    delta_filter2 = .006,
                    upper_wick = .0001,
                    lower_wick = .0001,
                    support = True
                       ):               

        s = time.time()
        c_col = df.columns.get_loc('c')
        o_col = df.columns.get_loc('o')
        l_col = df.columns.get_loc('l')
        h_col = df.columns.get_loc('h')
        uw_col = df.columns.get_loc('upper_wick')
        lw_col = df.columns.get_loc('lower_wick')
        dv_col = df.columns.get_loc('delta_vector')
        dm_col = df.columns.get_loc('delta_max')
        d_col = df.columns.get_loc('delta')
        ma1_col = df.columns.get_loc('smma_21')
        ma2_col = df.columns.get_loc('smma_50')
        ma3_col = df.columns.get_loc('smma_200')   



        df['ma_indicator'] = 0
        new_col = df.columns.get_loc('ma_indicator')

        trading_window_count = 0
        between_s_and_r_count = 0          
        resistance_delta_filter_count = 0           
        support_delta_filter_count = 0 
        resistance_delta_filter2_count = 0            
        support_delta_filter2_count = 0
        ma_indicator_count = 0            
        candle_size_count = 0             
        upper_wick_count = 0 
        lower_wick_count = 0
        bullish_candle_count = 0            
        sum_count = 0

        arr = df.values
        min_count = 0
        for i in range(lookup_range,arr.shape[0]):
            if i % 100000 == 0:
                print(i)
            #DEFINE SUPPORT/RESISTANCE as the MIN or MAX of a lookup range
            max_ = max(arr[i - lookup_range : i,c_col])
            min_ = min(arr[i - lookup_range : i,c_col])
            resistance_delta = max_ - arr[i,c_col]
            support_delta = arr[i,c_col] - min_ 
            pip_range = max_ - min_
            ma_indicator = 0
            if bullish_ma == True:
                if arr[i,ma1_col] > arr[i,ma2_col] and arr[i,ma2_col] > arr[i,ma3_col]:
                    ma_indicator = 1
                else:
                    ma_indicator = 0
            elif bullish_ma == False:
                if arr[i,ma1_col] < arr[i,ma2_col] and arr[i,ma2_col] < arr[i,ma3_col]:
                    ma_indicator = 1
                else:
                    ma_indicator = 0    

            #CANDLE BULL OR BEAR
            if arr[i,c_col] >= arr[i,o_col]:
                bull = True
            else:
                bull = False      

            #Within the delta range:
            if support == True:
                if support_delta >= delta_filter and support_delta <= delta_filter + delta_filter2:
                    near_reversal_indicator = 1
                else:
                    near_reversal_indicator = 0
            else:
                if resistance_delta >= delta_filter and resistance_delta <= delta_filter + delta_filter2:
                    near_reversal_indicator = 1
                else:
                    near_reversal_indicator = 0            

            #within support and resistance range:
            if delta_filter < 0:
                if arr[i,c_col] >= min_ - abs(delta_filter) and arr[i,c_col] <= max_ + abs(delta_filter):
                    between_s_and_r = 1
                else:
                    between_s_and_r = 0            
            else:
                if arr[i,c_col] >= min_ and arr[i,c_col] <= max_:
                    between_s_and_r = 1
                else:
                    between_s_and_r = 0

            if between_s_and_r and \
            near_reversal_indicator == 1 and \
            ma_indicator == 1 and \
            arr[i,d_col] >= candle_size and \
            arr[i,uw_col] >= upper_wick and \
            arr[i,lw_col] >= lower_wick and \
            bull == bullish_candle and \
            sum(arr[i-24:i,new_col]) == 0:
                arr[i,new_col] = 1



        e = time.time()
        df = pd.DataFrame(arr,columns = df.columns)
        print('SHAPE',df[df['ma_indicator'] == 1].shape)

        return df 





    cols = ['instrument',
                    'timeframe',
                    'buy_or_sell',
                    'trade_strategy',
                    'ending_val',
                    'starting_val',
                    'sl',
                    'tp',
                    'RR',
                    'win_rate',
                    'num_trades',
                    'risk_size',
                    'actual_number_of_trades',
            'lookup_range','bullish_ma','bullish_candle','support','candle_size','delta_filter','delta_filter2','upper_wick','lower_wick'
           ]





    df = ma_strategy(df,
                        lookup_range = lookup_range,
                       bullish_ma = bullish_ma,
                        bullish_candle = bullish_candle,
                        candle_size = candle_size,
                    delta_filter = delta_filter,
                    delta_filter2 = delta_filter2,
                    upper_wick = upper_wick,
                    lower_wick = lower_wick,
                     support = support
                       )    


    s = time.time()
    print('RUNNING SINGLE PARAMETER LOOP FOR PARAMS:',
          'lookup_range', lookup_range,
                       'bullish_ma', bullish_ma,
                        'bullish_candle', bullish_candle,
                        'candle_size',candle_size,
                    'delta_filter', delta_filter,
                    'delta_filter2', delta_filter2,
                    'upper_wick', upper_wick,
                    'lower_wick',lower_wick,
                     'support', support)
    

    df = ma_strategy(df,
                        lookup_range = lookup_range,
                       bullish_ma = bullish_ma,
                        bullish_candle = bullish_candle,
                        candle_size = candle_size,
                    delta_filter = delta_filter,
                    delta_filter2 = delta_filter2,
                    upper_wick = upper_wick,
                    lower_wick = lower_wick,
                     support = support
                       )    


    sls = [.0005,.00075,.001,.00125,.0015,.002,.0025]
    tps = [.001,.0015,.002,.0025,.003,.004,.005,.007,.009] 
    if 'JPY' in pair:
        sls = [.05,.075,.1,.125,.15,.2,.25]
        tps = [.1,.15,.2,.25,.3,.4,.5,.7,.9]          
    trade_strategy = 'ma_indicator'
    instrument = pair
    timeframe = 'M5'
    num_trades = 400
    risk_size = .01
    starting_val = 10000
    temp_lst = []
    actual_number_of_trades = df[df[trade_strategy] == 1].shape[0]
    for sl in sls:
        for tp in tps:
            ending_val,win_rate,wins,losses = strategy_tester_buy(df,col = trade_strategy,sl = sl,tp = tp,num_trades = num_trades,risk_size = risk_size,entry = 'c',pr = False)
            lst = [instrument,
                    timeframe,
                    'buy',
                    trade_strategy,
                    ending_val,
                    starting_val,
                    sl,
                    tp,
                    round(tp / sl,4),
                    round(win_rate,4),
                    num_trades,
                    risk_size,
                    actual_number_of_trades,
                   lookup_range,bullish_ma,bullish_candle,support,candle_size,delta_filter,delta_filter2,upper_wick,lower_wick

            ]

            temp_lst.append(lst)
            ending_val,win_rate,wins,losses = strategy_tester_sell(df,col = trade_strategy,sl = sl,tp = tp,num_trades = num_trades,risk_size = risk_size,entry = 'c',pr = False)


            lst = [instrument,
                    timeframe,
                    'sell',
                    trade_strategy,
                    ending_val,
                    starting_val,
                    sl,
                    tp,
                    round(tp / sl,4),
                    round(win_rate,4),
                    num_trades,
                    risk_size,
                    actual_number_of_trades,
                   lookup_range,bullish_ma,bullish_candle,support,candle_size,delta_filter,delta_filter2,upper_wick,lower_wick

            ]

            temp_lst.append(lst)






    final_df = pd.DataFrame(temp_lst,columns = cols)
    final_df = final_df.sort_values(by = 'ending_val',ascending = False)
    

    overall_best = final_df.sort_values(by = 'ending_val',ascending = False)


    print('OVERALL BEST BUY/VAL/WR/RR/TP/SL/NUM_TRADES:',overall_best['buy_or_sell'].iloc[0],'/',overall_best['ending_val'].iloc[0],'/',overall_best['win_rate'].iloc[0],'/',overall_best['RR'].iloc[0],'/',overall_best['tp'].iloc[0],'/',overall_best['sl'].iloc[0],'/',overall_best['actual_number_of_trades'].iloc[0]
         )


    e = time.time()
    print('TOTAL PARAM SIM LOOP TIME:',round((e-s)/60,2),'MINUTES')
    return final_df


def run_sim(pair,max_loops):
    def generate_random_param_list():
        def return_random_list_element(lst):
            return lst[random.randint(0,len(lst) - 1)]
        # param loop
        # PARAM DEFINITON
        if 'JPY' in pair:
            candle_sizes = [.005,.025,.05,.075,.1,.125,.15]
            delta_filters = [0,.05,.1,.15,.2,.3,.4]
            delta_filters2 = [.05,.1,.15,.2,.3,.4,.5,.7]
            upper_wicks = [0,0,0,0,.02,.04,.06]
            lower_wicks = [0,0,0,0,.02,.04,.06]
        else:
            candle_sizes = [.00005,.00025,.0005,.00075,.001,.00125,.0015]
            delta_filters = [0,.0005,.001,.0015,.002,.003,.004]
            delta_filters2 = [.0005,.001,.0015,.002,.003,.004,.005,.007]
            upper_wicks = [0,0,0,0,0.0002,.0004,.0006]
            lower_wicks = [0,0,0,0,.0002,.0004,.0006]



        lookup_ranges = [150,250,400,500,750,1000]
        bullish_mas = [True,False]
        bullish_candles = [True,False]
        supports = [True,False]        

        lookup_range = return_random_list_element(lookup_ranges)
        bullish_ma = return_random_list_element(bullish_mas)
        bullish_candle = return_random_list_element(bullish_candles)
        support = return_random_list_element(supports)
        candle_size = return_random_list_element(candle_sizes)
        delta_filter = return_random_list_element(delta_filters)
        delta_filter2 = return_random_list_element(delta_filters2)
        upper_wick = return_random_list_element(upper_wicks)
        lower_wick = return_random_list_element(lower_wicks)

        param_check_list = [lookup_range,bullish_ma,bullish_candle,support,candle_size,delta_filter,delta_filter2,upper_wick,lower_wick]

        return param_check_list,lookup_range,bullish_ma,bullish_candle,support,candle_size,delta_filter,delta_filter2,upper_wick,lower_wick


    print('RUNNING SIM...')
    file = pair + '_M5_2016-01-01_2022-01-31.csv'
    df = load_df(pair = pair,granularity ='M5',start = datetime(2016,1,1,0,0,0),end = datetime(2022,7,31,0,0,0))
    df_ = df.copy()
    file_name = 'SIM_CLEAN_MA_' + file
    file_name2 = 'CHECK_PARAMS_MA_' + file
    print(file_name,file_name2)
    try:
        final_df = pd.read_csv(file_name)
        loop_number = final_df['loop_number'].max()

        param_check_df = pd.read_csv(file_name2)
        param_check_lists = param_check_df.values.tolist()

    except:
        final_df = pd.DataFrame([0])
        param_check_df = pd.DataFrame([0])
        loop_number = 0
        param_check_lists = []
    
    while max_loops >= loop_number:
        checker = 0
        while checker == 0:
            param_check_list,lookup_range,bullish_ma,bullish_candle,support,candle_size,delta_filter,delta_filter2,upper_wick,lower_wick = generate_random_param_list()

            if param_check_list not in param_check_lists:
                param_check_lists.append(param_check_list)

                checker = 1
        try:
            if final_df.shape[0] == 1:
                loop_number = 1
            else:
                loop_number = final_df['loop_number'].max() + 1

            print('LOOP NUMBER:',loop_number)
            loop_df = run_single_parameter_sim_loop(df,
                                                    pair = pair,
                    lookup_range = lookup_range,
                   bullish_ma = bullish_ma,
                    bullish_candle = bullish_candle,
                    candle_size = candle_size,
                delta_filter = delta_filter,
                delta_filter2 = delta_filter2,
                upper_wick = upper_wick,
                lower_wick = lower_wick,
                 support = support)
            
            loop_df['loop_number'] = loop_number

            if final_df.shape[0] == 1:
                final_df = loop_df

            else:
                final_df = final_df.append(loop_df)

            final_df.to_csv(file_name,index = False)
            param_check_df = pd.DataFrame(param_check_lists)
            param_check_df.to_csv(file_name2,index = False)
       
        except:
            print('ERROR WITH THESE PARAMETERS... SKIPPING')


        try:
            overall_best = final_df[final_df['actual_number_of_trades'] >= 80].sort_values(by = 'ending_val',ascending = False)


            print('OVERALL BEST BUY/VAL/WR/RR/TP/SL/NUM_TRADES:',overall_best['buy_or_sell'].iloc[0],'/',overall_best['ending_val'].iloc[0],'/',overall_best['win_rate'].iloc[0],'/',overall_best['RR'].iloc[0],'/',overall_best['tp'].iloc[0],'/',overall_best['sl'].iloc[0],'/',overall_best['actual_number_of_trades'].iloc[0]
                 )
        except:
            print('NO RESULTS')
 
        print()
        print()
        print()
        
        
def run_sim_(pair,max_loops):
    def generate_random_param_list():
        def return_random_list_element(lst):
            return lst[random.randint(0,len(lst) - 1)]
        # param loop
        # PARAM DEFINITON
        if 'JPY' in pair:
            candle_sizes = [.005,.025,.05,.075,.1,.125,.15]
            delta_filters = [0,.05,.1,.15,.2,.3,.4]
            delta_filters2 = [.05,.1,.15,.2,.3,.4,.5,.7]
            upper_wicks = [0,0,0,0,.02,.04,.06]
            lower_wicks = [0,0,0,0,.02,.04,.06]
        else:
            candle_sizes = [.00005,.00025,.0005,.00075,.001,.00125,.0015]
            delta_filters = [0,.0005,.001,.0015,.002,.003,.004]
            delta_filters2 = [.0005,.001,.0015,.002,.003,.004,.005,.007]
            upper_wicks = [0,0,0,0,0.0002,.0004,.0006]
            lower_wicks = [0,0,0,0,.0002,.0004,.0006]



        lookup_ranges = [150,250,400,500,750,1000]
        bullish_mas = [True,False]
        bullish_candles = [True,False]
        supports = [True,False]        

        lookup_range = return_random_list_element(lookup_ranges)
        bullish_ma = return_random_list_element(bullish_mas)
        bullish_candle = return_random_list_element(bullish_candles)
        support = return_random_list_element(supports)
        candle_size = return_random_list_element(candle_sizes)
        delta_filter = return_random_list_element(delta_filters)
        delta_filter2 = return_random_list_element(delta_filters2)
        upper_wick = return_random_list_element(upper_wicks)
        lower_wick = return_random_list_element(lower_wicks)

        param_check_list = [lookup_range,bullish_ma,bullish_candle,support,candle_size,delta_filter,delta_filter2,upper_wick,lower_wick]

        return param_check_list,lookup_range,bullish_ma,bullish_candle,support,candle_size,delta_filter,delta_filter2,upper_wick,lower_wick


    print('RUNNING SIM...')
    file = pair + '_M5_2016-01-01_2022-01-31.csv'
    df = load_df(pair = pair,granularity ='M5',start = datetime(2016,1,1,0,0,0),end = datetime(2022,7,31,0,0,0))
    df_ = df.copy()
    file_name = 'SIM_CLEAN_MA_' + file
    file_name2 = 'CHECK_PARAMS_MA_' + file
    print(file_name,file_name2)
    try:
        final_df = pd.read_csv(file_name)
        loop_number = final_df['loop_number'].max()

        param_check_df = pd.read_csv(file_name2)
        param_check_lists = param_check_df.values.tolist()

    except:
        final_df = pd.DataFrame([0])
        param_check_df = pd.DataFrame([0])
        loop_number = 0
        param_check_lists = []
    
    while max_loops >= loop_number:
        checker = 0
        while checker == 0:
            param_check_list,lookup_range,bullish_ma,bullish_candle,support,candle_size,delta_filter,delta_filter2,upper_wick,lower_wick = generate_random_param_list()

            if param_check_list not in param_check_lists:
                param_check_lists.append(param_check_list)

                checker = 1

        if final_df.shape[0] == 1:
            loop_number = 1
        else:
            loop_number = final_df['loop_number'].max() + 1

        print('LOOP NUMBER:',loop_number)
        loop_df = run_single_parameter_sim_loop(df,
                                                pair = pair,
                lookup_range = lookup_range,
               bullish_ma = bullish_ma,
                bullish_candle = bullish_candle,
                candle_size = candle_size,
            delta_filter = delta_filter,
            delta_filter2 = delta_filter2,
            upper_wick = upper_wick,
            lower_wick = lower_wick,
             support = support)

        loop_df['loop_number'] = loop_number

        if final_df.shape[0] == 1:
            final_df = loop_df

        else:
            final_df = final_df.append(loop_df)

        final_df.to_csv(file_name,index = False)
        param_check_df = pd.DataFrame(param_check_lists)
        param_check_df.to_csv(file_name2,index = False)


        print('ERROR WITH THESE PARAMETERS... SKIPPING')


        try:
            overall_best = final_df[final_df['actual_number_of_trades'] >= 80].sort_values(by = 'ending_val',ascending = False)


            print('OVERALL BEST BUY/VAL/WR/RR/TP/SL/NUM_TRADES:',overall_best['buy_or_sell'].iloc[0],'/',overall_best['ending_val'].iloc[0],'/',overall_best['win_rate'].iloc[0],'/',overall_best['RR'].iloc[0],'/',overall_best['tp'].iloc[0],'/',overall_best['sl'].iloc[0],'/',overall_best['actual_number_of_trades'].iloc[0]
                 )
        except:
            print('NO RESULTS')
 
        print()
        print()
        print()