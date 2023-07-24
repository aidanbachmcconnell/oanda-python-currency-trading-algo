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


import os
import io
from io import StringIO 
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, timezone
import time 

#import datedelta
import calendar
import  csv
import json

from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.pricing import PricingStream
import numpy as np
import datetime
from scipy import stats
import json
from oandapyV20.contrib.requests import MarketOrderRequest
from oandapyV20.contrib.requests import TakeProfitDetails, StopLossDetails

def convert_to_simple_df(instrument,first = True,granularity = "H1",count = 5000,start = date(2020,1,1),end = date(2020,1,10)):
    #print(instrument,granularity,count)
    if first:
        params = {"count":count,
              "granularity": granularity
                 }
    else:
        params = {#"count":count,
              "granularity": granularity,
              "from": str(start.isoformat('T')),
              "to": str(end.isoformat('T'))
                 }

    r = instruments.InstrumentsCandles(instrument=instrument,
                                      params=params)
    
    access_token='2ad78612c58e604890fd961550f73cfd-28190cc09d5cbc67b2c14503575d6132'
    
    api = API(access_token)
    client=API(access_token)
    client.request(r)
    dct = r.response
    lst = dct['candles']
    if len(lst) == 0:
        print('NO RESULT RETRUNING EMPTY DF...')
        return pd.DataFrame([0])
    lst = []
    i = 0
    
    for candle in dct['candles']:

        lst.append([candle['time'],
                    candle['volume'],
                    float(candle['mid']['o']),
                    float(candle['mid']['h']),
                    float(candle['mid']['l']),
                    float(candle['mid']['c'])
                   ])
        
 
        i += 1


    df = pd.DataFrame(lst,columns = ['time','volume','o','h','l','c'])
    
    df['utc_timestamp'] = pd.to_datetime(df['time'], format = '%Y-%m-%d %H:%M:%S', errors ='coerce')
    df['est_timestamp'] = df['utc_timestamp'].dt.tz_convert('US/Eastern')   
   # print('ACTUAL SHAPE:',df.shape,' MIN UTC:',df['utc_timestamp'].min(),' MAX UTC:',df['utc_timestamp'].max())
    return df
def add_feature_cols(df):
    
    def smma(df,period = 14):
        close_col = df.columns.get_loc('c')
        df['smma_' + str(period)] = df['c']
        smma_col = df.columns.get_loc('smma_' + str(period))
        arr = df.values

        #SMMA CALC:

        #SUM1=SUM (CLOSE, N)

        #SMMA1 = SUM1/ N

        #The second and subsequent moving averages are calculated according to this formula:

        #SMMA (i) = (SUM1 – SMMA1+CLOSE (i))/ N    

       # Where:

        #SUM1 – is the total sum of closing prices for N periods;
        #SMMA1 – is the smoothed moving average of the first bar;
        #SMMA (i) – is the smoothed moving average of the current bar (except the first one);
        #CLOSE (i) – is the current closing price;
        #N – is the smoothing period.    

        for i in range(period,arr.shape[0]):
            if i == period:
                sum1 = sum(arr[:i,close_col])

                smma1 = sum1 / period

                arr[i,smma_col] = smma1
            elif i == period + 1:
                arr[i,smma_col] = (smma1 * (period - 1) + arr[i,close_col]) / period

            else:

                prev_sum = arr[i - 1,smma_col] * period
                arr[i,smma_col] = (prev_sum - arr[i - 1,smma_col] + arr[i,close_col]) / period


        return pd.DataFrame(arr,columns = df.columns)    
    s = time.time()
    
    df['o'] = df['o'].astype(float)
    df['h'] = df['h'].astype(float)
    df['l'] = df['l'].astype(float)
    df['c'] = df['c'].astype(float)
    
    df = smma(df,period = 21)
    df = smma(df,period = 50)
    df = smma(df,period = 200)

    df['delta_vector'] = 0

    df['upper_wick'] = 0
    df['lower_wick'] = 0
    
    dv_col = df.columns.get_loc('delta_vector')

    uw_col = df.columns.get_loc('upper_wick')
    lw_col = df.columns.get_loc('lower_wick')
    
    o_col = df.columns.get_loc('o')
    h_col = df.columns.get_loc('h')
    l_col = df.columns.get_loc('l')
    c_col = df.columns.get_loc('c')
    arr = df.values
    for i in range(arr.shape[0]):
        arr[i,dv_col] = arr[i,c_col] - arr[i,o_col]
        if arr[i,c_col] > arr[i,o_col]:
            arr[i,uw_col] = arr[i,h_col] - arr[i,c_col]
            arr[i,lw_col] = arr[i,o_col] - arr[i,l_col]
            
        elif arr[i,c_col] < arr[i,o_col]:
            arr[i,uw_col] = arr[i,h_col] - arr[i,o_col]
            arr[i,lw_col] = arr[i,c_col] - arr[i,l_col]   
            
    df = pd.DataFrame(arr,columns = df.columns)
    df['delta'] = abs(df['delta_vector'])
    e = time.time()
   # print('ADDING FEATURE COLS TIME:',round(e-s,3),'SECONDS')
    return df



def create_data(instrument,data_segments,count,pair,hours = 3,granularity = "M1"):

 #   print('CREATING DATASET...')
    start_ = time.time()
  #  print('PARAMS:',pair,granularity)

    total_df = convert_to_simple_df(instrument = instrument,first = True,granularity = granularity,count = count,start = datetime.datetime(2023,1,1,0,0,0),end = datetime.datetime(2023,1,1,0,0,0))
    cur = total_df['utc_timestamp'].iloc[0]
    df_lst = [total_df]
    for i in range(data_segments):
        df = convert_to_simple_df(instrument = instrument,first = False,granularity = granularity,count = count,start = cur - timedelta(hours = hours),end = cur)
        cur = cur - timedelta(hours = hours)
        df_lst.append(df)
       # total_df = pd.concat([total_df,df],axis = 0)
        #total_df = total_df.append(df)
    
    for item in enumerate(reversed(df_lst)):
        if item[0] == 0:
            total_df = item[1]
        else:
            total_df = pd.concat([total_df,item[1]],axis = 0)
    total_df = add_feature_cols(total_df)
   # print(total_df.shape)
#
    end_ = time.time()
   # print('TOTAL DATASET CONSTRUCTION TIME:',round(end_-start_,3),' SECONDS')
    return total_df



def signal_check(df,
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
    c_col = df.columns.get_loc('c')
    o_col = df.columns.get_loc('o')
    l_col = df.columns.get_loc('l')
    h_col = df.columns.get_loc('h')
    d_col = df.columns.get_loc('delta')
    uw_col = df.columns.get_loc('upper_wick')
    lw_col = df.columns.get_loc('lower_wick')
    ma1_col = df.columns.get_loc('smma_21')
    ma2_col = df.columns.get_loc('smma_50')
    ma3_col = df.columns.get_loc('smma_200')  
     
    
    
    i = df.shape[0] - 1
    #DEFINE SUPPORT/RESISTANCE as the MIN or MAX of a lookup range
    max_ = df.iloc[i - lookup_range : i,c_col].max()
    min_ = df.iloc[i - lookup_range : i,c_col].min()
    resistance_delta = max_ - df.iloc[i,c_col] #arr[i,c_col]
    support_delta = df.iloc[i,c_col] - min_ #arr[i,c_col] - min_ 
    pip_range = max_ - min_    
    if bullish_ma == True:
        if df.iloc[i,ma1_col] > df.iloc[i,ma2_col] and df.iloc[i,ma2_col] > df.iloc[i,ma3_col]:
            ma_indicator = 1
        else:
            ma_indicator = 0
    elif bullish_ma == False:
        if df.iloc[i,ma1_col] < df.iloc[i,ma2_col] and df.iloc[i,ma2_col] < df.iloc[i,ma3_col]:
            ma_indicator = 1
        else:
            ma_indicator = 0      
            
            
    #CANDLE BULL OR BEAR
    if df.iloc[i,c_col] >= df.iloc[i,o_col]:
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
        if df.iloc[i,c_col] >= min_ - abs(delta_filter) and df.iloc[i,c_col] <= max_ + abs(delta_filter):
            between_s_and_r = 1
        else:
            between_s_and_r = 0            
    else:
        if df.iloc[i,c_col] >= min_ and df.iloc[i,c_col] <= max_:
            between_s_and_r = 1
        else:
            between_s_and_r = 0

    if between_s_and_r and \
    near_reversal_indicator == 1 and \
    ma_indicator == 1 and \
    df.iloc[i,d_col] >= candle_size and \
    df.iloc[i,uw_col] >= upper_wick and \
    df.iloc[i,lw_col] >= lower_wick and \
    bull == bullish_candle:
        return 1
    else:
        return 0 
def check_all_pairs(param_df,all_pairs,str_time,placed_buy_in_last_24_candles = 0,placed_sell_in_last_24_candles = 0):
    lst = []
    count = 5000
    data_segments = 5
    hours = 250
    for pair in all_pairs:

        df = create_data(instrument = pair,data_segments = data_segments,count = count,pair = pair,hours = hours,granularity = "M5")
        last_utc = df['utc_timestamp'].iloc[-1]
        last_est = df['est_timestamp'].iloc[-1]
        last_close = df['c'].iloc[-1]

        
        #print(df.shape)    
        buy_or_sell = 'buy'
        params = param_df[(param_df['instrument'] == pair) & (param_df['buy_or_sell'] == buy_or_sell)]    

        buy_signal = signal_check(df,
            lookup_range = params['lookup_range'].iloc[0],
            bullish_ma = params['bullish_ma'].iloc[0],
            bullish_candle = params['bullish_candle'].iloc[0],
            candle_size = params['candle_size'].iloc[0],
            delta_filter = params['delta_filter'].iloc[0],
            delta_filter2 = params['delta_filter2'].iloc[0],
            upper_wick = params['upper_wick'].iloc[0],
            lower_wick = params['lower_wick'].iloc[0],
            support = params['support'].iloc[0]                
            )
        buy_or_sell = 'sell'
        params = param_df[(param_df['instrument'] == pair) & (param_df['buy_or_sell'] == buy_or_sell)]    
        

        sell_signal = signal_check(df,
            lookup_range = params['lookup_range'].iloc[0],
            bullish_ma = params['bullish_ma'].iloc[0],
            bullish_candle = params['bullish_candle'].iloc[0],
            candle_size = params['candle_size'].iloc[0],
            delta_filter = params['delta_filter'].iloc[0],
            delta_filter2 = params['delta_filter2'].iloc[0],
            upper_wick = params['upper_wick'].iloc[0],
            lower_wick = params['lower_wick'].iloc[0],
            support = params['support'].iloc[0]                
            )
      #  print(buy_signal,sell_signal)
      #  if buy_signal == 1 and placed_buy_in_last_24_candles == 0:
       #     print('BUYING')
        #if sell_signal == 1 and placed_sell_in_last_24_candles == 0:
         #   print('SELLING')
        #if buy_signal + sell_signal == 0 or placed_buy_in_last_24_candles + placed_sell_in_last_24_candles != 0:
         #   print('NOT PLACING TRADE')
            
        lst.append([pair,str_time,last_utc,last_est,last_close,buy_signal,sell_signal])
        
        
    return pd.DataFrame(lst,columns = ['pair','sys_time_5_min','last_utc','last_est','last_close','buy_signal','sell_signal'])
def get_account_balance():
    access_token='2ad78612c58e604890fd961550f73cfd-28190cc09d5cbc67b2c14503575d6132'
    accountID='101-001-20062555-002'
    
    api = oandapyV20.API(access_token=access_token)
    r = accounts.AccountDetails(accountID)
    acc_details=api.request(r)
    bal = acc_details['account']['balance']
    margin_available = acc_details['account']['marginAvailable']
    print('CURRENT BALANCE:',round(float(bal)),'CURRENT MARGIN AVAILABLE',round(float(margin_available)))
    return min(float(bal),float(margin_available))

def units_calc(trading_account,max_risk,sl):
    units=round((max_risk * trading_account / sl) * 10000)
    print('MAX RISK:',max_risk,'SL:',sl,'LOTS:',round(units / 100000,2),'UNITS:',units)
    return units
def create_market_order(instrument,units,TAKE_PROFIT_NUM_PIPS,STOP_LOSS_NUM_PIPS):
    def check_price(pair):
        access_token='2ad78612c58e604890fd961550f73cfd-28190cc09d5cbc67b2c14503575d6132'
        accountID='101-001-20062555-002'
        api = oandapyV20.API(access_token=access_token)
        bid_ask=[]
        params={"instruments":pair}

        r = pricing.PricingInfo(accountID=accountID, params=params)
        a = api.request(r)

        bid=float(a['prices'][0]['bids'][0]['price'])
        ask=float(a['prices'][0]['asks'][0]['price'])
        avg=(bid+ask)/2

        bid_ask.append(bid)
        bid_ask.append(ask)
        bid_ask.append(avg)



        return bid_ask
    
    if 'JPY' in instrument:
        multiplier = .01
    else:
        multiplier = .0001
    if units > 0:
        price = min(check_price(instrument))
        TAKE_PROFIT = round(price + (TAKE_PROFIT_NUM_PIPS * multiplier),3) if 'JPY' in instrument else round(price + (TAKE_PROFIT_NUM_PIPS * multiplier),5)
        STOP_LOSS = round(price - (STOP_LOSS_NUM_PIPS * multiplier),3) if 'JPY' in instrument else round(price - (STOP_LOSS_NUM_PIPS * multiplier),5)
    else:
        price = max(check_price(instrument))
        TAKE_PROFIT = round(price - (TAKE_PROFIT_NUM_PIPS * multiplier),3) if 'JPY' in instrument else round(price - (TAKE_PROFIT_NUM_PIPS * multiplier),5)
        STOP_LOSS = round(price + (STOP_LOSS_NUM_PIPS * multiplier),3) if 'JPY' in instrument else round(price + (STOP_LOSS_NUM_PIPS * multiplier),5)
        
        
    print('CREATING MARKET ORDER:',instrument,'|LOTS:',units / 100000,'|UNITS:',units,'|TP NUM PIPS:',TAKE_PROFIT_NUM_PIPS,
          '|SL NUM PIPS:',STOP_LOSS_NUM_PIPS,'|CUR PRICE:',price,'|STOP LOSS:',STOP_LOSS,'|TAKE PROFIT:',TAKE_PROFIT)
    print()
    access_token='2ad78612c58e604890fd961550f73cfd-28190cc09d5cbc67b2c14503575d6132'
    accountID='101-001-20062555-002'    
    api = oandapyV20.API(access_token=access_token)

    mktOrder = MarketOrderRequest(
        instrument = instrument,
        units = units,
        takeProfitOnFill=TakeProfitDetails(price=TAKE_PROFIT).data,
        stopLossOnFill=StopLossDetails(price=STOP_LOSS).data)

    # create the OrderCreate request
    r = orders.OrderCreate(accountID, data=mktOrder.data)
    try:
        # create the OrderCreate request
        rv = api.request(r)
    except oandapyV20.exceptions.V20Error as err:
        print(r.status_code, err)
    else:
        print(json.dumps(rv, indent=2))
        
    return rv

def enter_trade(dir_name,param_df,pair,buy_or_sell):
    import json
    # get stop loss and take profit
    sl = param_df[(param_df['instrument'] == pair) & (param_df['buy_or_sell'] == buy_or_sell)]['sl'].iloc[0]
    tp = param_df[(param_df['instrument'] == pair) & (param_df['buy_or_sell'] == buy_or_sell)]['tp'].iloc[0]


    # get balance and estimated units to trade
    if 'JPY' in pair:
        TAKE_PROFIT_NUM_PIPS = tp * 100
        STOP_LOSS_NUM_PIPS = sl * 100
    else:
        TAKE_PROFIT_NUM_PIPS = tp * 100 * 100
        STOP_LOSS_NUM_PIPS = sl * 100 * 100

    balance = get_account_balance()
    units = units_calc(trading_account = balance,max_risk = .01,sl = STOP_LOSS_NUM_PIPS)

    instrument =  pair
    if buy_or_sell != 'buy':
        units = units * -1
        
    order_dict = create_market_order(instrument,units,TAKE_PROFIT_NUM_PIPS,STOP_LOSS_NUM_PIPS)

    new_order_dict = json.dumps(order_dict)
    # open file for writing, "w" 
    dict_file = dir_name + "/order_dict_" + pair + '_' + buy_or_sell+ '_' + str(round(units /100000)) + '_' + str(TAKE_PROFIT_NUM_PIPS) + '_' + str(STOP_LOSS_NUM_PIPS) + ".json"
    print('SAVING ORDER DICT:',dict_file)
    f = open(dict_file,"w")

    # write json object to file
    f.write(new_order_dict)

    # close file
    f.close()
def run_trade_algo_once(dir_name,param_df):
    def run_trade_if_available(dir_name,check_df,param_df):

        buys = list(check_df[check_df['buy_signal'] == 1]['pair'])
        sells = list(check_df[check_df['sell_signal'] == 1]['pair'])
        print('BUY LEN:',len(buys),'SELL LEN:',len(sells))
        for pair in buys:
            print(pair)
            enter_trade(dir_name,param_df,pair,buy_or_sell = 'buy')
        for pair in sells:
            print(pair)
            enter_trade(dir_name,param_df,pair,buy_or_sell = 'sell')      
    s = time.time()
    all_pairs=['GBP_JPY','EUR_USD','USD_JPY','EUR_JPY','AUD_JPY','CAD_JPY','CHF_JPY',
           'GBP_USD','USD_CAD','USD_CHF','AUD_USD',
           'NZD_USD','AUD_CAD','AUD_CHF','AUD_NZD','CAD_CHF',
           'EUR_AUD','EUR_CAD','EUR_CHF','NZD_CAD','NZD_CHF'] 

    trade_check_path = dir_name + '/trade_check_data.csv'
    cur_time = datetime.datetime.now()
    utc = datetime.datetime.now(timezone.utc)
    str_time = str(cur_time.year) + '_' + str(cur_time.month) + '_' + str(cur_time.day) + '_' + str(cur_time.hour) + '_' + str((cur_time.minute // 5) * 5)
    print('CUR TIME:',cur_time,str_time,'UTC TIME:',utc)
   
    if utc.hour >= 18:
        sleep_time = 60 * 60
        print('UTC TIME IS:',utc,'SLEEPING FOR:',sleep_time,'SECONDS')
    try:
        trade_check_df = pd.read_csv(trade_check_path)

        if str_time in list(trade_check_df['sys_time_5_min']):
            seconds_at_start_of_next_candle = (((cur_time.minute + 5)//5) * 5) * 60
            second_right_now = (cur_time.minute * 60) + cur_time.second
            sleep_seconds = seconds_at_start_of_next_candle - second_right_now
            print(str_time,'ALREADY EXISTS, PASSING',seconds_at_start_of_next_candle , second_right_now,sleep_seconds)
            time.sleep(sleep_seconds)
        else:
            print('DOES NOT EXIST, RUNNING TRADE CHECKER')
            check_df = check_all_pairs(param_df,all_pairs,str_time,placed_buy_in_last_24_candles = 0,placed_sell_in_last_24_candles = 0) 

            trade_check_df = pd.concat([trade_check_df,check_df],axis = 0)
            trade_check_df.to_csv(trade_check_path,index = False)
            run_trade_if_available(dir_name,check_df,param_df)

    except FileNotFoundError:
        print("FileNotFoundError: running loop once to get file")
        check_df = check_all_pairs(param_df,all_pairs,str_time,placed_buy_in_last_24_candles = 0,placed_sell_in_last_24_candles = 0) 
        check_df.to_csv(trade_check_path,index = False)
        run_trade_if_available(dir_name,check_df,param_df)

    e = time.time()
    print((e-s),'TOTAL SECONDS') 