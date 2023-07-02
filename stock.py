from distutils.log import error
import pandas as pd
import sqlite3
import time
# import mysql.connector
from matplotlib import dates as mdates
from matplotlib import ticker as mticker
from mpl_finance import candlestick_ohlc #KD
from matplotlib.dates import DateFormatter
import datetime as dt
import matplotlib.pyplot as plt
import io
import numpy as np
import requests
from typing import Tuple, Dict

# def get_date_range_from_db(fname:str) -> Tuple[str, str]:
#     conn = sqlite3.connect(fname)
#     c = conn.cursor()
#     c.execute('select * from daily_price order by 日期 ASC LIMIT 1;')
#     date_from = dt.datetime.strptime(list(c)[0][17], '%Y-%m-%d %H:%M:%S')
#     c.execute('select * from daily_price order by 日期 DESC LIMIT 1;')
#     date_to = dt.datetime.strptime(list(c)[0][17], '%Y-%m-%d %H:%M:%S')
#     conn.close()
#     return date_from, date_to

# def trans_date(date_time:str) -> str:
#     return ''.join(str(date_time).split(' ')[0].split('-'))

# def prase_n_days(now_date:str, n:int):
#     print(now_date, n)
#     df_dict = {}
#     for i in range(n):
#         time.sleep(3)
#         now_date = now_date - dt.timedelta(days=1)
#         try:
#             df = crawler(trans_date(now_date))
#             print('successful' + ' ' + trans_date(now_date))
#             df_dict.update({trans_date(now_date):df})
#         except Exception as e:
#             print("error", e)
#         finally:
#             print('fails' + ' ' + trans_date(now_date))
#     print("df_dict_type:", type(df_dict), df_dict)
#     return df_dict

# def crawler(date_time:str) -> Dict:
#     page_url = 'https://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date=' + date_time + '&type=ALLBUT0999'
#     page = requests.get(page_url)
#     use_text = page.text.splitlines()
#     for i, text in enumerate(use_text):
#         if text == '"證券代號","證券名稱","成交股數","成交筆數","成交金額","開盤價","最高價","最低價","收盤價","漲跌(+/-)","漲跌價差","最後揭示買價","最後揭示買量","最後揭示賣價","最後揭示賣量","本益比",':
#             initital_point = i
#             break
#     stock_df = pd.read_csv(io.StringIO(''.join([text[:-1] + '\n' for text in use_text[initital_point:]])))
#     stock_df['證券代號'] = stock_df['證券代號'].apply(lambda x:x.replace('"', ''))
#     stock_df['證券代號'] = stock_df['證券代號'].apply(lambda x:x.replace('=', ''))
#     print("stock_df_type:", type(stock_df), stock_df)
#     return stock_df

# def sort_df_frame(df):
#     sorted_df = df.apply(pd.to_numeric, errors='coerce')
#     return sorted_df

# #查看資料庫內容
# dbname = 'daily_price.db'
# db_from, db_to = get_date_range_from_db(dbname)
# print('資料庫日期: {} 到 {}'.format(db_from.strftime('%Y-%m-%d'), db_to.strftime('%Y-%m-%d')))

# #確認資料庫缺幾天資料，爬蟲抓取
# today = str(dt.date.today())
# p_today = dt.datetime.strptime(today, '%Y-%m-%d')
# days = str(p_today - db_to)
# days = eval(days.split(' ')[0])
# # days = 4
# result_dict = prase_n_days(dt.datetime.now(), days - 1)
# for key in result_dict.keys():
#     result_dict[key].to_csv(str(key) + '.csv')

# #加入日期並存入現有資料庫
# result = result_dict.copy()
# df_frame = pd.concat([result[key].assign(
#     日期 = pd.to_datetime(key, format='%Y%m%d'),
#     成交股數 = sort_df_frame(result[key]['成交股數'].apply(lambda x: x.replace(",", ""))),
#     成交筆數 = sort_df_frame(result[key]['成交筆數'].apply(lambda x: x.replace(",", ""))),
#     成交金額 = sort_df_frame(result[key]['成交金額'].apply(lambda x: x.replace(",", ""))),
#     開盤價 = sort_df_frame(result[key]['開盤價']),
#     最高價 = sort_df_frame(result[key]['最高價']),
#     最低價 = sort_df_frame(result[key]['最低價']),
#     收盤價 = sort_df_frame(result[key]['收盤價']),
#     漲跌價差 = sort_df_frame(result[key]['漲跌價差']),
#     最後揭示買價 = sort_df_frame(result[key]['最後揭示買價']),
#     最後揭示買量 = sort_df_frame(result[key]['最後揭示買量']),
#     最後揭示賣價 = sort_df_frame(result[key]['最後揭示賣價']),
#     最後揭示賣量 = sort_df_frame(result[key]['最後揭示賣量']),
#     本益比 = sort_df_frame(result[key]['本益比'])
# ) for key, _ in result.items()])

# conn = sqlite3.connect(dbname)
# df_frame.to_sql('daily_price', con=conn, if_exists='append')
# conn.close()

dbname_2='daily_price.db'
db = sqlite3.connect(dbname_2)
stock=input('請輸入股票代號')


tsmc=pd.read_sql(con=db,sql="SELECT 日期,成交股數,開盤價,最高價,最低價,收盤價 FROM 'daily_price' where 證券代號 = '{}' ORDER BY 日期 ASC".format(stock))
stock_id=pd.read_sql(con=db,sql="SELECT 證券名稱 FROM 'daily_price' WHERE 證券代號='{}'".format(stock)).iloc[1,:]
stock_id=stock_id[0]
tsmc1=tsmc.copy()
tsmc1.index=tsmc1['日期']
tsmc1=tsmc1.iloc[:,1:]
tsmc1.columns=['Vol','Open','High','Low','Close']


def moving_average(data,per):
    return data['Close'].rolling(per).mean()
# #計算ＫＤ
#KD公式
#RSV=第Ｎ天收盤價-最近Ｎ天內最低價/最近Ｎ天最高價-最近N天最低價
#Ｋ值=2/3前一日Ｋ值+1/3ＲＳＶ
#Ｄ值=2/3前一天Ｄ值+1/3當日Ｋ值
def KD(data):
    df_copy=data.copy()
    df_copy['min']=df_copy['Low'].rolling(9).min()
    df_copy['max']=df_copy['High'].rolling(9).max()
    df_copy['RSV']=(df_copy['Close']-df_copy['min'])/(df_copy['max']-df_copy['min'])
    df_copy=df_copy.dropna()
    K_list = [50]
    for num,rsv in enumerate(list(df_copy['RSV'])):
        K_yestarday = K_list[num]
        K_today = 2/3 * K_yestarday + 1/3 * rsv
        K_list.append(K_today)
    df_copy['K']=K_list[1:]
    D_list = [50]
    for num,K in enumerate(list(df_copy['K'])):
        D_yestarday = D_list[num]
        D_today = 2/3 * D_yestarday + 1/3 * K
        D_list.append(D_today)
    df_copy['D'] = D_list[1:]
    use_df = pd.merge(data,df_copy[['K','D']],left_index=True,right_index=True,how='left')
    return use_df


df=KD(tsmc1)
df_plot=tsmc1[['Open','High','Low','Close']]
data_plot=df_plot.copy()
data_plot['Datetime']=df_plot.index
data_plot=data_plot.reset_index()
data_plot=data_plot[['Datetime','Open','High','Low','Close']]
data_plot['Datetime']=pd.to_datetime(data_plot['Datetime']).dt.normalize()
data_plot['Datetime']=mdates.date2num(data_plot['Datetime'])
# data_plot1=data_plot.copy()
# data_plot1['Datetime']=pd.to_datetime(data_plot1['Datetime']).dt.normalize()#將2000-01-01 00:00:00轉為2000-01-01(去除時分秒)
# data_plot1['Datetime']=data_plot1['Datetime'].strip(0)
# data_plot1['Datetime']=data_plot1['Datetime'].strftime("%Y-%m-%d")
#畫Ｋ線圖
ma_10=moving_average(df,10)
ma_50=moving_average(df,50)
length=len(data_plot['Datetime'].values[49:])
figure=plt.figure(facecolor='white',figsize=(15,10))
ax1=plt.subplot2grid((6,4),(0,0),rowspan=4,colspan=4,facecolor='white')
candlestick_ohlc(ax1,data_plot.values[-length:],width=0.6,colorup='red',colordown='green')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.plot(data_plot['Datetime'].values[-length:],ma_10[-length:],'black',label='10 MA Line',linewidth=1.5)
ax1.plot(data_plot['Datetime'].values[-length:],ma_50[-length:],'navy',label='50 MA Line',linewidth=1.5)
ax1.legend()
ax1.grid(True,color='black')
plt.ylabel('Stock price and Volume')
plt.suptitle('Stock Code:{}'.format(stock),color='black',fontsize=16)
#交易量
ax1v=ax1.twinx()
ax1v.fill_between(data_plot['Datetime'].values[-length:],0,df['Vol'].values[-length:],facecolor='aqua',alpha=0.4)
ax1v.grid(False)
ax1v.set_ylim(0,3*df.Vol.values.max())
#畫ＫＤ值
ax2=plt.subplot2grid((6,4),(4,0),sharex=ax1,rowspan=1,colspan=4,facecolor='white')
ax2.plot(data_plot['Datetime'].values[-length:],df['K'].values[-length:],color='black')
ax2.plot(data_plot['Datetime'].values[-length:],df['D'].values[-length:],color='red')
plt.show()
# db_name='daily_price.db'
# def get_data(fname, stock_id, period):
#     conn = sqlite3.connect(fname)
#     c = conn.cursor()
#     cmd = 'SELECT 日期, 收盤價 FROM daily_price WHERE 證券代號 = "{:s}" ORDER BY 日期 DESC LIMIT {:d};'.format(stock_id, period)
#     c.execute(cmd)
#     rows = c.fetchall()
#     rows.reverse()
#     conn.close()
#     return rows


# def calc_rsv(prices):
#     # 採用常見的 "9, 3, 3" 方式計算 KD 值
#     # (http://yhhuang1966.blogspot.com/2015/02/kd.html)
#     # (http://www.cmoney.tw/notes/note-detail.aspx?nid=6460)
#     window = prices[:8]  # 前 8 天股價
#     # 因不足 9 天，前 8 天的最高點、最低點、及 RSV 值皆為0; 第八天的 K 值 = D 值 = 50
#     highest = [0]*8
#     lowest = [0]*8
#     rsv_values = [0]*8
#     k_values = [0]*7 + [50]
#     d_values = [0]*7 + [50]
#     for i, p in enumerate(prices[8:]):  # 從第 9 天開始計算 RSV 及 KD 值
#         window.append(p)
#         window = window[len(window)-9:]  # 計算範圍為最近 9 天
#         high = max(window)
#         low = min(window)
#         rsv = 100 * ((p - low) / (high - low))
#         k = ((1/3)*rsv) + ((2/3)*k_values[-1])
#         d = ((1/3)*k) + ((2/3)*d_values[-1])
#         highest.append(high)
#         lowest.append(low)
#         rsv_values.append(rsv)
#         k_values.append(k)
#         d_values.append(d)
#     return k_values, d_values




# def get_buy_signal(k_values, d_values):
#     buy = [0]*8  # 前 8 天沒有資料故無買進訊號
#     for i in range(8, len(k_values)):
#         # 策略: KD 黃金交叉 (前一天 k < d 且今天 k > d) 且在低檔 (30)
#         # (http://www.cmoney.tw/app/ItemContent.aspx?id=2739)
#         if k_values[i-1] < d_values[i-1] and k_values[i] > d_values[i] and k_values[i] < 30:
#             buy.append(1)
#         else:
#             buy.append(0)
#     return buy


# price_data = get_data(db_name, stock, 2642)
# dates = [d[0] for d in price_data]
# prices = [d[1] for d in price_data]
# print('股票名稱：{},股票代號：{}'.format(stock_id,stock))
# print('起始日期: {} (收盤價: {}), 結束日期: {} (收盤價: {}) ({} 天)'.format(
#     dates[0], prices[0], dates[-1], prices[-1], len(dates)))
# k, d = calc_rsv(prices)
# buy = get_buy_signal(k, d)
# print('本金 10 萬元. 期間有 {} 次買進訊號, 一次投入 1 萬元'.format(sum(buy)))
# profit = [10]
# ratios = [1] + [prices[i] / prices[i - 1] for i in range(1, len(prices))]
# # 因為買進訊號是根據當天盤後價格計算, 隔天才能真正加碼, 故將買進訊號往右平移一天當作當天加碼 1 萬元
# buy = [0] + buy[:-1]
# for b, r in zip(buy[1:], ratios[1:]):
#     profit.append(profit[-1] * r + b)
# print('回測結果: {}萬'.format(profit[-1]))
# interest_rate = 0.011 * (len(dates)/365)
# total = 10 + sum(buy)
# print('{} 萬元定存結果 (利率 1.1%): {}萬'.format(total, total*(1+interest_rate)))

