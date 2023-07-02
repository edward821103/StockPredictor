import numpy as np
import pandas as pd
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 讀取數據集
stock_id = '6239'
db = sqlite3.connect('daily_price.db')
stock_infos = pd.read_sql(con=db,sql="SELECT 日期,成交股數,開盤價,最高價,最低價,收盤價 FROM 'daily_price' where 證券代號 = '{}' ORDER BY 日期 ASC".format(stock_id))

# 數據預處理 # TODO 使用多點特徵
scaler = MinMaxScaler(feature_range=(0, 1))
print(stock_infos[['開盤價','收盤價']].values.reshape(-1, 1))
scaled_data = scaler.fit_transform(stock_infos[['開盤價','收盤價']].values.reshape(-1, 1))
print(scaled_data)
db.close()
# 切分數據集
train_data = scaled_data[:int(len(scaled_data) * 0.7), :]
test_data = scaled_data[int(len(scaled_data) * 0.7):, :]

# # 定義函數用於處理數據集
def create_dataset(dataset, time_step=1):
    X_data, Y_data = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X_data.append(a)
        Y_data.append(dataset[i + time_step, 0])
    return np.array(X_data), np.array(Y_data)

# 創建訓練集和測試集
time_step = 60
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

# 將數據集轉換成LSTM需要的格式
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 定義LSTM模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# 編譯模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 訓練模型
model.fit(X_train, Y_train, epochs=100, batch_size=64)

# 儲存模型
# model.save('model.h5')  # creates a HDF5 file 'model.h5'

# 使用模型進行預測
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

# 計算預測準確性
rmse = np.sqrt(np.mean(((test_predict - Y_test) ** 2)))
print('RMSE:', rmse)

# 計算上漲/下跌機率
test_direction = np.sign(test_predict[1:] - test_predict[:-1])
up_probability = np.mean(test_direction == 1)
down_probability = np.mean(test_direction == -1)
print('Up Probability:', up_probability)
print('Down Probability:', down_probability)
