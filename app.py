import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime
model = load_model(r'C:\Users\abhis\OneDrive\Desktop\ali\.ipynb_checkpoints\Stock_predictions_Model.keras')


st.header('📈 Stock Market Predictor')


stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = datetime.datetime.today().strftime('%Y-%m-%d')


data = yf.download(stock, start, end)


if data.empty:
    st.error(" No data found for this symbol. Please check the stock symbol and try again.")
    st.stop()


st.subheader('Stock Data')
st.write(data)

if len(data) < 200:
    st.warning("⚠️ Not enough data to perform prediction. Try another stock with more historical data.")
    st.stop()

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])


scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)


data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test = data_test.reset_index(drop=True)
data_test = data_test.dropna()

if data_test.empty:
    st.error("Not enough data after merging for prediction.")
    st.stop()

data_test = data_test.values.reshape(-1, 1)
data_test_scale = scaler.fit_transform(data_test)


st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig3)


x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

if len(x) == 0:
    st.error(" Not enough data to generate predictions. Try a different stock.")
    st.stop()

x, y = np.array(x), np.array(y)
predict = model.predict(x)


scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale


st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)


st.subheader(' Volatility Analysis (Bollinger Bands)')
rolling_mean = data['Close'].rolling(window=20).mean()
rolling_std = data['Close'].rolling(window=20).std()
upper_band = rolling_mean + (rolling_std * 2)
lower_band = rolling_mean - (rolling_std * 2)

fig5 = plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Closing Price', color='blue')
plt.plot(upper_band, label='Upper Band', linestyle='--', color='red')
plt.plot(lower_band, label='Lower Band', linestyle='--', color='green')
plt.fill_between(data.index, lower_band.values.flatten(), upper_band.values.flatten(), color='gray', alpha=0.2)
plt.title('Bollinger Bands')
plt.legend()
st.pyplot(fig5)


trend_direction = None
trend_days = 1  

for i in range(len(predict) - 2, -1, -1):
    if predict[i] < predict[i + 1]:  
        if trend_direction in [None, 'up']:
            trend_direction = 'up'
            trend_days += 1
        else:
            break
    elif predict[i] > predict[i + 1]: 
        if trend_direction in [None, 'down']:
            trend_direction = 'down'
            trend_days += 1
        else:
            break
    else:
        if trend_direction is None:
            trend_direction = 'flat'
        else:
            break


st.subheader("🔮 Trend Prediction")
last_date = data.index[-1].date()
if trend_direction == 'up':
    st.success(f"It is *highly predicted* that the stock will go **UP** for around **{trend_days} days**, starting after **{last_date}**.")
elif trend_direction == 'down':
    st.error(f"It is *highly predicted* that the stock will go **DOWN** for around **{trend_days} days**, starting after **{last_date}**.")
else:
    st.info(f"The trend seems stable or unclear based on data ending on **{last_date}**.")
