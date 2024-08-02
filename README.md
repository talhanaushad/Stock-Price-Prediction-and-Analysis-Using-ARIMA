# Stock-Price-Prediction-and-Analysis-Using-ARIMA
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import yfinance as yf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
def old_stock_price(stock_symbol,start_date, end_date ): 
    data = yf.download(stock_symbol,start = start_date,end = end_date) 
    return data 
stock_data = old_stock_price("GOOG","2020-01-01","2024-01-01") 
stock_data.head()
plt.figure(figsize=(12,6))
plt.plot(stock_data["Close"])
plt.title("Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.show()
train_size = int(len(stock_data)*.8)
train_data = stock_data[:train_size]
test_data = stock_data[train_size:]
train_data = train_data["Close"]
test_data = test_data["Close"]
def find_best_arima_order(data,p_values,d_values,q_values):
    best_score,best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(data, order=(p,d,q))
                    model_fit = model.fit()
                    mse = mean_squared_error(data,model_fit.fittedvalues)
                    if mse < best_score:
                        best_score,best_cfg = mse,(p,d,q)
                except:
                    continue
    return best_cfg            
p_values = range(0,3)
d_values = range(0,3)
q_values = range(0,3)
best_cfg = find_best_arima_order(train_data,p_values,d_values,q_values)
print(f"the best is {best_cfg}.")
model = ARIMA(train_data,order=(best_cfg))
model_fit = model.fit()
forecasted_values = model_fit.forecast(steps = len(test_data))
mse = mean_squared_error(test_data,forecasted_values)
print("The Mean Square Error is {:.2f}".format(mse))
forecasted_values = model_fit.forecast(steps = 180)
forecasted_dates = pd.date_range(start=test_data.index[-1],periods=180,freq='D')
forecast_df = pd.DataFrame({"Date": forecasted_dates,"Forecast": forecasted_values})
forecast_df.head()
plt.figure(figsize=(12,6))
plt.plot(test_data.index,test_data,label = "Actual",color = "b")
plt.plot(forecast_df["Date"],forecast_df["Forecast"],label = "Forecasted",color = "r")
plt.title("Forecasted Closing Price for Next 6 Months")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.show()
