import matplotlib.pyplot as plt
import csv 
import os
import numpy as np
import ast
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pylab import *
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
from sklearn import *
from scipy.stats import chisquare
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import acf, pacf

#Task 1: Read and Visualise data 
#read csv into pandas data frame
with open("ovbarve.csv", "r") as f:
    time_series_data = []    
    for line in f.readlines():
        time_series_data.append(ast.literal_eval(line))

df = pd.DataFrame(np.array(time_series_data), columns = ["time_series"])
train_data, test_data = train_test_split(df, test_size=0.25, shuffle=False)


X = df.time_series
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#function to plot time series
def plot_time_series(train_data, test_data):
    
    fig = plt.figure()
    train_data.time_series.plot(figsize=(15,8), fontsize=14)
    test_data.time_series.plot(figsize=(15,8), fontsize=14)
    plt.ylabel('VALUESof time series data')
    plt.title("Time Series plot")
    plt.xlabel('Time')
    plt.gca().legend(('training data', 'testing data'))
    #fig.savefig("time_series.png")
    plt.show()

#Visualise the time series data
plot_time_series(train_data, test_data)



#TASK2
#Task 2: Simple Moving Average Model
window_sizes = [i for i in range(1, 1000)]
rms_errors_for_different_window = []


#Calculate the simple moving average for different values of window size
for window_size in window_sizes:
    
    y_hat_avg = test_data.copy()
    y_hat_avg['moving_avg_forecast'] = train_data['time_series'].rolling(window_size).mean().iloc[-1]
    rms = sqrt(mean_squared_error(test_data.time_series, y_hat_avg.moving_avg_forecast))
    #print("RMSE for the window size "+str(window_size)+" is :"+str(rms_error))    
    rms_errors_for_different_window.append(rms)

def plot_rmse(y_array, x_array):
    fig = plt.figure()
    plt.plot(x_array, y_array)
    plt.ylabel('RMSE')
    plt.title("RMSE v/s Window sizes")
    plt.xlabel('Window sizes')
    plt.legend('Rmse values')
    #fig.savefig("rmse_vs_k_for_sma.png")
    plt.show()

plot_rmse(rms_errors_for_different_window, window_sizes)
#Choose the minimum RMSE error for given windows sizes
best_value_of_window_size = rms_errors_for_different_window.index(min(rms_errors_for_different_window)) + 1

#Plot Simple moving average for the best value of window size
print(best_value_of_window_size)
y_hat_avg = test_data.copy()
y_hat_avg['moving_avg_forecast'] = train_data['time_series'].rolling(best_value_of_window_size).mean().iloc[-1]
plt.figure(figsize=(16,8))
plt.plot(train_data['time_series'], label='Train')
plt.plot(test_data['time_series'], label='Test')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
plt.gca().legend(('training data', 'testing data', 'SMA-fit')) 
plt.title("Simple Moving Average")
plt.show()

#TASK3
#Task3 Exponential Moving Average
def frange(start, stop, step):
     i = start
     while i < stop:
         yield i
         i += step
smoothing_levels = [i for i in frange(0.1, 0.9, 0.1)]
rms_errors_for_different_window = []

for smoothing_level in smoothing_levels:

    y_hat_avg = test_data.copy()
    fit2 = SimpleExpSmoothing(np.asarray(train_data['time_series'])).fit(smoothing_level,optimized=False)
    y_hat_avg['SES'] = fit2.forecast(len(test_data))
     
    rms = sqrt(mean_squared_error(test_data.time_series, y_hat_avg.SES))
    #print(rms)


    rms_errors_for_different_window.append(rms)

def plot_rmse(y_array, x_array):
    fig = plt.figure()
    plt.plot(x_array, y_array)
    plt.ylabel('RMSE')
    plt.title("RMSE v/s smoothing_levels")
    plt.xlabel('smoothing_levels')
    plt.legend('Rmse values')
    #fig.savefig("rmse_vs_k_for_sma.png")
    plt.show()

plot_rmse(rms_errors_for_different_window, smoothing_levels)
best_value_of_smoothin_level = rms_errors_for_different_window.index(min(rms_errors_for_different_window)) + 1

#Plot Simple moving average for the best value of window size
print(best_value_of_smoothin_level)


y_hat_avg = test_data.copy()
fit2 = SimpleExpSmoothing(np.asarray(train_data['time_series'])).fit(best_value_of_smoothin_level,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test_data))
plt.figure(figsize=(16,8))
plt.plot(train_data['time_series'], label='Train')
plt.plot(test_data['time_series'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.gca().legend(('training data', 'testing data', 'Exponential-fit'))
plt.title("Exponential Moving Average")
plt.show()


#TASK4
#Auto regressive integerated Moving Average (ARIMA)

lag_acf = acf(train_data['time_series'], nlags=50)
lag_pacf = pacf(train_data['time_series'], nlags=50, method='ols')

#Plot ACF: 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_data['time_series'])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_data['time_series'])),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#plt.show()

#Plot PACF:
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_data['time_series'])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_data['time_series'])),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
#plt.show()


y_hat_avg = test_data.copy()

ar_mod = AR(train_data.time_series)
ar_res = ar_mod.fit(2)
y_hat_avg['SARIMA'] = ar_res.predict(start=1500, end=2000, dynamic=True)

#fit1 = sm.tsa.statespace.SARIMAX(train_data.time_series, order=(2,1,0)).fit()
#y_hat_avg['SARIMA'] = fit1.predict(start=1500, end=2000, dynamic=True)
plt.figure(figsize=(16,8))
plt.plot( train_data['time_series'], label='Train')
plt.plot(test_data['time_series'], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='arima_fit')
plt.title("ARIMA")
plt.show()


residual = y_hat_avg['SARIMA'] - test_data.time_series

sm.qqplot(residual)
plt.title("QQ plot for time series")
#plt.savefig("QQ plot of residual.png")
plt.show()


#residual scatter plot 
plt.scatter( y_hat_avg['SARIMA'], residual,color='g')
plt.ylabel('residual');
plt.title("Residual scatter plot ")
plt.xlabel('predictions')
#plt.savefig("residual scatter.png")
plt.show()

#residual histogram

plt.hist(residual, bins=30)
plt.title("Histogram of residual")        
#plt.savefig("residual histogram.png")
plt.show()

#Chi square test
print(stats.normaltest(residual))

