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

os.chdir("graphs")
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
    fig.savefig("time_series.png")
    plt.show()

#Visualise the time series data
plot_time_series(train_data, test_data)

#TASK2
#Task 2: Simple Moving Average Model
window_sizes = [i for i in range(1, 150)]
rms_errors_for_different_window = []


#Calculate the simple moving average for different values of window size
for window_size in window_sizes:
    
    y_hat_avg = train_data.copy().shift(1)
    y_hat_avg['moving_avg_forecast'] = train_data['time_series'].rolling(window_size).mean()
    y_hat_avg = y_hat_avg.loc[window_size:,]
    rms = sqrt(mean_squared_error(y_hat_avg.time_series, y_hat_avg['moving_avg_forecast']))
    rms_errors_for_different_window.append(rms)

def plot_rmse(y_array, x_array):
    fig = plt.figure()
    plt.plot(x_array, y_array)
    plt.ylabel('RMSE')
    plt.title("RMSE v/s Window sizes")
    plt.xlabel('Window sizes')
    plt.legend('Rmse values')
    fig.savefig("rmse_vs_k_for_sma.png")
    plt.show()

plot_rmse(rms_errors_for_different_window, window_sizes)
#Choose the minimum RMSE error for given windows sizes
best_value_of_window_size = rms_errors_for_different_window.index(min(rms_errors_for_different_window)) + 1
#Plot Simple moving average for the best value of window size
print("RMSE SMA Training ",min(rms_errors_for_different_window))
print("best value of K is given by ",best_value_of_window_size)
y_hat_avg = train_data.copy().shift(1)
y_hat_avg['moving_avg_forecast'] = train_data['time_series'].rolling(best_value_of_window_size).mean()
plt.figure(figsize=(16,8))
plt.plot(train_data['time_series'], label='Train')
plt.plot(test_data['time_series'], label='Test')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
plt.gca().legend(('training data', 'testing data', 'SMA-fit')) 
plt.title("Simple Moving Average")
plt.savefig("time_series_moving_average_train.png")
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

    ema = train_data.time_series.ewm(alpha=smoothing_level, adjust=False).mean()
    rms = sqrt(mean_squared_error(train_data.time_series, ema))
    rms_errors_for_different_window.append(rms)

def plot_rmse(y_array, x_array):
    fig = plt.figure()
    plt.plot(x_array, y_array)
    plt.ylabel('RMSE')
    plt.title("RMSE v/s smoothing_levels")
    plt.xlabel('smoothing_levels')
    plt.legend('Rmse values')
    fig.savefig("rmse_vs_k_for_exponential.png")
    plt.show()

plot_rmse(rms_errors_for_different_window, smoothing_levels)
best_value_of_smoothin_level = rms_errors_for_different_window.index(min(rms_errors_for_different_window)) + 1
print("RMSE exponential Training ",min(rms_errors_for_different_window))
#Plot Simple moving average for the best value of window size
print("best value of Alpha is given by ",best_value_of_smoothin_level/10)

ema = train_data.time_series.ewm(alpha=smoothing_level, adjust=False).mean()
plt.figure(figsize=(16,8))
plt.plot(train_data['time_series'], label='Train')
plt.plot(test_data['time_series'], label='Test')
plt.plot(ema, label='SES')
plt.gca().legend(('training data', 'testing data', 'Exponential-fit'))
plt.title("Exponential Moving Average")
plt.savefig("time_series_exponential_average_train.png")
plt.show()


plt.plot(train_data['time_series'], label='Train')
#TASK4
#Auto regressive(AR)

lag_acf = acf(train_data['time_series'], nlags=1)
lag_pacf = pacf(train_data['time_series'], nlags=80, method='ols')

#Plot ACF:
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_data['time_series'])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_data['time_series'])),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.tight_layout()
plt.savefig("ACF.png")
plt.show()

#Plot PACF:
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_data['time_series'])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_data['time_series'])),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.savefig("PACF.png")
plt.show()


y_hat_avg = train_data.copy()

ar_mod = AR(train_data.time_series)
ar_res = ar_mod.fit(1)
y_hat_avg['AR'] = ar_res.predict(start=1, end=1500, dynamic=False)

#fit1 = sm.tsa.statespace.ARX(train_data.time_series, order=(2,1,0)).fit()
#y_hat_avg['AR'] = fit1.predict(start=1500, end=2000, dynamic=True)
plt.figure(figsize=(16,8))
plt.plot( train_data['time_series'], label='Train')
plt.plot(test_data['time_series'], label='Test')
plt.plot(y_hat_avg['AR'], label='AR')
plt.legend(loc='arima_fit')
plt.title("AR")
plt.savefig("time_series_auto_regressive_train.png")
plt.show()

residual = y_hat_avg['AR'] - train_data.time_series

sm.qqplot(residual)
plt.title("QQ plot for time series")
plt.savefig("QQ plot of residual.png")
plt.show()


#residual scatter plot 
plt.scatter( y_hat_avg['AR'], residual,color='g')
plt.ylabel('residual');
plt.title("Residual scatter plot ")
plt.xlabel('predictions')
plt.savefig("residual scatter.png")
plt.show()

#residual histogram

plt.hist(residual, bins=30)
plt.title("Histogram of residual")        
plt.savefig("residual histogram.png")
plt.show()

#Chi square test
print(stats.normaltest(residual, nan_policy='omit'))


#Task5
#Testing data

y_hat_avg = test_data.copy().shift(1)
y_hat_avg['moving_avg_forecast'] = test_data['time_series'].rolling(best_value_of_window_size).mean()

plt.figure(figsize=(16,8))
plt.plot(train_data['time_series'], label='Train')
plt.plot(test_data['time_series'], label='Test')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
plt.gca().legend(('training data', 'testing data', 'SMA-fit')) 
plt.title("Simple Moving Average")
plt.savefig("time_series_moving_average_test.png")
plt.show()
y_hat_avg = y_hat_avg.loc[best_value_of_window_size+1500:,]
rms = sqrt(mean_squared_error(y_hat_avg.time_series, y_hat_avg['moving_avg_forecast']))
print("RMSE SImple Moving Average", rms)

#y_hat_avg = test_data.copy()
#fit2 = SimpleExpSmoothing(np.asarray(train_data['time_series'])).fit(best_value_of_smoothin_level,optimized=False)
#y_hat_avg['SES'] = fit2.forecast(len(test_data))
ema = test_data.time_series.ewm(alpha=smoothing_level, adjust=False).mean()
plt.figure(figsize=(16,8))
plt.plot(train_data['time_series'], label='Train')
plt.plot(test_data['time_series'], label='Test')
plt.plot(ema, label='SES')  
plt.gca().legend(('training data', 'testing data', 'Exponential-fit'))
plt.title("Exponential Moving Average")
plt.savefig("time_series_exponential_average_test.png")
plt.show()

rms = sqrt(mean_squared_error(test_data.time_series, ema))
print("RMSE Exponential", rms)

y_hat_avg = test_data.copy()
y_hat_avg = y_hat_avg.reset_index()
#print(y_hat_avg)
ar_mod = AR(y_hat_avg.time_series)
ar_res = ar_mod.fit(1)
y_hat_avg['AR'] = ar_res.predict(start=1, end=499, dynamic=False)
#print(y_hat_avg['AR'])
#fit1 = sm.tsa.statespace.ARX(train_data.time_series, order=(2,1,0)).fit()
#y_hat_avg['AR'] = fit1.predict(start=1500, end=2000, dynamic=True)
plt.figure(figsize=(16,8))
plt.plot(y_hat_avg['time_series'], label='Test')
plt.plot(y_hat_avg['AR'], label='AR')
plt.legend(loc='arima_fit')
plt.title("AR testing")
plt.savefig("time_series_auto_regressive_test.png")
plt.show()

rms = sqrt(mean_squared_error(y_hat_avg.time_series[1:], y_hat_avg.AR.dropna()))
print("RMSE Auto Regressive Model", rms)
