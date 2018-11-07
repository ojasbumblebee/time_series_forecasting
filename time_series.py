import matplotlib.pyplot as plt
import csv 
import os
import numpy as np
import ast
import pandas as pd
from pylab import *
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
from sklearn import *
from scipy.stats import chisquare
import scipy.stats as stats
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

#task 1
#read csv into pandas data frame
with open("ovbarve.csv", "r") as f:
    time_series_data = []    
    for line in f.readlines():
        time_series_data.append(ast.literal_eval(line))

df = pd.DataFrame(np.array(time_series_data), columns = ["time_series"])
train_data, test_data = train_test_split(df, test_size=0.25, shuffle=False)


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


#Task 1: Simple Moving Average Model
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


#Task2 Exponential Moving Average

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
"""
#Task 1: Simple Moving Average Model

#Function to calculate the simple moving average
def simple_moving_average(time_series_data, window_size=3):
    weigths = np.repeat(1.0, window_size)/window_size
    smas = np.convolve(time_series_data, weigths, 'valid')
    return smas # as a numpy array


#calculate the rootmean square error
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

window_sizes = [i for i in range(1, 100)]
rms_errors_for_diiferent_window = []


#Calculate the simple moving average for different values of window size
for window_size in window_sizes:
    smas = simple_moving_average(time_series_data, window_size)
    rms_error = rmse(smas, time_series_data[window_size-1:])
    #print("RMSE for the window size "+str(window_size)+" is :"+str(rms_error))
    rms_errors_for_diiferent_window.append(rms_error)

#plot root mean square error
def plot_rmse(y_array, x_array):
    fig = plt.figure()
    plt.plot(x_array, y_array)
    plt.ylabel('RMSE')
    plt.title("RMSE v/s Window sizes")
    plt.xlabel('Window sizes')
    #fig.savefig("rmse_vs_k_for_sma.png")
    plt.show()

plot_rmse(rms_errors_for_diiferent_window, window_sizes)

'''simple exponential smoothing go back to last N values
 y_t = a * y_t + a * (1-a)^1 * y_t-1 + a * (1-a)^2 * y_t-2 + ... + a*(1-a)^n * 
y_t-n'''


def exponential_smoothing(time_series_data, alpha_value):
    
    ouput = sum([alpha_value * (1 - alpha_value) ** i * x for i, x in 
                enumerate(reversed(time_series_data))])
    return ouput


def frange(start, stop, step):
     i = start
     while i < stop:
         yield i
         i += step
window_sizes = [i for i in frange(0.1, 0.9, 0.1)]

rms_errors_for_different_alpha = []
alpha = 0
for _alpha in range(1,10):
    alpha += 0.1
    smoothing_number = []
    for index_of_data in range(1,len(time_series_data)):
        smoothing_number.append(exponential_smoothing(time_series_data[:index_of_data], alpha)) # use a=0.6 or 0.5 your choice, which gives less rms error

    rms_errors_for_different_alpha.append(rmse(smoothing_number, time_series_data[1:]))

plot_rmse(rms_errors_for_different_alpha, window_sizes)
"""

"""
#plot correlation matrix
#fig = plt.figure()
print(df.corr())
pd.scatter_matrix(df)
plt.savefig("correlation_scatter_plot.png")
plt.show()

# remove outliers at 3 std deviations and outside
print(df.shape)

df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
print(df.shape)
input("press enter to continue")

#Task 2
#Linear regression

regression_variables = list(df.keys())
y = df[regression_variables[-1]]

for count,variable in enumerate(regression_variables[:-1]):

    #prepare data by reshaping vector eg. [1, 2, 3] to [[1], [2], [3]]    
    current_independent_variable = [[i] for i in df[variable]]
    
    X = sm.add_constant(current_independent_variable) # adding a constant
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X) 
    print_model = model.summary()
    print(print_model)
    
    #calculate residual array
    residual =  y - predictions
    print("variance of residuals is :", np.var(residual))

    #residual qqplot
    sm.qqplot(residual)
    plt.title("QQ plot for single variable: X"+str(count))
    plt.savefig("QQ plot of residual for linear regression of: X"+str(count)+" .png")
    plt.show()
    

    #residual scatter plot 
    plt.scatter(predictions, residual,color='g')
    plt.ylabel('residual');
    plt.title("Residual scatter plot for single variable :"+str(count))
    plt.xlabel('x - current_independent_variable')
    plt.savefig("residual scatter for linear regression of: X"+str(count)+" .png")
    plt.show()

    #residual histogram

    plt.hist(residual, bins=30)
    plt.title("Histogram of residual:"+str(count))        
    plt.savefig("residual histogram for linear regression of: X"+str(count)+" .png")
    plt.show()

    #Chi square test
    print(stats.normaltest(residual))

    #Plot regression fit using sklearn fitting again (remove this) 
    lm=linear_model.LinearRegression()
    lm.fit(current_independent_variable, y)
    

    plt.scatter(current_independent_variable, y,color='g')
    plt.plot(current_independent_variable, lm.predict(current_independent_variable),color='k')
    plt.title("regression fit:"+str(count))            
    plt.savefig("Regression fit for linear regression of: X"+str(count)+" .png")
    plt.show()
    
    print('Coeff of determination:'+str(count), lm.score(current_independent_variable, y))
    print('correlation is:'+str(count), math.sqrt(lm.score(current_independent_variable, y)))

input("press enter to continue")

#Task 2: Polymnomial regression
#fit for higher order polynomial:
regression_variable = list(df.keys())
multi_variables = list(zip(df[regression_variable[0]], [i**2 for i in df[regression_variable[0]]]))
X = sm.add_constant(multi_variables) # adding a constant
     
model = sm.OLS(y, X).fit()
predictions = model.predict(X) 

#residual calculation
residual =  y - predictions 
print("variance of residuals is :", np.var(residual))

#residual scatter plot
plt.scatter(predictions, residual,color='g')
plt.title("Residual scatter plot X1:  y = a0 + a1*x1 + a2*(x1**2)")
plt.savefig("Residual_scatter_plot_for_ploynomial.png")
plt.show()

#residual histogram
plt.hist(residual, bins=30)
plt.title("Histogram of residual for polynomial regression X1:  y = a0 + a1*x1 + a2*(x1**2)")        
plt.savefig("residual histogram for polynomial regression of x1.png")
plt.show()

#Chi square test
print(stats.normaltest(residual))


print_model = model.summary()
print(print_model)
#regression fit plots
fig = plt.figure(figsize=(20,12))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)
plt.title("Residual scatter for polynomial regression plot X1:  y = a0 + a1*x1 + a2*(x1**2)")
plt.savefig("Regression_fit_polynomial.png")
fig.show()

#qq plot of residuals
sm.qqplot(residual)
plt.title("QQplot for polynomial regression plot X1:  y = a0 + a1*x1 + a2*(x1**2)")
plt.savefig("QQplot for polynomial.png")
plt.show()
input("press enter to continue")

#Task 3:
#Multipvariable regression

regression_variable = list(df.keys())
multi_variables = tuple(zip(df[regression_variable[0]], df[regression_variable[1]], df[regression_variable[2]], df[regression_variable[3]], df[regression_variable[4]]))
X = sm.add_constant(multi_variables) # adding a constant
     
model = sm.OLS(y, X).fit()
predictions = model.predict(X) 

#residual calculation and qqplot 
residual =  y - predictions
print("variance of residuals is :", np.var(residual))
#residual scatter plot 
plt.scatter(predictions, residual,color='g')
plt.title("Residual scatter plot :  y = a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5")
plt.savefig("Residual_scatter_plot_for_multi_variable.png")
plt.show() 

#residual histogram
plt.hist(residual, bins=30)
plt.title("Histogram of residual for polynomial regression :  y = a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5")        
plt.savefig("residual histogram for multi_variable regression.png")
plt.show()

#Chi square test
print(stats.normaltest(residual))   

print_model = model.summary()
print(print_model)
fig = plt.figure(figsize=(20,12))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)
plt.title("Regression fit for multi variable regression plot :  y = a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5")
plt.savefig("Regression_fit_multi_variable.png")
fig.show()

sm.qqplot(residual)
plt.title("QQplot for multi variable regression plot :  y = a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5")
plt.savefig("QQplot_for_multi_variable.png")
plt.show()

input("press enter to continue")


#Task 3: remove a non dependent variable x2
#Multipvariable regression

regression_variable = list(df.keys())
multi_variables = tuple(zip(df[regression_variable[0]], df[regression_variable[2]], df[regression_variable[3]], df[regression_variable[4]]))
X = sm.add_constant(multi_variables) # adding a constant
     
model = sm.OLS(y, X).fit()
predictions = model.predict(X) 

#residual calculation and qqplot 
residual =  y - predictions

#residual scatter plot 
plt.scatter(predictions, residual,color='g')
plt.title("Residual scatter plot :  y = a0 + a1*x1 + a3*x3 + a4*x4 + a5*x5")
plt.savefig("Residual_scatter_plot_for_multi_variable_withoutx2.png")
plt.show() 

#residual histogram
plt.hist(residual, bins=30)
plt.title("Histogram of residual for polynomial regression :  y = a0 + a1*x1 + a3*x3 + a4*x4 + a5*x5")        
plt.savefig("residual histogram for multi_variable regression_withoutx2.png")
plt.show()

#Chi square test
print(stats.normaltest(residual))   

print_model = model.summary()
print(print_model)
fig = plt.figure(figsize=(20,12))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)
plt.title("Regression fit for multi variable regression plot :  y = a0 + a1*x1 + a3*x3 + a4*x4 + a5*x5")
plt.savefig("Regression_fit_multi_variable_withoutx2.png")
fig.show()

sm.qqplot(residual)
plt.title("QQplot for multi variable regression plot :  y = a0 + a1*x1 + a3*x3 + a4*x4 + a5*x5")
plt.savefig("QQplot_for_multi_variable_withoutx2.png")
plt.show()

input("press enter to continue")

"""

