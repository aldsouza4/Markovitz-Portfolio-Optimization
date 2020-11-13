import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns

st = int(input("enter no. of stocks:- \n "))
assets = []
for i in range(0, st):
    x = input("enter stock ticker:- ")
    assets.append(x)

pf_data = pd.DataFrame()
for a in assets:
    pf_data[a] = wb.DataReader(a, data_source='yahoo', start='2010-1-1')['Adj Close']

(pf_data / pf_data.iloc[0] * 100).plot(figsize=(10, 5))
log_returns = np.log(pf_data / pf_data.shift(1))

# log_returns.mean() * 250
# log_returns.cov() * 250

num_assets = len(assets)

num_ports = 1000
pfolio_returns = []
pfolio_volatilities = []
all_weights = np.zeros((num_ports, num_assets))
sharpe_array = np.zeros(num_ports)
for x in range(num_ports):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    pfolio_returns.append(np.sum(weights * log_returns.mean()) * 250)
    pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 250, weights))))
    all_weights[x, :] = weights
    sharpe_array[x] = pfolio_returns[x] / pfolio_volatilities[x]

pfolio_returns = np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)
all_weights = np.array(all_weights)
sharpe_array = np.array(sharpe_array)
print('\n Maximum Sharpe Ratio in the array: {}'.format(sharpe_array.max()))

k = sharpe_array.argmax()

max_sr_returns = pfolio_returns[sharpe_array.argmax()]
max_sr_volatilities = pfolio_volatilities[sharpe_array.argmax()]
plt.figure(figsize=(12, 8))
plt.scatter(pfolio_volatilities, pfolio_returns, c=sharpe_array, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_volatilities, max_sr_returns, c='red', s=50, edgecolors='black')
plt.show()
new_weights = all_weights[k, :] * 100
for i in range(0, st):
    print("Weight of {:<15} is {:<6} % ".format(assets[i], round(new_weights[i], 2)))


def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_returns.mean() * weights) * 250
    vol = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 250, weights)))
    sr = ret / vol
    return np.array([ret, vol, sr])


def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * -1


def check_sum(weights):
    return np.sum(weights) - 1


cons = ({'type': 'eq', 'fun': check_sum})
bounds = []
for i in range(0, st):
    x = (0, 1)
    bounds.append(x)
tuple(bounds)
xx = 1 / st
initial_guess = []
for i in range(0, num_assets):
    initial_guess.append(xx)

from scipy.optimize import minimize

opt_results = minimize(neg_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=cons)

opt_results.x
fresult = get_ret_vol_sr(opt_results.x)

print("\n The EXPECTED return is {} %".format(round(fresult[0] * 100, 2)))
print("\n The EXPECTED volitility is {} %".format(round(fresult[1] * 100, 2)))
print("\n  The MAX sharpe ratio is {} \n ".format(round(fresult[2], 2)))

weight_1 = 1 / st


weights_1 = []
for i in range(0, num_assets):
    weights_1.append(weight_1)

ret = np.sum(weights_1 * log_returns.mean()) * 250

print("The return if All the STOCKS have equal Weightage is {}  % \n ".format(round(ret * 100, 2)))

var = np.dot(weights_1, np.dot(log_returns.cov() * 250, weights_1))

voltis = np.sqrt(np.dot(weights_1, np.dot(log_returns.cov() * 250, weights_1)))

print("The Volitility if All the STOCKS have equal Weightage is {} % \n ".format(round(voltis * 100, 2)))