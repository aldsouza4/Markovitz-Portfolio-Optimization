import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns


class PortfolioOptimization(object):

    def __init__(self):
        self.stock_list = []

    def enter_stocks(self):
        # self.num_stocks = int(input("enter no. of stocks:- \n "))
        # for i in range(num_stocks):
        #     x = input("Enter ticker of stock {} :-".format(i))
        #     self.stock_list.append(x)
        self.stock_list = ['STLTECH.NS', 'RAIN.NS', 'TCS.NS']
        self.num_stocks = len(self.stock_list)

    def get_data(self):
        self.enter_stocks()

        self.pf_data = pd.DataFrame()
        for a in self.stock_list:
            self.pf_data[a] = wb.DataReader(a, data_source='yahoo', start='2011-1-1')['Adj Close']

    def log_returns(self):
        self.get_data()

        (self.pf_data / self.pf_data.iloc[0] * 100).plot(figsize=(10, 5))
        self.log_returns = np.log(self.pf_data / self.pf_data.shift(1))

    def portfolio_simulation(self, num_ports=1000, show = True):
        self.log_returns()

        num_ports = 1000
        self.pfolio_returns = []
        self.pfolio_volatilities = []
        self.all_weights = np.zeros((num_ports, self.num_stocks))
        sharpe_array = np.zeros(num_ports)

        for x in range(num_ports):
            weights = np.random.random(self.num_stocks)
            weights /= np.sum(weights)
            self.pfolio_returns.append(np.sum(weights * self.log_returns.mean()) * 250)
            self.pfolio_volatilities.append(
                np.sqrt(np.dot(weights.transpose(), np.dot(self.log_returns.cov() * 250, weights))))
            self.all_weights[x, :] = weights
            sharpe_array[x] = self.pfolio_returns[x] / self.pfolio_volatilities[x]

        self.pfolio_returns = np.array(self.pfolio_returns)
        self.pfolio_volatilities = np.array(self.pfolio_volatilities)
        self.all_weights = np.array(self.all_weights)
        self.sharpe_array = np.array(sharpe_array)
        self.max_sr_returns = self.pfolio_returns[self.sharpe_array.argmax()]
        self.max_sr_volatilities = self.pfolio_volatilities[self.sharpe_array.argmax()]

        if show:
            print('\n Maximum Sharpe Ratio in the array: {}'.format(self.sharpe_array.max()))

    def plot(self):
        self.portfolio_simulation(show=False)

        plt.figure(figsize=(12, 8))
        plt.scatter(self.pfolio_volatilities, self.pfolio_returns*100, c=self.sharpe_array, cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility (%)')
        plt.ylabel('Return (%)')
        plt.scatter(self.max_sr_volatilities, self.max_sr_returns*100, c='red', s=50, edgecolors='black')
        plt.title("Volatility vs return")
        plt.show()

    def weight_optimized_pft(self, show = False):
        self.portfolio_simulation(show=False)

        new_weights = self.all_weights[self.sharpe_array.argmax(), :] * 100
        if show:
            for i in range(self.num_stocks):
                print("Weight of {:<15} is {:<6} % ".format(self.stock_list[i], round(new_weights[i], 2)))

    def final_result(self, plot=False, show=True):
        if plot:
            self.plot()

        self.weight_optimized_pft()

        new_weights = self.all_weights[self.sharpe_array.argmax(), :] * 100
        portfolio_returns = np.sum(self.log_returns.mean() * new_weights) * 250
        portfolio_volatility = np.sqrt(np.dot(new_weights.T, np.dot(self.log_returns.cov() * 250, new_weights)))

        max_sharpe = portfolio_returns / portfolio_volatility
        equal_weight = 1 / self.num_stocks
        equal_weights_list = []
        for i in range(self.num_stocks):
            equal_weights_list.append(equal_weight)

        ret_equal = np.sum(equal_weights_list * self.log_returns.mean()) * 250
        volatility_equal = np.sqrt(np.dot(equal_weights_list, np.dot(self.log_returns.cov() * 250, equal_weights_list)))

        result_dict = {'expect_return': round(portfolio_returns, 2),
                       'expect_volatility': round(portfolio_volatility, 2), 'max_sharpe': round(max_sharpe, 2),
                       'equal_portfolio_return': round(ret_equal * 100, 2),
                       'equal_portfolio_volatility': round(volatility_equal * 100, 2)}

        if show:
            print("\n The EXPECTED return is {} %".format(round(portfolio_returns, 2)))
            print("\n The EXPECTED volatility is {} %".format(round(portfolio_volatility, 2)))
            print("\n The MAX sharpe ratio is {} \n ".format(round(max_sharpe, 2)))

            print("The return if All the STOCKS have equal Weightage is {}  % \n ".format(round(ret_equal * 100, 2)))
            print("The Volatility if All the STOCKS have equal Weightage is {} % \n ".format(round(volatility_equal * 100, 2)))

        return result_dict


if __name__ == '__main__':
    t = PortfolioOptimization()
    print(t.final_result(show=False))
