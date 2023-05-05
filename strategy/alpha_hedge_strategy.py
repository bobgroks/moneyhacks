from strategy.strategy_utils import Factor_calculator, Signal_generator_IC
import numpy as np
from strategy.alpha191 import *


class Strategy:

    def __init__(self, bar, rolling_window, factor_lis, hyper_param_lis):
        # considering we will remove the bar[:rolling_window-1] because of normalization.
        # And we need rolling_window IC to calculate the IC. So we need 2*rolling_window-1 data point at least.
        # Which means the bar should contain the latest 2*rolling_window-1 snapshot
        self.bar = bar
        self.rolling_window = rolling_window
        self.factor_lis = factor_lis
        self.hyper_param_lis = hyper_param_lis
        factor_calculator = Factor_calculator(bar)
        self.data = factor_calculator.calculate_factor(self.factor_lis, self.hyper_param_lis)


    def generate_signals(self):
        signal_generator = Signal_generator_IC(self.data, self.factor_lis)
        # print('-----normalize start-----')
        signal_generator.normalize_data(rolling_window=self.rolling_window, ts_normalized=True, double_normalized=False)
        # print('-----data check-----')
        signal_generator.preprocess()
        # print('-----average IC combination start-----')
        signal_generator.average_IC_combination(ic_type='spearmanr')
        # print('-----calculate portfolio return-----')
        signal_generator.get_portfolio_ret_df(rolling_window=self.rolling_window, rolling_type='ewm', buy_threshold=0.8,
                                              sell_threshold=0.2, combine_type='weighted')
        # generate signals

        signal_generator.ts_lis = np.array(signal_generator.portfolio_ret_df.index)
        signal_generator.symbol_lis = np.array(signal_generator.normalized_data.index.get_level_values(1).unique())
        mu = signal_generator.portfolio_ret_df.loc[signal_generator.ts_lis[-1]] # The latest one
        signals = signal_generator._generate_signals(mu, 0.8, 0.2)

        return signals, signal_generator.symbol_lis


