import pandas as pd
import numpy as np
from tqdm import tqdm
from warnings import filterwarnings
import inspect
filterwarnings('ignore')
from strategy.alpha191 import *
from numba import jit, prange
from joblib import Parallel, delayed

@jit(nopython=True,nogil=True,parallel=True)
def calc_zscore_2d(series,rolling_window):
    res=series.copy()#初始填充原始值，不是nan
    symbol_num=len(series[0,:])
    for i in prange(rolling_window,len(series)):
        temp=series[i+1-rolling_window:i+1,:]
        # s_mean=np.nanmean(temp,axis=0)
        # s_std=np.nanstd(temp,axis=0)
        for j in prange(symbol_num):
            s_mean=np.nanmean(temp[:,j])
            s_std=np.nanstd(temp[:,j])
            res[i,j] = (series[i,j]-s_mean)/max(s_std,10e-9)
    return res

def calc_zscore_2d_parallel(series,rolling_window,n_jobs=-2):
    symbol_num=len(series[0,:])
    last_num=symbol_num%10
    chunk_slice=list(np.arange(symbol_num-last_num).reshape(-1,10))
    chunk_slice.append(list(range(symbol_num-last_num,symbol_num)))
    task_res=Parallel(n_jobs=n_jobs)(delayed(calc_zscore_2d)(series[:,select_slice],rolling_window) for select_slice in chunk_slice)
    res=np.hstack(task_res)
    return res

@jit(nopython=True,nogil=True,parallel=True)
def calc_zscore_cross_section(series):
    res=series.copy()
    symbol_num=len(series[0,:])
    for i in prange(len(series)):
        temp = series[i, :]
        s_mean = np.nanmean(temp)
        s_std = np.nanstd(temp)
        for j in prange(symbol_num):
            res[i, j] = (series[i, j] - s_mean) / max(s_std, 10e-9)
    return res


class Factor_calculator:
    """
    外部定义函数，函数名即为因子名
    """

    def __init__(self, bar):
        self.bar = bar
        self.bar['return_1'] = (self.bar['close'].unstack().diff(1).shift(-1) / self.bar['close'].unstack()).stack()
        self.bar = self.bar.dropna(subset=['return_1'])

        symbol_lis = self.bar.return_1.unstack().isna().sum(axis=0)[
            self.bar.return_1.unstack().isna().sum(axis=0) == 0].index.tolist()
        print('数据有缺失', list(set(self.bar.index.get_level_values(1).unique()).difference(set(symbol_lis))))
        self.bar = self.bar.loc[(slice(None), symbol_lis), :]

        m = len(self.bar.open.unstack().columns)
        n = len(self.bar.open.unstack().index)
        benchmark_Open = pd.DataFrame(np.array(self.bar.open.unstack()['BTCUSDT'].values.tolist() * m).reshape(m, n).T)
        benchmark_close = pd.DataFrame(
            np.array(self.bar.close.unstack()['BTCUSDT'].values.tolist() * m).reshape(m, n).T)

        self.data_dict = {'Open': self.bar.open.unstack(), 'close': self.bar.close.unstack(),
                          'high': self.bar.high.unstack(),
                          'low': self.bar.low.unstack(), 'volume': self.bar.volume.unstack(),
                          'vwap': self.bar.vwap.unstack(),
                          'amount': self.bar.usd_v.unstack(),
                          'returns': bar.close.unstack() / bar.close.unstack().shift(1) - 1,
                          'benchmark_Open': benchmark_Open, 'benchmark_close': benchmark_close}

    def calculate_factor(self, factor_lis, hyper_param_lis=None):
        for func in tqdm(factor_lis):
            try:
                for res in hyper_param_lis:
                    factor, best_param, study, factor_stable = res
                    if func == factor and best_param != None:
                        sig = inspect.signature(eval(func))
                        for param in sig.parameters:
                            if param in ['Open', 'close', 'high', 'low', 'vwap', 'volume', 'amount', 'returns',
                                         'benchmark_Open', 'benchmark_close']:
                                best_param[param] = self.data_dict[param]
                        self.bar[func] = self._generate_bar_factor(func, best_param).stack()
            except:
                print(f'{func}出现问题')

        return self.bar

    def _generate_bar_factor(self, func, params):
        return eval(f"{func}(**params)")


# IC加权信号生成模块
from scipy.stats import spearmanr


class Signal_generator_IC:
    """
    如果切换成cvxpy模式需要外部定义class继承，并且定义cvxpy_solve函数
    """

    def __init__(self, data, selected_factor_lis):

        self.selected_factor_lis = selected_factor_lis
        self.data = data.loc[:, self.selected_factor_lis + ['open', 'high', 'low', 'close', 'return_1']]

    def process(self, rolling_window_1, rolling_window_2, rolling_type,
                buy_threshold, sell_threshold, ts_normalized=True, double_normalized=False, method='signal',
                ic_type='pearson', combine_type='vote'):
        print('-----normalize start-----')
        self.normalize_data(rolling_window=rolling_window_1, ts_normalized=ts_normalized,
                            double_normalized=double_normalized)
        print('-----data check-----')
        self.preprocess()
        print('-----average IC combination start-----')
        self.average_IC_combination(ic_type=ic_type)
        print('-----calculate portfolio return-----')
        self.get_portfolio_ret_df(rolling_window=rolling_window_2, rolling_type=rolling_type,
                                  buy_threshold=buy_threshold, sell_threshold=sell_threshold, combine_type=combine_type)
        print('-----generate signals-----')
        if method == 'signal':
            self.get_trade_and_signals_matrix(buy_threshold=buy_threshold, sell_threshold=sell_threshold)
        elif method == 'cvxpy':
            self.cvxpy_solve()

        return self.signals_matrix, self.trade_p_matrix, self.ts_lis, self.symbol_lis

    def normalize_data(self, rolling_window, ts_normalized=True, double_normalized=False):

        if ts_normalized == True:
            for factor in tqdm(self.selected_factor_lis):
                self.data[factor] = pd.DataFrame(calc_zscore_2d(self.data[factor].unstack().values, rolling_window),
                                                 index=self.data[factor].unstack().index,
                                                 columns=self.data[factor].unstack().columns).stack()
            # 第一个rolling_window内部的因子值剔除
            self.normalized_data = self.data.unstack().iloc[rolling_window - 1:].stack()
        else:
            self.normalized_data = self.data

        if double_normalized:
            # 截面标准化
            for factor in tqdm(self.selected_factor_lis):
                self.normalized_data[factor] = pd.DataFrame(
                    calc_zscore_cross_section(self.normalized_data[factor].unstack().values),
                    index=self.normalized_data[factor].unstack().index,
                    columns=self.normalized_data[factor].unstack().columns).stack()

    def preprocess(self):
        # 将未来收益率的缺失值（当前tick）删去
        self.normalized_data = self.normalized_data.dropna(subset=['return_1'])
        print('inf', np.isinf(self.normalized_data).sum()[np.isinf(self.normalized_data).sum() > 0])
        print('na', self.normalized_data.isna().sum()[self.normalized_data.isna().sum() > 0])
        # 将inf值替换成nan 再ffill
        # self.normalized_data.loc[np.isinf(self.normalized_data['alpha84']), 'alpha84'] = np.nan
        self.normalized_data.loc[:, self.selected_factor_lis] = self.normalized_data.loc[:,
                                                                self.selected_factor_lis].unstack().ffill(
            axis=0).dropna(how='all').fillna(0).stack()

    def average_IC_combination(self, ic_type):
        time_idx_lis = pd.unique(self.normalized_data.index.get_level_values(0))
        symbol_lis = pd.unique(self.normalized_data.index.get_level_values(1))

        n = len(time_idx_lis)
        m = len(symbol_lis)
        k = len(self.selected_factor_lis)

        self.factor_ic_df = pd.DataFrame(index=time_idx_lis, columns=self.selected_factor_lis)
        X_array = np.asarray(self.normalized_data.loc[:, self.selected_factor_lis]).reshape(n, m, k).astype(np.float32)
        y_array = np.asarray(self.normalized_data.loc[:, 'return_1']).reshape(n, m).astype(np.float32)

        for i in tqdm(range(n)):
            X = X_array[i]
            y = y_array[i]
            ic_lis = []
            for j in range(k):
                if ic_type == 'spearmanr':
                    ic_lis.append(spearmanr(X[:, j], y)[0])
                elif ic_type == 'pearson':
                    ic_lis.append(np.corrcoef(X[:, j], y)[0][1])
            self.factor_ic_df.iloc[i] = ic_lis

    def get_portfolio_ret_df(self, rolling_window, rolling_type, buy_threshold, sell_threshold, combine_type):

        # 这个tick的截面回归结果需要用到下一个tick的收益率
        # 因此我们当前tick结束后，所能用到最新的截面回归结果是上一个tick都
        factor_ic_df = self.factor_ic_df.shift(1).bfill()

        self.factor_exposure = self.normalized_data.loc[:, self.selected_factor_lis]

        # common_risk = factor_ret_df.rolling(rolling_window, min_periods=1).cov().iloc[rolling_window-1:]
        # specified_risk = specified_ret_df.apply(lambda x: x.rolling(rolling_window, min_periods=1).var())
        # specified_ret_df_rolling = specified_ret_df.rolling(rolling_window, min_periods=1).mean().iloc[rolling_window-1:]

        if rolling_type == 'ewm' and rolling_window != 1:
            self.factor_ic_df_rolling = factor_ic_df.ewm(rolling_window, min_periods=1).mean()
        else:
            self.factor_ic_df_rolling = factor_ic_df.rolling(rolling_window, min_periods=1).mean()

        self.portfolio_ret_df = pd.DataFrame(index=pd.unique(self.factor_exposure.index.get_level_values(0)),
                                             columns=pd.unique(self.factor_exposure.index.get_level_values(1)))

        if combine_type == 'weighted':
            for time_idx, row in self.factor_exposure.groupby(level=0):
                self.portfolio_ret_df.loc[time_idx] = np.dot(self.factor_ic_df_rolling.loc[time_idx], row.T)


        elif combine_type == 'vote':
            for time_idx, row in self.factor_exposure.groupby(level=0):
                portfolio_ret_df_lis = np.array([0] * row.shape[0]).astype('float32')
                for factor in self.selected_factor_lis:
                    # 计算每一个因子的该时刻的ic加权后的信号
                    portfolio_ret_df_lis += self._generate_signals(
                        row.loc[:, factor] * np.sign(self.factor_ic_df_rolling.loc[time_idx, factor]),
                        buy_threshold, sell_threshold) * np.abs(self.factor_ic_df_rolling.loc[time_idx, factor])
                self.portfolio_ret_df.loc[time_idx] = portfolio_ret_df_lis

    def get_trade_and_signals_matrix(self, buy_threshold, sell_threshold):
        n, m = self.portfolio_ret_df.shape
        self.ts_lis = np.array(self.portfolio_ret_df.index)
        self.signals_matrix = np.empty((n, m))
        self.symbol_lis = np.array(self.normalized_data.index.get_level_values(1).unique())
        self.trade_p_matrix = self.normalized_data.loc[self.ts_lis, ['open', 'high', 'low', 'close']].unstack().shift(
            -1).ffill().stack().values.reshape(n, m, 4)
        for i in range(len(self.ts_lis)):
            # 该时刻预期收益率序列
            mu = self.portfolio_ret_df.loc[self.ts_lis[i]]
            signals = self._generate_signals(mu, buy_threshold, sell_threshold)
            self.signals_matrix[i] = signals

    def _generate_signals(self, mu, buy_threshold, sell_threshold):
        """
        多因子组合信号生成
        """
        signals = np.select(condlist=[mu > mu.quantile(buy_threshold), mu < mu.quantile(sell_threshold)],
                            choicelist=[1, -1], default=0)
        return signals

    def cvxpy_solve(self):
        pass