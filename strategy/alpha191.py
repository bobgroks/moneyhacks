import numpy as np
from numpy import log
from scipy.stats import rankdata


def Log(sr):
    # 自然对数函数
    return np.log(sr)


def Rank(sr):
    # 列-升序排序并转化成百分比
    return sr.rank(axis=1, method='min', pct=True)


def Delta(sr, period):
    # period日差分
    return sr.diff(period)


def Delay(sr, period):
    # period阶滞后项
    return sr.shift(period)


def Corr(x, y, window):
    # window日滚动相关系数
    return x.rolling(window).corr(y)


def Cov(x, y, window):
    # window日滚动协方差
    return x.rolling(window).cov(y)


def Sum(sr, window):
    # window日滚动求和
    return sr.rolling(window).sum()


def Prod(sr, window):
    # window日滚动求乘积
    return sr.rolling(window).apply(lambda x: np.prod(x))


def Mean(sr, window):
    # window日滚动求均值
    return sr.rolling(window).mean()


def Std(sr, window):
    # window日滚动求标准差
    return sr.rolling(window).std()


def Tsrank(sr, window):
    # window日序列末尾值的顺位
    return sr.rolling(window).apply(lambda x: rankdata(x)[-1])


def Tsmax(sr, window):
    # window日滚动求最大值
    return sr.rolling(window).max()


def Tsmin(sr, window):
    # window日滚动求最小值
    return sr.rolling(window).min()


def Sign(sr):
    # 符号函数
    return np.sign(sr)


def Max(sr1, sr2):
    return np.maximum(sr1, sr2)


def Min(sr1, sr2):
    return np.minimum(sr1, sr2)


def Rowmax(sr):
    return sr.max(axis=1)


def Rowmin(sr):
    return sr.min(axis=1)


def Sma(sr, n, m):
    # sma均值
    return sr.ewm(alpha=m / n, adjust=False).mean()


def Abs(sr):
    # 求绝对值
    return sr.abs()


def Sequence(n):
    # 生成 1~n 的等差序列
    return np.arange(1, n + 1)


def Regbeta(sr, x):
    window = len(x)
    return sr.rolling(window).apply(lambda y: np.polyfit(x, y, deg=1)[0])


def Decaylinear(sr, window):
    weights = np.array(range(1, window + 1))
    sum_weights = np.sum(weights)
    return sr.rolling(window).apply(lambda x: np.sum(weights * x) / sum_weights)


def Lowday(sr, window):
    return sr.rolling(window).apply(lambda x: len(x) - x.values.argmin())


def Highday(sr, window):
    return sr.rolling(window).apply(lambda x: len(x) - x.values.argmax())


def Wma(sr, window):
    weights = np.array(range(window - 1, -1, -1))
    weights = np.power(0.9, weights)
    sum_weights = np.sum(weights)

    return sr.rolling(window).apply(lambda x: np.sum(weights * x) / sum_weights)


def Count(cond, window):
    return cond.rolling(window).apply(lambda x: x.sum())


def Sumif(sr, window, cond):
    sr[~cond] = 0
    return sr.rolling(window).sum()


def Returns(df):
    return df.rolling(2).apply(lambda x: x.iloc[-1] / x.iloc[0]) - 1


def alpha001(Open, close, volume, window=6):  # 平均1751个数据
    ##### (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - Open) / Open)), 6))####
    return (-1 * Corr(Rank(Delta(log(volume), 1)), Rank(((close - Open) / Open)), window))

def alpha002(close, low, high, delta_interval=1):  # 1783
    ##### -1 * delta((((close-low)-(high-close))/(high-low)),1))####
    return -1 * Delta((((close - low) - (high - close)) / (high - low)), delta_interval)

def alpha003(close, low, high, window=6):
    ##### SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6) ####
    cond1 = (close == Delay(close, 1))
    cond2 = (close > Delay(close, 1))
    cond3 = (close < Delay(close, 1))
    part = close.copy(deep=True)
    part[cond1] = 0
    part[cond2] = close - Min(low, Delay(close, 1))
    part[cond3] = close - Max(high, Delay(close, 1))
    return Sum(part, window)

def alpha004(close, volume, window_1=8, window_2=8, window_3=2, window_4=20):
    #####((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) <((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME /MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))
    cond1 = ((Sum(close, window_1) / window_1 + Std(close, window_2)) < Sum(close, window_3) / window_3)
    cond2 = ((Sum(close, window_1) / window_1 + Std(close, window_2)) > Sum(close, window_3) / window_3)
    cond3 = ((Sum(close, window_1) / window_1 + Std(close, window_2)) == Sum(close, window_3) / window_3)
    cond4 = (volume / Mean(volume, window_4) >= 1)
    part = close.copy(deep=True)
    part[cond1] = -1
    part[cond2] = 1
    part[cond3] = -1
    part[cond3 & cond4] = 1

    return part

def alpha005(volume, high, window_1=5, window_2=3):  # 1447
    ####(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))###
    return -1 * Tsmax(Corr(Tsrank(volume, window_1), Tsrank(high, window_1), window_1), window_2)

def alpha006(Open, high, delta_interval=4):  # 1779
    ####(RANK(SIGN(DELTA((((Open * 0.85) + (HIGH * 0.15))), 4)))* -1)###
    return -1 * Rank(Sign(Delta(((Open * 0.85) + (high * 0.15)), delta_interval)))

def alpha007(close, volume, vwap, window=3):  # 1782
    ####((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))###
    return ((Rank(Tsmax((vwap - close), window)) + Rank(Tsmin((vwap - close), window))) * Rank(
        Delta(volume, window)))

def alpha008(high, low, vwap, delta_interval=4):  # 1779
    ####RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)###
    return Rank(Delta(((((high + low) / 2) * 0.2) + (vwap * 0.8)), delta_interval) * -1)

def alpha009(high, low, volume, n=7, m=2):  # 1790
    ####SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)###
    return Sma(((high + low) / 2 - (Delay(high, 1) + Delay(low, 1)) / 2) * (
                high - low) / volume, n, m)

def alpha010(returns, close, window_1=20, window_2=5):
    ####(RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))###
    cond = (returns < 0)
    part = returns.copy(deep=True)
    part[cond] = Std(returns, window_1)
    part[~cond] = close
    part = part ** 2

    return Rank(Tsmax(part, window_2))

def alpha011(close, low, high, volume, window=6):  # 1782
    ####SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)###
    return Sum(((close - low) - (high - close)) / (high - low) * volume, window)

def alpha012(Open, close, vwap, window=10):  # 1779
    ####(RANK((Open - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))###
    return (Rank((Open - (Sum(vwap, window) / window)))) * (-1 * (Rank(Abs((close - vwap)))))

def alpha013(high, low, vwap):  # 1790
    ####(((HIGH * LOW)^0.5) - VWAP)###
    return (((high * low) ** 0.5) - vwap)

def alpha014(close):  # 1776
    ####CLOSE-DELAY(CLOSE,5)###
    return close - Delay(close, 5)

def alpha015(Open, close):  # 1790
    ####Open/DELAY(CLOSE,1)-1###
    return Open / Delay(close, 1) - 1

def alpha016(volume, vwap, window=5):  # 1736
    ####(-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))###
    return (-1 * Tsmax(Rank(Corr(Rank(volume), Rank(vwap), window)), window))

def alpha017(close, vwap, window=15, delta_interval=5):  # 1776
    ####RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)###
    return Rank((vwap - Tsmax(vwap, window))) ** Delta(close, delta_interval)

def alpha018(close):  # 1776
    ####CLOSE/DELAY(CLOSE,5)###
    return close / Delay(close, 5)

def alpha019(close, delay_interval=5):
    ####(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))###
    cond1 = (close < Delay(close, delay_interval))
    cond2 = (close == Delay(close, delay_interval))
    cond3 = (close > Delay(close, delay_interval))
    part = close.copy(deep=True)
    part[cond1] = (close - Delay(close, delay_interval)) / Delay(close, delay_interval)
    part[cond2] = 0
    part[cond3] = (close - Delay(close, delay_interval)) / close

    return part

def alpha020(close, delay_interval=6):  # 1773
    ####(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100###
    return (close - Delay(close, delay_interval)) / Delay(close, delay_interval) * 100

def alpha021(close, window=6):  # reg？
    ####REGBETA(MEAN(CLOSE,6),SEQUENCE(6))###
    return Regbeta(Mean(close, window), Sequence(window))

def alpha022(close, window=6, delay_interval=3, n=12, m=1):  # 1736
    ####SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)###
    return Sma(((close - Mean(close, window)) / Mean(close, window) - Delay(
        (close - Mean(close, window)) / Mean(close, window), delay_interval)), n, m)

def alpha023(close, delay_interval=1, window=20, n=20, m=1):
    ####SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) / (SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) + SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100###
    cond = (close > Delay(close, delay_interval))
    part1 = close.copy(deep=True)
    part1[cond] = Std(close, window)
    part1[~cond] = 0
    part2 = close.copy(deep=True)
    part2[~cond] = Std(close, window)
    part2[cond] = 0

    return 100 * Sma(part1, n, m) / (Sma(part1, n, m) + Sma(part2, n, m))

def alpha024(close, delay_interval=5, n=5, m=1):  # 1776
    ####SMA(CLOSE-DELAY(CLOSE,5),5,1)###
    return Sma(close - Delay(close, delay_interval), n, m)

def alpha025(close, volume, returns, delta_interval=7, window_1=20, window_2=9, window_3=250):  # 886
    ####((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))###
    return ((-1 * Rank(
        (Delta(close, delta_interval) * (1 - Rank(Decaylinear((volume / Mean(volume, window_1)), window_2)))))) * (
                        1 + Rank(Sum(returns, window_3))))

def alpha026(close, vwap, window_1=7, delay_interval=5, window_2=230):
    ####((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))###
    return ((((Sum(close, window_1) / window_1) - close)) + ((Corr(vwap, Delay(close, delay_interval), window_2))))

def alpha027(close, delay_interval_1=3, delay_interval_2=6, window=12):
    ####WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)###
    A = (close - Delay(close, delay_interval_1)) / Delay(close, delay_interval_2) * 100 + (
                close - Delay(close, delay_interval_2)) / Delay(close, delay_interval_2) * 100
    return Wma(A, window)

def alpha028(close, low, high, window=9, n=3, m=1):  # 1728
    ####3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)###
    return 3 * Sma((close - Tsmin(low, window)) / (Tsmax(high, window) - Tsmin(low, window)) * 100, n,
                   m) - 2 * Sma(
        Sma((close - Tsmin(low, window)) / (Tsmax(high, window) - Tsmin(low, window)) * 100, n, m), n, m)

def alpha029(close, volume, delay_interval=6):  # 1773
    ####(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME###
    return (close - Delay(close, delay_interval)) / Delay(close, delay_interval) * volume

def alpha030():  # reg？
    ####WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML， 60))^2,20)###
    return 0

def alpha031(close, window=12):  # 1714
    ####(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100###
    return (close - Mean(close, window)) / Mean(close, window) * 100

def alpha032(high, volume, window=3):  # 1505
    ####(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))###
    return (-1 * Sum(Rank(Corr(Rank(high), Rank(volume), window)), window))

def alpha033(low, returns, volume, window_1=5, window_2=240, window_3=20):  # 904  数据量较少
    ####((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *TSRANK(VOLUME, 5))###
    return ((((-1 * Tsmin(low, window_1)) + Delay(Tsmin(low, window_1), window_1)) * Rank(
        ((Sum(returns, window_2) - Sum(returns, window_3)) / (window_2-window_3)))) * Tsrank(volume, window_1))

def alpha034(close, window=12):  # 1714
    ####MEAN(CLOSE,12)/CLOSE###
    return Mean(close, window) / close

def alpha035(Open, volume, delta_interval=1, window_1=15, window_2=17, window_3=7):  # 1790    (Open * 0.65) +(Open *0.35)有问题
    ####(MIN(RANK(DECAYLINEAR(DELTA(Open, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((Open * 0.65) +(Open *0.35)), 17),7))) * -1)###
    return (Min(Rank(Decaylinear(Delta(Open, delta_interval), window_1)),
                Rank(Decaylinear(Corr((volume), ((Open * 0.65) + (Open * 0.35)), window_2), window_3))) * -1)

def alpha036(volume, vwap, window_1=6, window_2=2):  # 1714
    ####RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP),6), 2))###
    return Rank(Sum(Corr(Rank(volume), Rank(vwap), window_1), window_2))

def alpha037(Open, returns, window=5, delay_interval=10):  # 1713
    ####(-1 * RANK(((SUM(Open, 5) * SUM(RET, 5)) - DELAY((SUM(Open, 5) * SUM(RET, 5)), 10))))###
    return (-1 * Rank(
        ((Sum(Open, window) * Sum(returns, window)) - Delay((Sum(Open, window) * Sum(returns, window)), delay_interval))))

def alpha038(high, close, window=20, delta_interval=2):
    ####(((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
    cond = ((Sum(high, window) / window) < high)
    part = close.copy(deep=True)
    part[cond] = -1 * Delta(high, delta_interval)
    part[~cond] = 0

    return part

def alpha039(Open, close, vwap, volume, delta_interval=2, window_1=8, window_2=180, window_3=37, window_4=14, window_5=12):  # 1666
    ####((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (Open * 0.7)),SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)###
    return ((Rank(Decaylinear(Delta((close), delta_interval), window_1)) - Rank(
        Decaylinear(Corr(((vwap * 0.3) + (Open * 0.7)), Sum(Mean(volume, window_2), window_3), window_4), window_5))) * -1)

def alpha040(close, volume, delay_interval=1, window=26):
    ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100###
    cond = (close > Delay(close, delay_interval))
    part1 = close.copy(deep=True)
    part1[cond] = volume
    part1[~cond] = 0
    part2 = close.copy(deep=True)
    part2[~cond] = volume
    part2[cond] = 0

    return Sum(part1, window) / Sum(part2, window) * 100

def alpha041(vwap, delta_interval=3, window=5):  # 1782
    ####(RANK(MAX(DELTA((VWAP), 3), 5))* -1)###
    return (Rank(Tsmax(Delta((vwap), delta_interval), window)) * -1)

def alpha042(high, volume, window=10):  # 1399  数据量较少
    ####((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))###
    return ((-1 * Rank(Std(high, window))) * Corr(high, volume, window))

def alpha043(close, volume, delay_interval=1, window=6):
    ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)###
    cond1 = (close > Delay(close, delay_interval))
    cond2 = (close < Delay(close, delay_interval))
    cond3 = (close == Delay(close, delay_interval))
    part = close.copy(deep=True)  # pd.Series(np.zeros(close.shape))
    part[cond1] = volume
    part[cond2] = -volume
    part[cond3] = 0

    return Sum(part, window)

def alpha044(low, volume, vwap, window_1=10, window_2=7, window_3=6, window_4=4, delta_interval=3, window_5=10, window_6=15):  # 1748
    ####(TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP),3), 10), 15))###
    return (Tsrank(Decaylinear(Corr(((low)), Mean(volume, window_1), window_2), window_3), window_4) + Tsrank(
        Decaylinear(Delta((vwap), delta_interval), window_5), window_6))

def alpha045(close, Open, vwap, volume, window_1=150, window_2=15, delta_interval=1):  # 1070  数据量较少
    ####(RANK(DELTA((((CLOSE * 0.6) + (Open *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))###
    return (Rank(Delta((((close * 0.6) + (Open * 0.4))), delta_interval)) * Rank(
        Corr(vwap, Mean(volume, window_1), window_2)))

def alpha046(close, window_1=3, window_2=6, window_3=12, window_4=24):  # 1630
    ####(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)###
    return (Mean(close, window_1) + Mean(close, window_2) + Mean(close, window_3) + Mean(close, window_4)) / (
                4 * close)

def alpha047(high, close, low, window=6, n=9, m=1):  # 1759
    ####SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)###
    return Sma((Tsmax(high, window) - close) / (Tsmax(high, window) - Tsmin(low, window)) * 100, n, m)

def alpha048(close, volume, delay_interval_1=0, delay_interval_2=1, delay_interval_3=2, window_1=5, window_2=20):  # 1657
    ####(-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))###
    return (-1 * ((Rank(((Sign((close - Delay(close, delay_interval_1+1))) + Sign(
        (Delay(close, delay_interval_2) - Delay(close, delay_interval_2+1)))) + Sign(
        (Delay(close, delay_interval_3) - Delay(close, delay_interval_3+1)))))) * Sum(volume, window_1)) / Sum(volume, window_2))

def alpha049(high, low, close, delay_interval=1, window=12):
    ####SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12) / (SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12) + SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    cond = ((high + low) > (Delay(high, delay_interval) + Delay(low, delay_interval)))
    part1 = close.copy(deep=True)  # pd.Series(np.zeros(close.shape))
    part1[cond] = 0
    part1[~cond] = Max(Abs(high - Delay(high, delay_interval)), Abs(low - Delay(low, delay_interval)))
    part2 = close.copy(deep=True)
    part2[~cond] = 0
    part2[cond] = Max(Abs(high - Delay(high, delay_interval)), Abs(low - Delay(low, delay_interval)))

    return Sum(part1, window) / (Sum(part1, window) + Sum(part2, window))

def alpha050(high, low, close, delay_interval=1, window=12):
    ####SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))###
    cond = ((high + low) <= (Delay(high, delay_interval) + Delay(low, delay_interval)))
    part1 = close.copy(deep=True)
    part1[cond] = 0
    part1[~cond] = Max(Abs(high - Delay(high, delay_interval)), Abs(low - Delay(low, delay_interval)))
    part2 = close.copy(deep=True)
    part2[~cond] = 0
    part2[cond] = Max(Abs(high - Delay(high, delay_interval)), Abs(low - Delay(low, delay_interval)))

    return (Sum(part1, window) - Sum(part2, window)) / (Sum(part1, window) + Sum(part2, window))

def alpha051(high, low, close, delay_interval=1, window=12):
    ####SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12) / (SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))###
    cond = ((high + low) <= (Delay(high, delay_interval) + Delay(low, delay_interval)))
    part1 = close.copy(deep=True)
    part1[cond] = 0
    part1[~cond] = Max(Abs(high - Delay(high, delay_interval)), Abs(low - Delay(low, delay_interval)))
    part2 = close.copy(deep=True)
    part2[~cond] = 0
    part2[cond] = Max(Abs(high - Delay(high, delay_interval)), Abs(low - Delay(low, delay_interval)))

    return Sum(part1, window) / (Sum(part1, window) + Sum(part2, window))

def alpha052(high, low, close, delay_interval=1,  window=26):  # 1611
    ####SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)*100###
    return Sum(Max(high - Delay((high + low + close) / 3, delay_interval), 0), window) / Sum(
        Max(Delay((high + low + close) / 3, delay_interval) - low, 0), window) * 100

def alpha053(close, delay_interval=1, window=12):
    ####COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100###
    cond = (close > Delay(close, delay_interval))
    return Count(cond, window) / window * 100

def alpha054(close, Open, window=10):  # 1729
    ####(-1 * RANK((STD(ABS(CLOSE - Open)) + (CLOSE - Open)) + CORR(CLOSE, Open,10)))###
    return (-1 * Rank(
        ((Abs(close - Open)).std() + (close - Open)) + Corr(close, Open, window)))

def alpha055(Open, high, close, low):  # 公式有问题
    ###SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-Open)/2+DELAY(CLOSE,1)-DELAY(Open,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2 + ABS(DELAY(CLOSE,1)-DELAY(Open,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(Open,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(Open,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
    A = Abs(high - Delay(close, 1))
    B = Abs(low - Delay(close, 1))
    C = Abs(high - Delay(low, 1))
    cond1 = ((A > B) & (A > C))
    cond2 = ((B > C) & (B > A))
    cond3 = ((C >= A) & (C >= B))
    part0 = 16 * (close + (close - Open) / 2 - Delay(Open, 1))
    part1 = close.copy(deep=True)
    part1.loc[:, :] = 0
    part1[cond1] = Abs(high - Delay(close, 1)) + Abs(low - Delay(close, 1)) / 2 + Abs(
        Delay(close, 1) - Delay(Open, 1)) / 4
    part1[cond2] = Abs(low - Delay(close, 1)) + Abs(high - Delay(close, 1)) / 2 + Abs(
        Delay(close, 1) - Delay(Open, 1)) / 4
    part1[cond3] = Abs(high - Delay(low, 1)) + Abs(Delay(close, 1) - Delay(Open, 1)) / 4
    part2 = Max(Abs(high - Delay(close, 1)), Abs(low - Delay(close, 1)))

    return Sum(part0 / part1 * part2, 20)

def alpha056(Open, high, low, volume, close, window_1=12, window_2=19, window_3=40, window_4=13):
    ####(RANK((Open - TSMIN(Open, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),SUM(MEAN(VOLUME,40), 19), 13))^5)))###
    A = Rank((Open - Tsmin(Open, window_1)))
    B = Rank((Rank(Corr(Sum(((high + low) / 2), window_2), Sum(Mean(volume, window_3), window_2), window_4)) ** 5))
    cond = (A < B)
    part = close.copy(deep=True)
    part[cond] = 1
    part[~cond] = 0
    return part

def alpha057(close, low, high, window=9, n=3, m=1):  # 1736
    ####SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)###
    return Sma((close - Tsmin(low, window)) / (Tsmax(high, window) - Tsmin(low, window)) * 100, n, m)

def alpha058(close, window=20, delay_interval=1):
    ####COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100###
    cond = (close > Delay(close, delay_interval))

    return Count(cond, window) / window * 100

def alpha059(close, low, window=20, delay_interval=1):
    ####SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)###
    cond1 = (close == Delay(close, delay_interval))
    cond2 = (close > Delay(close, delay_interval))
    cond3 = (close < Delay(close, delay_interval))
    part = close.copy(deep=True)
    part[cond1] = 0
    part[cond2] = close - Min(low, Delay(close, delay_interval))
    part[cond3] = close - Max(low, Delay(close, delay_interval))

    return Sum(part, window)

def alpha060(close, low, high, volume, window=20):  # 1635
    ####SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)###
    return Sum(((close - low) - (high - close)) / (high - low) * volume, window)

def alpha061(vwap, low, volume, delta_interval=1, window_1=12, window_2=80, window_3=8, window_4=17):  # 1790
    ####(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)###
    return (Max(Rank(Decaylinear(Delta(vwap, delta_interval), window_1)),
                Rank(Decaylinear(Rank(Corr((low), Mean(volume, window_2), window_3)), window_4))) * -1)

def alpha062(high, volume, window=5):  # 1479
    ####(-1 * CORR(HIGH, RANK(VOLUME), 5))###
    return (-1 * Corr(high, Rank(volume), window))

def alpha063(close, delay_interval=1, n=6, m=1):  # 1789
    ####SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100###
    return Sma(Max(close - Delay(close, delay_interval), 0), n, m) / Sma(Abs(close - Delay(close, delay_interval)), n,
                                                                      m) * 100

def alpha064(close, vwap, volume, window_1=4, window_2=4, window_3=60, window_4=4, window_5=13, window_6=14):  # 1774
    ####(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)###
    return (Max(Rank(Decaylinear(Corr(Rank(vwap), Rank(volume), window_1), window_2)),
                Rank(Decaylinear(Tsmax(Corr(Rank(close), Rank(Mean(volume, window_3)), window_4), window_5), window_6))) * -1)

def alpha065(close, window=6):  # 1759
    ####MEAN(CLOSE,6)/CLOSE###
    return Mean(close, window) / close

def alpha066(close, window=6):  # 1759
    ####(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100###
    return (close - Mean(close, window)) / Mean(close, window) * 100

def alpha067(close, n=24, m=1, delay_interval=1):  # 1759
    ####SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100###
    a1 = Sma(Max(close - Delay(close, delay_interval), 0), n, m)
    a2 = Sma(Abs(close - Delay(close, delay_interval)), n, m)
    return a1 / a2 * 100

def alpha068(high, low, volume, delay_interval=1, n=15, m=2):  # 1790
    ####SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)###
    return Sma(((high + low) / 2 - (Delay(high, delay_interval) + Delay(low, delay_interval)) / 2) * (
                high - low) / volume, n, m)

def alpha069(Open, close, high, low, window=20):
    ####(SUM(DTM,20)>SUM(DBM,20)？ (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)： (SUM(DTM,20)=SUM(DBM,20)？0： (SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))###
    ####DTM (Open<=DELAY(Open,1)?0:MAX((HIGH-Open),(Open-DELAY(Open,1))))
    ####DBM (Open>=DELAY(Open,1)?0:MAX((Open-LOW),(Open-DELAY(Open,1))))
    cond1 = (Open <= Delay(Open, 1))
    cond2 = (Open >= Delay(Open, 1))

    DTM = close.copy(deep=True)
    DTM[cond1] = 0
    DTM[~cond1] = Max((high - Open), (Open - Delay(Open, 1)))

    DBM = close.copy(deep=True)
    DBM[cond2] = 0
    DBM[~cond2] = Max((Open - low), (Open - Delay(Open, 1)))

    cond3 = (Sum(DTM, window) > Sum(DBM, window))
    cond4 = (Sum(DTM, window) == Sum(DBM, window))
    cond5 = (Sum(DTM, window) < Sum(DBM, window))
    part = close.copy(deep=True)
    part[cond3] = (Sum(DTM, window) - Sum(DBM, window)) / Sum(DTM, window)
    part[cond4] = 0
    part[cond5] = (Sum(DTM, window) - Sum(DBM, window)) / Sum(DBM, window)
    return part

def alpha070(amount, window=6):  # 1759
    ####STD(AMOUNT,6)###
    return Std(amount, window)

def alpha071(close, window=24):  # 1630
    ####(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100###
    return (close - Mean(close, window)) / Mean(close, window) * 100

def alpha072(high, close, low, window=6, n=15, m=1):  # 1759
    ####SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)###
    return Sma((Tsmax(high, window) - close) / (Tsmax(high, window) - Tsmin(low, window)) * 100, n, m)

def alpha073(close, volume, vwap, window_1, window_2, window_3, ):  # 1729
    ####((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) - RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)###
    return ((Tsrank(Decaylinear(Decaylinear(Corr((close), volume, 10), 16), 4), 5) - Rank(
        Decaylinear(Corr(vwap, Mean(volume, 30), 4), 3))) * -1)

def alpha074(low, vwap, volume, window_1=20, window_2=40, window_3=7, window_4=6):  # 1402
    ####(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))###
    return (Rank(Corr(Sum(((low * 0.35) + (vwap * 0.65)), window_1), Sum(Mean(volume, window_2), window_1), window_3)) + Rank(
        Corr(Rank(vwap), Rank(volume), window_4)))

def alpha075(close, Open, benchmark_close, benchmark_Open, window=50):
    ####COUNT(CLOSE>Open & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOpen,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOpen,50)###
    return Count(((close > Open) & (benchmark_close < benchmark_Open)), window) / Count(
        (benchmark_close < benchmark_Open), window)

def alpha076(close, volume, window=20, delay_interval=1):  # 1650
    ####STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)###
    return Std(Abs((close / Delay(close, delay_interval) - 1)) / volume, window) / Mean(
        Abs((close / Delay(close, delay_interval) - 1)) / volume, window)

def alpha077(high, low, vwap, volume, window_1=20, window_2=40, window_3=3, window_4=6):  # 1797
    #### MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)),RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))###
    return Min(Rank(Decaylinear(((((high + low) / 2) + high) - (vwap + high)), window_1)),
               Rank(Decaylinear(Corr(((high + low) / 2), Mean(volume, window_2), window_3), window_4)))

def alpha078(high, low, close, window=12):  # 1637
    ####((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))###
    return ((high + low + close) / 3 - Mean((high + low + close) / 3, window)) / (
                0.015 * Mean(Abs(close - Mean((high + low + close) / 3, window)), window))

def alpha079(close, delay_interval=1, n=12, m=1):  # 1789
    ####SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100###
    return Sma(Max(close - Delay(close, delay_interval), 0), n, m) / Sma(Abs(close - Delay(close, delay_interval)), n,
                                                                       m) * 100

def alpha080(volume, delay_interval=5):  # 1776
    ####(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100###
    return (volume - Delay(volume, delay_interval)) / Delay(volume, delay_interval) * 100

def alpha081(volume, n=21, m=2):  # 1797
    ####SMA(VOLUME,21,2)###
    return Sma(volume, n, m)

def alpha082(high, close, low, window=6, n=20, m=1):  # 1759
    ####SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)###
    return Sma((Tsmax(high, window) - close) / (Tsmax(high, window) - Tsmin(low, window)) * 100, n, m)

def alpha083(high, volume, window=5):  # 1766
    ####(-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))###
    return (-1 * Rank(Cov(Rank(high), Rank(volume), window)))

def alpha084(close, volume, delay_interval=1, window=20):
    ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)###
    cond1 = (close > Delay(close, delay_interval))
    cond2 = (close < Delay(close, delay_interval))
    cond3 = (close == Delay(close, delay_interval))
    part = close.copy(deep=True)
    part[cond1] = volume
    part[cond2] = 0
    part[cond3] = -volume
    return Sum(part, window)

def alpha085(close, volume, delta_interval=7, window_1=20, window_2=8):  # 1657
    ####(TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))###
    return (Tsrank((volume / Mean(volume, window_1)), window_1) * Tsrank((-1 * Delta(close, delta_interval)), window_2))

def alpha086(close, delay_interval_1=10, delay_interval_2=1):
    ####((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) :(((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ?1 : ((-1 * 1) *(CLOSE - DELAY(CLOSE, 1)))))
    A = (((Delay(close, delay_interval_1*2) - Delay(close, delay_interval_1)) / delay_interval_1) - ((Delay(close, delay_interval_1) - close) / delay_interval_1))
    cond1 = (A > 0.25)
    cond2 = (A < 0.0)
    cond3 = ((0 <= A) & (A <= 0.25))
    part = close.copy(deep=True)
    part[cond1] = -1
    part[cond2] = 1
    part[cond3] = -1 * (close - Delay(close, delay_interval_2))
    return part

def alpha087(Open, high, low, vwap, window_1=7, window_2=11, delta_interval=4):  # 1741
    ####((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) /(Open - ((HIGH + LOW) / 2))), 11), 7)) * -1)###
    return ((Rank(Decaylinear(Delta(vwap, delta_interval), window_1)) + Tsrank(Decaylinear(
        ((((low * 0.9) + (low * 0.1)) - vwap) / (Open - ((high + low) / 2))), window_2),
                                                                window_1)) * -1)

def alpha088(close, delay_interval=20):  # 1745
    ####(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100###
    return (close - Delay(close, delay_interval)) / Delay(close, delay_interval) * 100

def alpha089(close, n=12, m=2):  # 1797
    ####2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))###
    return 2 * (Sma(close, 13, 2) - Sma(close, 27, 2) - Sma(
        Sma(close, 13, 2) - Sma(close, 27, 2), n, m))

def alpha090(vwap, volume, window=5):  # 1745
    ####(RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)###
    return (Rank(Corr(Rank(vwap), Rank(volume), window)) * -1)

def alpha091(close, volume, low, window_1=5, window_2=40):  # 1745
    ####((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)###
    return ((Rank((close - Tsmax(close, window_1))) * Rank(Corr((Mean(volume, window_2)), low, window_1))) * -1)

def alpha092(close, vwap, volume, delta_interval=2, window_1=3, window_2=180, window_3=13, window_4=5, window_5=15):  # 1786
    ####(MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)###
    return (Max(Rank(Decaylinear(Delta(((close * 0.35) + (vwap * 0.65)), delta_interval), window_1)),
                Tsrank(Decaylinear(Abs(Corr((Mean(volume, window_2)), close, window_3)), window_4), window_5)) * -1)

def alpha093(Open, close, low, delay_interval=1, window=20):
    ####SUM((Open>=DELAY(Open,1)?0:MAX((Open-LOW),(Open-DELAY(Open,1)))),20)###
    cond = (Open >= Delay(Open, delay_interval))
    part = close.copy(deep=True)
    part[cond] = 0
    part[~cond] = Max((Open - low), (Open - Delay(Open, delay_interval)))
    return Sum(part, window)

def alpha094(close, volume, delay_interval=1, window=30):
    ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)###
    cond1 = (close > Delay(close, delay_interval))
    cond2 = (close < Delay(close, delay_interval))
    cond3 = (close == Delay(close, delay_interval))
    part = close.copy(deep=True)
    part[cond1] = volume
    part[cond2] = -1 * volume
    part[cond3] = 0
    return Sum(part, window)

def alpha095(amount, window=20):  # 1657
    ####STD(AMOUNT,20)###
    return Std(amount, 20)

def alpha096(close, low, high, window=9, n=3, m=1):  # 1736
    ####SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)###
    return Sma(Sma((close - Tsmin(low, window)) / (Tsmax(high, window) - Tsmin(low, window)) * 100, n, m), n,
               m)

def alpha097(volume, window=10):  # 1729
    ####STD(VOLUME,10)###
    return Std(volume, window)

def alpha098(close, window=100, delta_interval=3):
    ####((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) /DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))###
    cond = (Delta(Sum(close, window) / window, window) / Delay(close, window) <= 0.05)
    part = close.copy(deep=True)
    part[cond] = -1 * (close - Tsmin(close, window))
    part[~cond] = -1 * Delta(close, delta_interval)
    return part

def alpha099(close, volume, window=5):  # 1766
    ####(-1 * Rank(Cov(Rank(close), Rank(volume), 5)))###
    return (-1 * Rank(Cov(Rank(close), Rank(volume), window)))

def alpha100(volume, window=20):  # 1657
    ####Std(volume,20)###
    return Std(volume, window)

def alpha101(close, volume, high, vwap, window_1=30, window_2=37, window_3=15, window_4=11):
    ###((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),RANK(VOLUME), 11))) * -1)
    rank1 = Rank(Corr(close, Sum(Mean(volume, window_1), window_2), window_3))
    rank2 = Rank(Corr(Rank(((high * 0.1) + (vwap * 0.9))), Rank(volume), window_4))
    cond = (rank1 < rank2)
    part = close.copy(deep=True)
    part[cond] = 1
    part[~cond] = 0
    return part

def alpha102(volume, delay_interval=1, n=6, m=1):  # 1790
    ####SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100###
    return Sma(Max(volume - Delay(volume, delay_interval), 0), n, m) / Sma(Abs(volume - Delay(volume, delay_interval)), n,
                                                                        m) * 100

def alpha103(low, window=20):
    ####((20-LOWDAY(LOW,20))/20)*100###
    return ((window - Lowday(low, window)) / window) * 100

def alpha104(close, high, volume, window_1=5, window_2=20, delta_interval=5):  # 1657
    ####(-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))###
    return (-1 * (Delta(Corr(high, volume, window_1), delta_interval) * Rank(Std(close, window_2))))

def alpha105(Open, volume, window=10):  # 1729
    ####(-1 * CORR(RANK(Open), RANK(VOLUME), 10))###
    return (-1 * Corr(Rank(Open), Rank(volume), window))

def alpha106(close, delay_interval=20):  # 1745
    ####CLOSE-DELAY(CLOSE,20)###
    return close - Delay(close, delay_interval)

def alpha107(Open, high, close, low, delay_interval=1, ):  # 1790
    ####(((-1 * RANK((Open - DELAY(HIGH, 1)))) * RANK((Open - DELAY(CLOSE, 1)))) * RANK((Open - DELAY(LOW, 1))))###
    return (((-1 * Rank((Open - Delay(high, delay_interval)))) * Rank((Open - Delay(close, delay_interval)))) * Rank(
        (Open - Delay(low, delay_interval))))

def alpha108(high, vwap, volume, window_1=2, window_2=120, window_3=6):  # 1178
    ####((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)###
    return ((Rank((high - Tsmin(high, window_1))) ** Rank(Corr((vwap), (Mean(volume, window_2)), window_3))) * -1)

def alpha109(high, low, n=10, m=2):  # 1797
    ####SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)###
    return Sma(high - low, n, m) / Sma(Sma(high - low, n, m), n, m)

def alpha110(high, close, low, delay_interval=1, window=20):  # 1650
    ####SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100###
    return Sum(Max(high - Delay(close, delay_interval), 0), window) / Sum(Max(Delay(close, delay_interval) - low, 0),
                                                                   window) * 100

def alpha111(close, low, high, volume, n=11, m=2):  # 1789
    ####SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)###
    return Sma(volume * ((close - low) - (high - close)) / (high - low), n,
               m) - Sma(volume * ((close - low) - (high - close)) / (high - low),
                        4, 2)

def alpha112(close, delay_interval=1, window=12):
    ####(SUM((CLOSE-DELAY(CLOSE,1)>0? CLOSE-DELAY(CLOSE,1):0),12) - SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12) + SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
    cond = (close - Delay(close, delay_interval) > 0)
    part1 = close.copy(deep=True)
    part1[cond] = close - Delay(close, delay_interval)
    part1[~cond] = 0
    part2 = close.copy(deep=True)
    part2[~cond] = Abs(close - Delay(close, delay_interval))
    part2[cond] = 0
    return (Sum(part1, window) - Sum(part2, window)) / (Sum(part1, window) + Sum(part2, window)) * 100

def alpha113(close, volume, delay_interval=5, window_1=20, window_2=2):  # 1587
    ####(-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5),SUM(CLOSE, 20), 2))))###
    return (-1 * ((Rank((Sum(Delay(close, delay_interval), window_1) / window_1)) * Corr(close, volume, window_2)) * Rank(
        Corr(Sum(close, delay_interval), Sum(close, window_1), window_2))))

def alpha114(close, high, low, vwap, volume, window=5, delay_interval=2):  # 1751
    ####((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) /(SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))###
    return ((Rank(Delay(((high - low) / (Sum(close, window) / window)), delay_interval)) * Rank(Rank(volume))) / (
                ((high - low) / (Sum(close, window) / window)) / (vwap - close)))

def alpha115(close, high, low, volume, window_1=30, window_2=10, window_3=4, window_4=10, window_5=7):  # 1527
    ####(RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) /2), 4), TSRANK(VOLUME, 10), 7)))###
    return (Rank(Corr(((high * 0.9) + (close * 0.1)), Mean(volume, window_1), window_2)) ** Rank(
        Corr(Tsrank(((high + low) / 2), window_3), Tsrank(volume, window_4), window_5)))

def alpha116(close, window=20):
    ####REGBETA(CLOSE,SEQUENCE,20)###
    return Regbeta(close, Sequence(window))

def alpha117(close, high, low, returns, volume, window_1=32, window_2=16):  # 1786
    ####((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))###
    return ((Tsrank(volume, window_1) * (1 - Tsrank(((close + high) - low), window_2))) * (
                1 - Tsrank(returns, window_1)))

def alpha118(high, Open, low, window=20):  # 1657
    ####SUM(HIGH-Open,20)/SUM(Open-LOW,20)*100###
    return Sum(high - Open, window) / Sum(Open - low, window) * 100

def alpha119(Open, vwap, volume):  # 1626
    ####(RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(Open), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))###
    return (Rank(Decaylinear(Corr(vwap, Sum(Mean(volume, 5), 26), 5), 7)) - Rank(
        Decaylinear(Tsrank(Tsmin(Corr(Rank(Open), Rank(Mean(volume, 15)), 21), 9), 7), 8)))

def alpha120(vwap, close):  # 1797
    ####(RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))###
    return (Rank((vwap - close)) / Rank((vwap + close)))

def alpha121(vwap, volume, ):  # 972   数据量较少
    ####((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) *-1)###
    return ((Rank((vwap - Tsmin(vwap, 12))) ** Tsrank(
        Corr(Tsrank(vwap, 20), Tsrank(Mean(volume, 60), 2), 18), 3)) * -1)

def alpha122(close, n=13, m=2, delay_interval=1):  # 1790
    ####(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)###
    return (Sma(Sma(Sma(Log(close), n, m), n, m), n, m) - Delay(
        Sma(Sma(Sma(Log(close), n, m), n, m), n, m), delay_interval)) / Delay(
        Sma(Sma(Sma(Log(close), n, m), n, m), n, m), delay_interval)

def alpha123(close, high, low, volume, window_1=20, window_2=60, window_3=9, window_4=6):
    ####((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME,6))) * -1)###
    A = Rank(Corr(Sum(((high + low) / 2), window_1), Sum(Mean(volume, window_2), window_1), window_3))
    B = Rank(Corr(low, volume, window_4))
    cond = (A < B)
    part = close.copy(deep=True)
    part[cond] = -1
    part[~cond] = 0
    return part

def alpha124(close, vwap, window_1=30, window_2=2):  # 1592
    ####(CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)###
    return (close - vwap) / Decaylinear(Rank(Tsmax(close, window_1)), window_2)

def alpha125(close, vwap, volume, window_1=80, window_2=17, window_3=20, window_4=16, delta_interval=3):  # 1678
    ####(RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))###
    return (Rank(Decaylinear(Corr((vwap), Mean(volume, window_1), window_2), window_3)) / Rank(
        Decaylinear(Delta(((close * 0.5) + (vwap * 0.5)), delta_interval), window_4)))

def alpha126(close, high, low):  # 1797
    ####(CLOSE+HIGH+LOW)/3###
    return (close + high + low) / 3

def alpha127(close, window=12):  # 公式有问题，我们假设mean周期为12
    ####(MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2),12)^(1/2)###
    return (Mean((100 * (close - Tsmax(close, window)) / (Tsmax(close, window))) ** 2, window)) ** (1 / 2)

def alpha128(high, low, close, volume, window=14, delay_interval=1):
    #### 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))
    A = (high + low + close) / 3
    cond = (A > Delay(A, delay_interval))
    part1 = close.copy(deep=True)
    part1[cond] = A * volume
    part1[~cond] = 0
    part2 = close.copy(deep=True)
    part2[~cond] = A * volume
    part2[cond] = 0
    return 100 - (100 / (1 + Sum(part1, window) / Sum(part2, window)))

def alpha129(close, delay_interval=1, window=12):
    ####SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)###
    cond = ((close - Delay(close, delay_interval)) < 0)
    part = close.copy(deep=True)
    part[cond] = Abs(close - Delay(close, delay_interval))
    part[~cond] = 0
    return Sum(part, window)

def alpha130(high, low, volume, vwap, window_1=40, window_2=9, window_3=10, window_4=7, window_5=3):  # 1657
    ####(RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))###
    return (Rank(Decaylinear(Corr(((high + low) / 2), Mean(volume, window_1), window_2), window_3)) / Rank(
        Decaylinear(Corr(Rank(vwap), Rank(volume), window_4), window_5)))

def alpha131(vwap, close, volume, delta_interval=1, window_1=50, window_2=18):  # 1030
    ####(RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))###
    return (Rank(Delta(vwap, delta_interval)) ** Tsrank(Corr(close, Mean(volume, window_1), window_2), window_2))

def alpha132(amount, window=20):  # 1657
    ####MEAN(AMOUNT,20)###
    return Mean(amount, window)

def alpha133(high, low, window=20):
    ####((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100###
    return ((window - Highday(high, window)) / window) * 100 - ((window - Lowday(low, window)) / window) * 100

def alpha134(close, volume, delay_interval=12):  # 1760
    ####(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME###
    return (close - Delay(close, delay_interval)) / Delay(close, delay_interval) * volume

def alpha135(close, delay_interval_1=20, delay_interval_2=1, n=20, m=1):  # 1744
    ####SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)###
    return Sma(Delay(close / Delay(close, delay_interval_1), delay_interval_2), n, m)

def alpha136(Open, volume, returns, delta_interval=3, window=10):  # 1729
    ####((-1 * RANK(DELTA(RET, 3))) * CORR(Open, VOLUME, 10))###
    return ((-1 * Rank(Delta(returns, delta_interval))) * Corr(Open, volume, window))

def alpha137(Open, close, low, high, delay_interval=1):
    ####16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-Open)/2+DELAY(CLOSE,1)-DELAY(Open,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(Open,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(Open,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(Open,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
    A = Abs(high - Delay(close, delay_interval))
    B = Abs(low - Delay(close, delay_interval))
    C = Abs(high - Delay(low, delay_interval))
    D = Abs(Delay(close, delay_interval) - Delay(Open, delay_interval))
    cond1 = ((A > B) & (A > C))
    cond2 = ((B > C) & (B > A))
    cond3 = ((C >= A) & (C >= B))
    part0 = 16 * (close + (close - Open) / 2 - Delay(Open, delay_interval))
    part1 = close.copy(deep=True)
    part1[cond1] = A + B / 2 + D / 4
    part1[~cond1] = 0
    part1[cond2] = B + A / 2 + D / 4
    part1[cond3] = C + D / 4
    return part0 / part1 * Max(A, B)

def alpha138(low, vwap, volume, delta_interval=3, window_1=20, window_2=8, window_3=60,
             window_4=17, window_5=5, window_6=19, window_7=16, window_8=7):  # 1448
    ####((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)###
    return ((Rank(Decaylinear(Delta((((low * 0.7) + (vwap * 0.3))), delta_interval), window_1)) - Tsrank(
        Decaylinear(Tsrank(Corr(Tsrank(low, window_2), Tsrank(Mean(volume, window_3), window_4), window_5), window_6), window_7), window_8)) * -1)

def alpha139(Open, volume, window=10):  # 1729
    ####(-1 * CORR(Open, VOLUME, 10))###
    return (-1 * Corr(Open, volume, window))

def alpha140(Open, low, high, close, volume, window_1=8, window_2=60, window_3=20, window_4=7, window_5=3):  # 1797
    ####MIN(RANK(DECAYLINEAR(((RANK(Open) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))###
    return Min(Rank(Decaylinear(((Rank(Open) + Rank(low)) - (Rank(high) + Rank(close))), window_1)),
               Tsrank(Decaylinear(Corr(Tsrank(close, window_1), Tsrank(Mean(volume, window_2), window_3), window_1), window_4), window_5))

def alpha141(high, volume, window_1=15, window_2=8):  # 1637
    ####(RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)###
    return (Rank(Corr(Rank(high), Rank(Mean(volume, window_1)), window_2)) * -1)

def alpha142(close, volume, window_1=10, window_2=20, window_3=5, delta_interval_1=1, delta_interval_2=1):  # 1657
    ####(((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME,20)), 5)))###
    return (((-1 * Rank(Tsrank(close, window_1))) * Rank(Delta(Delta(close, delta_interval_1), delta_interval_2))) * Rank(
        Tsrank((volume / Mean(volume, window_2)), window_3)))

def alpha143():  # what fuck
    ####CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*:###

    return 0

def alpha144(close, amount, window=20, delay_interval=1):
    ####SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)###
    cond = (close < Delay(close, delay_interval))
    part1 = Abs(close / Delay(close, delay_interval) - 1) / amount
    return Sumif(part1, window, cond) / Count(cond, window)

def alpha145(volume, window_1=9, window_2=26, window_3=12):  # 1617
    ####(MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100###
    return (Mean(volume, window_1) - Mean(volume, window_2)) / Mean(volume, window_3) * 100

def alpha146(close, delay_interval=1, n=61, m=2, window=20):  # 1650
    ####MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,61,2)###
    return Mean((close - Delay(close, delay_interval)) / Delay(close, delay_interval) - Sma(
        (close - Delay(close, delay_interval)) / Delay(close, delay_interval), n, m), window) * (
                       (close - Delay(close, delay_interval)) / Delay(close, delay_interval) - Sma(
                   (close - Delay(close, delay_interval)) / Delay(close, delay_interval), n, m)) / Sma(((close - Delay(
        close, delay_interval)) / Delay(close, delay_interval) - ((close - Delay(close, delay_interval)) / Delay(close, delay_interval) - Sma(
        (close - Delay(close, delay_interval)) / Delay(close, delay_interval), n, m))) ** 2, n, m)

def alpha147(close, window=12):
    ####REGBETA(MEAN(CLOSE,12),SEQUENCE(12))###
    return Regbeta(Mean(close, window), Sequence(window))

def alpha148(Open, close, volume, window_1=60, window_2=9, window_3=6, window_4=14):
    ####((RANK(CORR((Open), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((Open - TSMIN(Open, 14)))) * -1)###
    cond = (Rank(Corr((Open), Sum(Mean(volume, window_1), window_2), window_3)) < Rank((Open - Tsmin(Open, window_4))))
    part = close.copy(deep=True)
    part[cond] = -1
    part[~cond] = 0
    return part

def alpha149():
    ####REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)
    return 0

def alpha150(close, high, low, volume):  # 1797
    ####(CLOSE+HIGH+LOW)/3*VOLUME###
    return (close + high + low) / 3 * volume

def alpha151(close, delay_interval=20, n=20, m=1):  # 1745
    ####SMA(CLOSE-DELAY(CLOSE,20),20,1)###
    return Sma(close - Delay(close, delay_interval), n, m)

def alpha152(close, n=9, m=1, window_1=12, window_2=26, delay_interval=1):  # 1559
    ####SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)###
    return Sma(Mean(Delay(Sma(Delay(close / Delay(close, n), m), n, m), delay_interval), window_1) - Mean(
        Delay(Sma(Delay(close / Delay(close, n), m), n, m), delay_interval), window_2), n, m)

def alpha153(close, window=3):  # 1630
    ####(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4###
    return (Mean(close, window) + Mean(close, window*2) + Mean(close, window*4) + Mean(close, window*8)) / 4

def alpha154(vwap, volume, close, window_1=16, window_2=180, window_3=18):
    ####(((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))###
    cond = (((vwap - Tsmin(vwap, window_1))) < (Corr(vwap, Mean(volume, window_2), window_3)))
    part = close.copy(deep=True)
    part[cond] = 1
    part[~cond] = 0
    return part

def alpha155(volume):  # 1797
    ####SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)###
    return Sma(volume, 13, 2) - Sma(volume, 27, 2) - Sma(
        Sma(volume, 13, 2) - Sma(volume, 27, 2), 10, 2)

def alpha156(Open, vwap, low,  delta_interval_1=5, delta_interval_2=2, window=3):  # 1776
    ####(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((Open * 0.15) + (LOW *0.85)),2) / ((Open * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)###
    return (Max(Rank(Decaylinear(Delta(vwap, delta_interval_1), window)), Rank(Decaylinear(
        ((Delta(((Open * 0.15) + (low * 0.85)), delta_interval_2) / ((Open * 0.15) + (low * 0.85))) * -1),
        window))) * -1)

def alpha157(close, returns, delta_interval=5, delay_interval=6, window_1=2, window_2=1, window_3=1, window_4=5):  # 1764
    ####(MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + TSRANK(DELAY((-1 * RET), 6), 5))###
    return (Tsmin(Prod(Rank(Rank(Log(Sum(Tsmin(Rank(Rank((-1 * Rank(Delta((close - 1), delta_interval))))), window_1), window_2)))), window_3),
                  window_4) + Tsrank(Delay((-1 * returns), delay_interval), window_4))

def alpha158(high, close, low, n=15, m=2):  # 1797
    ####((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE###
    return ((high - Sma(close, n, m)) - (low - Sma(close, n, m))) / close

def alpha159(close, low, high, window=6):  # 1630
    ####((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)###
    return ((close - Sum(Min(low, Delay(close, 1)), window)) / Sum(
        Max(high, Delay(close, 1)) - Min(low, Delay(close, 1)), window) * window*2 * window*4 + (
                        close - Sum(Min(low, Delay(close, 1)), window*2)) / Sum(
        Max(high, Delay(close, 1)) - Min(low, Delay(close, 1)), window*2) * window * window*4 + (
                        close - Sum(Min(low, Delay(close, 1)), window*4)) / Sum(
        Max(high, Delay(close, 1)) - Min(low, Delay(close, 1)), window*4) * window * window*4) * 100 / (
                       6 * 12 + 6 * 24 + 12 * 24)

def alpha160(close, window=20, delay_interval=1, n=20, m=1):
    ####SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)###
    cond = (close <= Delay(close, delay_interval))
    part = close.copy(deep=True)
    part[cond] = Std(close, window)
    part[~cond] = 0
    return Sma(part, n, m)

def alpha161(high, close, low, delay_interval=1, window=12):  # 1714
    ####MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)###
    return Mean(Max(Max((high - low), Abs(Delay(close, delay_interval) - high)),
                    Abs(Delay(close, delay_interval) - low)), window)

def alpha162(close, n=12, m=1, delay_interval=1, window=12):  # 1789
    ####(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))###
    return (Sma(Max(close - Delay(close, delay_interval), 0), n, m) / Sma(Abs(close - Delay(close, delay_interval)), n,
                                                                        m) * 100 - Tsmin(
        Sma(Max(close - Delay(close, delay_interval), 0), n, m) / Sma(Abs(close - Delay(close, delay_interval)), n,
                                                                    m) * 100, window)) / (
                       Sma(Sma(Max(close - Delay(close, delay_interval), 0), n, m) / Sma(
                           Abs(close - Delay(close, delay_interval)), n, m) * 100, n, m) - Tsmin(
                   Sma(Max(close - Delay(close, delay_interval), 0), n, m) / Sma(
                       Abs(close - Delay(close, delay_interval)), n, m) * 100, window))

def alpha163(close, high, vwap, volume, returns, window=20):  # 1657
    ####RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))###
    return Rank(((((-1 * returns) * Mean(volume, window)) * vwap) * (high - close)))

def alpha164(close, high, low, delay_interval=1, window=12, n=13, m=2):
    ####SMA(( ((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1) - MIN( ((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1) ,12) )/(HIGH-LOW)*100,13,2)###
    cond = (close > Delay(close, delay_interval))
    part = close.copy(deep=True)
    part[cond] = 1 / (close - Delay(close, delay_interval))
    part[~cond] = 1
    return Sma((part - Tsmin(part, window)) / (high - low) * 100, n, m)

def alpha165(close, window=48):  # rowmax
    ####MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)###
    p1 = Rowmax(Sum(close - Mean(close, window), window))
    p2 = Rowmin(Sum(close - Mean(close, window), window))
    p3 = Std(close, window)
    return -1 * (1 / p3.div(p2, axis=0)).sub(p1, axis=0)

def alpha166(close):  # 公式有问题
    ####-20* ( 20-1 ) ^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)
    p1 = -20 * (20 - 1) ** 1.5 * Sum(
        close / Delay(close, 1) - 1 - Mean(close / Delay(close, 1) - 1, 20), 20)
    p2 = ((20 - 1) * (20 - 2) * (Sum(Mean(close / Delay(close, 1), 20) ** 2, 20)) ** 1.5)
    return p1 / p2

def alpha167(close, delay_interval=1, window=12):
    ####SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)###
    cond = (close > Delay(close, delay_interval))
    part = close.copy(deep=True)
    part[cond] = close - Delay(close, delay_interval)
    part[~cond] = 0
    return Sum(part, window)

def alpha168(volume, window=20):  # 1657
    ####(-1*VOLUME/MEAN(VOLUME,20))###
    return (-1 * volume / Mean(volume, window))

def alpha169(close, n=9, m=1, delay_interval=1, window_1=12, window_2=26, ):  # 1610
    ####SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)###
    return Sma(Mean(Delay(Sma(close - Delay(close, delay_interval), n, m), 1), window_1) - Mean(
        Delay(Sma(close - Delay(close, delay_interval), n, m), delay_interval), window_2), 10, 1)

def alpha170(close, volume, high, vwap, window_1=20, window_2=5, delay_interval=5):  # 1657
    ####((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) /5))) - RANK((VWAP - DELAY(VWAP, 5))))###
    return ((((Rank((1 / close)) * volume) / Mean(volume, window_1)) * (
                (high * Rank((high - close))) / (Sum(high, window_2) / window_2))) - Rank(
        (vwap - Delay(vwap, delay_interval))))

def alpha171(low, close, Open, high):  # 1789
    ####((-1 * ((LOW - CLOSE) * (Open^5))) / ((CLOSE - HIGH) * (CLOSE^5)))###
    return ((-1 * ((low - close) * (Open ** 5))) / ((close - high) * (close ** 5)))

def alpha172(high, low, close, delay_interval=1, window_1=14, window_2=6):
    ####MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
    TR = Max(Max(high - low, Abs(high - Delay(close, delay_interval))), Abs(low - Delay(close, delay_interval)))
    HD = high - Delay(high, delay_interval)
    LD = Delay(low, delay_interval) - low
    cond1 = ((LD > 0) & (LD > HD))
    cond2 = ((HD > 0) & (HD > LD))
    part1 = close.copy(deep=True)
    part1[cond1] = LD
    part1[~cond1] = 0
    part2 = close.copy(deep=True)
    part2[cond2] = HD
    part2[~cond2] = 0
    return Mean(Abs(Sum(part1, window_1) * 100 / Sum(TR, window_1) - Sum(part2, window_1) * 100 / Sum(TR, window_1)) / (
                Sum(part1, window_1) * 100 / Sum(TR, window_1) + Sum(part2, window_1) * 100 / Sum(TR, window_1)) * 100, window_2)

def alpha173(close, n=13, m=2):  # 1797
    ####3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)###
    return 3 * Sma(close, n, m) - 2 * Sma(Sma(close, n, m), n, m) + Sma(
        Sma(Sma(Log(close), n, m), n, m), n, m)

def alpha174(close, delay_interval=1, window=20, n=20, m=1):
    ####SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)###
    cond = (close > Delay(close, delay_interval))
    part = close.copy(deep=True)
    part[cond] = Std(close, window)
    part[~cond] = 0
    return Sma(part, n, m)

def alpha175(close, high, low, delay_interval=1, window=6):  # 1759
    ####MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)###
    return Mean(Max(Max((high - low), Abs(Delay(close, delay_interval) - high)),
                    Abs(Delay(close, delay_interval) - low)), window)

def alpha176(close, low, high, volume, window_1=12, window_2=6):  # 1678
    ####CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)###
    return Corr(Rank(((close - Tsmin(low, window_1)) / (Tsmax(high, window_1) - Tsmin(low, window_1)))),
                Rank(volume), window_2)

def alpha177(high, window=20):
    ####((20-HIGHDAY(HIGH,20))/20)*100###
    return ((window - Highday(high, window)) / window) * 100

def alpha178(close, volume, delay_interval=1):  # 1790
    ####(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME###
    return (close - Delay(close, delay_interval)) / Delay(close, delay_interval) * volume

def alpha179(low, vwap, volume, window_1=4, window_2=50, window_3=12):  # 1421   数据量较少
    ####(RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))###
    return (Rank(Corr(vwap, volume, window_1)) * Rank(Corr(Rank(low), Rank(Mean(volume, window_2)), window_3)))

def alpha180(volume, close, delta_interval=7, window_1=20, window_2=60):  # 指标有问题
    ####((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 *VOLUME)))
    cond = (Mean(volume, window_1) < volume)
    part = close.copy(deep=True)
    part[cond] = (-1 * Tsrank(Abs(Delta(close, delta_interval)), window_2)) * Sign(Delta(close, delta_interval))
    part[~cond] = -1 * volume
    return part

def alpha181(close, benchmark_close, delay_interval=1, window=20):  # 1532  公式有问题，假设后面的sum周期为20
    ####SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)###
    return Sum(((close / Delay(close, delay_interval) - 1) - Mean((close / Delay(close, delay_interval) - 1), 20)) - (
                benchmark_close - Mean(benchmark_close, window)) ** 2, window) / Sum(
        ((benchmark_close - Mean(benchmark_close, window)) ** 3), window)

def alpha182(Open, close, benchmark_close, benchmark_Open, window=20):
    ####COUNT((CLOSE>Open & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOpen)OR(CLOSE<Open & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOpen),20)/20###
    return Count((((close > Open) & (benchmark_close > benchmark_Open)) | (
                (close < Open) & (benchmark_close < benchmark_Open))), window) / window

def alpha183(close, window=24):
    ###MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)###
    p1 = Rowmax(Sum(close - Mean(close, window), window))
    p2 = Rowmin(Sum(close - Mean(close, window), window))
    p3 = Std(close, window)
    return -1 * (1 / p3.div(p2, axis=0)).sub(p1, axis=0)

def alpha184(Open, close, delay_interval=1, window=200):  # 983   数据量较少
    ####(RANK(CORR(DELAY((Open - CLOSE), 1), CLOSE, 200)) + RANK((Open - CLOSE)))###
    return (Rank(Corr(Delay((Open - close), delay_interval), close, window)) + Rank((Open - close)))

def alpha185(Open, close):  # 1797
    ####RANK((-1 * ((1 - (Open / CLOSE))^2)))###
    return Rank((-1 * ((1 - (Open / close)) ** 2)))

def alpha186(high, low, close, delay_interval=1, window_1=14, window_2=6):
    ####(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2
    TR = Max(Max(high - low, Abs(high - Delay(close, delay_interval))), Abs(low - Delay(close, delay_interval)))
    HD = high - Delay(high, delay_interval)
    LD = Delay(low, delay_interval) - low
    cond1 = ((LD > 0) & (LD > HD))
    cond2 = ((HD > 0) & (HD > LD))
    part1 = close.copy(deep=True)
    part1[cond1] = LD
    part1[~cond1] = 0
    part2 = close.copy(deep=True)
    part2[cond2] = HD
    part2[~cond2] = 0
    return (Mean(Abs(Sum(part1, window_1) * 100 / Sum(TR, window_1) - Sum(part2, window_1) * 100 / Sum(TR, window_1)) / (
                Sum(part1, window_1) * 100 / Sum(TR, window_1) + Sum(part2, window_1) * 100 / Sum(TR, window_1)) * 100, window_2) + Delay(Mean(
        Abs(Sum(part1, window_1) * 100 / Sum(TR, window_1) - Sum(part2, window_1) * 100 / Sum(TR, window_1)) / (
                    Sum(part1, window_1) * 100 / Sum(TR, window_1) + Sum(part2, window_1) * 100 / Sum(TR, window_1)) * 100, window_2), window_2)) / 2

def alpha187(Open, close, high, delay_interval=1, window=20):
    ####SUM((Open<=DELAY(Open,1)?0:MAX((HIGH-Open),(Open-DELAY(Open,1)))),20)###
    cond = (Open <= Delay(Open, delay_interval))
    part = close.copy(deep=True)
    part[cond] = 0
    part[~cond] = Max((high - Open), (Open - Delay(Open, delay_interval)))
    return Sum(part, window)

def alpha188(high, low, n=11, m=2):  # 1797
    ####((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100###
    return ((high - low - Sma(high - low, n, m)) / Sma(high - low, n, m)) * 100

def alpha189(close, window=6):  # 1721
    ####MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)###
    return Mean(Abs(close - Mean(close, window)), window)

def alpha190():
    ####LOG((COUNT( CLOSE/DELAY(CLOSE,1)>((CLOSE/DELAY(CLOSE,19))^(1/20)-1) ,20)-1)*(SUMIF((CLOSE/DELAY(CLOSE,1)-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE,1)<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((COUNT((CLOSE/DELAY(CLOSE,1)<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE,1)-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE,1)>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))
    return 0

def alpha191(close, high, low, volume, window_1=20, window_2=5):  # 1721
    ####((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)###
    return ((Corr(Mean(volume, window_1), low, window_2) + ((high + low) / 2)) - close)
