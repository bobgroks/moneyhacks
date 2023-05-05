#!usr/bin/env python3.11
import os
import sys
import asyncio
import threading
from typing import Optional, Dict, Union, List, Tuple, Set, Any, ClassVar
from collections import namedtuple, deque
from dataclasses import dataclass
import sched


import pandas as pd
import numpy as np
from dotenv import load_dotenv
from binance import Client, AsyncClient, ThreadedWebsocketManager
from binance.client import BaseClient
from binance.exceptions import BinanceAPIException
import time

from models.timing import Timestamp
from models.log import logger
# from models.trade import Trade
from utils import pprint_resp, FixedLengthList
from strategy.alpha_hedge_strategy import Strategy


load_dotenv()
TESTNET = int(os.getenv("TESTNET", 0))
TESTNET = 2 # futures testnet 
if TESTNET == 1: # Spot testnet
  BINANCE_PRIVATE_KEY = os.getenv('BINANCE_TESTNET_SECRET_KEY') 
  BINANCE_PUBLIC_KEY = os.getenv('BINANCE_TESTNET_PUBLIC_KEY') 
elif TESTNET == 2: # Futures testnet
  BINANCE_PRIVATE_KEY = os.getenv('BINANCE_FUTURES_TESTNET_SECRET_KEY') 
  BINANCE_PUBLIC_KEY = os.getenv('BINANCE_FUTURES_TESTNET_PUBLIC_KEY') 
elif TESTNET == 0: # Live trading
  BINANCE_PRIVATE_KEY = os.getenv('BINANCE_SECRET_KEY')
  BINANCE_PUBLIC_KEY = os.getenv('BINANCE_PUBLIC_KEY')
else:
  print('???')
  exit(0)

# TODO: replace python-binance with our own implementation and use __slots__
  


@dataclass
class Pair:
  m: ClassVar[int] = 0
  # trades: List['Trade'] = []
  symbol: str
  recent_trade: float = None # maybe recent_trades: fixed length queue
  amount: float = 0.0
  bid: float = None
  ask: float = None 
  # bid: List[Tuple[float, float]] = None
  # ask: List[Tuple[float, float]] = None
  tick_size: float = 0.0
  step_size: float = 0.0
  notional: float = 0.0
  olhcv: FixedLengthList = None

  @property
  def position(self):
    pass

  @property
  def ready(self):
    return all([self.bid, self.ask, self.recent_trade]) and len(self.olhcv) == self.olhcv.max_length

  def __post_init__(self):
    self.m += 1

# @dataclass
# class Trade(Pair):



class PairContainer:
  def __init__(self):
    ...


class BinanceAsync(AsyncClient):
  def __init__(self):
    self.ws = ThreadedWebsocketManager(api_key=BINANCE_PUBLIC_KEY, api_secret=BINANCE_PRIVATE_KEY, testnet=TESTNET)
    self.strategy: 'Strategy' = None
    self.quote = {'BUSD', 'USDT'}
    # self.symbol_lis: Tuple[str, ...] = ["BTCUSDT", "ETHUSDT", "BTCBUSD"]
    self.symbol_lis: Tuple[str, ...] = ["BTCUSDT", "ETHUSDT"]
    self.symbols: Dict[str, Pair] = {}
    self.symbol_tradable: Tuple[str, ...] 
    self.fee_tier: int 

    super().__init__(api_key=BINANCE_PUBLIC_KEY, api_secret=BINANCE_PRIVATE_KEY, testnet=TESTNET)
    acc, info, olhcv = self.loop.run_until_complete(self._get_init_info())
    assert len(olhcv[self.symbol_lis[0]]) == 23, f'{len(olhcv[self.symbol_lis[0]])}'

    # 'feeTier': 0, 'canTrade': True, 'canDeposit': True, 'canWithdraw': True, 'updateTime': 0, 'multiAssetsMargin': False, 'totalInitialMargin': '4.19151110', 'totalMaintMargin': '0.33532088', 'totalWalletBalance': '15000.22725745', 'totalUnrealizedProfit': '1.26932214', 'totalMarginBalance': '15001.49657959', 'totalPositionInitialMargin': '4.19151110', 'totalOpenOrderInitialMargin': '0.00000000', 'totalCrossWalletBalance': '15000.22725745', 'totalCrossUnPnl': '1.26932214', 'availableBalance': '14997.30506849', 'maxWithdrawAmount': '14997.30506849',
    self.fee_tier = acc['feeTier']
    self.assets = [tok for tok in acc['assets'] if float(tok['walletBalance']) != 0.0]
    self.symbol_tradable = [pair['pair'] for pair in info['symbols'] if pair['contractType'] == 'PERPETUAL' and pair['status'] == 'TRADING']
    #['BTCUSDT', 'ETHUSDT', 'BCHUSDT', 'XRPUSDT', 'EOSUSDT', 'LTCUSDT', 'TRXUSDT', 'ETCUSDT', 'LINKUSDT', 'XLMUSDT', 'ADAUSDT', 'DASHUSDT', 'ZECUSDT', 'XTZUSDT', 'BNBUSDT', 'ATOMUSDT', 'ONTUSDT', 'IOTAUSDT', 'BATUSDT', 'VETUSDT', 'QTUMUSDT', 'IOSTUSDT', 'THETAUSDT', 'ALGOUSDT', 'ZILUSDT', 'KNCUSDT', 'ZRXUSDT', 'COMPUSDT', 'OMGUSDT', 'DOGEUSDT', 'SXPUSDT', 'KAVAUSDT', 'BANDUSDT', 'RLCUSDT', 'WAVESUSDT', 'MKRUSDT', 'SNXUSDT', 'DOTUSDT', 'DEFIUSDT', 'YFIUSDT', 'BALUSDT', 'CRVUSDT', 'RUNEUSDT', 'SUSHIUSDT', 'EGLDUSDT', 'SOLUSDT', 'STORJUSDT', 'UNIUSDT', 'AVAXUSDT', 'FTMUSDT', 'ENJUSDT', 'FLMUSDT', 'TOMOUSDT', 'RENUSDT', 'KSMUSDT', 'NEARUSDT', 'AAVEUSDT', 'FILUSDT', 'LRCUSDT', 'MATICUSDT', 'OCEANUSDT', 'AXSUSDT', 'ZENUSDT', 'SKLUSDT', 'GRTUSDT', '1INCHUSDT', 'UNFIUSDT', 'BTCBUSD', 'CHZUSDT', 'SANDUSDT', 'ANKRUSDT', 'LITUSDT', 'REEFUSDT', 'COTIUSDT', 'CHRUSDT', 'MANAUSDT', 'ALICEUSDT', 'ONEUSDT', 'LINAUSDT', 'STMXUSDT', 'DENTUSDT', 'CELRUSDT', 'HOTUSDT', 'MTLUSDT', 'OGNUSDT', '1000SHIBUSDT', 'ETHBUSD', 'BTCDOMUSDT', 'MASKUSDT', 'BNBBUSD', 'ADABUSD', 'IOTXUSDT', 'AUDIOUSDT', 'C98USDT', 'ATAUSDT', 'SOLBUSD', 'DYDXUSDT', '1000XECUSDT', 'GALAUSDT', 'CELOUSDT', 'ARUSDT', 'KLAYUSDT', 'ARPAUSDT', 'ENSUSDT', 'PEOPLEUSDT', 'ANTUSDT', 'ROSEUSDT', 'DUSKUSDT', 'FLOWUSDT', 'IMXUSDT', 'API3USDT', 'GMTUSDT', 'APEUSDT', 'WOOUSDT', 'JASMYUSDT', 'DARUSDT', 'GALUSDT', 'APEBUSD', 'TRXBUSD', 'GALABUSD', 'FTMBUSD', 'GALBUSD', 'ANCBUSD', '1000LUNCBUSD', 'OPUSDT', 'DOTBUSD', 'LINKBUSD', 'WAVESBUSD', 'XMRUSDT', 'LTCBUSD', 'MATICBUSD', '1000SHIBBUSD', 'LDOBUSD', 'AUCTIONBUSD', 'INJUSDT', 'STGUSDT', 'SPELLUSDT', '1000LUNCUSDT', 'LUNA2USDT', 'LDOUSDT', 'CVXUSDT', 'FOOTBALLUSDT', 'APTUSDT', 'QNTUSDT', 'APTBUSD', 'BLUEBIRDUSDT', 'AGIXBUSD', 'FXSUSDT', 'HOOKUSDT', 'MAGICUSDT', 'TUSDT', 'RNDRUSDT', 'HIGHUSDT', 'MINAUSDT', 'ASTRUSDT', 'PHBUSDT', 'AGIXUSDT', 'FETUSDT', 'GMXUSDT', 'CFXUSDT', 'STXUSDT', 'COCOSUSDT', 'BNXUSDT', 'ACHUSDT', 'SSVUSDT', 'CKBUSDT', 'PERPUSDT', 'LQTYUSDT', 'USDCUSDT', 'ARBUSDT', 'IDUSDT', 'IDBUSD', 'ARBBUSD', 'JOEUSDT', 'AMBUSDT', 'LEVERUSDT', 'TRUUSDT', 'RDNTUSDT', 'ETHBTC', 'HFTUSDT', 'XVSUSDT', 'BLURUSDT', 'EDUUSDT', 'IDEXUSDT']

    for symbol in self.symbol_lis:
      dat = [i for i in info['symbols'] if i['symbol'] == symbol][0]
      tick_size = dat['filters'][0]['tickSize']
      step_size = dat['filters'][1]['stepSize']
      notional = dat['filters'][-2]['notional']
      self.symbols[symbol] = Pair(symbol=symbol, tick_size=tick_size, step_size=step_size, notional=notional, olhcv=FixedLengthList(max_length=23, lis=olhcv[symbol]))


    # open price stream
    print('running ws')
    self.run_ws()

    while not all([pair.ready for pair in self.symbols.values()]):
      pass
    
    print('finished init')

    # self.latency: float
    # self.timestamp_offset: float
    # self.loop.run_until_complete(self._test_server())

  async def _test_server(self):
    async def time_diff():
      st = time.time_ns() * 1e-6
      t = await self.get_server_time()
      ed = time.time_ns() * 1e-6
      offset = t["serverTime"] - (ed-st)/2 - st
      return (offset, ed-st)
    tasks: List[asyncio.Task] = [time_diff() for _ in range(100)]
    ret = await asyncio.gather(*tasks)
    ret.sort(key=lambda x: x[0])
    self.timestamp_offset, self.latency = ret[len(ret)//2]

    # ret = [r['serverTime'] for r in ret]


  async def _get_init_info(self):
    tasks = []
    tasks.append(asyncio.create_task(self.futures_account()))
    tasks.append(asyncio.create_task(self.futures_exchange_info()))

    t = Timestamp.now()
    end_str = str(t)
    for _ in range(23):
      t.apply_delta(delta='04:00:00', add=False)
    start_str = str(t)

    for symbol in self.symbol_lis:
      tasks.append(asyncio.create_task(self.futures_historical_klines(symbol=symbol, interval='4h', end_str=end_str, start_str=start_str)))

    ret = await asyncio.gather(*tasks)
    ret = ret[:2] + [{k: v for k,v in zip(self.symbol_lis, ret[2:len(self.symbol_lis)+2])}]
    return ret
  

  def run_ws(self):
    def _handle_price_update(msg):
      pair = msg['data']['s']
      price = msg['data']['p']
      self.symbols[pair].recent_trade = price
      # self.symbol_data[msg['data']['s']][0] = msg['data']['p']

    def _handle_olhc_update(msg):
      # [1683086400000, '1860.71', '1863.57', '1858.00', '1860.90', '17355.431', 1683100799999, '32287286.76144', 3085, '13575.887', '25255013.44799', '0']
      # {'e': 'kline', 'E': 1683096379232, 's': 'ETHUSDT', 'k': {'t': 1683086400000, 'T': 1683100799999, 's': 'ETHUSDT', 'i': '4h', 'f': 116551861, 'L': 116554954, 'o': '1860.71', 'c': '1860.90', 'h': '1863.57', 'l': '1858.00', 'v': '17397.634', 'n': 3094, 'x': False, 'q': '32365830.36189', 'V': '13606.948', 'Q': '25312822.90064', 'B': '0'}
      pair = msg['data']['s']
      olhcv = msg['data']['k']
      latest_olhcv = self.symbols[pair].olhcv[-1]
      if olhcv['t'] == latest_olhcv[0]:
        latest_olhcv[2] = olhcv['h']
        latest_olhcv[3] = olhcv['l']
        latest_olhcv[4] = olhcv['c']
        latest_olhcv[5] = olhcv['v']
        latest_olhcv[7] = olhcv['q']
        latest_olhcv[8] = olhcv['n']
        latest_olhcv[9] = olhcv['V']
        latest_olhcv[10] = olhcv['Q']
      else:
        self.symbols[pair].olhcv.append([olhcv['t'], olhcv['o'], olhcv['h'], olhcv['l'], olhcv['c'], olhcv['v'], olhcv['T'], olhcv['q'], olhcv['n'], olhcv['V'], olhcv['Q'], '0'])


    def _handle_bid_ask_update(msg):
      pair = msg['data']['s']
      bid = msg['data']['b']
      ask = msg['data']['a']
      self.symbols[pair].bid = bid
      self.symbols[pair].ask = ask

    def _handle_acc_update(msg):
      print(msg) # log this

    trade_streams = [pair.lower()+'@trade' for pair in self.symbol_lis]
    bid_ask_streams = [pair.lower()+'@bookTicker' for pair in self.symbol_lis]
    olhc_streams = [pair.lower()+'@kline_4h' for pair in self.symbol_lis]
    self.ws.start()
    self.ws.start_futures_multiplex_socket(callback=_handle_bid_ask_update, streams=bid_ask_streams)
    self.ws.start_futures_multiplex_socket(callback=_handle_olhc_update, streams=olhc_streams)
    self.ws.start_futures_multiplex_socket(callback=_handle_price_update, streams=trade_streams)
    self.ws.start_futures_user_socket(callback=_handle_acc_update)
    ws_thread = threading.Thread(target=self.ws.join)
    ws_thread.start()
    
  def run_strategy(self):
    # prepare bar
    bar = pd.DataFrame()
    for symbol, pair in self.symbols.items():
      pair_bar = pd.DataFrame(pair.olhcv,columns=["ots","open","high","low","close","volume","ts"\
                                                 ,"usd_v","n_trades","taker_buy_v","taker_buy_usd","ig"]).drop(['ots','ig'],axis=1) 
      pair_bar = pair_bar.set_index('ts')
      pair_bar['symbol'] = symbol
      pair_bar = pair_bar.set_index(['symbol', pair_bar.index])
      bar = pd.concat([bar, pair_bar])
                                  
    # init strategy
    if not self.strategy:
      self.strategy = Strategy(bar=bar, symbol_lis=self.symbol_lis, rolling_window=12)
    exit()
    ...


  def close(self):
    # close all trades, make sure all money is in self.quote

    ...


if __name__ == '__main__':
  try:
    # init bot
    abot = BinanceAsync()
    # start schedule trades
    scheduler = sched.scheduler()
    # scheduler.enter(4 * 3600, 1, abot.run_strategy, argument=())
    # maybe init threads or processes
    abot.run_strategy()
    while True:
      try:
        scheduler.run()
      except KeyboardInterrupt:
          # Stop the scheduler if the user presses Ctrl+C
        asyncio.run(abot.close_connection())
        break
    # asyncio.run(abot.close_connection())
    # bot = Binance()
    # print(bot.test('a', 1))
    # print(bot.get_server_time()['serverTime'])
    #pprint_resp(abot.loop.run_until_complete(abot.futures_account_balance()))
    #pprint_resp(abot.loop.run_until_complete(abot.futures_place_batch_order()))
    # print(abot.loop.run_until_complete(abot.futures_time())["serverTime"])
    #abot.loop.run_until_complete(abot.close_connection())
    #print(bot.get_server_time()['serverTime'])
    #print(time.time()*1000)
    # pprint_resp(asyncio.run(bot.get_account()))

  except BinanceAPIException as e:
    print(e.status_code)
    print(e.message)
"""
  import requests

  url = "https://api.binance.com/api/v3/time"
  response = requests.get(url)

  if response.status_code == 200:
    data = response.json()
    server_time = data['serverTime']
    print(Time(server_time).dt)

  else:
    print(f"Error: {response.status_code} - {response.reason}")

  # try:
  #  bot.get_all_orders()
  # except BinanceAPIException as e:
  #  print(e.status_code)
  #  print(e.message)
"""