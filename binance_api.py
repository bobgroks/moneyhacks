#!usr/bin/env python3.11
import os
import asyncio
from typing import Optional, Dict, Union, List

from dotenv import load_dotenv
from binance import Client, AsyncClient
from binance.client import BaseClient
from binance.exceptions import BinanceAPIException

from models.timing import Time
from models.log import api_call_logger
from models.trade import Trade
from models.account import Account
from utils import pprint_resp

load_dotenv()
TESTNET = int(os.getenv("TESTNET", 0))
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

# BINANCE_PRIVATE_KEY = os.getenv('BINANCE_TESTNET_SECRET_KEY') if TESTNET else os.getenv('BINANCE_SECRET_KEY')
# BINANCE_PUBLIC_KEY = os.getenv('BINANCE_TESTNET_PUBLIC_KEY') if TESTNET else os.getenv('BINANCE_PUBLIC_KEY')

# NOTE: when serverTime returns, it's faster than next line local time

@api_call_logger
class Binance(Client):
  def __init__(self):
    super().__init__(BINANCE_PUBLIC_KEY, BINANCE_PRIVATE_KEY, testnet=TESTNET)
    # {'makerCommission': 0, 'takerCommission': 0, 'buyerCommission': 0, 'sellerCommission': 0, 'commissionRates': {'maker': '0.00000000', 'taker': '0.00000000', 'buyer': '0.00000000', 'seller': '0.00000000'}, 'canTrade': True, 'canWithdraw': False, 'canDeposit': False, 'brokered': False, 'requireSelfTradePrevention': False, 'updateTime': 1674836417445, 'accountType': 'SPOT', 'balances': [{'asset': 'BNB', 'free': '1000.00000000', 'locked': '0.00000000'}, {'asset': 'BTC', 'free': '1.00000000', 'locked': '0.00000000'}, {'asset': 'BUSD', 'free': '10000.00000000', 'locked': '0.00000000'}, {'asset': 'ETH', 'free': '100.00000000', 'locked': '0.00000000'}, {'asset': 'LTC', 'free': '500.00000000', 'locked': '0.00000000'}, {'asset': 'TRX', 'free': '500000.00000000', 'locked': '0.00000000'}, {'asset': 'USDT', 'free': '10000.00000000', 'locked': '0.00000000'}, {'asset': 'XRP', 'free': '50000.00000000', 'locked': '0.00000000'}], 'permissions': ['SPOT']}
    # self.account = Account(self.get

  def test(self, symbol: str, amount: float): import time; time.sleep(1); return 2+4
  
  def trade(self, symbol: str, amount: float):
    ret = self.order_market_buy(symbol, amount)
    return ret
  
  def hard_reset(self):
    ...


class BinanceAsync(AsyncClient):
  def __init__(self):
    super().__init__(api_key=BINANCE_PUBLIC_KEY, api_secret=BINANCE_PRIVATE_KEY, testnet=TESTNET)
    






    self.quote = 'BUSD'

    acc = [tok for tok in self.loop.run_until_complete(self.futures_account_balance()) if float(tok['balance']) != 0.0]
    assert len(acc) == 1 and acc["asset"] == self.quote, f"{len(acc)} should be 1 and {acc['asset']} should be {self.quote}"
    self._balance = float(acc["balance"])
    self.pairs = ["BTCBUSD", "ETHBUSD", ...]

  

  async def futures_coin_account(self, **params):
    return await self.futures_coin_account(**params)
  
  def order_constructor(self, tok: str, amt: float, batch=False):
    if batch:
      self.futures_place_batch_order()




  @property
  def balance():
    # current balance after chg
    ...

  def trade(self, tok : List[int]):
    # tasks = []
    # task.append(self.loop.create_task(trade))
    # asyncio.gather(tasks)
    ...


  def close(self):
    # close all trades, make sure all money is in self.quote

    ...


  

if __name__ == '__main__':
  try:
    #abot = BinanceAsync()
    bot = Binance()
    print(bot.test('a', 1))
    print(bot.get_server_time()['serverTime'])
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