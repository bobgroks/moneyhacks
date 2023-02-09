#!usr/bin/env python3.11
import os
import asyncio

from dotenv import load_dotenv
from binance import Client, AsyncClient 
from binance.exceptions import BinanceAPIException

from logger import log_method_call



load_dotenv()
TESTNET = bool(os.getenv("TESTNET", 0))
BINANCE_PRIVATE_KEY = os.getenv('BINANCE_TESTNET_PRIVATE_KEY') if TESTNET else os.getenv('BINANCE_PRIVATE_KEY')
BINANCE_PUBLIC_KEY = os.getenv('BINANCE_TESTNET_PUBLIC_KEY') if TESTNET else os.getenv('BINANCE_PUBLIC_KEY')

class Binance(Client):
  def __init__(self):
    super().__init__(BINANCE_PUBLIC_KEY, BINANCE_PRIVATE_KEY, testnet=TESTNET)
    self.account = self.get_account()
    #{'makerCommission': 0, 'takerCommission': 0, 'buyerCommission': 0, 'sellerCommission': 0, 'commissionRates': {'maker': '0.00000000', 'taker': '0.00000000', 'buyer': '0.00000000', 'seller': '0.00000000'}, 'canTrade': True, 'canWithdraw': False, 'canDeposit': False, 'brokered': False, 'requireSelfTradePrevention': False, 'updateTime': 1674836417445, 'accountType': 'SPOT', 'balances': [{'asset': 'BNB', 'free': '1000.00000000', 'locked': '0.00000000'}, {'asset': 'BTC', 'free': '1.00000000', 'locked': '0.00000000'}, {'asset': 'BUSD', 'free': '10000.00000000', 'locked': '0.00000000'}, {'asset': 'ETH', 'free': '100.00000000', 'locked': '0.00000000'}, {'asset': 'LTC', 'free': '500.00000000', 'locked': '0.00000000'}, {'asset': 'TRX', 'free': '500000.00000000', 'locked': '0.00000000'}, {'asset': 'USDT', 'free': '10000.00000000', 'locked': '0.00000000'}, {'asset': 'XRP', 'free': '50000.00000000', 'locked': '0.00000000'}], 'permissions': ['SPOT']}

    #log start time time_res = client.get_server_time()

  def _update(self):
    #init again
    pass
  @log_method_call
  def test(self, symbol: str, amount: float):
    a = 1+1
    return a
  
  def trade(self, symbol: str, amount: float):
    ret = self.order_market_buy(symbol, amount)
    return ret

class BinanceAsync(AsyncClient):
  def __init__(self):
    super().__init__(BINANCE_PUBLIC_KEY, BINANCE_PRIVATE_KEY, testnet=TESTNET)





if __name__ == '__main__':
  try:
    bot = Binance()
  except BinanceAPIException as e:
    print(e.status_code)
    print(e.message)
  print(dir(bot.test))
  print(help(bot.test.__getattribute__(str())))
  
  #try:
  #  bot.get_all_orders()
  #except BinanceAPIException as e:
  #  print(e.status_code)
  #  print(e.message)


