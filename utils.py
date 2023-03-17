from typing import Union
import os
import json


def pprint_resp(resp: Union[str, dict]):
  if isinstance(resp, str):
    resp = json.load(resp)
  print(json.dumps(resp, indent=2))
      
  

  



TESTNET = bool(os.getenv("TESTNET", 0))
BINANCE_PRIVATE_KEY = os.getenv('BINANCE_TESTNET_PRIVATE_KEY') if TESTNET else os.getenv('BINANCE_PRIVATE_KEY')
BINANCE_PUBLIC_KEY = os.getenv('BINANCE_TESTNET_PUBLIC_KEY') if TESTNET else os.getenv('BINANCE_PUBLIC_KEY')

