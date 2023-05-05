from typing import Union, List
import os
import json


def pprint_resp(resp: Union[str, dict]):
  if isinstance(resp, str):
    resp = json.load(resp)
  print(json.dumps(resp, indent=2))
      
class FixedLengthList(list):
  def __init__(self, max_length: int, lis: List[List[float]] = []):
    assert len(lis) <= max_length, 'lol...'
    self.max_length = max_length
    list.__init__(self, lis)

  def append(self, l: List[float]):
    list.append(self, l)
    if len(self) > self.max_length: del self[0]


  

  



TESTNET = bool(os.getenv("TESTNET", 0))
BINANCE_PRIVATE_KEY = os.getenv('BINANCE_TESTNET_PRIVATE_KEY') if TESTNET else os.getenv('BINANCE_PRIVATE_KEY')
BINANCE_PUBLIC_KEY = os.getenv('BINANCE_TESTNET_PUBLIC_KEY') if TESTNET else os.getenv('BINANCE_PUBLIC_KEY')


if __name__ == '__main__':
  l = FixedLengthList(max_length=4, lis=[1,2,3])
  l.append(1)
  l.append(1)
  print(dir(l))
