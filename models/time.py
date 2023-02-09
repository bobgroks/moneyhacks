from .. import api
import time
import datetime

class Time:
  def __init__(self, dt):
    self.dt = dt
    #self.local_dt = 
    #self.

  @classmethod
  def from_milliseconds(cls, ms:str, loc):

    return Time(datetime.datetime.fromtimestamp(float(ms)/1000.0))

if __name__ == '__main__':
  print(api.Binance.get_server_time())
  trade = Time.from_milliseconds('1499827319559')
  print(trade.dt)

