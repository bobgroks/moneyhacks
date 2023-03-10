from typing import Optional
from loguru import logger
from functools import wraps
import time
from datetime import datetime
from dateutil import parser
import pytz
from dataclasses import dataclass

logger.add("date.log")

@dataclass
class Time:
  dt: datetime
  ms: int
  tz: str = 'UTC'
  
  @classmethod
  def init_time(cls, time_:str):
    try:
      dt = parser.parse(time_)
    except Exception as e:
      dt = datetime.now()
      #log that e
    ms = int(dt.timestamp() * 1000)
    return cls(dt, ms)

  @classmethod
  def init_from_ms(cls, ms:str):
    return cls(datetime.fromtimestamp(float(ms)/1000.0), ms)
  
  @staticmethod
  def get_cur_time() -> datetime:
    return datetime.now()

  def switch_tz(self, tz: str):
    raise NotImplementedError

  def __repr__(self):
    return f'''Time: {str(self.dt)} \
Milliseconds: {self.ms} \
Timezone: {self.tz}'''

class WithTimeit(object):
  def __enter__(self): 
    self.st = time.monotonic_ns()
  def __exit__(self, exc_type, exc_val, exc_tb): 
    print(f"With block took: {(time.monotonic_ns()-self.st)*1e-6:.2f} ms")


def timeit(is_logging=False):
  def decorator(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        if is_logging:
          pass
          #log it 
        else:
          print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
  return decorator

def async_timeit(func):
  @wraps(func)
  async def timeit_wrapper(*args, **kwargs):
      start_time = time.monotonic_ns()
      result = await func(*args, **kwargs)
      end_time = time.monotonic_ns()
      total_time = end_time - start_time
      print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
      return result
  return timeit_wrapper

@timeit(is_logging=False)
def waste():
  time.sleep(1)

if __name__ == '__main__':
  waste()
  #print(type(Time.get_cur_time()))
  
  #print(api.Binance.get_server_time())
  #trade = TimeConverter.from_milliseconds('1499827319559')
  #print(trade.dt)

