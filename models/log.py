from typing import Optional
from functools import wraps
import os
import time
import pytz
import cProfile
import line_profiler
from loguru import logger
"""
Level name	Severity value	Logger method
TRACE	5	logger.trace()
DEBUG	10	logger.debug()
INFO	20	logger.info()
SUCCESS	25	logger.success()
WARNING	30	logger.warning()
ERROR	40	logger.error()
CRITICAL	50	logger.critical()
"""

# logging.error(), logging.exception() or logging.critical() 

os.getenv("TESTNET")

logger.add("date.log")

def logger_wraps(*, entry=True, exit=True, resp=True, level: str = 'info'):

  def wrapper(func):
    name = func.__name__

    @wraps(func)
    def wrapped(*args, **kwargs):
      logger_ = logger.opt(depth=1)
      if entry:
        logger_.log(level, f"Entering '{name}' (args={args}, kwargs={kwargs}) at {time.time()}")
      result = func(*args, **kwargs)
      if exit:
        logger_.log(level, "Exiting '{}' (result={})", name, result)
      return result

    return wrapped

  return wrapper



class BlockTimeit(object):
  def __enter__(self):
    self.st = time.monotonic_ns()
  def __exit__(self, exc_type, exc_val, exc_tb):
    print(f"With block took: {(time.monotonic_ns()-self.st)*1e-6:.2f} ms")


def timeit(is_logging=False):
  def decorator(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
      start_time = time.monotonic_ns()
      result = func(*args, **kwargs)
      end_time = time.monotonic_ns()
      total_time = end_time - start_time
      if is_logging:
        pass
        #log it
      else:
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time*1000:.4f} ms or {total_time*1000000:.4f} s')
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
  from ..binance_api import Binance
  #from dotenv import load_dotenv
  #load_dotenv()
  #TESTNET = int(os.getenv("TESTNET", 1))
  #BINANCE_PRIVATE_KEY = os.getenv('BINANCE_TESTNET_SECRET_KEY') if TESTNET else os.getenv('BINANCE_SECRET_KEY')
  #BINANCE_PUBLIC_KEY = os.getenv('BINANCE_TESTNET_PUBLIC_KEY') if TESTNET else os.getenv('BINANCE_PUBLIC_KEY')
  #print(type(Time.get_cur_time()))
  bot = Binance()
  print(bot.get_account())

  #print(api.Binance.get_server_time())
  #trade = TimeConverter.from_milliseconds('1499827319559')
  #print(trade.dt)
