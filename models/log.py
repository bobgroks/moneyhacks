from typing import Optional, Callable
from functools import wraps
import os
import time
import asyncio
import pytz
import cProfile
import line_profiler
from loguru import logger
from pathlib import Path
from timeit import timeit
"""
Level name	Severity value	Logger method
TRACE	5	logger.trace()
DEBUG	10	logger.debug()
INFO	20	logger.info()         api_call_logger decorator
SUCCESS	25	logger.success()
WARNING	30	logger.warning() 
ERROR	40	logger.error()        need to produce detailed tracing
CRITICAL	50	logger.critical() need to call hard_reset() and detailed tracing
"""

# logging.error(), logging.exception() or logging.critical() 

os.getenv("TESTNET")

info_path = Path('../var/{time:MM-DD}.log')
info_format = "<green>{time:XSSS!UTC}</green> | <level>{level: <8}</level> | <level>{message}</level>"
#logger_loop = asyncio.new_event_loop()
#format_ = "<green>{time:XSSS!UTC}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
error_path = Path('../var/error.log')
error_format = "<green>{time:XSSS!UTC}</green> | <level>{level: <8}</level> | <level>{message}</level>"
logger.add(sink=info_path, enqueue=True, format=info_format,level='INFO')
logger.add(sink=error_path, enqueue=True, format=error_format, backtrace=True, diagnose=True, level='ERROR')
#logger.info(f"{time.time()*1000:.0f}")

def loggy(*, logger):
  def log(fxn: Callable):
    @wraps(fxn)
    def method_wrapper(*args, **kwargs):
      st = int(time.time()*1000)
      logger.info(f"{st}: CALL {fxn.__name__} with args={args} kwargs={kwargs}")
      res = fxn(*args, **kwargs)
      ed = int(time.time()*1000)
      el = ed - st
      logger.info(f"{ed}: RESP {fxn.__name__} | time elapsed {el} | with args={args} kwargs={kwargs} | ret={res}")
      return res
    return method_wrapper

def api_call_logger(cls):
  for name, fxn in vars(cls).items():
    if callable(fxn):
      setattr(cls, name, loggy(fxn))
  return cls








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



class ContextLogger(object):
  def __enter__(self):
    self.st = time.monotonic_ns()
  def __exit__(self, exc_type, exc_val, exc_tb):
    print(f"Code block took: {(time.monotonic_ns()-self.st)*1e-6:.2f} ms")


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


@timeit(is_logging=False)
def waste():
  time.sleep(1)

if __name__ == '__main__':
  #from ..binance_api import Binance
  #from dotenv import load_dotenv
  #load_dotenv()
  #TESTNET = int(os.getenv("TESTNET", 1))
  #BINANCE_PRIVATE_KEY = os.getenv('BINANCE_TESTNET_SECRET_KEY') if TESTNET else os.getenv('BINANCE_SECRET_KEY')
  #BINANCE_PUBLIC_KEY = os.getenv('BINANCE_TESTNET_PUBLIC_KEY') if TESTNET else os.getenv('BINANCE_PUBLIC_KEY')
  #print(type(Time.get_cur_time()))
  #bot = Binance()
  #print(bot.get_account())

  #print(api.Binance.get_server_time())
  #trade = TimeConverter.from_milliseconds('1499827319559')
  logger.info('fuck')
  #print(trade.dt)