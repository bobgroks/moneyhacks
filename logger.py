import os
import inspect
from functools import wraps
import logging
import datetime

DEBUG = bool(os.getenv("DEBUG", 0))
#logging.basicConfig(format='%(name)s %(levelname)s %(message)s')
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('application.log')
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter.default_msec_format = '%s.%03d'
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_method_call(fn):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        logger.info(f'Method {fn.__name__} returned: {result}')
        return result
    return wrapper


def logger_wrapper(func):
  @wraps(func)
  def decorate():
    ...


if __name__ == "__main__":
  logger.error('fuck')
