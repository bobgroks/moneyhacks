from typing import List, ClassVar
from dataclasses import dataclass
'''
from timing import Timestamp

@dataclass
class Trade:
  time_created = Timestamp.now()
  # is_executed: bool = False
  # open_trades: ClassVar[int] = 0

  def __post_init__(self):
    if self.time_created is None:
      self.time_created = Timestamp.now()
    type(self).open_trades += 1



  def close(self):
    # see if trade has been executed
    # cancel trade if still open
    # close trade
    type(self).open_trades -= 1
    del self

if __name__ == "__main__":
  t1 = Trade()
  t2 = Trade()
  t3 = Trade()
  print(Trade.open_trades)
  t3.close()
  print(Trade.open_trades)
'''