from typing import List
from dataclasses import dataclass
from models.timing import Time

@dataclass
class Trade:
  is_executed: bool
  time_created: Time



  def close(self):
    # see if trade has been executed
    # cancel trade if still open
    # close trade
    pass
