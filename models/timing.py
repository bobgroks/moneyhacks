import time
from typing import Optional, Union, Type
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from functools import total_ordering
# from dateutil import parser

@total_ordering
@dataclass(order=True)
class Timestamp:
  ms: int
  tz: timezone = timezone.utc
  _dt: Optional[datetime] = None

  def __init__(self, ms:Union[str, int, float]):
    self.ms = ms if isinstance(ms, int) else int(ms)
    assert len(str(self.ms)) == 13, f"The value of self.ms {self.ms} is not 12 digits long."

  @property
  def dt(self) -> datetime:
    if self._dt is None:
      self._dt = self.ms_to_dt(self.ms) 
    return self._dt
  
  @dt.setter
  def dt(self, new_dt):
    self._dt = new_dt
    self.ms = self.dt_to_ms(new_dt)

  @property
  def time_elapsed_ms(self) -> int:
    return self.now() - self

  @classmethod
  def now(cls, tz: timezone=timezone.utc) -> 'Timestamp':
    return cls(time.time_ns() * 1e-6)
  
  def apply_delta(self, delta: str, add: bool): # e.g. "04:00:00"
    delta: datetime = datetime.strptime(delta, '%H:%M:%S')
    if add:
      self.dt = self.dt + timedelta(hours=delta.hour, minutes=delta.minute, seconds=delta.second)
    else:
      self.dt = self.dt - timedelta(hours=delta.hour, minutes=delta.minute, seconds=delta.second)

  @staticmethod
  def ms_to_dt(ms: Union[int,str], tz=timezone.utc):
    return datetime.fromtimestamp(float(ms)/1000.0, tz=tz)

  @staticmethod
  def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


  def switch_tz(self, tz: str):
    raise NotImplementedError("don't need this yet")

  def __str__(self) -> str: return f"{self.ms}"
  def __repr__(self) -> str: return f"ms: {self.ms}, time: {str(self.dt)} {self.tz}"
  def __sub__(self, other: Union['Timestamp', str, int, float]) -> int: return self.ms - other.ms if isinstance(other, Timestamp) else self.ms - int(other)
  def __rsub__(self, other: Union['Timestamp', str, int, float]) -> int: return other.ms - self.ms if isinstance(other, Timestamp) else int(other) - self.ms
  def __eq__(self, other: 'Timestamp'): return self.ms == other.ms

  

if __name__ == "__main__":
  # print(api.Binance.get_server_time())
  trade = Timestamp('1499827319559')
  trade2 = Timestamp('1499827329559')
  trade3 = Timestamp('1499827339559')
  ls = [trade2, trade, trade3]
  print(ls)
  print(sorted(ls))
  print(trade2)
  print(str(trade2))
  print(trade2 - trade)
  n = Timestamp.now()
  import time
  time.sleep(1)
  print(n.time_elapsed_ms)

