import time
from typing import Optional, Union
from dataclasses import dataclass
from datetime import datetime, timezone
# from dateutil import parser


@dataclass(order=True)
class Time:
  ms: int
  tz: timezone = timezone.utc
  _dt: Optional[datetime] = None

  def __init__(self, ms:Union[str, int, float]):
    if isinstance(ms, str):
      self.ms = int(ms)
    elif isinstance(ms, float):
      self.ms = int(ms)
    else:
      self.ms = ms

  @property
  def dt(self) -> datetime:
    if self._dt is None:
      self._dt = self.ms_to_dt(self.ms) 
    return self._dt

  @property
  def time_elapsed(self) -> datetime:
    return datetime.now(tz=self.tz) - self.dt

  @property
  def time_elapsed_ms(self) -> int:
    return int(time.time_ns()*1e-6 - self.ms)

  @staticmethod
  def ms_to_dt(ms: Union[int,str], tz=timezone.utc):
    return datetime.fromtimestamp(float(ms)/1000.0, tz=tz)

  def switch_tz(self, tz: str):
    raise NotImplementedError("don't need this yet")

  def __str__(self) -> str: return f"ms: {self.ms}, time: {str(self.dt)} {self.tz}"
  def __sub__(self, other: Union['Time', str, int]) -> int: return self.ms - other.ms

  

if __name__ == "__main__":
  # print(api.Binance.get_server_time())
  trade = Time('1499827319559')
  trade2 = Time('1499827329559')
  trade3 = Time('1499827339559')
  ls = [trade2, trade, trade3]
  print(ls)
  print(sorted(ls))
  print(trade2)
  print(str(trade2))
