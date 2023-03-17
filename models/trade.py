from dataclasses import dataclass
from functools import total_ordering


@total_ordering
@dataclass
class Person:
  name: str
  age: int

  def __eq__(self, other):
    return self.age == other.age

  def __lt__(self, other):
    return self.age < other.age


class Trade:
  def __init__(self):
    pass

  def close(self):
    pass
