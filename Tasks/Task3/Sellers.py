from dataclasses import dataclass, field
from typing import List
from random import randrange


@dataclass
class Seller:
    units_sold: List[int] = field(default_factory=list)
    shoes_available: int = 0
    current_price: int = 0
    earning: int = 0
    goal: int = 100

    def pick_price(self) -> None:
        pass


@dataclass
class ZISeller(Seller):
    units_sold: List[int] = field(default_factory=list)
    shoes_available: int = 0
    current_price: int = 0
    earning: int = 0
    goal: int = 100

    def pick_price(self) -> None:
        self.units_sold = randrange(0, 101)
