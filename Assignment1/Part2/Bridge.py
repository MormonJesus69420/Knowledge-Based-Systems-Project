from dataclasses import dataclass, field
from typing import List
from Part2.Car import Car


@dataclass
class Bridge:
    capacity: int = field(default=5)
    length: int = field(default=10)
    cars: List[Car] = field(default_factory=list, repr=False, init=False)

    def has_collapsed(self) -> bool:
        return self.capacity <= len(self.cars)

    def move_cars(self) -> List[Car]:
        for c in self.cars:
            c.distance_on_bridge += 1

        finished_cars = [c for c in self.cars if c.distance_on_bridge >= self.length]
        self.cars = [c for c in self.cars if c.distance_on_bridge < self.length]

        return finished_cars

    def collapse(self) -> List[Car]:

        temp = self.cars
        self.cars = list()

        return temp
