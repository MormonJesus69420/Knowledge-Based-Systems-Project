from dataclasses import dataclass, field
from typing import List
from Car2 import Car


@dataclass
class Bridge:
    """Bridge class simulating the behaviour of bridge in simulation.

    On can set specific length and capacity for the bridge to change the overall
    behaviour of bridge in the simulation and see how it impacts the scores for
    cars.
    """

    capacity: int = field(default=5)
    """Set amount of cars that the bridge can accommodate before collapsing."""

    length: int = field(default=10)
    """Length of bridge deciding how much time a car will use to cross it."""

    cars: List[Car] = field(default_factory=list, repr=False, init=False)
    """List of all of the cars that are currently on the bridge."""

    def has_collapsed(self) -> bool:
        """Simple method to check if bridge has collapsed.

        Returns:
            bool: True if bridge has collapsed, False otherwise.
        """

        return len(self.cars) > self.capacity

    def move_cars(self) -> List[Car]:
        """ Moves cars across the bridge and returns cars that have crossed it.

        Returns:
            List[Car2]: List of cars that have crossed the bridge this turn.
        """

        finished_cars = list()
        for c in self.cars:
            c.distance_on_bridge += c.speed
            if c.distance_on_bridge >= self.length:
                finished_cars.append(c)

        self.cars = [c for c in self.cars if c not in finished_cars]

        return finished_cars

    def collapse_bridge(self) -> List[Car]:
        """"""
        temp = self.cars
        self.cars = list()

        return temp
