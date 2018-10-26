from Car2 import Car, Action
from Bridge2 import Bridge
from dataclasses import dataclass, field
from typing import List
from matplotlib import pyplot
from random import shuffle
import numpy as np


@dataclass
class Simulation:
    """A class for simulating fixed capacity bridge crossing problem.

    It takes in a list of cars that will try crossing a bridge with a fixed
    capacity. It tries to teach them to not drive on the bridge if it is at its
    full capacity, without telling the cars what the capacity is.
    """

    bridge: Bridge = field(repr=False)
    """Bridge used in simulation"""

    carpool: List[Car] = field(repr=False)
    """List of cars that are not on bridge or in queue."""

    lamb: float = field(default=1.5)
    """Lambda value for Poisson distribution"""

    debug: bool = field(default=False, repr=False)
    """Boolean that decides whether or not to print some extra info to console."""

    queue: List[Car] = field(default_factory=list, repr=False, init=False)
    """List of cars waiting to cross the bridge."""

    def get_new_cars(self) -> None:
        """Method for getting cars from carpool into queue."""

        new_cars = np.random.poisson(lam=self.lamb)

        if new_cars >= len(self.carpool):  # All cars from carpool come to queue
            self.queue.extend(self.carpool)
            self.carpool = list()
        else:  # Fewer cars than cars in carpool came
            shuffle(self.carpool)
            while new_cars > 0:
                car = self.carpool.pop(0)
                car.distance_on_bridge = 0
                self.queue.append(car)
                new_cars -= 1

    def take_action(self, car: Car) -> None:
        """Lets car in queue take an action, if it drives it gets put on bridge."""

        car.take_action(len(self.bridge.cars))

        if car.action == Action.DRIVE:
            self.bridge.cars.append(self.queue.pop(0))

    def give_reward(self, car: Car) -> None:
        """Gives rewards to car based on its action, if bridge collapses reset cars.

        Firstly it finds the reward for car and calls reward action method,
        before checking if bridge has collapsed, if it did it moves cars from
        bridge back into carpool.
        """

        reward = -200 if self.bridge.has_collapsed() else car.get_reward()

        car.reward_action(reward)

        if self.bridge.has_collapsed():
            self.carpool += self.bridge.collapse_bridge()

    def show_graph(self) -> None:
        """Shows graph with scores for each car after simulation."""

        handles = list()
        count = 1

        for c in self.carpool:
            if self.debug:
                c.print_data()

            temp, = pyplot.plot(c.scores, label=f"Car {count}")
            handles.append(temp)
            count += 1

        pyplot.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))

        pyplot.ylabel("Score")
        pyplot.xlabel("Actions")

        pyplot.show()

    def simulate_turns(self, no_turns: int) -> None:
        """Simulates a given amount of turns for the cars.

        This method starts by instantiating Q matrices for cars based on bridge
        capacity and then it simulates turns. For each turn it takes some cars
        from carpool and moves them to queue (number depends of Poisson
        distribution). Cars on bridge get to move and then the cars in queue get
        to decide whether to move or not, and are rewarded based on their
        actions. After the turns it shows score graph for cars.

        Arguments:
            no_turns (int): Number of turns to simulate.
        """

        for c in self.carpool:
            c.q_matrix = [0] * (self.bridge.capacity + 2)

        for _ in range(no_turns):
            self.get_new_cars()

            self.carpool += self.bridge.move_cars()

            for c in self.queue:
                self.take_action(c)
                self.give_reward(c)

        self.carpool += self.bridge.cars
        self.carpool += self.queue

        self.show_graph()


if __name__ == "__main__":
    a = list()
    for _ in range(15):
        a.append(Car())

    s = Simulation(Bridge(), a, debug=True)
    s.simulate_turns(100000)
