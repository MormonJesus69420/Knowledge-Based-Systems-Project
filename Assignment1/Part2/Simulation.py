from Part2.Car import Car, Action
from Part2.Bridge import Bridge
from dataclasses import dataclass, field
from typing import List
from matplotlib import pyplot
from random import shuffle
import numpy as np


@dataclass
class Simulation:
    bridge: Bridge = field(repr=False)
    carpool: List[Car] = field(repr=False)
    lamb: float = field(default=1.5)
    queue: List[Car] = field(default_factory=list, repr=False, init=False)
    debug: bool = field(default=False, repr=False, init=False)

    def get_new_cars(self) -> None:
        new_cars = np.random.poisson(lam=self.lamb)
        if new_cars >= len(self.carpool):
            self.queue = self.carpool
            self.carpool = list()
        else:
            shuffle(self.carpool)
            while new_cars > 0:
                self.queue.append(self.carpool.pop(0))
                new_cars -= 1

    def take_action(self, car: Car) -> None:
        """Lets each car that hasn't crossed bridge take an action."""

        car.take_action(len(self.bridge.cars) - 1)

        if car.action == Action.DRIVE:
            self.bridge.cars.append(self.queue.pop(0))

    def give_reward(self, car: Car) -> None:
        """Gives rewards to each car based on current turn state.

        First it finds reward value based on turn state, then calls
        reward_action in each car as long as it hasn't driven over the bridge.
        """

        reward = -200 if self.bridge.has_collapsed() else car.get_reward()

        car.reward_action(reward)

        if self.bridge.has_collapsed():
            self.carpool += self.bridge.collapse()

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

        pyplot.legend(handles=handles)

        pyplot.ylabel("Score")
        pyplot.xlabel("Bridge crossings")

        pyplot.show()

    def simulate_turns(self, no_turns: int) -> None:
        """Simulates a given amount of games for the cars.

        This method runs the whole simulation calling methods in class to play
        out games and turns in the game. For each game it lets cars take action,
        checks turn state, and gives rewards. After each game it cleans up and
        runs again until it has ran the required amount of games. In the end it
        shows plot of car scores.

        Arguments:
            no_turns (int): Number of games to simulate.
        """

        for c in self.carpool:
            c.q_matrix = [0] * (self.bridge.capacity + 1)

        for _ in range(no_turns):
            self.get_new_cars()

            self.carpool += self.bridge.move_cars()

            for c in self.queue:
                self.take_action(c)
                self.give_reward(c)

        self.carpool += self.bridge.cars
        self.carpool += self.queue

        self.show_graph()


a = list()
for _ in range(15):
    a.append(Car())

s = Simulation(Bridge(), a)
s.simulate_turns(100)
