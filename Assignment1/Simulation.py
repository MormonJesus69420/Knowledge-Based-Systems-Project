from dataclasses import dataclass, field
from typing import List
from enum import IntEnum
from matplotlib import pyplot

from Car import Action, Car, RandomCar


class State(IntEnum):
    CRASH = 0
    BOTH_WAIT = 1
    CAR_1_DROVE = 2
    CAR_2_DROVE = 3

    def __str__(self) -> str:
        return self.name.lower().replace("_", " ")


@dataclass
class Simulation:
    car1: Car
    car2: Car
    state: State = State.BOTH_WAIT
    car1_scores: List[int] = field(default_factory=list, repr=False, init=False)
    car2_scores: List[int] = field(default_factory=list, repr=False, init=False)

    def simulate_turn(self) -> None:
        pass

    def check_state(self) -> None:
        if self.car1.action == self.car2.action == Action.DRIVE:
            self.state = State.CRASH
        elif self.car1.action == self.car2.action == Action.WAIT:
            self.state = State.BOTH_WAIT
        elif self.car1.action == Action.DRIVE:
            self.state = State.CAR_1_DROVE
            self.car1.drove_over = True
            self.car1.action = Action.WAIT
            self.car1.score += 100
        else:
            self.state = State.CAR_2_DROVE
            self.car2.drove_over = True
            self.car2.action = Action.WAIT
            self.car2.score += 100

    def give_rewards(self) -> None:
        pass

    def print_results(self) -> None:
        print(f"State: {str(self.state)} Scores: {self.car1.score} & {self.car2.score}")

    def cleanup_game(self):
        self.car1_scores.append(self.car1.score)
        self.car2_scores.append(self.car2.score)
        self.state = State.BOTH_WAIT
        self.car1.drove_over = self.car2.drove_over = False

    def print_graph(self) -> None:
        car1, = pyplot.plot(self.car1_scores, label="Car 1 scores")
        car2, = pyplot.plot(self.car2_scores, label="Car 2 scores")
        pyplot.legend(handles=[car1, car2])

        pyplot.ylabel("Score")

        pyplot.show()

    def simulate_games(self, no_games: int) -> None:
        for _ in range(no_games):
            print("New game")
            while (not self.car1.drove_over or not self.car2.drove_over) and self.state != State.CRASH:
                self.simulate_turn()
                self.check_state()
                self.give_rewards()

                self.print_results()

            self.cleanup_game()

        self.print_graph()


@dataclass
class RandomSimulation(Simulation):
    def __init__(self):
        super().__init__(RandomCar(), RandomCar())

    def simulate_turn(self):
        if not self.car1.drove_over:
            self.car1.make_choice()
        if not self.car2.drove_over:
            self.car2.make_choice()

    def give_rewards(self):
        if self.state == State.CRASH:
            r1 = r2 = -200
        elif self.state == State.BOTH_WAIT:
            r1 = r2 = -10
        elif self.state == State.CAR_1_DROVE:
            r1 = 100
            r2 = -10
        else:
            r1 = -10
            r2 = 100

        if not self.car1.drove_over:
            self.car1.score += r1
        if not self.car2.drove_over:
            self.car2.score += r2


r = RandomSimulation()
r.simulate_games(1000)
