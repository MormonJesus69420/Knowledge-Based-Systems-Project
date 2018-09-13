from dataclasses import dataclass, field
from typing import List
from enum import IntEnum
from matplotlib import pyplot

from Car import Action, Car, RandomCar, LearningCar


class State(IntEnum):
    """An enum class used to represent states in simulation.

    Attributes:
        CRASH (int): Crash state variable (is 0).
        BOTH_WAIT (int): Both cars wait state variable (is 1).
        CAR_1_DROVE (int): Car 1 drove while 2 waited state variable (is 2).
        CAR_2_DROVE (int): Car 2 drove while 1 waited state variable (is 3).
    """

    CRASH = 0
    BOTH_WAIT = 1
    CAR_1_DROVE = 2
    CAR_2_DROVE = 3

    def __str__(self) -> str:
        """Method for getting name of state in readable format.

        Returns:
            str: Name of state in nice format.
        """

        return self.name.lower().replace("_", " ")


@dataclass
class Simulation:
    """A class for simulating one lane bridge crossing problem.

    It takes in two cars and uses them to simulate games finally showing a graph
    with scores for each cars gained through simulation. Simulates games where
    each car tries to cross a one lane bridge. Each game lasts until both cars
    have safely crossed the bridge or crashed.

    Attributes:
        car1 (Car): First car to be used in simulation.
        car2 (Car): Second car to be used in simulation.
        car1_scores (List[int]): List of scores for car 1 after each game
        (default is an empty list).
        car2_scores (List[int]): List of scores for car 2 after each game
        (default is an empty list).
        state (State): State for current turn in game (default is BOTH_WAIT).
        debug (bool): Whether to print status messages to (default is False).
    """

    car1: Car
    car2: Car
    car1_scores: List[int] = field(default_factory=list, repr=False, init=False)
    car2_scores: List[int] = field(default_factory=list, repr=False, init=False)
    state: State = State.BOTH_WAIT
    debug: bool = False

    def take_actions(self) -> None:
        """Lets each car that hasn't crossed bridge take an action."""

        if not self.car1.drove_over:
            self.car1.take_action()
        if not self.car2.drove_over:
            self.car2.take_action()

    def check_state(self) -> None:
        """Checks game state after each car took action."""

        c1 = self.car1
        c2 = self.car2

        if c1.action == c2.action == Action.DRIVE:
            self.state = State.CRASH
        elif c1.action == c2.action == Action.WAIT:
            self.state = State.BOTH_WAIT
        elif c1.action == Action.DRIVE:
            self.state = State.CAR_1_DROVE
            c1.drove_over = True
        else:
            self.state = State.CAR_2_DROVE
            c2.drove_over = True

    def give_rewards(self) -> None:
        """Gives rewards to each car based on current turn state.

        First it finds reward value based on turn state, then calls
        reward_action in each car as long as it hasn't driven over the bridge.
        """

        s = self.state
        r1 = -200 if s is State.CRASH else 100 if s is State.CAR_1_DROVE else -10
        r2 = -200 if s is State.CRASH else 100 if s is State.CAR_2_DROVE else -10

        if not self.car1.drove_over or s is State.CAR_1_DROVE:
            self.car1.reward_action(r1)
        if not self.car2.drove_over or s is State.CAR_2_DROVE:
            self.car2.reward_action(r2)

    def print_results(self) -> None:
        """Prints out state and scores for cars for this round to console."""

        print(f"State: {str(self.state)} Scores: {self.car1.score} & {self.car2.score}")

    def cleanup_game(self) -> None:
        """Adds scores for cars to list, resets state and drove_over for cars."""

        self.car1_scores.append(self.car1.score)
        self.car2_scores.append(self.car2.score)
        self.state = State.BOTH_WAIT
        self.car1.drove_over = self.car2.drove_over = False

    def show_graph(self) -> None:
        """Shows graph with scores for each car after simulation."""

        car1, = pyplot.plot(self.car1_scores, label="Car 1")
        car2, = pyplot.plot(self.car2_scores, label="Car 2")
        pyplot.legend(handles=[car1, car2])

        pyplot.ylabel("Score")
        pyplot.xlabel("Games")

        pyplot.show()

    def simulate_games(self, no_games: int) -> None:
        """Simulates a given amount of games for the cars.

        This method runs the whole simulation calling methods in class to play
        out games and turns in the game. For each game it lets cars take action,
        checks turn state, and gives rewards. After each game it cleans up and
        runs again until it has ran the required amount of games. In the end it
        shows plot of car scores.

        Arguments:
            no_games (int): Number of games to simulate.
        """

        for _ in range(no_games):
            if self.debug:
                print("New game")
            while not (self.car1.drove_over and self.car2.drove_over) and self.state != State.CRASH:
                self.take_actions()
                self.check_state()
                self.give_rewards()
                if self.debug:
                    self.print_results()

            self.cleanup_game()

        self.show_graph()


print("Zero Intelligence cars")
random = Simulation(RandomCar(), RandomCar())
random.simulate_games(1000)

print("Q-Learning cars")
learn = Simulation(LearningCar(), LearningCar())
learn.simulate_games(1000)
