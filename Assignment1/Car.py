from dataclasses import dataclass, field
from enum import IntEnum
from random import choice
from numpy import array, zeros


class Action(IntEnum):
    """An enum class used to represent actions for cars.

    Attributes:
        WAIT (int): Wait action variable (is 0).
        DRIVE (int): Drive action variable (is 1).
    """

    WAIT = 0
    DRIVE = 1

    def __str__(self) -> str:
        """Method for getting name of action in readable format.

        Returns:
            str: Name of action in nice format.
        """

        return self.name.title()


def make_new_q_matrix() -> array:
    """Method used to initialize Q matrix for learning car.

    Returns:
        array: A 2x2 NumPy array with zeroes.
    """

    return zeros((2, 2))


def random_action() -> Action:
    """Method for getting a random action.

    Returns:
        Action: A random action value.
    """

    return choice([Action.WAIT, Action.DRIVE])


@dataclass
class Car:
    """A super class for representing a car in simulation.

    Attributes:
        action (Action): Action currently chosen by car (default is WAIT).
        drove_over (bool): Whether car has driven over bridge (default is False).
        score (int): Score earned by car in simulation.
    """

    action: Action = field(default=Action.WAIT, repr=False, init=False)
    drove_over: bool = field(default=False, repr=False, init=False)
    score: int = 0

    def take_action(self) -> None:
        """Takes action for car, NOT IMPLEMENTED."""

        pass

    def reward_action(self, reward: int) -> None:
        """Rewards action taken by car, NOT IMPLEMENTED.

        Arguments:
            reward (int): Reward value, can be negative.
        """

        pass


@dataclass
class RandomCar(Car):
    """Zero intelligence car, takes random actions and learns nothing."""

    def take_action(self) -> None:
        """Takes random action for car."""

        self.action = random_action()

    def reward_action(self, reward: int) -> None:
        """Rewards action taken by car, simply updates score.

        Arguments:
            reward (int): Reward value, can be negative.
        """

        self.score += reward

        self.action = Action.WAIT if self.drove_over else self.action  # Done here so that simulate class is versatile.


@dataclass
class LearningCar(Car):
    """Learning car, takes actions based on Q-Learning algorithm.

    Attributes:
        q_matrix (array): Q matrix for car (default is 2x2 zero matrix).
        state (Action): Current state for car (default is either WAIT or DRIVE).
        learn_factor (float): Learning factor for car (default is 0.9).
        decay_factor (float): Decay factor for car (default is 0.5).
    """

    q_matrix: array = field(default_factory=make_new_q_matrix, repr=False, init=False)
    state: Action = field(default_factory=random_action, repr=False, init=False)
    learn_factor: float = 0.9
    decay_factor: float = 0.5

    def take_action(self) -> None:
        """Takes action for car based on its current state and Q matrix.

        If all actions have equal Q value, take random action. Take action with
        biggest Q value otherwise.
        """

        col = self.q_matrix[:, self.state]  # Get column for current state
        if col[Action.WAIT] == col[Action.DRIVE]:  # Check if equal in column
            self.action = random_action()  # Random choice
        else:
            self.action = Action(col.argmax())  # Choose best value from column

    def reward_action(self, reward: int) -> None:
        """Rewards action taken by car using Q-Learning algorithm.

        Using reward argument and Q-Learning algorithm updates learns the car to
        take actions. Updates score and state for car.

        Arguments:
            reward (int): Reward value, can be negative.
        """

        a = self.action
        s = self.state
        q = self.q_matrix[s, a]  # Q value for state change
        q_next = self.q_matrix[a, self.q_matrix[a].argmax()]  # Q value for best action in new state

        # Update value in Q matrix
        self.q_matrix[s, a] += self.learn_factor * (reward + self.decay_factor * q_next - q)

        self.state = self.action  # Action is the new state
        self.score += reward  # Update score
        self.action = Action.WAIT if self.drove_over else self.action  # Done here so that simulate class is versatile.
