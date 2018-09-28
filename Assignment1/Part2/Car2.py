from dataclasses import dataclass, field
from typing import List
from enum import IntEnum
from random import choice


def reward_matrix():
    return [-1, 100]


class Action(IntEnum):
    """An enum class used to represent actions for cars."""

    WAIT = 0
    """WAIT (int): Wait action variable (is 0)."""

    DRIVE = 1
    """DRIVE (int): Drive action variable (is 1)."""

    def __str__(self) -> str:
        """Method for getting name of action in readable format.

        Returns:
            str: Name of action in nice format.
        """

        return self.name.title()


@dataclass
class Car:
    """A class for representing a car in simulation."""

    score: int = 0

    speed: int = field(default=1, repr=False)
    """speed (int): Car's speed, how many tiles it moves during a turn."""

    scores: List[int] = field(default_factory=list, repr=False, init=False)
    """scores (List[int]): List of scores for car (default is list())."""

    state: int = field(default=0, repr=False, init=False)
    """state (int): Current state (default is 0)."""

    action: Action = field(default=Action.WAIT, repr=False, init=False)
    """action (Action): Action currently chosen by car (default is WAIT)."""

    q_matrix: List[float] = field(default_factory=list, repr=False, init=False)
    """q_matrix (List[float]): List of q values. 
    
    Must be set in simulation based on bridge's capacity (default is list()).
    """

    learn_factor: float = field(default=0.9, repr=False, init=False)
    """learn_factor (float): Learning factor for car (default is 0.9)."""

    decay_factor: float = field(default=0.5, repr=False, init=False)
    """decay_factor (float): Decay factor for car (default is 0.5)."""

    distance_on_bridge: int = field(default=0, repr=False, init=False)
    """distance_on_bridge (int): How far car has driven across bridge (default is 0)."""

    r_matrix: List[int] = field(default_factory=reward_matrix, repr=False, init= False)

    def take_action(self, cars_on_bridge: int) -> None:
        """Takes action for car based on bridge's current state and Q matrix.

        If all possible actions have equal Q value, it takes random action, highest values action otherwise.
        """

        self.state = cars_on_bridge

        wait_val = self.q_matrix[cars_on_bridge]
        drive_val = self.q_matrix[cars_on_bridge + 1]

        if wait_val == drive_val:
            self.action = choice([Action.WAIT, Action.DRIVE])  # Random choice
        else:
            self.action = Action.DRIVE if wait_val < drive_val else Action.WAIT

    def get_reward(self) -> int:
        return self.r_matrix[self.action]

    def reward_action(self, reward: int = None) -> None:
        """Rewards action taken by car using Q-Learning algorithm.

        Using reward argument and Q-Learning algorithm learns the car to take
        actions. Updates score for car.

        Arguments:
            reward (int): Reward value, can be negative.
        """

        a = self.action  # Current action
        s = self.state  # Current state
        q = self.q_matrix[s + a]  # Q value for state change

        if s + a + 1 >= len(self.q_matrix):  # Terminating state (bridge collapsed)
            q_next = 0
        else:  # Non-terminating state (bridge is still up)
            q_next = max(self.q_matrix[s + a], self.q_matrix[s + a + 1])

        # Update value in Q matrix
        self.q_matrix[s + a] += self.learn_factor * (reward + self.decay_factor * q_next - q)

        self.score += reward
        self.scores.append(self.score)  # Update score list

    def print_data(self) -> None:
        """Prints out score, state, action and Q-matrix for car to console."""

        print(f"Score: {sum(self.scores)}")
        print(f"Q matrix: {self.q_matrix}\n")
