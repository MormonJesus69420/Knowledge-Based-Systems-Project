from dataclasses import dataclass, field
from enum import IntEnum
from random import choice


class Action(IntEnum):
    WAIT = 0
    DRIVE = 1

    def __str__(self) -> str:
        return self.name.title()


@dataclass
class Car:
    action: Action = field(default=Action.WAIT, repr=False, init=False)
    drove_over: bool = field(default=False, repr=False, init=False)
    score: int = 0

    def make_choice(self):
        pass


@dataclass
class RandomCar(Car):
    def make_choice(self):
        self.action = choice([Action.WAIT, Action.DRIVE])

