from dataclasses import dataclass, field
from Luggage import Luggage
from random import choice
from typing import List


@dataclass
class Passenger:
    on_person: List[Luggage] = None
    luggage: List[Luggage] = None
    wrong_side: bool = True
    own_weight: int = 100
    payed: int = 0

    def pick_luggage(self) -> None:
        # Assume that what was on person is left on shore
        self.on_person = list()

        # Choose randomly from bags left to take
        for bag in reversed(self.luggage):
            if choice([True, False]):
                self.on_person.append(bag)

        # Remove taken bags from bags to take, set - set return non-duplicates
        self.luggage = list(set(self.luggage) - set(self.on_person))

    def get_weight(self) -> int:
        return self.own_weight + sum(bag.weight for bag in self.on_person)

    def pay_for_trip(self) -> None:
        self.payed += 100
        self.payed += self.get_weight() * 10

    def pay_for_return(self) -> None:
        self.payed += 100
