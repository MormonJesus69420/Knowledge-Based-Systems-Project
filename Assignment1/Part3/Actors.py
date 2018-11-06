import random as r
from dataclasses import dataclass, field
from datetime import timedelta
from enum import IntEnum

import numpy as np
from typing import List


class Propensity(IntEnum):
    """An enum class used as an identifier for propensities for auctioneers."""

    RAISE = 0
    """RAISE (int): Propensity identifier for raising (is 0)."""
    LOWER = 1
    """LOWER (int): Propensity identifier for lowering (is 1)."""

    def __str__(self) -> str:
        """Method for getting name of action in readable format.

        Returns:
            str: Name of propensity in nice format.
        """

        return self.name.title()


def get_propensities() -> List[float]:
    return list([0.5, 0.5])


@dataclass
class Auctioneer:
    """A class for representing an auction participant in simulation."""

    propensities: List[float] = field(default_factory=get_propensities, repr=False, init=False)
    """propensities (List[float]): List of propensities for choosing raise or lower (default is [0.5, 0.5])."""

    bid: int = field(default=r.randint(0, 100), repr=False, init=False)
    """bid (int): Starting bid, chosen randomly in start (default is 0 - 100)."""

    phi: float = field(default=0.25)
    """phi (float): Decides how much propensity is changed after accepting/rejectin the bid (default is 0.25)."""

    delta: int = field(default=5)
    """delta (int): Decides how much the bid is changed in Roth Evet (default is 5)."""

    def balance_propensities(self) -> None:
        _raise = self.propensities[Propensity.RAISE]
        _lower = self.propensities[Propensity.LOWER]

        total = _raise + _lower
        self.propensities[Propensity.RAISE] = _raise / total
        self.propensities[Propensity.LOWER] = _lower / total

    def reject(self) -> None:
        pass

    def accept(self) -> None:
        pass

    def zero_intelligence_bidding(self):
        self.bid = r.randint(0, 60)

    def roth_erev_bidding(self) -> None:
        val = np.random.uniform(0, 1)
        highest = max(self.propensities[Propensity.RAISE], self.propensities[Propensity.LOWER])
        if val <= highest:
            self.bid += self.delta if highest == self.propensities[Propensity.RAISE] else -self.delta
        else:
            self.bid -= self.delta if highest == self.propensities[Propensity.LOWER] else -self.delta


@dataclass
class Taxi(Auctioneer):
    travel_time: timedelta = field(default=None, repr=False)

    customer_distance: int = field(default=5, repr=False)

    def busy(self, minute: timedelta) -> None:
        self.travel_time -= minute

    def reject(self) -> None:
        _raise = self.propensities[Propensity.RAISE]
        _lower = self.propensities[Propensity.LOWER]

        self.propensities[Propensity.LOWER] = (1 - self.phi) * _lower + (_lower * _raise)
        self.balance_propensities()

    def accept(self) -> None:
        _raise = self.propensities[Propensity.RAISE]
        _lower = self.propensities[Propensity.LOWER]

        self.propensities[Propensity.RAISE] = (1 - self.phi) * _raise + (_lower * _raise)
        self.balance_propensities()


@dataclass
class Passenger(Auctioneer):
    travel_time: timedelta = field(default=None)

    group_size: int = field(default=0)

    def __post_init__(self):
        self.delta = 10
        self.propensities = list([1.0, 0])

    def reject(self) -> None:
        _raise = self.propensities[Propensity.RAISE]
        _lower = self.propensities[Propensity.LOWER]

        self.propensities[Propensity.RAISE] = (1 - self.phi) * _raise + (_lower * _raise)
        self.balance_propensities()

    def accept(self) -> None:
        _raise = self.propensities[Propensity.RAISE]
        _lower = self.propensities[Propensity.LOWER]

        self.propensities[Propensity.LOWER] = (1 - self.phi) * _lower + (_lower * _raise)
        self.balance_propensities()
