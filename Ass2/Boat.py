from dataclasses import dataclass
from Passenger import Passenger


@dataclass
class Boat:
    max_weight: int = 500
    current_weight: int = 100

    def board(self, p1: Passenger, p2: Passenger) -> None:
        self.current_weight += p1.get_weight()
        self.current_weight += p2.get_weight()

    def de_board(self) -> None:
        self.current_weight = 100

    def is_sinking(self) -> bool:
        return self.current_weight > self.max_weight
