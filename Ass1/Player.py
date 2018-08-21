from dataclasses import dataclass
from Move import Move


@dataclass
class Player:
    move: Move = None
    wins: int = 0
