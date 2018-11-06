import random as r
from dataclasses import dataclass, field
from datetime import timedelta


@dataclass
class Location:
    name: str

    capacity: int

    closing_hour: timedelta

    clients: int = field(default=0)

    def close(self):
        self.clients = r.randint(0, self.capacity)
