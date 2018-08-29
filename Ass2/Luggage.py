from dataclasses import dataclass


@dataclass
class Luggage:
    weight: int

    # Hash needed for using set method on list of luggages.
    def __hash__(self):
        return hash(self.weight)
